#!/usr/bin/env python3
"""
Lens Factory — Generate concept steering lenses for ComfyUI Concept Steer.

Creates direction vectors that steer image generation toward learned aesthetic
concepts. Two extraction methods:

  DPO:  Optimize a separating hyperplane between positive/negative embeddings.
        Fast, simple, works well. Operates on output embeddings only.

  SAE:  Train a Sparse Autoencoder on the text encoder's residual stream,
        decompose activations into interpretable features, find which features
        fire differentially for the concept, then reconstruct a clean direction
        from those features. Optionally refine with DPO for robust separation.
        Gives interpretable, disentangled concept directions.

Modes:
  1. auto       : From a built-in concept preset (cinematic, ethereal, etc.)
  2. text-pairs : From explicit positive/negative text pairs (JSON file)
  3. few-shot   : From example images (positive directory + optional negatives)
  4. sae        : SAE-decomposed direction from contrastive pairs (interpretable)

Usage:
  python lens_factory.py auto cinematic --target zimage
  python lens_factory.py sae cinematic --target zimage --sae-features 30
  python lens_factory.py text-pairs pairs.json --concept mystyle --target zimage
  python lens_factory.py few-shot ./positive_images/ --concept my_style
  python lens_factory.py list-presets
  python lens_factory.py list-lenses
  python lens_factory.py batch-all --target zimage --method sae
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
LENS_DIR = PACKAGE_ROOT / "lenses"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model paths (override via environment variables) ─────────────────────────

QWEN_ENCODER_PATH = os.environ.get(
    "QWEN_ENCODER_PATH",
    "",  # User must set this or pass --encoder-path
)
SIGLIP_MODEL_ID = os.environ.get(
    "SIGLIP_MODEL_ID", "google/siglip-base-patch16-224")
HF_CACHE = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

SIGLIP_DIM = 768
QWEN_HIDDEN_DIM = 2560


# ═════════════════════════════════════════════════════════════════════════════
#  Core Direction Training
# ═════════════════════════════════════════════════════════════════════════════

def train_dpo_direction(
    h_positive: torch.Tensor,
    h_negative: torch.Tensor,
    dim: int,
    betas: list[float] = [0.1, 0.3, 0.5, 1.0, 2.0],
    steps: int = 5000,
    lr: float = 5e-3,
    min_accuracy: float = 0.98,
    verbose: bool = True,
) -> dict:
    """Train a DPO direction that separates positive from negative embeddings.

    Sweeps over beta values and selects the one with the highest minimum margin
    (most robust separation) among those achieving >= min_accuracy.

    Args:
        h_positive: (N, dim) positive concept embeddings
        h_negative: (N, dim) negative concept embeddings
        dim: embedding dimension
        betas: DPO temperature values to sweep
        steps: optimization steps per beta
        lr: learning rate
        min_accuracy: minimum accuracy threshold
        verbose: print progress

    Returns:
        dict with 'direction', 'beta', 'accuracy', 'mean_margin', 'min_margin'
    """
    if verbose:
        print(f"  DPO training: {h_positive.shape[0]} pairs x {dim}d")

    results = {}
    # Exit inference_mode — ComfyUI wraps node execution in inference_mode()
    # which is stricter than no_grad and cannot be overridden by enable_grad().
    # detach().clone() MUST happen inside this block — cloning an inference
    # tensor outside inference_mode(False) still produces an inference tensor.
    with torch.inference_mode(False):
        h_pref = h_positive.detach().clone().float().to(DEVICE)
        h_rej = h_negative.detach().clone().float().to(DEVICE)

        for beta in betas:
            d = torch.randn(dim, device=DEVICE, requires_grad=True)
            d.data = F.normalize(d.data, dim=0)
            opt = torch.optim.Adam([d], lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=steps)

            best_loss = float("inf")
            for step in range(steps):
                d_norm = F.normalize(d, dim=0)
                margins = (h_pref @ d_norm) - (h_rej @ d_norm)
                loss = -torch.log(torch.sigmoid(beta * margins) + 1e-8).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                scheduler.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()

            d_final = F.normalize(d.detach(), dim=0)
            with torch.no_grad():
                final_margins = (h_pref @ d_final) - (h_rej @ d_final)
                acc = (final_margins > 0).float().mean().item()
                mean_m = final_margins.mean().item()
                min_m = final_margins.min().item()

            results[beta] = {
                "direction": d_final.cpu(),
                "accuracy": acc,
                "mean_margin": mean_m,
                "min_margin": min_m,
                "final_loss": best_loss,
            }
            if verbose:
                print(
                    f"    beta={beta:.1f}: acc={acc:.0%}  mean={mean_m:+.3f}  min={min_m:+.3f}")

    # Select best beta: highest min_margin among those with accuracy >= threshold
    candidates = [b for b, r in results.items() if r["accuracy"]
                  >= min_accuracy]
    if not candidates:
        candidates = list(results.keys())
    best_beta = max(candidates, key=lambda b: results[b]["min_margin"])
    r = results[best_beta]

    if verbose:
        print(f"  Best beta={best_beta}: acc={r['accuracy']:.0%}, "
              f"min_margin={r['min_margin']:+.3f}, mean_margin={r['mean_margin']:+.3f}")

    return {
        "direction": r["direction"],
        "beta": best_beta,
        "accuracy": r["accuracy"],
        "mean_margin": r["mean_margin"],
        "min_margin": r["min_margin"],
        "all_results": results,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Sparse Autoencoder
# ═════════════════════════════════════════════════════════════════════════════

class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for decomposing residual stream activations.

    Learns to reconstruct hidden states through a sparse bottleneck,
    revealing interpretable features in the activation space.

    Architecture:
      - Pre-encoder centering: subtract decoder bias (learned data mean)
      - Encoder: Linear + ReLU → sparse activations
      - Decoder: Linear (unit-norm columns enforced during training)
      - decode_sparse(): Linear WITHOUT bias → bias-free concept directions

    Args:
        d_input: Input dimension (model hidden_size)
        d_sae: SAE dimension (typically 8x expansion for rich features)
        l1_coeff: L1 sparsity penalty coefficient
    """

    def __init__(self, d_input: int, d_sae: int, l1_coeff: float = 1e-2):
        super().__init__()
        self.d_input = d_input
        self.d_sae = d_sae
        self.l1_coeff = l1_coeff
        self.encoder = nn.Linear(d_input, d_sae)
        self.decoder = nn.Linear(d_sae, d_input)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.encoder.weight)
            self.decoder.weight.copy_(self.encoder.weight.T)
            self.decoder.weight.data /= (
                self.decoder.weight.data.norm(
                    dim=0, keepdim=True).clamp(min=1e-8)
            )
            nn.init.zeros_(self.encoder.bias)
            nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input, pre-centering by subtracting decoder bias."""
        return F.relu(self.encoder(x - self.decoder.bias))

    def decode_sparse(self, z: torch.Tensor) -> torch.Tensor:
        """Decode WITHOUT bias — pure concept direction reconstruction.

        Used for extracting concept directions that aren't contaminated
        by the learned data mean stored in the decoder bias.
        """
        return F.linear(z, self.decoder.weight, bias=None)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decoder(z), z


def train_sae(
    data: torch.Tensor,
    d_sae: int,
    l1_coeff: float = 1e-2,
    epochs: int = 200,
    lr: float = 5e-4,
    batch_size: int = 512,
    label: str = "sae",
    verbose: bool = True,
) -> SparseAutoencoder:
    """Train a Sparse Autoencoder on activation data.

    Applies:
      - L1 warmup over first 20% of training (prevents premature feature death)
      - Unit-norm decoder columns after each step (prevents L1 cheating)
      - Cosine LR schedule

    Args:
        data: (N, d_input) activation vectors
        d_sae: SAE hidden dimension (8x expansion recommended)
        l1_coeff: sparsity penalty strength
        epochs: training epochs
        lr: learning rate
        batch_size: training batch size
        label: display label for progress logging
        verbose: print training progress

    Returns:
        Trained SparseAutoencoder in eval mode
    """
    d_input = data.shape[1]

    if verbose:
        print(
            f"  Training SAE: {d_input}d → {d_sae}d ({data.shape[0]:,} vectors, {epochs} epochs)")

    # Exit inference_mode — ComfyUI wraps node execution in inference_mode()
    # which is stricter than no_grad and cannot be overridden by enable_grad().
    # ALL nn.Module creation AND detach().clone() must happen inside this block —
    # cloning an inference tensor outside still produces an inference tensor.
    with torch.inference_mode(False):
        data = data.detach().clone()

        sae = SparseAutoencoder(d_input, d_sae, l1_coeff).to(DEVICE)
        opt = torch.optim.Adam(sae.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=lr * 0.1)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        for epoch in range(1, epochs + 1):
            total_mse = total_l1 = 0
            l1_scale = min(1.0, epoch / (epochs * 0.2)
                           )  # warmup over first 20%

            for (batch,) in loader:
                batch = batch.to(DEVICE)
                x_hat, z = sae(batch)
                mse = (x_hat - batch).pow(2).mean()
                l1 = z.abs().mean() * l1_coeff * l1_scale
                (mse + l1).backward()
                opt.step()
                opt.zero_grad()

                # Enforce unit-norm decoder columns
                with torch.no_grad():
                    norms = sae.decoder.weight.data.norm(
                        dim=0, keepdim=True).clamp(min=1e-8)
                    sae.decoder.weight.data /= norms

                total_mse += mse.item()
                total_l1 += l1.item()

            sched.step()

            if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == epochs):
                n = len(loader)
                with torch.no_grad():
                    z_check = sae.encode(data[:2000].to(DEVICE))
                    l0 = (z_check > 0).float().sum(1).mean()
                    dead = (z_check.sum(0) == 0).sum()
                    cos_sim = F.cosine_similarity(
                        data[:2000].to(DEVICE), sae.decoder(z_check)
                    ).mean()
                print(
                    f"    [{label}] {epoch:>3d}/{epochs}: mse={total_mse/n:.6f} "
                    f"l1={total_l1/n:.6f} L0={l0:.0f}/{d_sae} dead={dead} cos={cos_sim:.4f}"
                )

    return sae.eval()


# ═════════════════════════════════════════════════════════════════════════════
#  Residual Stream Activation Collection
# ═════════════════════════════════════════════════════════════════════════════

# Diverse prompts for collecting representative activations
_SAE_TOPICS = [
    "quantum physics", "medieval history", "cooking recipes", "machine learning",
    "philosophy of mind", "space exploration", "gardening tips", "jazz music",
    "climate change", "ancient mythology", "software engineering", "marine biology",
    "romantic poetry", "cryptocurrency", "yoga and meditation", "civil engineering",
    "horror fiction", "child psychology", "organic chemistry", "political science",
    "abstract art", "veterinary medicine", "renewable energy", "game theory",
    "linguistics", "astrophysics", "culinary arts", "neuroscience",
    "architecture", "environmental law", "data visualization", "anthropology",
    "fashion design", "volcanology", "cryptography", "musical composition",
    "forensic science", "urban planning", "calligraphy", "astrobiology",
]
_SAE_STARTERS = [
    "Explain", "Describe", "What is", "Tell me about", "How does",
    "Why is", "Discuss", "Compare", "Analyze", "Write about",
    "The history of", "An introduction to", "The importance of",
    "A detailed look at", "Understanding", "Exploring",
]
_SAE_TONES = [
    "", "in simple terms", "for a scientist", "enthusiastically",
    "critically", "with examples", "briefly", "in detail",
    "from a historical perspective", "with humor",
]


def generate_diverse_prompts(n: int = 500, seed: int = 42) -> list[str]:
    """Generate diverse prompts for collecting representative activations.

    Uses combinatorial explosion of topics × starters × tones to
    create a broad distribution of language patterns.
    """
    import random as _rng

    rng = _rng.Random(seed)
    prompts: set[str] = set()
    while len(prompts) < n:
        starter = rng.choice(_SAE_STARTERS)
        topic = rng.choice(_SAE_TOPICS)
        tone = rng.choice(_SAE_TONES)
        prompts.add(f"{starter} {topic} {tone}".strip())
    return sorted(prompts)[:n]


def collect_layer_activations(
    texts: list[str],
    model,
    tokenizer,
    layer_idx: int,
    hidden_dim: int,
    max_length: int = 64,
    pool: str = "all",
    verbose: bool = True,
) -> torch.Tensor:
    """Collect activations from a specific transformer layer.

    Hooks into the residual stream at `layer_idx` and runs texts forward,
    collecting hidden states for SAE training.

    Args:
        texts: input texts to process
        model: the language model (e.g. Qwen3Model)
        tokenizer: corresponding tokenizer
        layer_idx: which layer to hook (typically ~60% depth)
        hidden_dim: model's hidden dimension
        max_length: max sequence length for tokenization
        pool: 'all' = all token activations (for SAE training)
              'mean' = mean-pool per text (for concept extraction)
              'last' = last token per text

    Returns:
        Tensor of activations shaped per pool mode
    """
    collected = []

    def _hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        collected.append(h.detach().cpu().float())

    handle = model.layers[layer_idx].register_forward_hook(_hook)

    if verbose:
        print(
            f"  Collecting activations from layer {layer_idx} ({len(texts)} texts)...")

    for i, text in enumerate(texts):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(DEVICE)
        with torch.no_grad():
            model(inputs.input_ids, attention_mask=inputs.attention_mask)
        if verbose and (i + 1) % 100 == 0:
            n_vecs = sum(a.shape[0] * a.shape[1] for a in collected)
            print(f"    {i+1}/{len(texts)} prompts, {n_vecs:,} vectors")

    handle.remove()

    if pool == "all":
        result = torch.cat([a.reshape(-1, hidden_dim) for a in collected])
    elif pool == "mean":
        result = torch.stack([a.squeeze(0).mean(0) for a in collected])
    elif pool == "last":
        result = torch.stack([a.squeeze(0)[-1] for a in collected])
    else:
        raise ValueError(f"Unknown pool mode: {pool}")

    if verbose:
        print(
            f"  Collected: {result.shape} (norm={result.norm(dim=-1).mean():.2f})")
    return result


def extract_concept_features(
    positive_texts: list[str],
    negative_texts: list[str],
    sae_model: SparseAutoencoder,
    model,
    tokenizer,
    layer_idx: int,
    hidden_dim: int,
    top_k: int = 30,
    verbose: bool = True,
) -> dict:
    """Extract concept direction via SAE feature decomposition.

    Pipeline:
      1. Encode positive & negative texts → layer activations (mean-pooled)
      2. Pass activations through SAE encoder → sparse feature activations
      3. Compute differential features: pos_feats - neg_feats
      4. Select top-K features by absolute differential magnitude
      5. Create sparse vector with only those features
      6. Decode via decode_sparse() (no bias) → clean concept direction
      7. Scale to match raw activation magnitude

    This gives you an *interpretable* direction — you know exactly which
    SAE features contribute and by how much.

    Args:
        positive_texts: texts embodying the concept
        negative_texts: texts NOT embodying the concept
        sae_model: trained SparseAutoencoder
        model: the language model
        tokenizer: model's tokenizer
        layer_idx: which layer to hook
        hidden_dim: model hidden dimension
        top_k: number of differential features to keep
        verbose: print progress

    Returns:
        dict with 'direction', 'raw_direction', 'feature_indices',
        'feature_weights', 'cos_raw_sae', 'top_k'
    """
    if verbose:
        print(f"  Extracting concept features (top-{top_k} from SAE)...")

    # 1. Collect layer activations (mean-pooled per text)
    pos_acts = collect_layer_activations(
        positive_texts, model, tokenizer, layer_idx, hidden_dim,
        pool="mean", verbose=False,
    )
    neg_acts = collect_layer_activations(
        negative_texts, model, tokenizer, layer_idx, hidden_dim,
        pool="mean", verbose=False,
    )

    # Raw direction (for magnitude reference)
    raw_dir = pos_acts.mean(0) - neg_acts.mean(0)

    # 2. Encode through SAE
    _dev = next(sae_model.parameters()).device
    with torch.no_grad():
        pos_feats = sae_model.encode(pos_acts.to(_dev)).cpu()
        neg_feats = sae_model.encode(neg_acts.to(_dev)).cpu()

    # 3. Differential features
    diff = pos_feats.mean(0) - neg_feats.mean(0)

    # 4. Top-K by absolute magnitude
    top_idx = torch.argsort(-diff.abs())[:top_k]
    sparse = torch.zeros_like(diff)
    sparse[top_idx] = diff[top_idx]

    # 5. Decode sparse (bias-free)
    with torch.no_grad():
        sae_dir = sae_model.decode_sparse(
            sparse.unsqueeze(0).to(_dev)).cpu()[0]

    # 6. Scale to match raw magnitude
    sae_dir_scaled = sae_dir * \
        (raw_dir.norm() / sae_dir.norm().clamp(min=1e-8))
    unit_dir = sae_dir / sae_dir.norm().clamp(min=1e-8)

    cos_sim = (F.normalize(raw_dir, dim=0) @ unit_dir).item()

    if verbose:
        print(f"    Top-{top_k} features selected")
        print(f"    Raw direction norm: {raw_dir.norm():.2f}")
        print(f"    SAE direction norm: {sae_dir.norm():.2f}")
        print(f"    cos(raw, sae): {cos_sim:.3f}")
        # Show top 5 features
        sorted_idx = torch.argsort(-diff[top_idx].abs())[:5]
        print(f"    Top-5 features: {[int(top_idx[i]) for i in sorted_idx]}")
        print(
            f"    Top-5 weights:  {[f'{diff[top_idx[i]]:.3f}' for i in sorted_idx]}")

    return {
        "direction": unit_dir,
        "raw_direction": sae_dir_scaled,
        "feature_indices": top_idx,
        "feature_weights": diff[top_idx],
        "cos_raw_sae": cos_sim,
        "top_k": top_k,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Encoder Loading
# ═════════════════════════════════════════════════════════════════════════════

_siglip_model = None
_siglip_processor = None
_qwen_encoder = None
_qwen_tokenizer = None


def load_siglip():
    """Load SigLIP model + processor (cached across calls)."""
    global _siglip_model, _siglip_processor
    if _siglip_model is not None:
        return _siglip_model, _siglip_processor

    from transformers import AutoModel, AutoProcessor

    print(f"Loading SigLIP ({SIGLIP_MODEL_ID})...")
    _siglip_model = AutoModel.from_pretrained(
        SIGLIP_MODEL_ID, cache_dir=HF_CACHE
    ).to(DEVICE).eval()
    _siglip_processor = AutoProcessor.from_pretrained(
        SIGLIP_MODEL_ID, cache_dir=HF_CACHE
    )
    print(f"  SigLIP loaded ({SIGLIP_DIM}d)")
    return _siglip_model, _siglip_processor


def load_qwen_encoder(encoder_path: str = ""):
    """Load Qwen 3.4B text encoder from safetensors (cached across calls).

    Args:
        encoder_path: Path to qwen_3_4b.safetensors. Falls back to
                      QWEN_ENCODER_PATH env var (re-read at call time).
    """
    global _qwen_encoder, _qwen_tokenizer
    if _qwen_encoder is not None:
        return _qwen_encoder, _qwen_tokenizer

    from transformers import Qwen3Config, Qwen3Model, AutoTokenizer
    from safetensors import safe_open

    # Re-read env var at call time so nodes that set it after import work
    path = encoder_path or os.environ.get(
        "QWEN_ENCODER_PATH", "") or QWEN_ENCODER_PATH

    # Auto-discover from common locations if not explicitly set
    if not path or not os.path.isfile(path):
        _search_paths = [
            # Relative to ComfyUI install (custom_nodes/../models/)
            *[
                str(p)
                for comfy_root in [
                    Path(__file__).resolve().parent.parent.parent,
                    Path(os.environ.get("COMFYUI_PATH", "")),
                ]
                if comfy_root and comfy_root.is_dir()
                for p in [
                    comfy_root / "models" / "text_encoders" / "qwen_3_4b.safetensors",
                    comfy_root / "models" / "clip" / "qwen_3_4b.safetensors",
                ]
            ],
            # User home directories
            str(Path.home() / "Models" / "text_encoders" / "qwen_3_4b.safetensors"),
            str(Path.home() / ".cache" / "comfyui" / "models" /
                "text_encoders" / "qwen_3_4b.safetensors"),
        ]
        for candidate in _search_paths:
            if candidate and os.path.isfile(candidate):
                path = candidate
                print(f"  Auto-discovered Qwen encoder: {path}")
                break

    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"Qwen encoder not found at '{path}'. "
            f"Set QWEN_ENCODER_PATH env var or pass --encoder-path.\n"
            f"  Searched: {', '.join(p for p in _search_paths if p)}"
        )

    print(f"Loading Qwen 3.4B from {Path(path).name}...")
    t0 = time.time()

    config = Qwen3Config(
        hidden_size=QWEN_HIDDEN_DIM,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=9728,
        vocab_size=151936,
        head_dim=128,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
    )

    _qwen_encoder = Qwen3Model(config)
    sf = safe_open(path, framework="pt")
    state = {}
    for key in sf.keys():
        new_key = key.replace(
            "model.", "") if key.startswith("model.") else key
        state[new_key] = sf.get_tensor(key)

    missing, unexpected = _qwen_encoder.load_state_dict(state, strict=False)
    assert len(missing) == 0, f"Missing keys: {missing[:5]}"
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected[:5]}"
    del state, sf

    _qwen_encoder = _qwen_encoder.to(DEVICE, dtype=torch.bfloat16).eval()

    _qwen_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct", cache_dir=HF_CACHE
    )

    print(f"  Qwen 3.4B loaded in {time.time() - t0:.1f}s")
    return _qwen_encoder, _qwen_tokenizer


# ═════════════════════════════════════════════════════════════════════════════
#  Text Encoding
# ═════════════════════════════════════════════════════════════════════════════

def encode_texts_qwen(texts: list[str], batch_size: int = 8, max_length: int = 256) -> torch.Tensor:
    """Encode texts through Qwen 3.4B -> mean-pooled 2560d embeddings."""
    model, tokenizer = load_qwen_encoder()
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(DEVICE)
            out = model(inputs.input_ids, attention_mask=inputs.attention_mask)
            hidden = out.last_hidden_state.float()
            mask = inputs.attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            all_embeds.append(pooled.cpu())
    return torch.cat(all_embeds, dim=0)


def encode_texts_siglip(texts: list[str], batch_size: int = 16) -> torch.Tensor:
    """Encode texts through SigLIP -> 768d embeddings."""
    model, processor = load_siglip()
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [t[:300] for t in texts[i: i + batch_size]]
            inp = processor(
                text=batch, return_tensors="pt",
                padding=True, truncation=True, max_length=64,
            ).to(DEVICE)
            out = model.get_text_features(**inp)
            if hasattr(out, "pooler_output"):
                feats = out.pooler_output
            elif hasattr(out, "shape"):
                feats = out
            else:
                text_out = model.text_model(**inp)
                feats = text_out.pooler_output
            all_embeds.append(feats.float().cpu())
    return torch.cat(all_embeds, dim=0)


# ═════════════════════════════════════════════════════════════════════════════
#  Image Encoding (Few-Shot)
# ═════════════════════════════════════════════════════════════════════════════

def encode_images_siglip(image_paths: list[str], batch_size: int = 8) -> torch.Tensor:
    """Encode images through SigLIP vision encoder -> 768d embeddings."""
    from PIL import Image

    model, processor = load_siglip()
    all_embeds = []

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i: i + batch_size]
            images = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                except Exception as e:
                    print(f"  Skipping {Path(p).name}: {e}")
            if not images:
                continue
            inp = processor(images=images, return_tensors="pt").to(DEVICE)
            feats = model.get_image_features(**inp)
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            elif not hasattr(feats, "shape"):
                feats = model.vision_model(**inp).pooler_output
            all_embeds.append(feats.float().cpu())

    if not all_embeds:
        raise ValueError("No images could be loaded!")
    return torch.cat(all_embeds, dim=0)


def collect_images(
    directory: str,
    extensions: set = {".jpg", ".jpeg", ".png", ".webp", ".bmp"},
) -> list[str]:
    """Recursively collect image paths from a directory."""
    images = []
    dirpath = Path(directory)
    if not dirpath.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    for f in sorted(dirpath.rglob("*")):
        if f.suffix.lower() in extensions and f.is_file():
            images.append(str(f))
    return images


# ═════════════════════════════════════════════════════════════════════════════
#  Cross-Modal Bridge (SigLIP -> Qwen)
# ═════════════════════════════════════════════════════════════════════════════

class ProjectionHead(nn.Module):
    """Projection from SigLIP (768d) -> Qwen (2560d) space."""

    def __init__(self, in_dim: int = 768, out_dim: int = 2560, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def train_siglip_bridge(
    sig_embeds: torch.Tensor,
    qwen_embeds: torch.Tensor,
    epochs: int = 300,
    lr: float = 3e-4,
    temperature: float = 0.07,
    verbose: bool = True,
) -> ProjectionHead:
    """Train contrastive projection SigLIP -> Qwen space.

    Uses InfoNCE loss to align SigLIP and Qwen embeddings for the same texts.
    """
    if verbose:
        print(
            f"  Training SigLIP->Qwen bridge ({sig_embeds.shape[0]} pairs, {epochs} epochs)...")

    # Exit inference_mode — ComfyUI wraps node execution in inference_mode()
    # which is stricter than no_grad and cannot be overridden by enable_grad().
    # ALL nn.Module creation AND detach().clone() must happen inside this block —
    # cloning an inference tensor outside still produces an inference tensor.
    with torch.inference_mode(False):
        sig_train = sig_embeds.detach().clone().to(DEVICE)
        qwen_target = F.normalize(
            qwen_embeds.detach().clone().to(DEVICE), dim=-1)

        proj = ProjectionHead(SIGLIP_DIM, QWEN_HIDDEN_DIM,
                              hidden_dim=1024).to(DEVICE)
        optimizer = torch.optim.AdamW(
            proj.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)

        for epoch in range(epochs):
            z_sig = proj(sig_train)
            logits = (z_sig @ qwen_target.T) / temperature
            labels = torch.arange(len(sig_train), device=DEVICE)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if verbose and (epoch + 1) % 100 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}")

    proj.eval()
    if verbose:
        with torch.no_grad():
            z_projected = proj(sig_train).cpu()
            sims = z_projected @ F.normalize(qwen_embeds, dim=-1).T
            top1 = (sims.argmax(dim=1) == torch.arange(
                len(sims))).float().mean().item()
            print(f"  Bridge trained — top-1 retrieval: {top1:.0%}")

    return proj


# ═════════════════════════════════════════════════════════════════════════════
#  Lens Generators
# ═════════════════════════════════════════════════════════════════════════════

def generate_lens_from_text_pairs(
    concept: str,
    positive_texts: list[str],
    negative_texts: list[str],
    target: str = "zimage",
    include_bridge: bool = True,
    output_dir: Optional[Path] = None,
    dpo_steps: int = 5000,
) -> Path:
    """Generate a lens from explicit positive/negative text pairs.

    Args:
        concept: concept name (e.g. "cinematic")
        positive_texts: texts embodying the concept
        negative_texts: neutral/opposite texts (same count)
        target: "zimage" (2560d Qwen) or "sd15" (768d SigLIP)
        include_bridge: train SigLIP->Qwen bridge (for cross-modal use)
        output_dir: override output directory
        dpo_steps: DPO optimization steps

    Returns:
        Path to exported lens file
    """
    assert len(positive_texts) == len(negative_texts), \
        f"Need equal positive/negative pairs: {len(positive_texts)} vs {len(negative_texts)}"

    print(f"\n{'='*60}")
    print(f"  Lens Factory: '{concept}' ({target})")
    print(f"  {len(positive_texts)} text pairs")
    print(f"{'='*60}\n")

    t0 = time.time()

    if target == "zimage":
        print("[1/4] Encoding positive texts through Qwen 3.4B...")
        h_pos = encode_texts_qwen(positive_texts)
        print("[2/4] Encoding negative texts through Qwen 3.4B...")
        h_neg = encode_texts_qwen(negative_texts)

        print("[3/4] Training DPO direction (2560d)...")
        dpo = train_dpo_direction(
            h_pos, h_neg, dim=QWEN_HIDDEN_DIM, steps=dpo_steps)

        bridge_data = {}
        if include_bridge:
            print("[4/4] Training SigLIP -> Qwen bridge...")
            all_texts = []
            for p, n in zip(positive_texts, negative_texts):
                all_texts.append(p)
                all_texts.append(n)
            sig_embeds = encode_texts_siglip(all_texts)
            qwen_interleaved = torch.zeros(len(all_texts), QWEN_HIDDEN_DIM)
            qwen_interleaved[0::2] = h_pos
            qwen_interleaved[1::2] = h_neg
            proj = train_siglip_bridge(sig_embeds, qwen_interleaved)

            with torch.no_grad():
                proj_pos = proj(sig_embeds[0::2].to(DEVICE))
                proj_neg = proj(sig_embeds[1::2].to(DEVICE))
                d_dev = dpo["direction"].to(DEVICE)
                cross_margins = (proj_pos @ d_dev) - (proj_neg @ d_dev)
                cross_acc = (cross_margins > 0).float().mean().item()
                print(f"  Cross-modal direction transfer: {cross_acc:.0%}")

            bridge_data = {
                "proj_sig2qwen_state": proj.cpu().state_dict(),
                "proj_sig2qwen_config": {
                    "in_dim": SIGLIP_DIM,
                    "out_dim": QWEN_HIDDEN_DIM,
                    "hidden_dim": 1024,
                },
                "cross_modal_direction_acc": cross_acc,
                "siglip_model": SIGLIP_MODEL_ID,
                "siglip_dim": SIGLIP_DIM,
            }
        else:
            print("[4/4] Skipping bridge (not requested)")

        lens_data = {
            "direction": dpo["direction"],
            "direction_dim": QWEN_HIDDEN_DIM,
            "dpo_beta": dpo["beta"],
            "dpo_accuracy": dpo["accuracy"],
            "dpo_mean_margin": dpo["mean_margin"],
            "dpo_min_margin": dpo["min_margin"],
            "encoder_name": "qwen_3_4b",
            "encoder_hidden_dim": QWEN_HIDDEN_DIM,
            "encoder_type": "Qwen3Model",
            "encoder_layers": 36,
            "target_model": "z_image_turbo",
            "cap_embedder_out_dim": 3840,
            "concept": concept,
            "n_training_pairs": len(positive_texts),
            "training_mode": "text_pairs",
            **bridge_data,
        }
        out_dir = output_dir or (LENS_DIR / "zimage")

    elif target == "sd15":
        print("[1/3] Encoding positive texts through SigLIP...")
        h_pos = encode_texts_siglip(positive_texts)
        print("[2/3] Encoding negative texts through SigLIP...")
        h_neg = encode_texts_siglip(negative_texts)

        print("[3/3] Training DPO direction (768d)...")
        dpo = train_dpo_direction(
            h_pos, h_neg, dim=SIGLIP_DIM, steps=dpo_steps)

        lens_data = {
            "direction": dpo["direction"],
            "direction_dim": SIGLIP_DIM,
            "dpo_beta": dpo["beta"],
            "dpo_accuracy": dpo["accuracy"],
            "dpo_mean_margin": dpo["mean_margin"],
            "dpo_min_margin": dpo["min_margin"],
            "encoder_name": "siglip-base-patch16-224",
            "encoder_dim": SIGLIP_DIM,
            "target_model": "sd15",
            "concept": concept,
            "n_training_pairs": len(positive_texts),
            "training_mode": "text_pairs",
        }
        out_dir = output_dir or (LENS_DIR / "sd15")

    else:
        raise ValueError(f"Unknown target: {target}. Use 'zimage' or 'sd15'.")

    # Save lens
    out_dir.mkdir(parents=True, exist_ok=True)
    lens_name = f"{concept}_{target}_dpo"
    lens_path = out_dir / f"{lens_name}.pt"
    torch.save(lens_data, lens_path)

    # Metadata JSON
    meta = {
        "concept": concept,
        "target": target,
        "direction_dim": lens_data["direction_dim"],
        "training_mode": "text_pairs",
        "n_pairs": len(positive_texts),
        "dpo_beta": dpo["beta"],
        "dpo_accuracy": dpo["accuracy"],
        "dpo_mean_margin": dpo["mean_margin"],
        "dpo_min_margin": dpo["min_margin"],
        "training_time_s": round(time.time() - t0, 1),
    }
    meta_path = out_dir / f"{lens_name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nLens exported: {lens_path}")
    print(f"  Size: {lens_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Total time: {time.time() - t0:.1f}s")
    return lens_path


def generate_lens_from_images(
    concept: str,
    positive_dir: str,
    negative_dir: Optional[str] = None,
    target: str = "sd15",
    n_negative_random: int = 0,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate a lens from few-shot example images.

    Computes mean SigLIP embedding of positive images, subtracts mean of
    negatives (or origin), and normalizes to get a direction vector.

    For 'zimage' target, projects through a pre-trained SigLIP->Qwen bridge
    if one exists in an existing lens file.
    """
    print(f"\n{'='*60}")
    print(f"  Lens Factory (Few-Shot): '{concept}' ({target})")
    print(f"{'='*60}\n")

    t0 = time.time()

    pos_images = collect_images(positive_dir)
    print(
        f"[1/3] Found {len(pos_images)} positive images — encoding with SigLIP...")
    if len(pos_images) < 2:
        raise ValueError(
            f"Need at least 2 positive images, found {len(pos_images)}")
    h_pos = encode_images_siglip(pos_images)
    mean_pos = F.normalize(h_pos.mean(dim=0), dim=0)
    print(f"  Positive centroid norm: {h_pos.mean(0).norm():.3f}")
    print(
        f"  Intra-positive cosine: {F.cosine_similarity(h_pos, mean_pos.unsqueeze(0)).mean():.3f}")

    if negative_dir:
        neg_images = collect_images(negative_dir)
        print(f"[2/3] Found {len(neg_images)} negative images — encoding...")
        h_neg = encode_images_siglip(neg_images)
        mean_neg = F.normalize(h_neg.mean(dim=0), dim=0)
    elif n_negative_random > 0:
        print(
            f"[2/3] Generating {n_negative_random} random negative vectors...")
        h_neg = torch.randn(n_negative_random, SIGLIP_DIM)
        h_neg = F.normalize(h_neg, dim=-1)
        mean_neg = F.normalize(h_neg.mean(dim=0), dim=0)
    else:
        print("[2/3] No negatives — using origin as contrast")
        h_neg = None
        mean_neg = torch.zeros(SIGLIP_DIM)

    raw_direction = mean_pos - mean_neg
    direction = F.normalize(raw_direction, dim=0)

    pos_scores = h_pos @ direction
    if negative_dir:
        neg_scores = h_neg @ direction
        acc = ((pos_scores > neg_scores.mean()).float().mean().item())
    else:
        acc = (pos_scores > 0).float().mean().item()
    print(f"  Direction accuracy: {acc:.0%}")
    print(f"  Positive mean score: {pos_scores.mean():.3f}")

    if target == "sd15":
        print("[3/3] Exporting SigLIP-space lens (768d)...")
        lens_data = {
            "direction": direction,
            "d_in_siglip": direction,
            "direction_dim": SIGLIP_DIM,
            "encoder_name": "siglip-base-patch16-224",
            "concept": concept,
            "n_positive_images": len(pos_images),
            "n_negative_images": len(neg_images) if negative_dir else 0,
            "training_mode": "few_shot_images",
            "accuracy": acc,
            "target_model": "sd15",
        }
        out_dir = output_dir or (LENS_DIR / "sd15")
        lens_name = f"{concept}_sd15_fewshot"

    elif target == "zimage":
        print("[3/3] Projecting to Qwen space for Z Image...")
        # Look for an existing bridge in any zimage lens
        bridge_found = False
        zimage_dir = LENS_DIR / "zimage"
        if zimage_dir.is_dir():
            for lens_file in sorted(zimage_dir.glob("*.pt")):
                try:
                    existing = torch.load(
                        lens_file, map_location="cpu", weights_only=False)
                    if "proj_sig2qwen_state" in existing:
                        print(f"  Loading bridge from {lens_file.name}...")
                        proj = ProjectionHead(
                            SIGLIP_DIM, QWEN_HIDDEN_DIM, hidden_dim=1024)
                        proj.load_state_dict(existing["proj_sig2qwen_state"])
                        proj = proj.to(DEVICE).eval()
                        with torch.no_grad():
                            direction_qwen = proj(direction.unsqueeze(
                                0).to(DEVICE)).squeeze(0).cpu()
                        direction_qwen = F.normalize(direction_qwen, dim=0)
                        print(
                            f"  Projected direction to {QWEN_HIDDEN_DIM}d via bridge")
                        bridge_found = True
                        break
                except Exception:
                    continue

        if not bridge_found:
            print("  No bridge found — storing SigLIP direction only")
            direction_qwen = direction

        lens_data = {
            "direction": direction_qwen,
            "d_in_siglip": direction,
            "direction_dim": direction_qwen.shape[0],
            "encoder_name": "qwen_3_4b" if bridge_found else "siglip",
            "concept": concept,
            "n_positive_images": len(pos_images),
            "n_negative_images": len(neg_images) if negative_dir else 0,
            "training_mode": "few_shot_images",
            "accuracy": acc,
            "target_model": "z_image_turbo",
            "siglip_model": SIGLIP_MODEL_ID,
            "siglip_dim": SIGLIP_DIM,
        }
        out_dir = output_dir or (LENS_DIR / "zimage")
        lens_name = f"{concept}_zimage_fewshot"

    else:
        raise ValueError(f"Unknown target: {target}")

    out_dir.mkdir(parents=True, exist_ok=True)
    lens_path = out_dir / f"{lens_name}.pt"
    torch.save(lens_data, lens_path)

    meta = {
        "concept": concept,
        "target": target,
        "training_mode": "few_shot_images",
        "n_positive_images": len(pos_images),
        "n_negative_images": len(neg_images) if negative_dir else 0,
        "direction_dim": int(lens_data["direction_dim"]),
        "accuracy": float(acc),
        "training_time_s": round(time.time() - t0, 1),
    }
    meta_path = out_dir / f"{lens_name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nLens exported: {lens_path}")
    print(f"  Size: {lens_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Total time: {time.time() - t0:.1f}s")
    return lens_path


# ═════════════════════════════════════════════════════════════════════════════
#  SAE Lens Generator
# ═════════════════════════════════════════════════════════════════════════════

# keyed by f"{layer}_{expansion}"
_sae_cache: dict[str, SparseAutoencoder] = {}


def generate_lens_sae(
    concept: str,
    positive_texts: list[str],
    negative_texts: list[str],
    target: str = "zimage",
    layer: Optional[int] = None,
    sae_expansion: int = 8,
    sae_epochs: int = 200,
    n_activation_prompts: int = 500,
    top_k: int = 30,
    refine_dpo: bool = True,
    dpo_steps: int = 5000,
    include_bridge: bool = True,
    output_dir: Optional[Path] = None,
    sae_save_path: Optional[Path] = None,
    sae_load_path: Optional[Path] = None,
) -> Path:
    """Generate a lens using SAE feature decomposition of the residual stream.

    Full pipeline:
      1. Load the text encoder (Qwen 3.4B)
      2. Collect diverse activations at the target layer
      3. Train a Sparse Autoencoder on those activations
      4. Run contrastive texts through the model + SAE
      5. Find differential features (which SAE features fire for + but not -)
      6. Reconstruct a clean direction from top-K features (bias-free)
      7. Optionally refine with DPO for robust separation
      8. Export lens with feature metadata for interpretability

    Args:
        concept: concept name
        positive_texts: texts embodying the concept
        negative_texts: neutral/opposite texts (same count)
        target: "zimage" (2560d Qwen) or "sd15" (768d SigLIP)
        layer: target layer to hook (default: 60% depth)
        sae_expansion: SAE hidden multiplier (default: 8x)
        sae_epochs: SAE training epochs
        n_activation_prompts: number of diverse prompts for activation collection
        top_k: number of SAE features to keep in concept direction
        refine_dpo: also run DPO on the output embeddings and blend
        dpo_steps: DPO optimization steps (if refine_dpo)
        include_bridge: train SigLIP->Qwen bridge
        output_dir: override output directory
        sae_save_path: save trained SAE to this path for reuse
        sae_load_path: load pre-trained SAE instead of training new one

    Returns:
        Path to exported lens file
    """
    assert len(positive_texts) == len(negative_texts), \
        f"Need equal positive/negative pairs: {len(positive_texts)} vs {len(negative_texts)}"

    if target != "zimage":
        raise ValueError(
            "SAE mode currently only supports 'zimage' target (Qwen 3.4B). "
            "SigLIP models are too small for meaningful SAE decomposition."
        )

    print(f"\n{'='*60}")
    print(f"  Lens Factory [SAE]: '{concept}' ({target})")
    print(f"  {len(positive_texts)} text pairs, top-{top_k} features")
    print(f"{'='*60}\n")

    t0 = time.time()

    # ── Step 1: Load Qwen encoder ────────────────────────────────────────
    print("[1/7] Loading Qwen 3.4B encoder...")
    model, tokenizer = load_qwen_encoder()

    num_layers = 36
    hidden_dim = QWEN_HIDDEN_DIM
    target_layer = layer if layer is not None else int(num_layers * 0.6)
    d_sae = hidden_dim * sae_expansion

    print(f"  Target layer: {target_layer}/{num_layers}")
    print(
        f"  SAE dimensions: {hidden_dim}d → {d_sae}d ({sae_expansion}x expansion)")

    # ── Step 2: Load or train SAE ────────────────────────────────────────
    cache_key = f"{target_layer}_{sae_expansion}"

    if sae_load_path and Path(sae_load_path).exists():
        print(f"\n[2/7] Loading pre-trained SAE from {sae_load_path}...")
        sae_model = SparseAutoencoder(hidden_dim, d_sae).to(DEVICE)
        sae_state = torch.load(
            sae_load_path, map_location=DEVICE, weights_only=True)
        sae_model.load_state_dict(sae_state)
        sae_model.eval()
        print(f"  SAE loaded ({hidden_dim}d → {d_sae}d)")
    elif cache_key in _sae_cache:
        print(
            f"\n[2/7] Using cached SAE (layer {target_layer}, {sae_expansion}x)...")
        sae_model = _sae_cache[cache_key]
    else:
        print(f"\n[2/7] Collecting activations for SAE training...")
        prompts = generate_diverse_prompts(n_activation_prompts)
        print(f"  Generated {len(prompts)} diverse prompts")

        all_acts = collect_layer_activations(
            prompts, model, tokenizer, target_layer, hidden_dim,
            pool="all", verbose=True,
        )
        print(f"  Activation matrix: {all_acts.shape}")

        print(f"\n[3/7] Training Sparse Autoencoder...")
        sae_model = train_sae(
            all_acts, d_sae,
            l1_coeff=1e-2,
            epochs=sae_epochs,
            lr=5e-4,
            batch_size=512,
            label=f"layer-{target_layer}",
        )

        # Cache for reuse across concepts in same session
        _sae_cache[cache_key] = sae_model

        # Free activation memory
        del all_acts
        gc.collect()
        torch.cuda.empty_cache()

        if sae_save_path:
            sae_save_path = Path(sae_save_path)
            sae_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sae_model.state_dict(), sae_save_path)
            print(f"  SAE saved to {sae_save_path}")

    # ── Step 3: Extract concept features via SAE ─────────────────────────
    step = 4 if cache_key not in _sae_cache or sae_load_path else 3
    print(f"\n[{step}/7] Extracting concept features via SAE decomposition...")

    sae_result = extract_concept_features(
        positive_texts, negative_texts,
        sae_model, model, tokenizer,
        target_layer, hidden_dim,
        top_k=top_k,
        verbose=True,
    )

    sae_direction = sae_result["direction"]
    print(f"  SAE concept direction extracted ({top_k} features)")

    # ── Step 4: Optional DPO refinement ──────────────────────────────────
    dpo_data = {}
    if refine_dpo:
        print(f"\n[{step+1}/7] Encoding texts for DPO refinement...")
        h_pos = encode_texts_qwen(positive_texts)
        h_neg = encode_texts_qwen(negative_texts)

        print(f"[{step+2}/7] Training DPO direction on output embeddings...")
        dpo = train_dpo_direction(
            h_pos, h_neg, dim=QWEN_HIDDEN_DIM, steps=dpo_steps)

        dpo_direction = dpo["direction"]

        # Blend: use SAE direction as the interpretable core, DPO as refinement
        # Compute cosine similarity between the two
        cos_sae_dpo = (sae_direction @ dpo_direction).item()
        print(f"  cos(SAE, DPO) = {cos_sae_dpo:.3f}")

        # Final direction: normalize the average of both (equal weight)
        # This gives an interpretable direction that also separates well
        blended = F.normalize(sae_direction + dpo_direction, dim=0)
        print(f"  Blended SAE+DPO direction")

        # Verify blend separates well
        with torch.no_grad():
            pos_scores = h_pos @ blended
            neg_scores = h_neg @ blended
            blend_acc = (pos_scores.mean() > neg_scores.mean()).float().item()
            blend_margin = (pos_scores.mean() - neg_scores.mean()).item()
            print(f"  Blended direction margin: {blend_margin:+.3f}")

        dpo_data = {
            "dpo_direction": dpo_direction,
            "dpo_beta": dpo["beta"],
            "dpo_accuracy": dpo["accuracy"],
            "dpo_mean_margin": dpo["mean_margin"],
            "dpo_min_margin": dpo["min_margin"],
            "cos_sae_dpo": cos_sae_dpo,
            "blend_margin": blend_margin,
        }

        final_direction = blended
    else:
        final_direction = sae_direction

    # ── Step 5: Optional cross-modal bridge ──────────────────────────────
    bridge_data = {}
    if include_bridge:
        print(
            f"\n[{step+3 if refine_dpo else step+1}/7] Training SigLIP -> Qwen bridge...")
        all_texts = []
        for p, n in zip(positive_texts, negative_texts):
            all_texts.append(p)
            all_texts.append(n)
        sig_embeds = encode_texts_siglip(all_texts)

        if refine_dpo:
            qwen_interleaved = torch.zeros(len(all_texts), QWEN_HIDDEN_DIM)
            qwen_interleaved[0::2] = h_pos
            qwen_interleaved[1::2] = h_neg
        else:
            qwen_interleaved_pos = encode_texts_qwen(positive_texts)
            qwen_interleaved_neg = encode_texts_qwen(negative_texts)
            qwen_interleaved = torch.zeros(len(all_texts), QWEN_HIDDEN_DIM)
            qwen_interleaved[0::2] = qwen_interleaved_pos
            qwen_interleaved[1::2] = qwen_interleaved_neg

        proj = train_siglip_bridge(sig_embeds, qwen_interleaved)

        with torch.no_grad():
            proj_pos = proj(sig_embeds[0::2].to(DEVICE))
            proj_neg = proj(sig_embeds[1::2].to(DEVICE))
            d_dev = final_direction.to(DEVICE)
            cross_margins = (proj_pos @ d_dev) - (proj_neg @ d_dev)
            cross_acc = (cross_margins > 0).float().mean().item()
            print(f"  Cross-modal direction transfer: {cross_acc:.0%}")

        bridge_data = {
            "proj_sig2qwen_state": proj.cpu().state_dict(),
            "proj_sig2qwen_config": {
                "in_dim": SIGLIP_DIM,
                "out_dim": QWEN_HIDDEN_DIM,
                "hidden_dim": 1024,
            },
            "cross_modal_direction_acc": cross_acc,
            "siglip_model": SIGLIP_MODEL_ID,
            "siglip_dim": SIGLIP_DIM,
        }

    # ── Step 6: Export lens ──────────────────────────────────────────────
    out_dir = output_dir or (LENS_DIR / "zimage")
    out_dir.mkdir(parents=True, exist_ok=True)

    method_suffix = "sae_dpo" if refine_dpo else "sae"
    lens_name = f"{concept}_{target}_{method_suffix}"
    lens_path = out_dir / f"{lens_name}.pt"

    lens_data = {
        "direction": final_direction,
        "direction_dim": QWEN_HIDDEN_DIM,
        "encoder_name": "qwen_3_4b",
        "encoder_hidden_dim": QWEN_HIDDEN_DIM,
        "encoder_type": "Qwen3Model",
        "encoder_layers": num_layers,
        "target_model": "z_image_turbo",
        "cap_embedder_out_dim": 3840,
        "concept": concept,
        "n_training_pairs": len(positive_texts),
        "training_mode": "sae" if not refine_dpo else "sae_dpo",
        # SAE metadata (interpretability)
        "sae_layer": target_layer,
        "sae_expansion": sae_expansion,
        "sae_d_sae": d_sae,
        "sae_top_k": top_k,
        "sae_feature_indices": sae_result["feature_indices"],
        "sae_feature_weights": sae_result["feature_weights"],
        "sae_direction": sae_result["direction"],
        "sae_raw_direction": sae_result["raw_direction"],
        "cos_raw_sae": sae_result["cos_raw_sae"],
        **dpo_data,
        **bridge_data,
    }

    torch.save(lens_data, lens_path)

    # Metadata JSON
    meta = {
        "concept": concept,
        "target": target,
        "direction_dim": QWEN_HIDDEN_DIM,
        "training_mode": "sae" if not refine_dpo else "sae_dpo",
        "n_pairs": len(positive_texts),
        "sae_layer": target_layer,
        "sae_expansion": sae_expansion,
        "sae_top_k": top_k,
        "sae_features": [int(i) for i in sae_result["feature_indices"].tolist()],
        "cos_raw_sae": round(sae_result["cos_raw_sae"], 4),
        "training_time_s": round(time.time() - t0, 1),
    }
    if refine_dpo:
        meta.update({
            "dpo_beta": dpo_data["dpo_beta"],
            "dpo_accuracy": dpo_data["dpo_accuracy"],
            "cos_sae_dpo": round(dpo_data["cos_sae_dpo"], 4),
            "blend_margin": round(dpo_data["blend_margin"], 4),
        })
    meta_path = out_dir / f"{lens_name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nLens exported: {lens_path}")
    print(f"  Size: {lens_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Method: SAE{' + DPO' if refine_dpo else ''}")
    print(f"  Features: {top_k} from layer {target_layer}")
    print(f"  Total time: {time.time() - t0:.1f}s")
    return lens_path


def generate_lens_sae_from_preset(
    concept: str,
    target: str = "zimage",
    layer: Optional[int] = None,
    sae_expansion: int = 8,
    sae_epochs: int = 200,
    top_k: int = 30,
    refine_dpo: bool = True,
    dpo_steps: int = 5000,
    output_dir: Optional[Path] = None,
    sae_save_path: Optional[Path] = None,
    sae_load_path: Optional[Path] = None,
) -> Path:
    """Generate an SAE lens from a built-in concept preset."""
    if concept not in CONCEPT_PRESETS:
        available = ", ".join(sorted(CONCEPT_PRESETS.keys()))
        raise ValueError(f"Unknown preset '{concept}'. Available: {available}")

    preset = CONCEPT_PRESETS[concept]
    print(f"  Using preset: {concept} — {preset['description']}")

    return generate_lens_sae(
        concept=concept,
        positive_texts=preset["positive_prompts"],
        negative_texts=preset["negative_prompts"],
        target=target,
        layer=layer,
        sae_expansion=sae_expansion,
        sae_epochs=sae_epochs,
        top_k=top_k,
        refine_dpo=refine_dpo,
        dpo_steps=dpo_steps,
        include_bridge=True,
        output_dir=output_dir,
        sae_save_path=sae_save_path,
        sae_load_path=sae_load_path,
    )


CONCEPT_PRESETS = {
    "cinematic": {
        "description": "Hollywood film-like dramatic lighting, composition, and color grading",
        "positive_prompts": [
            "A sweeping aerial shot of a misty mountain range at golden hour, with dramatic lens flare cutting through the clouds and warm amber light painting the peaks in cinematic glory",
            "A lone figure silhouetted against a massive neon-lit cityscape at night, rain-slicked streets reflecting bokeh lights in a moody, atmospheric noir composition",
            "An epic wide-angle shot of a medieval castle on a cliff edge during a thunderstorm, lightning splitting the sky, shot on anamorphic lenses with dramatic depth of field",
            "A close-up portrait with shallow depth of field, dramatic rim lighting, and cool blue shadows contrasting warm key light, like a Villeneuve film still",
            "A vast desert landscape at magic hour with a caravan stretching to the horizon, dust particles catching golden light, shot with sweeping crane movement",
            "An underwater scene with beams of light piercing through deep blue water, a diver silhouetted against the surface, beautifully color-graded like a BBC nature documentary",
            "A rainy Tokyo alley at night with neon signs reflecting in puddles, a solitary figure with an umbrella, shot with anamorphic lens distortion and cinematic grain",
            "A dramatic overhead shot of a chess board, with pieces casting long shadows from a single directional light source, extreme contrast and depth",
            "A sweeping tracking shot through a lavish ballroom, crystal chandeliers creating bokeh, warm candlelight on faces, motion blur suggesting camera movement",
            "An abandoned industrial hallway with dusty light shafts from broken windows, the texture of decaying concrete beautifully lit with cool ambient light",
        ],
        "negative_prompts": [
            "A mountain range photograph taken during the day showing peaks and clouds in the sky with normal lighting conditions",
            "A person standing in a city at night with lights visible on buildings and streets that have some wet surfaces",
            "A castle on a hillside during cloudy weather with thunder visible in the distance, taken as a standard landscape photo",
            "A portrait of a person with even studio lighting from multiple angles, standard white balance, and a neutral background",
            "A desert scene during the afternoon showing sand dunes and a group of people walking, taken with a standard lens",
            "An underwater photo showing clear blue water with a swimmer visible near the surface, natural underwater colors",
            "A street in Tokyo at night showing various shop signs and people walking, taken with a standard camera and normal settings",
            "A chess board with pieces on a table, lit by overhead room lighting, showing typical indoor shadows",
            "A ballroom interior with chandeliers and people, taken with standard event photography settings and flash",
            "A corridor in an old building with windows, showing daylight coming in, standard exposure and white balance",
        ],
    },
    "ethereal": {
        "description": "Dreamy, otherworldly, soft-focus with luminous quality and gentle light",
        "positive_prompts": [
            "A ghostly figure draped in flowing translucent fabric, floating through a misty forest glade where morning light refracts into prismatic rainbows through dewdrops",
            "An otherworldly garden where bioluminescent flowers pulse with soft blue and violet light, their petals seeming to dissolve into wisps of luminous mist",
            "A celestial being with iridescent wings spread wide, hovering above a mirror-still lake that reflects a sky filled with aurora borealis in pastel colors",
            "A dreamy double-exposure of a woman's profile filled with a blooming cherry blossom forest, soft focus creating a halo of pink light around her silhouette",
            "An ancient temple ruins overgrown with glowing moss and floating particles of golden light, as if the very air is alive with magic and memory",
            "A soft-focus underwater scene where a dancer in flowing white fabric twirls among clouds of luminescent jellyfish in deep blue water",
            "A snow-covered landscape at twilight where the sky transitions from deep indigo to pale rose, every snowflake catching light like tiny floating diamonds",
            "A bride walking through a field of lavender at sunset, her veil caught by wind creating a gossamer wave, backlit by warm golden light that makes everything glow",
            "A mystical cave interior where shafts of light illuminate crystal formations that scatter rainbow refractions across the walls like captured starlight",
            "A child reaching toward fireflies in a dusky meadow, the tiny lights creating a magical constellation around outstretched fingers, soft bokeh everywhere",
        ],
        "negative_prompts": [
            "A person standing in a forest clearing during the morning wearing regular clothing, with trees and some fog visible around them",
            "A garden at night with some flowers that have bright colors, standard garden lighting and a path visible between the plant beds",
            "A person with costume wings standing near a lake with clouds reflected in the water, normal outdoor lighting during evening",
            "A portrait photo of a woman's side view combined with an overlay of tree branches, standard photo editing technique",
            "Old stone ruins covered in moss and grass, photographed during the day with sunlight coming through gaps in the walls",
            "A swimmer underwater in a pool with fluorescent lighting, wearing a white swimsuit, with some sea creatures visible nearby",
            "A winter landscape at dusk showing snow-covered ground and a sky that is getting dark, snowflakes falling normally",
            "A woman in a white dress walking through a purple flower field during sunset, gentle breeze moving her clothing",
            "The inside of a cave with natural light coming through an opening, showing rock formations and mineral deposits on walls",
            "A child playing outside in the evening with some bugs with lights flying around, lawn and trees in the background",
        ],
    },
    "dark_moody": {
        "description": "Dark, atmospheric, high contrast with deep shadows and emotional intensity",
        "positive_prompts": [
            "A brooding portrait shrouded in near-darkness, only a sliver of cold blue light revealing furrowed brows and intense eyes, deep shadows swallowing the rest",
            "An abandoned asylum corridor stretching into absolute blackness, a single flickering light creating harsh angular shadows on peeling walls and rusted beds",
            "A rain-battered window with rivulets distorting the view of a solitary street lamp in otherwise total darkness, the glass itself weeping",
            "A forest at midnight where gnarled tree trunks twist like tortured figures, a faint blood-red moon barely illuminating the canopy of dead branches",
            "A smoke-filled underground bar where a single spotlight cuts through the haze to illuminate a worn microphone on an empty stage, all else in shadow",
            "A decayed gothic cathedral interior where darkness pools in every crevice, a single votive candle guttering beside a cracked marble saint",
            "Storm clouds roiling over a dark sea, waves crashing against black rocks, the only light a distant lighthouse beam slashing through sheets of rain",
            "A noir-style street scene in near-total darkness, wet cobblestones reflecting a single red neon sign, a fedora'd silhouette disappearing into an alley",
            "A portrait of weathered hands clutching a faded photograph, lit only from below by a dying ember, extreme chiaroscuro rendering flesh and shadow",
            "A desolate winter landscape under a starless sky, bare trees like black veins against gunmetal grey, the ground frozen and cracked like shattered glass",
        ],
        "negative_prompts": [
            "A portrait of a person with normal indoor lighting, showing their face clearly with standard contrast and balanced exposure settings",
            "A hospital corridor with standard fluorescent lighting, clean floors, and medical equipment visible along the walls under even illumination",
            "A window on a rainy day showing a street outside with normal city lights, the rain visible on the glass with typical indoor lighting",
            "A forest scene at night with moonlight visible through the trees, standard night photography showing trunks and branches with some ground detail",
            "A bar or pub interior with typical lighting, showing a stage area with a microphone, tables, and chairs in standard ambient light",
            "A church interior showing architecture with standard photo lighting, pews, columns, and religious artifacts visible with normal exposure",
            "An ocean scene with rough waves and cloudy sky, a lighthouse in the distance, photographed with standard landscape settings during day",
            "A city street at night showing wet pavement, a neon sign, and people walking, standard street photography with normal exposure",
            "Close-up photo of hands holding an old photograph, taken with flash or standard indoor lighting with visible detail throughout the image",
            "A winter landscape during overcast day showing bare trees and frozen ground, standard exposure showing landscape in grey tones",
        ],
    },
    "vintage_film": {
        "description": "Analog film look with warm tones, grain, light leaks, and nostalgic imperfections",
        "positive_prompts": [
            "A sun-drenched summer afternoon captured on Kodak Portra 400, warm amber color cast, visible film grain, a child running through a sprinkler with delicious halation on highlights",
            "A faded Polaroid of a roadside diner at dusk, characteristic color shift to warm yellows and cool shadows, soft focus edges with that unmistakable instant film border",
            "A dreamy double-exposure on expired film, showing a woman's face superimposed over a field of wildflowers, light leaks bleeding orange and magenta across the frame",
            "A candid street photograph with the gritty texture of Tri-X pushed to 1600, deep blacks and bright whites, visible grain structure lending atmosphere to the scene",
            "A 1970s-style family photo with oversaturated Ektachrome colors, slightly off white balance, lens flare from shooting into sunlight, and a soft vignette at the edges",
            "A washed-out beach scene shot on expired Fuji Superia, characteristic green-shift in shadows, faded highlights, and that wonderful pastel quality of degraded emulsion",
            "A late-afternoon portrait with golden Kodachrome warmth, the subject bathed in rich amber tones, razor-sharp yet somehow nostalgic, like a rediscovered family treasure",
            "A moody night scene shot on high-speed film with extreme grain, neon lights creating soft halos, the entire image swimming in that magical photographic texture",
            "A garden party captured on medium format Hasselblad, creamy bokeh, waist-level finder perspective, rich but restrained colors with that unmistakable medium format depth",
            "A rainy window view shot on Cinestill 800T, tungsten-balanced film creating cool blue tones outdoors and warm halation around the streetlights, red halos on highlights",
        ],
        "negative_prompts": [
            "A digital photo of a sunny day with a child playing near a sprinkler, taken with a modern camera with clean, noise-free image quality and accurate colors",
            "A standard digital photo of a diner at sunset, taken with a smartphone showing clean edges, accurate white balance, and no film artifacts or borders",
            "Two photos layered together using digital editing, showing a woman and flowers, clean blend with no color artifacts or light effects from film processing",
            "A street photo taken with a digital camera, clean black and white conversion with smooth tones, no visible noise or grain texture in the image",
            "A family photo taken with a digital camera, accurate colors and white balance, no lens flare or color shifts, sharp and evenly exposed throughout",
            "A digital beach photo with accurate sand and water colors, clean exposure, no color shifts or fading, typical modern camera output quality",
            "A portrait taken during golden hour with a digital camera, correct white balance maintaining natural skin tones, clean and sharp throughout the image",
            "A night scene photographed with a modern camera on a tripod, clean long exposure with sharp neon signs, no grain or texture artifacts visible",
            "A garden event photographed with a modern digital medium format camera, clean bokeh and sharp focus, accurate color reproduction with no film effects",
            "A photo through a rainy window taken with digital camera, clean image with correct white balance, no color casts or halation around light sources",
        ],
    },
    "minimalist": {
        "description": "Clean, sparse compositions with lots of negative space, simple forms, and restrained palette",
        "positive_prompts": [
            "A single white feather resting on an infinite plane of pale grey, casting a whisper-thin shadow, the vast emptiness around it lending profound significance",
            "A geometric concrete staircase ascending against a pure white sky, the clean lines and sharp angles creating a study in form and negative space",
            "A solitary tree in a snow-covered field, its bare branches forming a delicate ink-drawing silhouette against a uniform overcast sky, nothing else in frame",
            "A perfectly smooth pebble centered on a sweep of fine sand, its oval shadow the only contrast in the frame, zen-like simplicity made tangible",
            "Two parallel lines of footprints in wet sand stretching to a vanishing point where sea meets pale sky, the composition stripped to its barest elements",
            "A single cup of black coffee on a plain white table, shot from directly above, the dark circle a stark punctuation mark in an ocean of white",
            "A long-exposure of a single wave washing over an empty white beach, the water reduced to a smooth gradient from foam to glass, nothing more",
            "A vertical thin reed reflected perfectly in still water, bisecting the frame into symmetrical halves of sky and mirror, pure geometric tranquility",
            "A small red door set into a massive white wall, the scale contrast and color isolation creating an almost abstract composition of proportion",
            "A single light bulb hanging from a long wire in an empty white room, its glow creating the gentlest gradient on otherwise featureless surfaces",
        ],
        "negative_prompts": [
            "A table with multiple feathers, pens, notebooks, and other objects scattered across a wooden surface, with a detailed patterned background visible",
            "A building with multiple staircases, railings, signs, and windows visible against a complex urban backdrop with other structures and sky elements",
            "A grove of multiple trees in a varied landscape showing grass, bushes, fences, paths, and other vegetation with complex textures throughout",
            "Several rocks and pebbles of different sizes on a beach with seaweed, shells, driftwood, and footprints visible in the sand and surrounding area",
            "A busy beach scene with many people, beach umbrellas, towels, coolers, and various equipment, multiple footprints and activities in the frame",
            "A kitchen counter with multiple coffee cups, utensils, a coffee maker, plates, and various items on a patterned counter with shelves behind",
            "A beach scene with multiple waves, surfers, seabirds, beach grass, and various coastal features creating a complex and detailed composition",
            "A pond with multiple reeds, lily pads, fish, and surrounding forest vegetation reflected in the water with ripples and varied textures visible",
            "A building facade with multiple doors, windows, shutters, and architectural details of different colors and styles in a busy street setting",
            "A room full of furniture, pictures, shelves, lamps, and decorative items with complex lighting from multiple sources creating varied shadows",
        ],
    },
    "vibrant_pop": {
        "description": "Highly saturated, bold colors with graphic punch and visual energy",
        "positive_prompts": [
            "An explosion of electric magenta, blazing yellow, and deep cobalt blue paint splashing against a pure white background, colors so intense they vibrate",
            "A tropical parrot in flight, its feathers a riot of scarlet, emerald, and sapphire against a perfectly saturated cyan sky, every color pushed to maximum",
            "A Tokyo street at night transformed into a neon dreamscape, hot pink kanji signs reflecting in puddles alongside electric green and ultraviolet blue",
            "A pop-art style portrait with Andy Warhol-level color saturation, skin in hot pink, hair in electric blue, background in screaming yellow, bold outlines",
            "Stacks of colorful macarons forming a rainbow tower, each layer an impossibly saturated pastel, the light making the colors seem to glow from within",
            "A field of tulips where each row is a different pure, saturated color — crimson, golden, violet, orange — stretching to the horizon like a painter's palette",
            "A vintage car in candy apple red parked against a wall painted in vivid turquoise, the complementary colors creating maximum chromatic energy",
            "Hot air balloons filling the frame in every conceivable color, their geometric panels creating a mosaic of pure hues against an impossibly blue sky",
            "A Mexican Day of the Dead altar covered in marigolds so orange they glow, sugar skulls painted in hot pink and lime green, papel picado in every color",
            "A coral reef underwater scene with maximum color: fluorescent anemones, electric blue tangs, yellow butterfish, and magenta coral pulsing with life",
        ],
        "negative_prompts": [
            "Paint being mixed on a palette showing various muted grey, beige, and brown tones on a neutral background in standard lighting conditions",
            "A bird in flight showing typical coloring against an overcast grey sky, standard nature photography with normal saturation and muted tones",
            "A city street at night showing typical yellow-toned street lamps and some commercial signs, standard urban night photography with normal exposure",
            "A standard portrait with normal skin tones against a grey background, typical studio photography with accurate white balance and natural colors",
            "Pastries arranged on a tray in natural bakery lighting, showing typical muted pastry colors in beige, light brown, and cream tones",
            "A field of flowers during an overcast day showing somewhat muted colors, standard landscape photography with clouds diffusing the light",
            "A car parked on a regular street showing standard automotive paint in a common neutral color against a typical urban background setting",
            "Hot air balloons in the distance during a hazy day, their colors appearing muted and desaturated due to atmospheric conditions and distance",
            "A traditional altar with flowers and decorations in natural indoor lighting, showing warm but normal colors without oversaturation",
            "An underwater photograph of a reef at moderate depth where water has filtered out warm colors, showing mainly blue and grey tones throughout",
        ],
    },
}


def generate_lens_from_preset(
    concept: str,
    target: str = "zimage",
    dpo_steps: int = 5000,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate a lens from a built-in concept preset."""
    if concept not in CONCEPT_PRESETS:
        available = ", ".join(sorted(CONCEPT_PRESETS.keys()))
        raise ValueError(f"Unknown preset '{concept}'. Available: {available}")

    preset = CONCEPT_PRESETS[concept]
    print(f"  Using preset: {concept} — {preset['description']}")

    return generate_lens_from_text_pairs(
        concept=concept,
        positive_texts=preset["positive_prompts"],
        negative_texts=preset["negative_prompts"],
        target=target,
        dpo_steps=dpo_steps,
        output_dir=output_dir,
    )


def list_presets():
    """Print available concept presets."""
    print(f"\n{'='*60}")
    print("  Available Concept Presets")
    print(f"{'='*60}\n")
    for name, preset in sorted(CONCEPT_PRESETS.items()):
        pos_count = len(preset["positive_prompts"])
        print(f"  {name:20s} — {preset['description']}")
        print(f"  {'':20s}   ({pos_count} pairs)")
    print()


def list_lenses():
    """Print all discovered lenses."""
    print(f"\n{'='*60}")
    print(f"  Installed Lenses ({LENS_DIR})")
    print(f"{'='*60}\n")
    if not LENS_DIR.is_dir():
        print("  (lens directory not found)")
        return
    for dirpath, _, filenames in os.walk(LENS_DIR):
        for fname in sorted(filenames):
            if fname.endswith((".pt", ".safetensors")):
                full = Path(dirpath) / fname
                rel = full.relative_to(LENS_DIR)
                size_mb = full.stat().st_size / 1e6
                meta_path = full.with_name(full.stem + "_metadata.json")
                desc = ""
                if meta_path.exists():
                    try:
                        m = json.loads(meta_path.read_text())
                        desc = f" — {m.get('concept', '')} ({m.get('training_mode', 'unknown')})"
                    except Exception:
                        pass
                print(f"  {str(rel):50s} {size_mb:6.1f} MB{desc}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Lens Factory — Generate concept steering lenses for Concept Steer / ComfyUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a preset lens
  python lens_factory.py auto cinematic --target zimage

  # From custom text pairs JSON
  python lens_factory.py text-pairs pairs.json --concept mystyle --target zimage

  # From example images (few-shot)
  python lens_factory.py few-shot ./positive_imgs/ --negative ./negative_imgs/ --concept dreamy

  # List presets
  python lens_factory.py list-presets

  # List installed lenses
  python lens_factory.py list-lenses

  # Generate ALL presets at once (DPO)
  python lens_factory.py batch-all --target zimage

  # Generate ALL presets with SAE + DPO
  python lens_factory.py batch-all --target zimage --method sae

  # Generate SAE lens from preset
  python lens_factory.py sae cinematic --target zimage --sae-features 30

  # SAE lens without DPO refinement
  python lens_factory.py sae cinematic --no-refine-dpo

  # Reuse a previously trained SAE
  python lens_factory.py sae ethereal --sae-load ./sae_layer22_8x.pt

Environment variables:
  QWEN_ENCODER_PATH   Path to qwen_3_4b.safetensors (required for zimage target)
  SIGLIP_MODEL_ID     SigLIP model ID (default: google/siglip-base-patch16-224)
  HF_HOME             Hugging Face cache directory
        """,
    )

    # Global options
    parser.add_argument(
        "--encoder-path",
        default="",
        help="Path to Qwen 3.4B safetensors (overrides QWEN_ENCODER_PATH env var)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── auto (from preset, DPO only) ──
    p_auto = sub.add_parser(
        "auto", help="Generate lens from built-in concept preset (DPO)")
    p_auto.add_argument(
        "concept", help="Preset name (e.g. cinematic, ethereal, dark_moody)")
    p_auto.add_argument("--target", default="zimage",
                        choices=["zimage", "sd15"])
    p_auto.add_argument("--steps", type=int, default=5000,
                        help="DPO training steps")
    p_auto.add_argument("--output-dir", type=Path,
                        default=None, help="Override output directory")

    # ── sae (from preset, SAE + optional DPO) ──
    p_sae = sub.add_parser(
        "sae", help="Generate lens via SAE feature decomposition (interpretable)")
    p_sae.add_argument(
        "concept", help="Preset name (e.g. cinematic, ethereal, dark_moody)")
    p_sae.add_argument("--target", default="zimage", choices=["zimage"])
    p_sae.add_argument("--layer", type=int, default=None,
                       help="Target layer to hook (default: 60%% depth = layer 22)")
    p_sae.add_argument("--sae-expansion", type=int, default=8,
                       help="SAE hidden dimension multiplier (default: 8x)")
    p_sae.add_argument("--sae-epochs", type=int, default=200,
                       help="SAE training epochs (default: 200)")
    p_sae.add_argument("--sae-features", type=int, default=30,
                       help="Number of top SAE features to keep (default: 30)")
    p_sae.add_argument("--n-prompts", type=int, default=500,
                       help="Number of diverse prompts for activation collection (default: 500)")
    p_sae.add_argument("--no-refine-dpo", action="store_true",
                       help="Skip DPO refinement (SAE-only direction)")
    p_sae.add_argument("--dpo-steps", type=int, default=5000,
                       help="DPO optimization steps (default: 5000)")
    p_sae.add_argument("--sae-save", type=Path, default=None,
                       help="Save trained SAE to this path for reuse")
    p_sae.add_argument("--sae-load", type=Path, default=None,
                       help="Load pre-trained SAE instead of training new one")
    p_sae.add_argument("--output-dir", type=Path, default=None)

    # ── text-pairs ──
    p_text = sub.add_parser(
        "text-pairs", help="Generate lens from custom text pairs JSON")
    p_text.add_argument(
        "pairs_file", help="JSON file with positive/negative text pairs")
    p_text.add_argument("--concept", required=True,
                        help="Concept name for the lens")
    p_text.add_argument("--target", default="zimage",
                        choices=["zimage", "sd15"])
    p_text.add_argument("--steps", type=int, default=5000)
    p_text.add_argument("--output-dir", type=Path, default=None)

    # ── few-shot ──
    p_img = sub.add_parser(
        "few-shot", help="Generate lens from example images")
    p_img.add_argument(
        "positive_dir", help="Directory of positive example images")
    p_img.add_argument("--negative", dest="negative_dir",
                       help="Directory of negative images")
    p_img.add_argument("--concept", required=True, help="Concept name")
    p_img.add_argument("--target", default="sd15", choices=["zimage", "sd15"])
    p_img.add_argument("--output-dir", type=Path, default=None)

    # ── list-presets ──
    sub.add_parser("list-presets", help="List available concept presets")

    # ── list-lenses ──
    sub.add_parser("list-lenses", help="List installed lenses")

    # ── batch-all ──
    p_batch = sub.add_parser(
        "batch-all", help="Generate lenses for ALL presets")
    p_batch.add_argument("--target", default="zimage",
                         choices=["zimage", "sd15"])
    p_batch.add_argument("--method", default="dpo", choices=["dpo", "sae"],
                         help="Training method: 'dpo' (fast) or 'sae' (interpretable, default: dpo)")
    p_batch.add_argument("--steps", type=int, default=5000)
    p_batch.add_argument("--sae-features", type=int, default=30,
                         help="Top SAE features to keep (sae method only)")
    p_batch.add_argument("--sae-save", type=Path, default=None,
                         help="Save trained SAE for reuse (sae method only)")
    p_batch.add_argument("--sae-load", type=Path, default=None,
                         help="Load pre-trained SAE (sae method only)")
    p_batch.add_argument("--output-dir", type=Path, default=None)

    args = parser.parse_args()

    # Apply encoder path override
    if hasattr(args, "encoder_path") and args.encoder_path:
        global QWEN_ENCODER_PATH
        QWEN_ENCODER_PATH = args.encoder_path

    if args.command == "list-presets":
        list_presets()
        return

    if args.command == "list-lenses":
        list_lenses()
        return

    if args.command == "auto":
        generate_lens_from_preset(
            args.concept, target=args.target,
            dpo_steps=args.steps, output_dir=args.output_dir,
        )

    elif args.command == "sae":
        generate_lens_sae_from_preset(
            args.concept,
            target=args.target,
            layer=args.layer,
            sae_expansion=args.sae_expansion,
            sae_epochs=args.sae_epochs,
            top_k=args.sae_features,
            refine_dpo=not args.no_refine_dpo,
            dpo_steps=args.dpo_steps,
            output_dir=args.output_dir,
            sae_save_path=args.sae_save,
            sae_load_path=args.sae_load,
        )

    elif args.command == "text-pairs":
        with open(args.pairs_file) as f:
            data = json.load(f)
        pos_key = "positive" if "positive" in data[0] else "enchanting"
        neg_key = "negative" if "negative" in data[0] else "neutral"
        positive_texts = [d[pos_key] for d in data]
        negative_texts = [d[neg_key] for d in data]
        generate_lens_from_text_pairs(
            args.concept, positive_texts, negative_texts,
            target=args.target, dpo_steps=args.steps,
            output_dir=args.output_dir,
        )

    elif args.command == "few-shot":
        generate_lens_from_images(
            args.concept, args.positive_dir,
            negative_dir=args.negative_dir, target=args.target,
            output_dir=args.output_dir,
        )

    elif args.command == "batch-all":
        method = args.method
        results = []
        for concept in sorted(CONCEPT_PRESETS.keys()):
            try:
                if method == "sae":
                    path = generate_lens_sae_from_preset(
                        concept, target=args.target,
                        top_k=args.sae_features,
                        refine_dpo=True,
                        dpo_steps=args.steps,
                        output_dir=args.output_dir,
                        sae_save_path=args.sae_save,
                        sae_load_path=args.sae_load,
                    )
                else:
                    path = generate_lens_from_preset(
                        concept, target=args.target,
                        dpo_steps=args.steps, output_dir=args.output_dir,
                    )
                results.append((concept, "OK", str(path)))
            except Exception as e:
                results.append((concept, "FAIL", str(e)))
                print(f"\nFailed to generate '{concept}': {e}\n")

        print(f"\n{'='*60}")
        print(f"  Batch Results ({method.upper()})")
        print(f"{'='*60}")
        for concept, status, info in results:
            print(f"  [{status:4s}] {concept:20s} -> {info}")
        print()


if __name__ == "__main__":
    main()
