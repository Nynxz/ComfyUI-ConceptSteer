#!/usr/bin/env python3
"""
Lens Factory — Generate concept steering lenses for ComfyUI Concept Steer.

Creates direction vectors that steer image generation toward learned aesthetic
concepts. Two extraction methods:

  Contrastive:  Optimize a separating hyperplane between positive/negative embeddings.
        Fast, simple, works well. Operates on output embeddings only.

  SAE:  Train a Sparse Autoencoder on the text encoder's residual stream,
        decompose activations into interpretable features, find which features
        fire differentially for the concept, then reconstruct a clean direction
        from those features. Optionally refine with Contrastive for robust separation.
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
HF_CACHE = os.environ.get("HF_HOME", str(
    Path.home() / ".cache" / "huggingface"))

SIGLIP_DIM = 768
QWEN_HIDDEN_DIM = 2560


# ═════════════════════════════════════════════════════════════════════════════
#  Core Direction Training
# ═════════════════════════════════════════════════════════════════════════════

def train_contrastive_direction(
    h_positive: torch.Tensor,
    h_negative: torch.Tensor,
    dim: int,
    betas: list[float] = [0.1, 0.3, 0.5, 1.0, 2.0],
    steps: int = 500,
    lr: float = 5e-3,
    min_accuracy: float = 0.98,
    verbose: bool = True,
) -> dict:
    """Train a Contrastive direction that separates positive from negative embeddings.

    Sweeps over beta values and selects the one with the highest minimum margin
    (most robust separation) among those achieving >= min_accuracy.

    Args:
        h_positive: (N, dim) positive concept embeddings
        h_negative: (N, dim) negative concept embeddings
        dim: embedding dimension
        betas: Contrastive temperature values to sweep
        steps: optimization steps per beta
        lr: learning rate
        min_accuracy: minimum accuracy threshold
        verbose: print progress

    Returns:
        dict with 'direction', 'beta', 'accuracy', 'mean_margin', 'min_margin'
    """
    if verbose:
        print(f"  Contrastive training: {h_positive.shape[0]} pairs x {dim}d")

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
    resample_dead_every: int = 0,
) -> SparseAutoencoder:
    """Train a Sparse Autoencoder on activation data.

    Applies:
      - LR linear warmup over first 5% of steps (prevents early divergence)
      - L1 warmup over first 20% of training (prevents premature feature death)
      - Gradient clipping (max_norm=1.0) for stability
      - Unit-norm decoder columns after each step (prevents L1 cheating)
      - Cosine LR schedule (after warmup)
      - Optional dead feature resampling (Anthropic-style)

    Dead feature resampling (when resample_dead_every > 0):
      Features that never fire across a diagnostic batch are "dead weight".
      Resampling reinitializes their encoder/decoder weights from high-loss
      data points, giving them a chance to learn something useful. This is
      critical for large-data training where L1 can kill features early.

      Resampling is calibrated: encoder rows are scaled to match the average
      norm of *living* encoder rows, and optimizer momentum is properly reset
      for resampled features.

    Args:
        data: (N, d_input) activation vectors
        d_sae: SAE hidden dimension (8x expansion recommended)
        l1_coeff: sparsity penalty strength
        epochs: training epochs
        lr: peak learning rate (reached after warmup)
        batch_size: training batch size
        label: display label for progress logging
        verbose: print training progress
        resample_dead_every: resample dead features every N epochs (0 = disabled).
            Recommended: every 2 epochs for large datasets (5-10 epochs total),
            every 25 epochs for small datasets (100+ epochs).

    Returns:
        Trained SparseAutoencoder in eval mode
    """
    d_input = data.shape[1]
    n_vectors = data.shape[0]
    steps_per_epoch = n_vectors // batch_size
    total_steps = steps_per_epoch * epochs

    # Dead feature resampling is disabled by default. It requires careful
    # calibration of encoder scaling relative to activation norms, and naive
    # resampling reliably causes divergence when activation norms are large
    # (e.g. norm ~150 for Qwen). Users can opt in via resample_dead_every > 0
    # in the node UI if they want to experiment.
    if resample_dead_every > 0 and verbose:
        print(f"  Dead feature resampling enabled every {resample_dead_every} epoch(s)")
    elif verbose:
        print(f"  Dead feature resampling: disabled (default)")

    # Warmup steps: 5% of total, minimum 50 steps
    warmup_steps = max(50, int(total_steps * 0.05))

    if verbose:
        act_norm = data.norm(dim=-1).mean().item()
        print(
            f"  Training SAE: {d_input}d → {d_sae}d "
            f"({n_vectors:,} vectors, {epochs} epochs, batch={batch_size})"
        )
        print(f"  Total steps: ~{total_steps:,} ({steps_per_epoch:,}/epoch)")
        print(f"  Activation norm: {act_norm:.1f}, LR warmup: {warmup_steps} steps")

    # Exit inference_mode — ComfyUI wraps node execution in inference_mode()
    # which is stricter than no_grad and cannot be overridden by enable_grad().
    # ALL nn.Module creation AND detach().clone() must happen inside this block —
    # cloning an inference tensor outside still produces an inference tensor.
    with torch.inference_mode(False):
        data = data.detach().clone()

        sae = SparseAutoencoder(d_input, d_sae, l1_coeff).to(DEVICE)
        opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))

        # Combined LR schedule: linear warmup then cosine decay
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps  # linear warmup from 0 → 1
            # Cosine decay from 1 → 0.1 over remaining steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Determine how often to log (every ~20% of training, at least every epoch)
        log_every = max(1, epochs // 5)

        t_start = time.time()
        global_step = 0

        for epoch in range(1, epochs + 1):
            total_mse = total_l1 = 0
            l1_scale = min(1.0, epoch / (epochs * 0.2))  # L1 warmup over first 20%

            for (batch,) in loader:
                batch = batch.to(DEVICE)
                x_hat, z = sae(batch)
                mse = (x_hat - batch).pow(2).mean()
                l1 = z.abs().mean() * l1_coeff * l1_scale
                loss = mse + l1
                loss.backward()

                # Gradient clipping — critical for stability with large activations
                torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)

                opt.step()
                opt.zero_grad()
                sched.step()
                global_step += 1

                # Enforce unit-norm decoder columns
                with torch.no_grad():
                    norms = sae.decoder.weight.data.norm(
                        dim=0, keepdim=True).clamp(min=1e-8)
                    sae.decoder.weight.data /= norms

                total_mse += mse.item()
                total_l1 += l1.item()

            # ── Dead feature resampling ──
            # Only resample if:
            #  - resample_dead_every > 0
            #  - we're on a resampling epoch
            #  - we're past epoch 2 (let training stabilize)
            #  - we have ≥3 epochs remaining (resampled features need time)
            #  - we're in the first 60% of training
            remaining_epochs = epochs - epoch
            can_resample = (
                resample_dead_every > 0
                and epoch % resample_dead_every == 0
                and epoch >= 3
                and remaining_epochs >= 3
                and epoch <= int(epochs * 0.6)
            )
            if can_resample:
                with torch.no_grad():
                    # Check which features are dead on a large sample
                    check_size = min(10_000, n_vectors)
                    z_check = sae.encode(data[:check_size].to(DEVICE))
                    dead_mask = z_check.sum(0) == 0  # (d_sae,)
                    n_dead = dead_mask.sum().item()
                    alive_mask = ~dead_mask

                    # Cap resampling at 10% of features per round to avoid
                    # destabilizing the model
                    max_resample = int(d_sae * 0.10)
                    if n_dead > max_resample:
                        dead_indices_all = dead_mask.nonzero().squeeze(-1)
                        perm = torch.randperm(n_dead)[:max_resample]
                        # Only resample a subset
                        resample_indices = dead_indices_all[perm]
                        n_resample = max_resample
                    else:
                        resample_indices = dead_mask.nonzero().squeeze(-1)
                        n_resample = n_dead

                    if n_resample > 0 and alive_mask.any():
                        # Find high-loss data points to seed dead features
                        x_sample = data[:check_size].to(DEVICE)
                        x_hat_sample = sae.decoder(z_check)
                        losses = (x_sample - x_hat_sample).pow(2).sum(dim=-1)
                        # Sample data points proportional to their loss
                        probs = losses / losses.sum()
                        seed_indices = torch.multinomial(
                            probs, n_resample, replacement=True
                        )

                        # Reinitialize dead decoder columns from high-loss points
                        new_dirs = F.normalize(x_sample[seed_indices], dim=-1)
                        sae.decoder.weight.data[:, resample_indices] = new_dirs.T

                        # CRITICAL: Set encoder rows to a TINY fraction of living
                        # norms. With activation norms of ~150, full-norm encoder
                        # rows produce activations in the hundreds, which
                        # immediately destabilizes training. Starting at 1% lets
                        # training gradually scale them up.
                        alive_enc_norms = sae.encoder.weight.data[alive_mask].norm(
                            dim=1
                        )
                        target_enc_norm = alive_enc_norms.median().item() * 0.01
                        sae.encoder.weight.data[resample_indices] = (
                            new_dirs * target_enc_norm
                        )

                        # Set encoder bias negative enough that resampled features
                        # don't fire immediately. They need a strong match to
                        # activate, and training will adjust the bias upward
                        # for useful features.
                        sae.encoder.bias.data[resample_indices] = -1.0

                        # Reset optimizer momentum for resampled features so
                        # Adam doesn't apply stale statistics
                        for param in sae.parameters():
                            if param not in opt.state:
                                continue
                            state = opt.state[param]
                            for key in ["exp_avg", "exp_avg_sq"]:
                                if key not in state:
                                    continue
                                s = state[key]
                                if param is sae.encoder.weight or param is sae.decoder.weight:
                                    if param is sae.encoder.weight:
                                        s[resample_indices] = 0
                                    else:
                                        s[:, resample_indices] = 0
                                elif param is sae.encoder.bias:
                                    s[resample_indices] = 0

                        if verbose:
                            print(
                                f"    [{label}] Resampled {n_resample} dead features "
                                f"(of {n_dead} dead) at epoch {epoch} "
                                f"(enc_norm={target_enc_norm:.2f})"
                            )

            # ── Logging ──
            should_log = (
                epoch == 1
                or epoch % log_every == 0
                or epoch == epochs
            )
            if verbose and should_log:
                n = len(loader)
                with torch.no_grad():
                    check_size = min(5_000, n_vectors)
                    z_check = sae.encode(data[:check_size].to(DEVICE))
                    l0 = (z_check > 0).float().sum(1).mean()
                    dead = (z_check.sum(0) == 0).sum()
                    cos_sim = F.cosine_similarity(
                        data[:check_size].to(DEVICE), sae.decoder(z_check)
                    ).mean()

                current_lr = sched.get_last_lr()[0]
                elapsed = time.time() - t_start
                eta = elapsed / epoch * (epochs - epoch) if epoch > 0 else 0
                print(
                    f"    [{label}] {epoch:>3d}/{epochs}: "
                    f"mse={total_mse/n:.4f} l1={total_l1/n:.6f} "
                    f"L0={l0:.0f}/{d_sae} dead={dead} cos={cos_sim:.4f} "
                    f"lr={current_lr:.2e} "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
                )

    return sae.eval()


# ═════════════════════════════════════════════════════════════════════════════
#  Pretrained SAE / Transcoder Loading
# ═════════════════════════════════════════════════════════════════════════════

# Default HuggingFace transcoder repo for Qwen3-4B
DEFAULT_TRANSCODER_REPO = "mwhanna/qwen3-4b-transcoders"
TRANSCODER_CACHE_DIR = PACKAGE_ROOT / "sae" / "transcoders"


def download_transcoder_layer(
    layer: int = 22,
    repo_id: str = DEFAULT_TRANSCODER_REPO,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Download a single transcoder layer from HuggingFace.

    Args:
        layer: layer index (default 22, our target layer)
        repo_id: HuggingFace repo ID
        cache_dir: local cache directory

    Returns:
        Path to downloaded safetensors file
    """
    cache_dir = Path(cache_dir or TRANSCODER_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = f"layer_{layer}.safetensors"
    local_path = cache_dir / f"{repo_id.replace('/', '_')}_{filename}"

    if local_path.exists():
        print(f"  Transcoder layer {layer} cached: {local_path}")
        return local_path

    print(f"  Downloading transcoder layer {layer} from {repo_id}...")
    print(f"  (this is ~1.7GB, only needed once)")

    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir / ".hf_cache"),
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        dl_path = Path(downloaded)
        if dl_path != local_path:
            import shutil
            shutil.copy2(dl_path, local_path)
        print(f"  Downloaded to {local_path}")
        return local_path

    except ImportError:
        raise RuntimeError(
            "huggingface_hub required for transcoder download.\n"
            "  pip install huggingface_hub\n"
            f"Or manually download: https://huggingface.co/{repo_id}/resolve/main/{filename}"
        )


# ── Transcoder Feature Dictionary ────────────────────────────────────────────
FEATURE_DICT_CACHE_DIR = PACKAGE_ROOT / "sae" / "transcoders" / "features"


def download_transcoder_feature_dict(
    layer: int = 22,
    repo_id: str = DEFAULT_TRANSCODER_REPO,
    cache_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Download the feature dictionary index + bin for a single layer.

    Returns:
        (index_path, bin_path) tuple
    """
    cache_dir = Path(cache_dir or FEATURE_DICT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    repo_prefix = repo_id.replace("/", "_")
    index_local = cache_dir / f"{repo_prefix}_index.json.gz"
    bin_local = cache_dir / f"{repo_prefix}_layer_{layer}.bin"

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub required for feature dictionary download")

    # Download index (~24 MB compressed)
    if not index_local.exists():
        print(f"  Downloading feature dictionary index from {repo_id}...")
        dl = hf_hub_download(
            repo_id=repo_id,
            filename="features/index.json.gz",
            cache_dir=str(cache_dir / ".hf_cache"),
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        dl_path = Path(dl)
        if dl_path != index_local:
            import shutil
            shutil.copy2(dl_path, index_local)
        print(f"  Index cached: {index_local}")

    # Download layer bin (~1 GB)
    if not bin_local.exists():
        print(
            f"  Downloading feature dictionary for layer {layer} (~1 GB, only needed once)...")
        dl = hf_hub_download(
            repo_id=repo_id,
            filename=f"features/layer_{layer}.bin",
            cache_dir=str(cache_dir / ".hf_cache"),
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        dl_path = Path(dl)
        if dl_path != bin_local:
            import shutil
            shutil.copy2(dl_path, bin_local)
        print(f"  Feature dict cached: {bin_local}")

    return index_local, bin_local


def load_transcoder_feature_labels(
    feature_indices: list[int],
    layer: int = 22,
    repo_id: str = DEFAULT_TRANSCODER_REPO,
    cache_dir: Optional[Path] = None,
) -> dict[int, str]:
    """Load semantic labels for specific features from the transcoder dictionary.

    Only reads the specific feature entries needed (random-access into the bin
    file), so it's fast even with 163K total features.

    Args:
        feature_indices: list of feature indices to look up
        layer: transcoder layer (default 22)
        repo_id: HuggingFace repo ID
        cache_dir: local cache directory

    Returns:
        dict mapping feature_index -> label string built from top logit tokens
    """
    import gzip
    import json
    import struct

    cache_dir = Path(cache_dir or FEATURE_DICT_CACHE_DIR)
    repo_prefix = repo_id.replace("/", "_")
    index_path = cache_dir / f"{repo_prefix}_index.json.gz"
    bin_path = cache_dir / f"{repo_prefix}_layer_{layer}.bin"

    # Ensure files exist
    if not index_path.exists() or not bin_path.exists():
        try:
            download_transcoder_feature_dict(layer, repo_id, cache_dir)
        except Exception as e:
            print(f"  Warning: Could not download feature dictionary: {e}")
            return {}

    # Load offsets
    with gzip.open(str(index_path), "rt") as f:
        index_data = json.load(f)

    layer_info = index_data.get(str(layer))
    if not layer_info:
        print(f"  Warning: Layer {layer} not in feature dictionary index")
        return {}

    offsets = layer_info["offsets"]
    labels: dict[int, str] = {}

    with open(bin_path, "rb") as f:
        for idx in feature_indices:
            if idx < 0 or idx >= len(offsets) - 1:
                continue
            try:
                f.seek(offsets[idx])
                chunk_len = offsets[idx + 1] - offsets[idx]
                chunk = f.read(chunk_len)
                # 4-byte compressed-size header, then gzip JSON
                decompressed = gzip.decompress(chunk[4:])
                entry = json.loads(decompressed)

                # Build label from top logits
                top_logits = entry.get("top_logits", [])
                if top_logits:
                    # Combine top 3 logit tokens as label
                    tokens = [t.strip() for t in top_logits[:3] if t.strip()]
                    label = " | ".join(tokens) if tokens else f"F{idx}"
                else:
                    label = f"F{idx}"

                # Add activation frequency hint
                freq = entry.get("activation_frequency", 0)
                if freq > 0:
                    if freq > 0.01:
                        label += " (common)"
                    elif freq < 0.0001:
                        label += " (rare)"

                labels[idx] = label
            except Exception:
                # Skip features we can't decode
                continue

    return labels


def load_sae_or_transcoder(
    path: str | Path,
    d_model: int = QWEN_HIDDEN_DIM,
    expected_expansion: Optional[int] = None,
) -> SparseAutoencoder:
    """Load a pretrained SAE or transcoder, auto-detecting the format.

    Supports:
      1. Our native SparseAutoencoder state dicts (.pt) — keys: encoder.weight, etc.
      2. Transcoder safetensors — keys: W_enc, W_dec, b_enc, b_dec
      3. Transcoder .pt — keys: W_enc, W_dec, b_enc, b_dec

    Auto-detects dimensions from weight shapes. Returns a SparseAutoencoder
    that can be used identically to our trained SAEs.

    Args:
        path: path to .safetensors or .pt file
        d_model: expected model hidden dim (for validation)
        expected_expansion: if set, validate expansion factor

    Returns:
        SparseAutoencoder in eval mode on DEVICE
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SAE/transcoder not found: {path}")

    # ── Load weights ──
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        # Load to CPU first to avoid 2x GPU peak during copy
        sf = safe_open(str(path), framework="pt", device="cpu")
        state = {k: sf.get_tensor(k) for k in sf.keys()}
    else:
        state = torch.load(str(path), map_location="cpu", weights_only=True)

    # ── Detect format ──
    keys = set(state.keys())

    # Format 1: Our native SparseAutoencoder (nn.Linear keys)
    if "encoder.weight" in keys:
        d_sae, d_in = state["encoder.weight"].shape
        sae = SparseAutoencoder(d_in, d_sae)
        sae.load_state_dict(state)
        sae = sae.to(DEVICE).eval()
        expansion = d_sae // d_in
        print(f"  Loaded native SAE: {d_in}d → {d_sae}d ({expansion}x)")
        return sae

    # Format 2: Transcoder format (W_enc, W_dec, etc.)
    if "W_enc" in keys:
        W_enc = state["W_enc"]  # [d_feature, d_model]
        # [d_feature, d_model] (already transposed in repo)
        W_dec = state["W_dec"]
        b_enc = state.get("b_enc")  # [d_feature] — may not exist
        b_dec = state.get("b_dec")  # [d_model] — may not exist

        d_feature, d_in = W_enc.shape
        expansion = d_feature // d_in

        print(f"  Loaded transcoder: {d_in}d → {d_feature}d ({expansion}x, "
              f"{d_feature:,} features)")

        if d_in != d_model:
            print(
                f"  WARNING: transcoder d_model={d_in} != expected {d_model}")

        if expected_expansion and expansion != expected_expansion:
            print(f"  NOTE: expansion {expansion}x != expected {expected_expansion}x "
                  f"(auto-adjusting)")

        # Wrap in our SparseAutoencoder interface (build on CPU, move once)
        sae = SparseAutoencoder(d_in, d_feature, l1_coeff=0)

        with torch.no_grad():
            # encoder: nn.Linear weight is [out, in], matches W_enc [d_feature, d_model]
            sae.encoder.weight.copy_(W_enc)
            if b_enc is not None:
                sae.encoder.bias.copy_(b_enc)
            else:
                sae.encoder.bias.zero_()

            # decoder: nn.Linear weight is [out, in] = [d_model, d_feature]
            # W_dec from transcoder is [d_feature, d_model], so transpose it
            sae.decoder.weight.copy_(W_dec.T)

            # IMPORTANT: Zero decoder bias for transcoders.
            # Our SparseAutoencoder.encode() pre-centers with (x - decoder.bias),
            # which is an Anthropic SAE convention. Transcoders have independent
            # b_dec that should NOT be used for pre-centering. Zeroing it makes
            # encode() = ReLU(W_enc @ x + b_enc), which is correct for transcoders.
            # decode_sparse() already ignores bias, so this doesn't affect direction
            # extraction.
            sae.decoder.bias.zero_()

        # Free CPU state dict before moving to GPU
        del state, W_enc, W_dec, b_enc, b_dec
        gc.collect()

        sae = sae.to(DEVICE).eval()
        return sae

    raise ValueError(
        f"Unknown SAE/transcoder format. Keys: {sorted(keys)[:10]}...\n"
        f"Expected either 'encoder.weight' (native) or 'W_enc' (transcoder)."
    )


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


# ═════════════════════════════════════════════════════════════════════════════
#  HuggingFace Dataset Activation Collection (for proper SAE training)
# ═════════════════════════════════════════════════════════════════════════════

# Recommended target sizes by expansion factor:
#   8×  (20K features)  → 500K–1M vectors
#   16× (40K features)  → 1M–2M vectors
#   64× (160K features) → 2M–5M vectors
# Rule of thumb: ~25–50 vectors per feature minimum.

DEFAULT_HF_DATASET = "HuggingFaceFW/fineweb"
DEFAULT_HF_SUBSET = "sample-10BT"  # 10B token sample — plenty for streaming


def collect_activations_from_dataset(
    model,
    tokenizer,
    layer_idx: int,
    hidden_dim: int,
    n_vectors: int = 500_000,
    dataset_name: str = DEFAULT_HF_DATASET,
    dataset_subset: str = DEFAULT_HF_SUBSET,
    text_column: str = "text",
    max_length: int = 128,
    batch_size: int = 16,
    seed: int = 42,
    verbose: bool = True,
    save_path: Optional[str] = None,
) -> torch.Tensor:
    """Stream real text from a HuggingFace dataset and collect layer activations.

    Uses streaming mode so no full dataset download is needed. Collects
    activations from all token positions (not just mean-pooled) to maximize
    data diversity per text sample.

    For SAE training, you want 25–50× more activation vectors than SAE features.
    With 8× expansion on 2560d (= 20,480 features), target ~500K–1M vectors.

    Args:
        model: Qwen3Model (or similar) with .layers attribute
        tokenizer: corresponding tokenizer
        layer_idx: which transformer layer to hook
        hidden_dim: model hidden dimension
        n_vectors: target number of activation vectors to collect
        dataset_name: HuggingFace dataset identifier
        dataset_subset: dataset config/subset name
        text_column: column name containing text
        max_length: max tokens per text (longer = more vectors per sample)
        batch_size: texts per forward pass (adjust for GPU memory)
        seed: random seed for dataset shuffling
        verbose: print progress
        save_path: if set, save activations to disk (for reuse without re-collecting)

    Returns:
        Tensor of shape (N, hidden_dim) where N >= n_vectors
    """
    from datasets import load_dataset

    if verbose:
        print(f"  Streaming activations from {dataset_name}/{dataset_subset}")
        print(f"  Target: {n_vectors:,} vectors from layer {layer_idx}")

    # Check for cached activations
    if save_path and os.path.isfile(save_path):
        if verbose:
            print(f"  Loading cached activations from {save_path}")
        cached = torch.load(save_path, map_location="cpu", weights_only=True)
        if cached.shape[0] >= n_vectors and cached.shape[1] == hidden_dim:
            if verbose:
                print(f"  Cached: {cached.shape} — sufficient, skipping collection")
            return cached[:n_vectors]
        else:
            if verbose:
                print(f"  Cached {cached.shape} insufficient (need {n_vectors}), re-collecting")

    # Load dataset in streaming mode — no full download
    ds = load_dataset(
        dataset_name,
        dataset_subset,
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    collected: list[torch.Tensor] = []
    total_vectors = 0
    total_texts = 0
    skipped = 0
    t0 = time.time()

    # Temporary buffer for hook outputs (cleared after each batch)
    _hook_buf: list[torch.Tensor] = []

    def _hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        _hook_buf.append(h.detach().cpu().float())

    handle = model.layers[layer_idx].register_forward_hook(_hook)

    try:
        batch_texts: list[str] = []

        for sample in ds:
            text = sample.get(text_column, "")
            if not text or len(text.strip()) < 20:
                skipped += 1
                continue

            # Truncate very long texts to save memory (still get max_length tokens)
            if len(text) > max_length * 8:
                text = text[:max_length * 8]

            batch_texts.append(text)

            if len(batch_texts) >= batch_size:
                # Tokenize and forward the batch
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                _hook_buf.clear()
                with torch.no_grad():
                    model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )

                # Hook produces one tensor of shape (B, T, D) per forward pass.
                # Apply attention mask to exclude padding token activations.
                if _hook_buf:
                    act = _hook_buf[0]  # (B, T, D)
                    mask = inputs["attention_mask"].cpu()  # (B, T)
                    if act.dim() == 3:
                        for i in range(act.shape[0]):
                            valid = mask[i].bool()
                            collected.append(act[i, valid])  # (n_valid, D)
                    else:
                        # Unexpected shape — just reshape and keep
                        collected.append(act.reshape(-1, hidden_dim))

                total_texts += len(batch_texts)
                total_vectors = sum(a.shape[0] for a in collected)
                batch_texts = []

                if verbose and total_texts % (batch_size * 10) == 0:
                    elapsed = time.time() - t0
                    rate = total_vectors / elapsed if elapsed > 0 else 0
                    print(
                        f"    {total_texts:,} texts → {total_vectors:,} vectors "
                        f"({total_vectors/n_vectors:.0%} of target, "
                        f"{rate:,.0f} vec/s)"
                    )

                if total_vectors >= n_vectors:
                    break

                # Periodic memory management — consolidate collected tensors
                if len(collected) > 500:
                    valid = [
                        a.reshape(-1, hidden_dim)
                        for a in collected
                        if a.dim() >= 1 and a.shape[-1] == hidden_dim
                    ]
                    collected = [torch.cat(valid)] if valid else []
                    gc.collect()

    finally:
        handle.remove()

    # Consolidate all activations
    valid = [
        a.reshape(-1, hidden_dim)
        for a in collected
        if a.dim() >= 1 and a.shape[-1] == hidden_dim
    ]
    if not valid:
        raise RuntimeError(
            f"No activations collected after {total_texts} texts. "
            f"Check layer_idx={layer_idx} and model architecture."
        )

    result = torch.cat(valid)[:n_vectors]

    elapsed = time.time() - t0
    if verbose:
        print(
            f"  Collected: {result.shape} in {elapsed:.1f}s "
            f"(norm={result.norm(dim=-1).mean():.2f}, skipped={skipped})"
        )

    # Optionally cache to disk
    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(result, save_path)
        mb = Path(save_path).stat().st_size / 1024 / 1024
        if verbose:
            print(f"  Saved activations: {save_path} ({mb:.1f} MB)")

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

    d_sae = sae_model.d_sae
    _dev = next(sae_model.parameters()).device
    is_large = d_sae > 40_000  # transcoder-scale

    if is_large:
        # ── Per-token encoding for transcoders ──────────────────────────
        # Transcoders were trained on individual token activations.
        # Mean-pooling before ReLU kills the signal:
        #   ReLU(W_enc @ mean(tokens)) ≠ mean(ReLU(W_enc @ token_i))
        # So we encode each token separately, then mean-pool the features.
        if verbose:
            print(f"    Using per-token encoding ({d_sae:,} features)")

        def _encode_texts_per_token(texts):
            """Collect per-token activations and encode through SAE, one text at a time."""
            feat_accum = torch.zeros(d_sae)
            n_tokens = 0
            raw_accum = torch.zeros(hidden_dim)

            for text in texts:
                # Collect all token activations for this single text
                acts = collect_layer_activations(
                    [text], model, tokenizer, layer_idx, hidden_dim,
                    pool="all", verbose=False,
                )  # [T, hidden_dim]

                raw_accum += acts.sum(0)
                n_tokens += acts.shape[0]

                # Encode through SAE per-token (batch it)
                with torch.no_grad():
                    feats = sae_model.encode(acts.to(_dev)).cpu()  # [T, d_sae]
                feat_accum += feats.sum(0)

            # Mean across all tokens
            mean_feats = feat_accum / max(n_tokens, 1)
            mean_raw = raw_accum / max(n_tokens, 1)
            return mean_feats, mean_raw, n_tokens

        pos_mean_feats, pos_mean_raw, n_pos = _encode_texts_per_token(positive_texts)
        neg_mean_feats, neg_mean_raw, n_neg = _encode_texts_per_token(negative_texts)

        raw_dir = pos_mean_raw - neg_mean_raw
        diff = pos_mean_feats - neg_mean_feats

        if verbose:
            print(f"    Encoded {n_pos} positive tokens, {n_neg} negative tokens")

    else:
        # ── Mean-pooled encoding for native SAEs ────────────────────────
        # Our trained SAEs are fitted to mean-pooled data, so this is correct.
        pos_acts = collect_layer_activations(
            positive_texts, model, tokenizer, layer_idx, hidden_dim,
            pool="mean", verbose=False,
        )
        neg_acts = collect_layer_activations(
            negative_texts, model, tokenizer, layer_idx, hidden_dim,
            pool="mean", verbose=False,
        )

        raw_dir = pos_acts.mean(0) - neg_acts.mean(0)

        with torch.no_grad():
            pos_feats = sae_model.encode(pos_acts.to(_dev)).cpu()
            neg_feats = sae_model.encode(neg_acts.to(_dev)).cpu()

        diff = pos_feats.mean(0) - neg_feats.mean(0)

    # Count how many features have meaningful differential activation
    n_nonzero = (diff.abs() > 1e-6).sum().item()
    if verbose:
        print(f"    Differential features with non-zero activation: "
              f"{n_nonzero:,} / {diff.shape[0]:,}")
        if n_nonzero > 0:
            nonzero_diffs = diff[diff.abs() > 1e-6]
            print(f"    Differential range: [{nonzero_diffs.min():.4f}, "
                  f"{nonzero_diffs.max():.4f}], "
                  f"mean abs: {nonzero_diffs.abs().mean():.4f}")

    # 4. Auto-scale top_k for large feature spaces
    #    At 8x expansion (20K features), top-30 = 0.15% coverage.
    #    Scale proportionally for larger expansions to maintain coverage.
    d_sae = diff.shape[0]
    if d_sae > 40_000 and top_k <= 50:
        # Auto-scale: maintain ~0.15% coverage
        scaled_k = max(top_k, min(int(d_sae * 0.0015), 200))
        if scaled_k != top_k and verbose:
            print(f"    Auto-scaling top_k: {top_k} → {scaled_k} "
                  f"(0.15% of {d_sae:,} features)")
        top_k = scaled_k

    # Also cap at actual non-zero features
    if n_nonzero < top_k:
        effective_k = max(n_nonzero, 2)  # At least 2 features
        if verbose and effective_k != top_k:
            print(f"    Capping top_k to {effective_k} "
                  f"(only {n_nonzero} features have differential activation)")
        top_k = effective_k

    # Select top-K by absolute magnitude
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
        # Try ComfyUI's folder_paths API first (picks up extra_model_paths.yaml)
        _comfy_dirs: list[str] = []
        try:
            import folder_paths  # type: ignore
            for folder_type in ("text_encoders", "clip"):
                try:
                    dirs = folder_paths.get_folder_paths(folder_type)
                    _comfy_dirs.extend(dirs)
                except Exception:
                    pass
        except ImportError:
            pass

        _search_paths = [
            # ComfyUI folder_paths API results
            *[str(Path(d) / "qwen_3_4b.safetensors") for d in _comfy_dirs],
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
    contrastive_steps: int = 500,
) -> Path:
    """Generate a lens from explicit positive/negative text pairs.

    Args:
        concept: concept name (e.g. "cinematic")
        positive_texts: texts embodying the concept
        negative_texts: neutral/opposite texts (same count)
        target: "zimage" (2560d Qwen) or "sd15" (768d SigLIP)
        include_bridge: train SigLIP->Qwen bridge (for cross-modal use)
        output_dir: override output directory
        contrastive_steps: Contrastive optimization steps

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

        print("[3/4] Training Contrastive direction (2560d)...")
        ctr = train_contrastive_direction(
            h_pos, h_neg, dim=QWEN_HIDDEN_DIM, steps=contrastive_steps)

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
                d_dev = ctr["direction"].to(DEVICE)
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
            "direction": ctr["direction"],
            "direction_dim": QWEN_HIDDEN_DIM,
            "contrastive_beta": ctr["beta"],
            "contrastive_accuracy": ctr["accuracy"],
            "contrastive_mean_margin": ctr["mean_margin"],
            "contrastive_min_margin": ctr["min_margin"],
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

        print("[3/3] Training Contrastive direction (768d)...")
        ctr = train_contrastive_direction(
            h_pos, h_neg, dim=SIGLIP_DIM, steps=contrastive_steps)

        lens_data = {
            "direction": ctr["direction"],
            "direction_dim": SIGLIP_DIM,
            "contrastive_beta": ctr["beta"],
            "contrastive_accuracy": ctr["accuracy"],
            "contrastive_mean_margin": ctr["mean_margin"],
            "contrastive_min_margin": ctr["min_margin"],
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
    lens_name = f"{concept}_{target}_contrastive"
    lens_path = out_dir / f"{lens_name}.pt"
    torch.save(lens_data, lens_path)

    # Metadata JSON
    meta = {
        "concept": concept,
        "target": target,
        "direction_dim": lens_data["direction_dim"],
        "training_mode": "text_pairs",
        "n_pairs": len(positive_texts),
        "contrastive_beta": ctr["beta"],
        "contrastive_accuracy": ctr["accuracy"],
        "contrastive_mean_margin": ctr["mean_margin"],
        "contrastive_min_margin": ctr["min_margin"],
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
    use_contrastive: bool = True,
    contrastive_steps: int = 500,
    use_vl_captions: bool = False,
    vl_model: Optional[str] = None,
    output_dir: Optional[Path] = None,
    transcoder_repo: Optional[str] = None,
) -> Path:
    """Generate a lens from few-shot example images.

    Three modes of operation (from weakest to strongest):

      1. Centroid (use_contrastive=False, use_vl_captions=False):
         Simple mean difference. Fast but weak — just subtracts centroids.

      2. Contrastive (use_contrastive=True, use_vl_captions=False):
         Encodes images via SigLIP, then runs contrastive paired-margin
         optimization on the embeddings for robust separation. Much
         stronger than centroid subtraction.

      3. VL Captioning (use_vl_captions=True):
         Uses a VL model to caption each image, then runs contrastive
         training on the captions through the native text encoder (Qwen).
         Bypasses the lossy SigLIP→Qwen bridge entirely. Best quality
         for zimage target but requires a VL model to be available.

    Args:
        concept: concept name
        positive_dir: directory of images embodying the concept
        negative_dir: optional directory of images without the concept
        target: "sd15" or "zimage"
        n_negative_random: number of random negatives if no negative_dir
        use_contrastive: apply contrastive optimization (recommended)
        contrastive_steps: optimization steps for contrastive refinement
        use_vl_captions: caption images with a VL model and train on text
        vl_model: VL model name/path (default: auto-detect)
        output_dir: override output directory
    """
    print(f"\n{'='*60}")
    print(f"  Lens Factory (Few-Shot): '{concept}' ({target})")
    mode_label = "VL-Caption" if use_vl_captions else (
        "Contrastive" if use_contrastive else "Centroid")
    print(f"  Mode: {mode_label}")
    print(f"{'='*60}\n")

    t0 = time.time()

    pos_images = collect_images(positive_dir)
    print(f"[1/4] Found {len(pos_images)} positive images")
    if len(pos_images) < 2:
        raise ValueError(
            f"Need at least 2 positive images, found {len(pos_images)}")

    neg_images_list = None
    if negative_dir:
        neg_images_list = collect_images(negative_dir)
        print(f"  Found {len(neg_images_list)} negative images")

    # ── VL Captioning path ───────────────────────────────────────────────
    if use_vl_captions:
        return _generate_lens_vl_caption(
            concept=concept,
            pos_images=pos_images,
            neg_images=neg_images_list,
            target=target,
            contrastive_steps=contrastive_steps,
            vl_model=vl_model,
            output_dir=output_dir,
            t0=t0,
            transcoder_repo=transcoder_repo,
        )

    # ── SigLIP embedding path ────────────────────────────────────────────
    print(f"[2/4] Encoding positive images with SigLIP...")
    h_pos = encode_images_siglip(pos_images)
    mean_pos = F.normalize(h_pos.mean(dim=0), dim=0)
    print(f"  Positive centroid norm: {h_pos.mean(0).norm():.3f}")
    print(
        f"  Intra-positive cosine: {F.cosine_similarity(h_pos, mean_pos.unsqueeze(0)).mean():.3f}")

    if negative_dir:
        print(f"  Encoding {len(neg_images_list)} negative images...")
        h_neg = encode_images_siglip(neg_images_list)
        mean_neg = F.normalize(h_neg.mean(dim=0), dim=0)
    elif n_negative_random > 0:
        print(
            f"  Generating {n_negative_random} random negative vectors...")
        h_neg = torch.randn(n_negative_random, SIGLIP_DIM)
        h_neg = F.normalize(h_neg, dim=-1)
        mean_neg = F.normalize(h_neg.mean(dim=0), dim=0)
    else:
        print("  No negatives — using origin as contrast")
        h_neg = None
        mean_neg = torch.zeros(SIGLIP_DIM)

    # ── Direction extraction ─────────────────────────────────────────────
    if use_contrastive and h_neg is not None and len(h_pos) >= 2 and len(h_neg) >= 2:
        # Contrastive: paired margin optimization on SigLIP embeddings
        # Need equal-sized sets — pair by index, wrapping shorter set
        n_pairs = max(len(h_pos), len(h_neg))
        h_pos_paired = h_pos[torch.arange(n_pairs) % len(h_pos)]
        h_neg_paired = h_neg[torch.arange(n_pairs) % len(h_neg)]

        print(
            f"[3/4] Contrastive optimization ({n_pairs} pairs, {contrastive_steps} steps)...")
        contrastive = train_contrastive_direction(
            h_pos_paired, h_neg_paired,
            dim=SIGLIP_DIM,
            steps=contrastive_steps,
            verbose=True,
        )
        direction = contrastive["direction"]
        acc = contrastive["accuracy"]
        extra_meta = {
            "contrastive_beta": contrastive["beta"],
            "contrastive_accuracy": contrastive["accuracy"],
            "contrastive_mean_margin": contrastive["mean_margin"],
            "contrastive_min_margin": contrastive["min_margin"],
        }
        print(f"  Contrastive accuracy: {acc:.0%}")
    else:
        # Centroid subtraction fallback
        print("[3/4] Computing centroid direction...")
        raw_direction = mean_pos - mean_neg
        direction = F.normalize(raw_direction, dim=0)
        pos_scores = h_pos @ direction
        if h_neg is not None:
            neg_scores = h_neg @ direction
            acc = ((pos_scores > neg_scores.mean()).float().mean().item())
        else:
            acc = (pos_scores > 0).float().mean().item()
        extra_meta = {}
        print(f"  Centroid accuracy: {acc:.0%}")

    print(f"  Positive mean score: {(h_pos @ direction).mean():.3f}")

    if target == "sd15":
        print("[4/4] Exporting SigLIP-space lens (768d)...")
        lens_data = {
            "direction": direction,
            "d_in_siglip": direction,
            "direction_dim": SIGLIP_DIM,
            "encoder_name": "siglip-base-patch16-224",
            "concept": concept,
            "n_positive_images": len(pos_images),
            "n_negative_images": len(neg_images_list) if negative_dir else 0,
            "training_mode": "few_shot_images",
            "accuracy": acc,
            "target_model": "sd15",
            **extra_meta,
        }
        out_dir = output_dir or (LENS_DIR / "sd15")
        lens_name = f"{concept}_sd15_fewshot"

    elif target == "zimage":
        print("[4/4] Projecting to Qwen space for Z Image...")
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
            print("  WARNING: No bridge found — storing SigLIP direction only.")
            print("  Consider using --use-vl-captions for native Qwen-space directions.")
            direction_qwen = direction

        lens_data = {
            "direction": direction_qwen,
            "d_in_siglip": direction,
            "direction_dim": direction_qwen.shape[0],
            "encoder_name": "qwen_3_4b" if bridge_found else "siglip",
            "concept": concept,
            "n_positive_images": len(pos_images),
            "n_negative_images": len(neg_images_list) if negative_dir else 0,
            "training_mode": "few_shot_images",
            "accuracy": acc,
            "target_model": "z_image_turbo",
            "siglip_model": SIGLIP_MODEL_ID,
            "siglip_dim": SIGLIP_DIM,
            **extra_meta,
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
        "training_method": mode_label.lower(),
        "n_positive_images": len(pos_images),
        "n_negative_images": len(neg_images_list) if negative_dir else 0,
        "direction_dim": int(lens_data["direction_dim"]),
        "accuracy": float(acc),
        "training_time_s": round(time.time() - t0, 1),
        **{k: float(v) if isinstance(v, (int, float)) else v
           for k, v in extra_meta.items()},
    }
    meta_path = out_dir / f"{lens_name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nLens exported: {lens_path}")
    print(f"  Size: {lens_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Total time: {time.time() - t0:.1f}s")
    return lens_path


# ── VL Captioning Pipeline for Few-Shot ─────────────────────────────────────

def _load_vl_model(model_name: Optional[str] = None):
    """Load a vision-language model for captioning.

    Tries (in order):
      1. User-specified model_name
      2. Qwen2.5-VL (best quality, needs ~8GB VRAM)
      3. InternVL2 (good quality, lighter)
      4. Florence-2 (light, ~1.5GB)

    Returns:
        (model, processor, caption_fn) where caption_fn(images) -> list[str]
    """
    if model_name is None:
        # Auto-detect best available
        candidates = [
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "microsoft/Florence-2-large",
        ]
    else:
        candidates = [model_name]

    for name in candidates:
        try:
            if "qwen" in name.lower() and "vl" in name.lower():
                return _load_qwen_vl(name)
            elif "florence" in name.lower():
                return _load_florence(name)
            else:
                # Try as generic transformers model
                return _load_generic_vl(name)
        except Exception as e:
            print(f"  Could not load {name}: {e}")
            continue

    raise RuntimeError(
        "No VL model available. Install one of:\n"
        "  pip install qwen-vl-utils transformers>=4.40\n"
        "  pip install transformers  (for Florence-2)\n"
        "Or specify --vl-model with a model path."
    )


def _load_qwen_vl(model_name: str):
    """Load Qwen2/Qwen2.5-VL for captioning."""
    from transformers import AutoProcessor

    # Qwen2.5-VL needs its own class; fall back to Qwen2VL for older models
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLCls
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as QwenVLCls

    print(f"  Loading VL model: {model_name}...")
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True)
    model = QwenVLCls.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )

    def caption_fn(image_paths: list[str], detail: str = "concept") -> list[str]:
        from PIL import Image

        prompt = (
            "Describe this image in one detailed paragraph for an AI image generator. "
            "Focus on visual style, lighting, composition, colors, mood, textures, "
            "and notable aesthetic qualities. Be specific about visual attributes. "
            "Do not mention any text, watermarks, or UI elements visible in the image."
        )

        captions = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ]}
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[
                               img], return_tensors="pt").to(model.device)
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=256)
            output = processor.batch_decode(
                ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            captions.append(output.strip())
            print(f"    {Path(path).name}: {output.strip()[:80]}...")
        return captions

    return model, processor, caption_fn


def _load_florence(model_name: str):
    """Load Florence-2 for captioning."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    print(f"  Loading VL model: {model_name}...")
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True,
        attn_implementation="eager",  # Florence-2 doesn't support SDPA
    ).to(DEVICE)

    def caption_fn(image_paths: list[str], detail: str = "concept") -> list[str]:
        from PIL import Image

        captions = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            prompt = "<MORE_DETAILED_CAPTION>"
            inputs = processor(text=prompt, images=img,
                               return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                ids = model.generate(
                    **inputs, max_new_tokens=256, num_beams=3,
                    do_sample=False,
                )
            output = processor.batch_decode(ids, skip_special_tokens=True)[0]
            # Florence returns task token + output, strip task token
            output = output.replace("<MORE_DETAILED_CAPTION>", "").strip()
            captions.append(output)
            print(f"    {Path(path).name}: {output[:80]}...")
        return captions

    return model, processor, caption_fn


def _load_generic_vl(model_name: str):
    """Attempt to load a generic VL model via transformers pipeline."""
    from transformers import pipeline

    print(f"  Loading VL model: {model_name}...")
    pipe = pipeline("image-to-text", model=model_name, device=DEVICE,
                    torch_dtype=torch.float16, trust_remote_code=True)

    def caption_fn(image_paths: list[str], detail: str = "concept") -> list[str]:
        from PIL import Image

        captions = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            result = pipe(img, max_new_tokens=256)
            caption = result[0]["generated_text"].strip()
            captions.append(caption)
            print(f"    {Path(path).name}: {caption[:80]}...")
        return captions

    return None, None, caption_fn


def _generate_lens_vl_caption(
    concept: str,
    pos_images: list[str],
    neg_images: Optional[list[str]],
    target: str,
    contrastive_steps: int,
    vl_model: Optional[str],
    output_dir: Optional[Path],
    t0: float,
    transcoder_repo: Optional[str] = None,
) -> Path:
    """Few-shot via VL captioning → contrastive training in native text-encoder space.

    This is the strongest few-shot method because it:
    1. Uses a VL model to extract rich text descriptions of what makes images special
    2. Trains in native Qwen text-encoder space (no lossy SigLIP→Qwen projection)
    3. Applies full contrastive paired-margin optimization

    The VL model captions each positive image with a detailed style description,
    then generates "neutral" counterparts by re-captioning with style stripped.
    """
    print(f"[2/4] Loading VL model for captioning...")
    vl_model_obj, vl_processor, caption_fn = _load_vl_model(vl_model)

    print(f"\n[3/4] Captioning {len(pos_images)} positive images...")
    pos_captions = caption_fn(pos_images)

    if neg_images and len(neg_images) >= 2:
        print(f"  Captioning {len(neg_images)} negative images...")
        neg_captions = caption_fn(neg_images)

        # Ensure equal pairs
        n = min(len(pos_captions), len(neg_captions))
        pos_captions = pos_captions[:n]
        neg_captions = neg_captions[:n]
    else:
        # Generate neutral counterparts for each positive caption
        print("  Generating neutral counterpart captions...")
        neg_captions = _generate_neutral_captions(pos_captions)

    # Free VL model VRAM before loading transcoder/encoder
    del caption_fn, vl_processor
    if vl_model_obj is not None:
        del vl_model_obj
    gc.collect()
    torch.cuda.empty_cache()
    print("  VL model unloaded — VRAM freed")

    print(f"\n  Positive captions: {len(pos_captions)}")
    print(f"  Negative captions: {len(neg_captions)}")

    # ── Route through transcoder or pure contrastive ─────────────────────
    if transcoder_repo and target == "zimage":
        # Strongest path: VL captions → transcoder feature decomposition
        print(f"\n[4/4] Training lens via transcoder decomposition from VL captions...")
        print(f"  Transcoder: {transcoder_repo}")
        lens_path = generate_lens_sae(
            concept=concept,
            positive_texts=pos_captions,
            negative_texts=neg_captions,
            target=target,
            include_bridge=True,
            output_dir=output_dir,
            contrastive_steps=contrastive_steps,
            transcoder_repo=transcoder_repo,
        )
    else:
        # Standard text-pair contrastive training
        if transcoder_repo and target != "zimage":
            print(f"  NOTE: transcoder only supported for zimage target, "
                  f"falling back to contrastive for {target}")
        print(f"\n[4/4] Training contrastive direction from VL captions...")
        lens_path = generate_lens_from_text_pairs(
            concept=concept,
            positive_texts=pos_captions,
            negative_texts=neg_captions,
            target=target,
            include_bridge=(target == "zimage"),
            output_dir=output_dir,
            contrastive_steps=contrastive_steps,
        )

    # Re-save metadata to note this was VL-captioned
    out_dir = output_dir or (
        LENS_DIR / ("zimage" if target == "zimage" else "sd15"))
    # Determine lens filename based on training mode
    if transcoder_repo and target == "zimage":
        method_suffix = "transcoder_contrastive"
    else:
        method_suffix = "contrastive"
    lens_name = f"{concept}_{target}_{method_suffix}"
    meta_path = out_dir / f"{lens_name}_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta["training_mode"] = (
            "vl_caption_transcoder_fewshot" if transcoder_repo and target == "zimage"
            else "vl_caption_fewshot"
        )
        meta["n_positive_images"] = len(pos_images)
        meta["n_negative_images"] = len(neg_images) if neg_images else 0
        meta["vl_model"] = vl_model or "auto"
        meta["transcoder_repo"] = transcoder_repo
        meta["training_time_s"] = round(time.time() - t0, 1)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"\n  VL-captioned lens total time: {time.time() - t0:.1f}s")
    return lens_path


def _generate_neutral_captions(positive_captions: list[str]) -> list[str]:
    """Generate neutral counterpart captions by stripping style descriptors.

    Uses simple text transforms to create 'same scene, neutral style' versions.
    These aren't perfect but give the contrastive optimizer enough contrast.
    """
    import re

    # Style/aesthetic words to strip
    style_words = {
        "dramatic", "cinematic", "moody", "ethereal", "dreamy", "stunning",
        "gorgeous", "breathtaking", "atmospheric", "magical", "mystical",
        "haunting", "evocative", "striking", "captivating", "luminous",
        "vibrant", "vivid", "intense", "bold", "rich", "deep", "lush",
        "delicate", "subtle", "soft", "gentle", "warm", "cool", "golden",
        "silver", "crimson", "azure", "emerald", "amber", "noir",
        "chiaroscuro", "bokeh", "lens flare", "rim lighting",
        "backlit", "silhouette", "high contrast", "low key", "high key",
        "anamorphic", "shallow depth of field", "tilt-shift",
        "painterly", "surreal", "otherworldly", "fantastical",
        "epic", "majestic", "grandiose", "sweeping",
        "gritty", "raw", "visceral", "brutal",
        "elegant", "refined", "luxurious", "opulent",
        "melancholic", "somber", "brooding", "dark",
        "whimsical", "playful", "cheerful", "joyful",
    }

    neutral = []
    for caption in positive_captions:
        # Remove style adjectives
        words = caption.split()
        filtered = [w for w in words if w.lower().strip(".,;:!?")
                    not in style_words]
        result = " ".join(filtered)

        # Simplify: prefix with neutral framing
        result = f"A standard photograph showing {result.lower()}" if result else \
            "A standard photograph of an everyday scene with normal lighting and composition"

        # Clean up double spaces
        result = re.sub(r'\s+', ' ', result).strip()
        neutral.append(result)

    return neutral


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
    refine_contrastive: bool = True,
    contrastive_steps: int = 500,
    include_bridge: bool = True,
    output_dir: Optional[Path] = None,
    sae_save_path: Optional[Path] = None,
    sae_load_path: Optional[Path] = None,
    transcoder_repo: Optional[str] = None,
) -> Path:
    """Generate a lens using SAE feature decomposition of the residual stream.

    Full pipeline:
      1. Load the text encoder (Qwen 3.4B)
      2. Collect diverse activations at the target layer
      3. Train a Sparse Autoencoder on those activations
      4. Run contrastive texts through the model + SAE
      5. Find differential features (which SAE features fire for + but not -)
      6. Reconstruct a clean direction from top-K features (bias-free)
      7. Optionally refine with Contrastive for robust separation
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
        refine_contrastive: also run Contrastive on the output embeddings and blend
        contrastive_steps: Contrastive optimization steps (if refine_contrastive)
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

    # ── Resolve transcoder vs SAE mode ────────────────────────────────
    use_transcoder = bool(transcoder_repo)
    mode_label = "Transcoder" if use_transcoder else "SAE"

    print(f"\n{'='*60}")
    print(f"  Lens Factory [{mode_label}]: '{concept}' ({target})")
    print(f"  {len(positive_texts)} text pairs, top-{top_k} features")
    if use_transcoder:
        print(f"  Transcoder repo: {transcoder_repo}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # ── Step 1: Load Qwen encoder ────────────────────────────────────────
    print("[1/7] Loading Qwen 3.4B encoder...")
    model, tokenizer = load_qwen_encoder()

    num_layers = 36
    hidden_dim = QWEN_HIDDEN_DIM
    target_layer = layer if layer is not None else int(num_layers * 0.6)

    print(f"  Target layer: {target_layer}/{num_layers}")

    # ── Step 2: Load transcoder OR load/train SAE ────────────────────────
    if use_transcoder:
        # Download and load pretrained transcoder — skip SAE training entirely
        print(
            f"\n[2/7] Downloading pretrained transcoder (layer {target_layer})...")
        tc_path = download_transcoder_layer(
            layer=target_layer, repo_id=transcoder_repo)
        print(f"  Loading transcoder from {tc_path}...")
        sae_model = load_sae_or_transcoder(tc_path, d_model=hidden_dim)

        # Auto-detect dimensions from loaded model
        d_sae = sae_model.encoder.weight.shape[0]
        sae_expansion = d_sae // hidden_dim
        cache_key = f"{target_layer}_{sae_expansion}_tc"
        _sae_cache[cache_key] = sae_model

        print(
            f"  Transcoder: {hidden_dim}d → {d_sae:,}d ({sae_expansion}x expansion)")
        print(f"  Skipping SAE training — using pretrained features")

    else:
        d_sae = hidden_dim * sae_expansion
        print(
            f"  SAE dimensions: {hidden_dim}d → {d_sae}d ({sae_expansion}x expansion)")

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

    # ── Step 4: Optional Contrastive refinement ──────────────────────────────────
    contrastive_data = {}
    if refine_contrastive:
        print(f"\n[{step+1}/7] Encoding texts for Contrastive refinement...")
        h_pos = encode_texts_qwen(positive_texts)
        h_neg = encode_texts_qwen(negative_texts)

        print(f"[{step+2}/7] Training Contrastive direction on output embeddings...")
        ctr = train_contrastive_direction(
            h_pos, h_neg, dim=QWEN_HIDDEN_DIM, steps=contrastive_steps)

        contrastive_direction = ctr["direction"]

        # ── Adaptive blending based on agreement ─────────────────────────
        # When SAE/transcoder and contrastive agree (high cos), blend equally.
        # When they disagree (low cos), lean on contrastive which has proven
        # separation (100% accuracy, high margin). Don't dilute a good
        # contrastive direction with a noisy/orthogonal SAE direction.
        cos_sae_contrastive = (sae_direction @ contrastive_direction).item()
        print(f"  cos(SAE, Contrastive) = {cos_sae_contrastive:.3f}")

        # Adaptive weight: sae_weight ranges from 0.0 (orthogonal) to 0.5 (aligned)
        # Linear ramp: weight = max(0, cos) clamped to [0, 0.5]
        # cos ≥ 0.5 → equal blend; cos ≈ 0 → pure contrastive; cos < 0 → pure contrastive
        sae_weight = max(0.0, min(cos_sae_contrastive, 0.5))
        ctr_weight = 1.0 - sae_weight

        if sae_weight < 0.05:
            # SAE direction is orthogonal/opposed — use pure contrastive
            blended = contrastive_direction.clone()
            print(f"  SAE direction orthogonal — using pure contrastive")
        else:
            blended = F.normalize(
                sae_weight * sae_direction + ctr_weight * contrastive_direction,
                dim=0,
            )
            print(f"  Blended: SAE weight={sae_weight:.2f}, "
                  f"Contrastive weight={ctr_weight:.2f}")

        # Verify blend separates well
        with torch.no_grad():
            pos_scores = h_pos @ blended
            neg_scores = h_neg @ blended
            blend_acc = (pos_scores.mean() > neg_scores.mean()).float().item()
            blend_margin = (pos_scores.mean() - neg_scores.mean()).item()
            print(f"  Blended direction margin: {blend_margin:+.3f}")

        contrastive_data = {
            "contrastive_direction": contrastive_direction,
            "contrastive_beta": ctr["beta"],
            "contrastive_accuracy": ctr["accuracy"],
            "contrastive_mean_margin": ctr["mean_margin"],
            "contrastive_min_margin": ctr["min_margin"],
            "cos_sae_contrastive": cos_sae_contrastive,
            "blend_margin": blend_margin,
            "blend_sae_weight": sae_weight,
            "blend_ctr_weight": ctr_weight,
        }

        final_direction = blended
    else:
        final_direction = sae_direction

    # ── Step 5: Optional cross-modal bridge ──────────────────────────────
    bridge_data = {}
    if include_bridge:
        print(
            f"\n[{step+3 if refine_contrastive else step+1}/7] Training SigLIP -> Qwen bridge...")
        all_texts = []
        for p, n in zip(positive_texts, negative_texts):
            all_texts.append(p)
            all_texts.append(n)
        sig_embeds = encode_texts_siglip(all_texts)

        if refine_contrastive:
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

    method_base = "transcoder" if use_transcoder else "sae"
    method_suffix = f"{method_base}_contrastive" if refine_contrastive else method_base
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
        "training_mode": (
            ("transcoder" if use_transcoder else "sae")
            + ("_contrastive" if refine_contrastive else "")
        ),
        "transcoder_repo": transcoder_repo,
        # SAE/transcoder metadata (interpretability)
        "sae_layer": target_layer,
        "sae_expansion": sae_expansion,
        "sae_d_sae": d_sae,
        "sae_top_k": sae_result["top_k"],
        "sae_feature_indices": sae_result["feature_indices"],
        "sae_feature_weights": sae_result["feature_weights"],
        "sae_direction": sae_result["direction"],
        "sae_raw_direction": sae_result["raw_direction"],
        "cos_raw_sae": sae_result["cos_raw_sae"],
        **contrastive_data,
        **bridge_data,
    }

    torch.save(lens_data, lens_path)

    # Metadata JSON
    meta = {
        "concept": concept,
        "target": target,
        "direction_dim": QWEN_HIDDEN_DIM,
        "training_mode": (
            ("transcoder" if use_transcoder else "sae")
            + ("_contrastive" if refine_contrastive else "")
        ),
        "transcoder_repo": transcoder_repo,
        "n_pairs": len(positive_texts),
        "sae_layer": target_layer,
        "sae_expansion": sae_expansion,
        "sae_top_k": sae_result["top_k"],        "sae_features": [int(i) for i in sae_result["feature_indices"].tolist()],
        "cos_raw_sae": round(sae_result["cos_raw_sae"], 4),
        "training_time_s": round(time.time() - t0, 1),
    }
    if refine_contrastive:
        meta.update({
            "contrastive_beta": contrastive_data["contrastive_beta"],
            "contrastive_accuracy": contrastive_data["contrastive_accuracy"],
            "cos_sae_contrastive": round(contrastive_data["cos_sae_contrastive"], 4),
            "blend_margin": round(contrastive_data["blend_margin"], 4),
        })
    meta_path = out_dir / f"{lens_name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nLens exported: {lens_path}")
    print(f"  Size: {lens_path.stat().st_size / 1e6:.1f} MB")
    print(
        f"  Method: {mode_label}{' + Contrastive' if refine_contrastive else ''}")
    print(f"  Features: {sae_result['top_k']} from layer {target_layer}")
    print(f"  Total time: {time.time() - t0:.1f}s")
    return lens_path


def generate_lens_sae_from_preset(
    concept: str,
    target: str = "zimage",
    layer: Optional[int] = None,
    sae_expansion: int = 8,
    sae_epochs: int = 200,
    top_k: int = 30,
    refine_contrastive: bool = True,
    contrastive_steps: int = 500,
    output_dir: Optional[Path] = None,
    sae_save_path: Optional[Path] = None,
    sae_load_path: Optional[Path] = None,
    transcoder_repo: Optional[str] = None,
) -> Path:
    """Generate an SAE/transcoder lens from a built-in concept preset."""
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
        refine_contrastive=refine_contrastive,
        contrastive_steps=contrastive_steps,
        include_bridge=True,
        output_dir=output_dir,
        sae_save_path=sae_save_path,
        sae_load_path=sae_load_path,
        transcoder_repo=transcoder_repo,
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
    # ─── Utility Presets (steer AWAY from these with negative strength) ──────
    "text_overlay": {
        "description": "Text, writing, watermarks, captions, and lettering overlaid on images (use negative strength to suppress)",
        "positive_prompts": [
            "A landscape photo with a large white watermark reading 'SAMPLE' stamped diagonally across the center of the image in bold sans-serif font",
            "A portrait with an Instagram-style text overlay at the bottom reading 'Follow for more' in white Helvetica with a drop shadow",
            "A city skyline photo with a stock photo watermark grid of repeating text covering the entire image in semi-transparent white letters",
            "A food photo with a recipe title in large decorative script font overlaid at the top, and ingredient list text at the bottom",
            "A nature scene with a motivational quote in cursive font overlaid in the center: 'Live Laugh Love' with a lens flare behind the text",
            "A product photo with price tags, sale banners reading '50% OFF', and promotional text scattered across the image in red and yellow",
            "A meme image with large white Impact font text at the top and bottom with black outlines, taking up a third of the image",
            "A screenshot of a social media post with username, timestamp, like count, and comment text overlaid on a photo in UI elements",
            "A photograph with a copyright notice, photographer name, and date stamp in the corner, plus a semi-transparent logo watermark",
            "A movie poster with the title in large metallic 3D letters, cast names at the top, tagline in italic, and credits block at the bottom",
        ],
        "negative_prompts": [
            "A landscape photograph showing mountains and sky with no overlaid elements, clean unedited image with nothing on top of the photo",
            "A portrait of a person with a plain background, no text or graphics added, just the raw photograph as captured by the camera",
            "A city skyline photograph during sunset showing buildings without any overlaid elements, a clean architectural photo",
            "A food photograph on a wooden table showing a plated meal, shot from above with natural lighting, no text or labels",
            "A nature scene showing a forest clearing with sunlight filtering through trees, purely photographic with no additions",
            "A product photograph on a white background showing the item clearly, clean commercial photography with no price tags or banners",
            "A candid photograph of a cat sitting on a windowsill, natural spontaneous moment without any added borders or text",
            "A photograph of a park with people walking on paths between trees, a casual snapshot without any interface elements",
            "A photograph of a sunset over the ocean, untouched raw photo with no stamps, logos, or text of any kind visible",
            "A movie set photograph showing actors on location during filming, behind-the-scenes photo without any graphic design",
        ],
    },
    "blur_defocus": {
        "description": "Blurry, out-of-focus, motion-blurred images (use negative strength for sharper output)",
        "positive_prompts": [
            "A completely out-of-focus photograph where nothing is sharp, all shapes are soft undefined blobs of color with no discernible edges",
            "A photo taken with extreme motion blur, the entire scene is smeared horizontally into streaks of color from camera shake during long exposure",
            "A portrait where the autofocus locked on the background, leaving the subject's face a soft blur while the wall behind is sharp",
            "An intentionally defocused night scene where city lights have expanded into massive soft circular bokeh discs filling the frame",
            "A photograph shot through frosted glass, the scene behind is a soft impressionistic blur of shapes and muted colors",
            "A photo taken from a moving car window, everything outside is motion-blurred into horizontal streaks, nothing is recognizable",
            "A macro photo where the depth of field is paper-thin, only one millimeter is sharp, the rest dissolves into creamy smooth blur",
            "A photograph where the lens is smeared with vaseline or a soft-focus filter, everything has a dreamy hazy glow with no sharp details",
            "A scene photographed during an earthquake, severe camera shake makes every element doubled and tripled in jagged motion blur",
            "A photo taken while the lens was zooming, creating radial zoom blur emanating from the center, stretching everything outward",
        ],
        "negative_prompts": [
            "A tack-sharp photograph where every detail is crisp, taken on a tripod with precise focus showing fine texture in every element",
            "A perfectly still photograph of a building showing razor-sharp edges, every brick and window frame rendered with perfect clarity",
            "A portrait with precise autofocus on the subject's eyes, every eyelash and skin pore visible in sharp detail at full resolution",
            "A night scene photograph taken on a tripod with long exposure, city lights are pinpoint sharp stars against a detailed skyline",
            "A photograph taken through clear glass showing a scene with full sharpness and clarity, every object well-defined and detailed",
            "A parked car photographed with optimal aperture showing every panel reflection and badge detail perfectly sharp and resolved",
            "A macro photograph with focus stacking showing an insect with every compound eye facet and wing vein in perfect razor-sharp focus",
            "A landscape photograph taken at optimal aperture showing fine detail from foreground rocks to distant mountains, all perfectly sharp",
            "A still life photograph on a tripod showing perfect stability, every object edge clean and well-defined without any blur whatsoever",
            "A photograph taken with optimal shutter speed freezing all motion, a bird in flight with every feather perfectly crisp and defined",
        ],
    },
    "noise_grain": {
        "description": "Noisy, grainy, low-quality sensor noise (use negative strength for cleaner output)",
        "positive_prompts": [
            "A photograph taken at ISO 25600 in near darkness, extreme luminance noise makes the image look like colored sand, detail drowned in static",
            "A heavily compressed JPEG image of a face showing severe compression artifacts, blockiness, color banding, and mosquito noise around edges",
            "A phone photo taken in very low light showing aggressive noise reduction smearing combined with remaining chroma noise in purple and green splotches",
            "A surveillance camera still showing extreme noise, scanlines, and interlacing artifacts, barely recognizable shapes in a sea of grain",
            "A photo that's been enlarged 400% showing massive pixel noise, interpolation artifacts, and complete loss of fine detail in a blocky mess",
            "A film photo shot on expired ISO 3200 film showing extreme grain structure the size of golf balls, with color shifts and fogging",
            "A webcam screenshot at 240p resolution showing extreme compression, noise, and aliasing, every surface shimmering with digital artifacts",
            "A night photo from a drone camera showing overwhelming sensor noise, hot pixels, and banding artifacts across the dark sky and dim landscape",
            "A photograph taken through a screen door or mesh adding a moire pattern of interference noise over the entire scene",
            "A deep crop from a low-resolution image blown up to poster size, every pixel visible as square blocks, absolute minimum quality",
        ],
        "negative_prompts": [
            "A photograph taken at ISO 100 in good light showing perfectly clean shadows, smooth gradients, and zero visible noise or grain",
            "A high-resolution photograph with perfect compression, smooth tonal transitions, no blocking artifacts, and pristine image quality",
            "A well-lit phone photograph in daylight showing smooth skin tones, clean colors, and excellent detail without any visible noise",
            "A high-definition security camera image showing a clear scene with clean edges, good resolution, and no visible noise or artifacts",
            "A high-resolution photograph viewed at native size showing fine detail, clean textures, and smooth tonal gradations throughout",
            "A modern digital photograph shot on medium format with extremely clean files, beautiful tonal range, and zero visible grain",
            "A high-quality video screenshot at 4K resolution showing clean detail, accurate colors, and smooth gradients without artifacts",
            "A photograph taken by a professional drone in good lighting showing clean landscape detail with smooth skies and sharp ground textures",
            "A clear photograph taken in normal conditions showing a clean scene without any interference patterns or overlay effects",
            "A well-exposed photograph at optimal settings showing the full resolution potential of the camera with no visible noise at all",
        ],
    },
    "hands_fingers": {
        "description": "Malformed hands, extra fingers, fused digits, wrong finger count (use negative strength to reduce hand artifacts)",
        "positive_prompts": [
            "A close-up of a hand with six fingers, the extra digit growing between the ring and pinky finger, all fingers slightly fused at the base",
            "Two hands clasped together where the fingers blend and merge at the joints, creating an ambiguous mass of too many finger-like protrusions",
            "A person holding a cup where their hand has only three thick fingers and a thumb, the fingers unnaturally short and wide like sausages",
            "A hand raised in greeting where each finger splits into two at the second knuckle, creating a branching tree-like structure of twelve fingertips",
            "A pianist's hands on keys where the fingers are different lengths on each hand, some curving impossibly backward, nails facing wrong directions",
            "Two hands framing a face where the left hand has four fingers and the right has seven, fingers varying wildly in thickness and length",
            "A hand holding a pen where the thumb emerges from the center of the palm, fingers overlap each other, and the wrist bends at a wrong angle",
            "Baby hands reaching out where tiny fingers merge together into webbed paddle-like shapes with too many nail beds visible on each hand",
            "A person making a peace sign but with extra fingers appearing between the V, fingers at inconsistent angles with knuckles in wrong places",
            "A close-up of interlocked fingers where it's impossible to tell which finger belongs to which hand, digits phasing through each other",
        ],
        "negative_prompts": [
            "A close-up photograph of a real human hand showing five distinct well-formed fingers with correct proportions, joints, and natural skin texture",
            "Two real hands clasped together in a clear pose where each finger is distinct, correctly jointed, and the grip is anatomically natural",
            "A person holding a coffee cup with a natural grip showing five normal fingers wrapped around the cup at anatomically correct angles",
            "A real hand held up showing all five fingers clearly separated with correct lengths, proportions, and natural finger spacing",
            "A pianist's real hands photographed on a keyboard showing ten natural fingers with correct anatomy, length ratios, and proper positioning",
            "Two real hands held up side by side showing matching anatomy, five fingers each, symmetrical proportions, natural skin and nail details",
            "A real hand holding a pen in a natural writing grip, thumb and index finger pinching, three other fingers supporting naturally as expected",
            "A real baby's hands photographed showing tiny but perfectly formed five fingers on each hand with correct proportions for an infant",
            "A real person making a peace sign showing exactly two raised fingers and three curled, with correct anatomy and natural hand proportions",
            "A real photograph of two people holding hands showing correct interlocking finger anatomy, each digit clearly belonging to one person",
        ],
    },
    "ai_artifacts": {
        "description": "Common AI generation artifacts: plastic skin, symmetry glitches, floating objects (use negative strength to reduce)",
        "positive_prompts": [
            "A portrait with unnaturally smooth waxy skin that looks like plastic, zero pores or blemishes, an uncanny-valley perfection with dead eyes",
            "A symmetrical face where the left and right halves are exact mirrors, including asymmetric elements like hair parting duplicated on both sides",
            "A room interior where objects float slightly above surfaces, shadows don't match light sources, and reflections show a different scene",
            "A group photo where one person has three arms, another's ear merges into their hair, and a hand appears disconnected from any body",
            "A landscape where the horizon line is inconsistent, water flows uphill, and a tree trunk passes through a solid rock seamlessly",
            "A close-up of teeth that are perfectly identical cloned rectangles, unnaturally white and uniform like a computer-generated dental model",
            "A portrait where earrings are different on each ear despite meant to be a pair, glasses frames pass through hair, and collar is asymmetric weirdly",
            "A street scene where text on signs is gibberish letter-like shapes, numbers are scrambled, and a clock shows impossible time with extra hands",
            "A photograph-like image where fabric patterns tile and repeat in impossibly regular ways, plaid squares are perfectly aligned even around folds",
            "A pet portrait where the animal has slightly too many legs, its tail splits partway, and the fur texture has unnaturally regular repeating patterns",
        ],
        "negative_prompts": [
            "A real photograph of a person's face showing natural skin texture with visible pores, slight asymmetry, and natural imperfections",
            "A real photograph of a person's face showing natural asymmetry between the two halves, slightly different eyebrows, natural hairline",
            "A real photograph of a room interior where all objects rest naturally on surfaces with physically correct shadows and consistent lighting",
            "A real group photograph of people with correct anatomy, each person with the right number of limbs, all body parts connected naturally",
            "A real landscape photograph with consistent horizon, water flowing naturally downhill, and trees growing from soil in physically normal ways",
            "A real photograph smile showing natural teeth with slight variations in size, shape, and color, normal human dental imperfections",
            "A real portrait photograph showing matching jewelry, glasses sitting naturally on the nose, and clothing fitting in a physically normal way",
            "A real street photograph showing legible text on signs, readable numbers, and clocks showing normal valid times with standard clock hands",
            "A real photograph of fabric showing natural drape where patterns distort around folds, stretch at seams, and follow the cloth's 3D shape",
            "A real photograph of a pet showing correct anatomy, proper number of legs, single tail, and natural fur with realistic variation in texture",
        ],
    },
}


def generate_lens_from_preset(
    concept: str,
    target: str = "zimage",
    contrastive_steps: int = 500,
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
        contrastive_steps=contrastive_steps,
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

  # Generate ALL presets at once (Contrastive)
  python lens_factory.py batch-all --target zimage

  # Generate ALL presets with SAE + Contrastive
  python lens_factory.py batch-all --target zimage --method sae

  # Generate SAE lens from preset
  python lens_factory.py sae cinematic --target zimage --sae-features 30

  # SAE lens without Contrastive refinement
  python lens_factory.py sae cinematic --no-refine-contrastive

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

    # ── auto (from preset, Contrastive only) ──
    p_auto = sub.add_parser(
        "auto", help="Generate lens from built-in concept preset (Contrastive)")
    p_auto.add_argument(
        "concept", help="Preset name (e.g. cinematic, ethereal, dark_moody)")
    p_auto.add_argument("--target", default="zimage",
                        choices=["zimage", "sd15"])
    p_auto.add_argument("--steps", type=int, default=500,
                        help="Contrastive training steps")
    p_auto.add_argument("--output-dir", type=Path,
                        default=None, help="Override output directory")

    # ── sae (from preset, SAE + optional Contrastive) ──
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
    p_sae.add_argument("--no-refine-contrastive", action="store_true",
                       help="Skip Contrastive refinement (SAE-only direction)")
    p_sae.add_argument("--contrastive-steps", type=int, default=500,
                       help="Contrastive optimization steps (default: 500)")
    p_sae.add_argument("--sae-save", type=Path, default=None,
                       help="Save trained SAE to this path for reuse")
    p_sae.add_argument("--sae-load", type=Path, default=None,
                       help="Load pre-trained SAE instead of training new one")
    p_sae.add_argument("--transcoder-repo", type=str, default=None,
                       help="HuggingFace repo for pretrained transcoders "
                            "(e.g. 'mwhanna/qwen3-4b-transcoders'). "
                            "Uses 64x expansion instead of training small SAE.")
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
    p_text.add_argument("--steps", type=int, default=500)
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
    p_img.add_argument("--method", default="contrastive",
                       choices=["contrastive", "vl_caption", "centroid"],
                       help="Training method (default: contrastive)")
    p_img.add_argument("--contrastive-steps", type=int, default=500,
                       help="Contrastive optimization steps (default: 500)")
    p_img.add_argument("--vl-model", default=None,
                       help="VL model for captioning (vl_caption mode, auto-detect if empty)")
    p_img.add_argument("--transcoder-repo", type=str, default=None,
                       help="HuggingFace repo for pretrained transcoders "
                            "(used with vl_caption mode for feature decomposition)")
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
    p_batch.add_argument("--method", default="contrastive", choices=["contrastive", "sae"],
                         help="Training method: 'contrastive' (fast) or 'sae' (interpretable, default: contrastive)")
    p_batch.add_argument("--steps", type=int, default=500)
    p_batch.add_argument("--sae-features", type=int, default=30,
                         help="Top SAE features to keep (sae method only)")
    p_batch.add_argument("--sae-save", type=Path, default=None,
                         help="Save trained SAE for reuse (sae method only)")
    p_batch.add_argument("--sae-load", type=Path, default=None,
                         help="Load pre-trained SAE (sae method only)")
    p_batch.add_argument("--transcoder-repo", type=str, default=None,
                         help="HuggingFace repo for pretrained transcoders")
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
            contrastive_steps=args.steps, output_dir=args.output_dir,
        )

    elif args.command == "sae":
        generate_lens_sae_from_preset(
            args.concept,
            target=args.target,
            layer=args.layer,
            sae_expansion=args.sae_expansion,
            sae_epochs=args.sae_epochs,
            top_k=args.sae_features,
            refine_contrastive=not args.no_refine_contrastive,
            contrastive_steps=args.contrastive_steps,
            output_dir=args.output_dir,
            sae_save_path=args.sae_save,
            sae_load_path=args.sae_load,
            transcoder_repo=getattr(args, "transcoder_repo", None),
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
            target=args.target, contrastive_steps=args.steps,
            output_dir=args.output_dir,
        )

    elif args.command == "few-shot":
        generate_lens_from_images(
            args.concept, args.positive_dir,
            negative_dir=args.negative_dir, target=args.target,
            use_contrastive=(args.method in ("contrastive", "vl_caption")),
            contrastive_steps=args.contrastive_steps,
            use_vl_captions=(args.method == "vl_caption"),
            vl_model=args.vl_model,
            output_dir=args.output_dir,
            transcoder_repo=getattr(args, "transcoder_repo", None),
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
                        refine_contrastive=True,
                        contrastive_steps=args.steps,
                        output_dir=args.output_dir,
                        sae_save_path=args.sae_save,
                        sae_load_path=args.sae_load,
                        transcoder_repo=getattr(args, "transcoder_repo", None),
                    )
                else:
                    path = generate_lens_from_preset(
                        concept, target=args.target,
                        contrastive_steps=args.steps, output_dir=args.output_dir,
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
