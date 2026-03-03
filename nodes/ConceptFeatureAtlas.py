"""
ConceptFeatureAtlas — Build a persistent visual catalog of SAE features.

Systematically analyzes every feature in a trained SAE to build a persistent
atlas: a JSON file that catalogs each feature with:
  - Decoder column statistics (norm, sparsity, top dimensions)
  - Firing patterns across a diverse prompt bank
  - Co-occurrence clusters (which features fire together)
  - Feature-to-feature cosine similarities
  - Semantic labels (from the feature dictionary if available)
  - Quality metrics (selectivity, dead features, redundant features)

The atlas is the foundation for the Feature Dashboard, giving it a
pre-computed database to browse rather than re-computing everything
per query.

Also includes SAE health diagnostics:
  - Reconstruction quality (how well does encode→decode preserve the signal?)
  - Dead feature ratio (features that never fire)
  - Feature orthogonality (are decoder columns independent or redundant?)
  - Decoder column norm distribution (are features balanced?)

These diagnostics help answer "is the SAE actually doing what we think?"
— a critical question for trustworthy mechanistic interpretability.

Usage in ComfyUI:
  [Feature Atlas] → STRING (atlas_path) + STRING (summary) + IMAGE (diagnostics)

  Then set dict_path in Feature Map/Dashboard to the atlas JSON.
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from io import BytesIO
from pathlib import Path
from comfy_api.latest import io

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_DIR = _PACKAGE_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


def _log(msg: str):
    print(f"[Concept Steer] {msg}")


# ── Matplotlib rendering ────────────────────────────────────────────────────

def _render_figure_to_tensor(fig) -> torch.Tensor:
    """Convert a matplotlib figure to a ComfyUI image tensor [1, H, W, 3]."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    buf.seek(0)
    from PIL import Image as PILImage
    img = PILImage.open(buf).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)
    buf.close()
    return tensor


def _setup_dark_style():
    """Apply dark theme to matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e94560",
        "axes.labelcolor": "#eaeaea",
        "text.color": "#eaeaea",
        "xtick.color": "#aaaaaa",
        "ytick.color": "#aaaaaa",
        "grid.color": "#2a2a4a",
        "grid.alpha": 0.5,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })
    return plt


# ── Diagnostics rendering ───────────────────────────────────────────────────

def _render_diagnostics(
    col_norms: torch.Tensor,
    firing_rates: torch.Tensor,
    recon_cos: float,
    recon_mse: float,
    dead_ratio: float,
    mean_pairwise_cos: float,
    top_redundant: list[tuple[int, int, float]],
    d_sae: int,
    d_model: int,
) -> torch.Tensor:
    """Render a multi-panel SAE health diagnostics chart."""
    plt = _setup_dark_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Panel 1: Decoder column norm distribution ──
    ax = axes[0, 0]
    norms_np = col_norms.numpy()
    # Use 'auto' bins with a cap to handle near-uniform distributions
    n_norm_bins = min(50, max(1, len(np.unique(np.round(norms_np, 4)))))
    try:
        ax.hist(norms_np, bins=n_norm_bins, color="#58a6ff", alpha=0.8,
                edgecolor="none")
    except ValueError:
        ax.hist(norms_np, bins="auto", color="#58a6ff", alpha=0.8,
                edgecolor="none")
    ax.axvline(x=1.0, color="#3fb950", linewidth=2, linestyle="--",
               label="Unit norm (ideal)")
    ax.axvline(x=norms_np.mean(), color="#d29922", linewidth=1.5,
               linestyle=":", label=f"Mean={norms_np.mean():.3f}")
    ax.set_xlabel("Column Norm")
    ax.set_ylabel("Count")
    ax.set_title("Decoder Column Norms", fontweight="bold")
    ax.legend(fontsize=8)

    # ── Panel 2: Feature firing rate distribution ──
    ax = axes[0, 1]
    fr_np = firing_rates.numpy()
    # Log-scale the bins since firing rates are usually sparse
    fr_nonzero = fr_np[fr_np > 0]
    if len(fr_nonzero) > 0:
        n_fr_bins = min(50, max(1, len(np.unique(np.round(fr_nonzero, 4)))))
        try:
            ax.hist(fr_nonzero, bins=n_fr_bins, color="#bc8cff", alpha=0.8,
                    edgecolor="none")
        except ValueError:
            ax.hist(fr_nonzero, bins="auto", color="#bc8cff", alpha=0.8,
                    edgecolor="none")
        ax.set_xlabel("Firing Rate")
    else:
        ax.text(0.5, 0.5, "No features fire!", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="#f85149")
    ax.set_ylabel("Count")
    ax.set_title(f"Feature Firing Rates ({dead_ratio:.0%} dead)", fontweight="bold")

    # ── Panel 3: SAE quality summary ──
    ax = axes[1, 0]
    ax.axis("off")
    quality_lines = [
        f"SAE Health Report",
        f"─" * 40,
        f"Dimensions: {d_model}d → {d_sae}d ({d_sae // d_model}× expansion)",
        f"",
        f"Reconstruction Quality:",
        f"  Cosine similarity:  {recon_cos:.4f}  {'✓' if recon_cos > 0.95 else '⚠' if recon_cos > 0.85 else '✗'}",
        f"  MSE:                {recon_mse:.6f}",
        f"",
        f"Feature Health:",
        f"  Dead features:      {dead_ratio:.1%}  {'✓' if dead_ratio < 0.2 else '⚠' if dead_ratio < 0.5 else '✗'}",
        f"  Mean pairwise cos:  {mean_pairwise_cos:.4f}  {'✓' if mean_pairwise_cos < 0.1 else '⚠' if mean_pairwise_cos < 0.2 else '✗'}",
        f"  Col norm std:       {norms_np.std():.4f}  {'✓' if norms_np.std() < 0.1 else '⚠'}",
        f"",
        f"Interpretation:",
    ]

    # Add interpretive notes
    if recon_cos > 0.95:
        quality_lines.append("  ✓ SAE faithfully reconstructs the conditioning")
    elif recon_cos > 0.85:
        quality_lines.append("  ⚠ Moderate reconstruction — features are approximate")
    else:
        quality_lines.append("  ✗ Poor reconstruction — SAE may need retraining")

    if dead_ratio > 0.5:
        quality_lines.append(f"  ⚠ {dead_ratio:.0%} dead features — try lower L1 or more data")
    elif dead_ratio > 0.2:
        quality_lines.append(f"  ℹ {dead_ratio:.0%} dead features — normal for {d_sae // d_model}× expansion")

    if mean_pairwise_cos > 0.15:
        quality_lines.append("  ⚠ Features may be redundant — try higher expansion")

    ax.text(0.05, 0.95, "\n".join(quality_lines),
            transform=ax.transAxes, fontfamily="monospace", fontsize=9,
            verticalalignment="top", color="#eaeaea")

    # ── Panel 4: Top redundant feature pairs ──
    ax = axes[1, 1]
    ax.axis("off")
    if top_redundant:
        redundant_lines = [
            "Most Similar Feature Pairs",
            "─" * 40,
            f"{'Pair':>12s}  {'Cosine':>8s}",
        ]
        for i, (f1, f2, cos) in enumerate(top_redundant[:12]):
            redundant_lines.append(f"  F{f1} ↔ F{f2}  {cos:.4f}")
        ax.text(0.05, 0.95, "\n".join(redundant_lines),
                transform=ax.transAxes, fontfamily="monospace", fontsize=9,
                verticalalignment="top", color="#eaeaea")
    else:
        ax.text(0.5, 0.5, "No highly similar pairs found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="#8b949e")

    plt.tight_layout()
    tensor = _render_figure_to_tensor(fig)
    plt.close(fig)
    return tensor


# ── The Node ─────────────────────────────────────────────────────────────────

class ConceptFeatureAtlasNode(io.ComfyNode):
    """Build a persistent atlas + diagnostics for an SAE."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        default_save = str(_PACKAGE_ROOT / "sae" / "feature_atlas.json")

        return io.Schema(
            node_id="conceptsteer.FeatureAtlas",
            display_name="Feature Atlas",
            description=(
                "Analyze a trained SAE to build a persistent feature atlas: "
                "decoder column statistics, firing patterns, co-occurrence "
                "clusters, redundancy analysis, and SAE health diagnostics. "
                "Run once per SAE. The atlas JSON is used by Feature Dashboard "
                "and enriches Feature Map labels. Also answers 'is this SAE "
                "actually working correctly?' with reconstruction quality "
                "metrics and dead feature analysis."
            ),
            category="Concept Steer/Research",
            inputs=[
                io.String.Input(
                    "sae_path",
                    default="",
                    tooltip="Path to the trained SAE weights (.pt)",
                ),
                io.String.Input(
                    "save_path",
                    default=default_save,
                    tooltip="Where to save the feature atlas JSON.",
                ),
                io.Int.Input(
                    "layer",
                    default=22,
                    min=1,
                    max=36,
                    tooltip="Transformer layer (must match SAE training)",
                ),
                io.Int.Input(
                    "sae_expansion",
                    default=8,
                    min=2,
                    max=16,
                    tooltip="SAE expansion factor (must match SAE training)",
                ),
                io.Int.Input(
                    "n_probe_prompts",
                    default=200,
                    min=50,
                    max=2000,
                    step=50,
                    tooltip=(
                        "Number of prompts to probe features with. "
                        "More = better coverage but slower. "
                        "200 is a good balance."
                    ),
                ),
                io.Int.Input(
                    "n_redundancy_samples",
                    default=500,
                    min=100,
                    max=5000,
                    step=100,
                    tooltip=(
                        "How many random feature pairs to check for "
                        "redundancy. Higher = more thorough but slower."
                    ),
                ),
                io.String.Input(
                    "dict_path",
                    default="",
                    tooltip=(
                        "Path to existing feature dictionary JSON. "
                        "If provided, labels are merged into the atlas."
                    ),
                ),
                io.String.Input(
                    "encoder_path",
                    default="",
                    tooltip="Path to Qwen encoder safetensors (or set env var)",
                ),
                io.String.Input(
                    "cached_activations_path",
                    default="",
                    tooltip=(
                        "Path to cached activations .pt file from training "
                        "(e.g. sae/activations_layer22_500k.pt). "
                        "When provided, firing statistics are computed from "
                        "these real activations instead of only probe prompts, "
                        "giving much more accurate dead/alive feature counts."
                    ),
                ),
                io.Int.Input(
                    "max_cached_vectors",
                    default=50000,
                    min=1000,
                    max=500000,
                    step=1000,
                    tooltip=(
                        "Max vectors to sample from cached activations for "
                        "firing analysis. Higher = more accurate but slower. "
                        "50K is a good balance for 20K features."
                    ),
                ),
                io.Boolean.Input(
                    "protect_existing",
                    default=True,
                    tooltip=(
                        "If the save path already exists, auto-rename to _v2, _v3, … "
                        "instead of overwriting. Disable only when intentionally replacing."
                    ),
                ),
            ],
            outputs=[
                io.String.Output("atlas_path"),
                io.String.Output("summary"),
                io.Image.Output("DIAGNOSTICS"),
            ],
        )

    @classmethod
    def execute(
        cls,
        sae_path: str = "",
        save_path: str = "",
        layer: int = 22,
        sae_expansion: int = 8,
        n_probe_prompts: int = 200,
        n_redundancy_samples: int = 500,
        dict_path: str = "",
        encoder_path: str = "",
        cached_activations_path: str = "",
        max_cached_vectors: int = 50000,
        protect_existing: bool = True,
    ):
        sae_path = sae_path.strip()
        save_path = save_path.strip()

        if not sae_path:
            _log("ERROR: SAE path required")
            return io.NodeOutput(
                "", "Error: SAE path required",
                torch.zeros(1, 64, 64, 3),
            )

        if not os.path.isabs(sae_path):
            sae_path = str(_PACKAGE_ROOT / sae_path)

        if not os.path.isfile(sae_path):
            _log(f"ERROR: SAE not found at {sae_path}")
            return io.NodeOutput(
                "", f"Error: SAE not found at {sae_path}",
                torch.zeros(1, 64, 64, 3),
            )

        if not save_path:
            save_path = str(_PACKAGE_ROOT / "sae" / "feature_atlas.json")

        if encoder_path.strip():
            os.environ["QWEN_ENCODER_PATH"] = encoder_path.strip()

        # ── Imports ──
        try:
            from lens_factory import (
                SparseAutoencoder,
                load_qwen_encoder,
                DEVICE,
                QWEN_HIDDEN_DIM,
            )
            from probe_features import PROBE_PROMPTS, generate_extra_prompts
        except ImportError as e:
            _log(f"ERROR: Import failed: {e}")
            return io.NodeOutput(
                "", f"Error: {e}",
                torch.zeros(1, 64, 64, 3),
            )

        t0 = time.time()
        hidden_dim = QWEN_HIDDEN_DIM
        d_sae = hidden_dim * sae_expansion

        # ── Load SAE ──
        _log(f"Loading SAE from {os.path.basename(sae_path)}...")
        sae = SparseAutoencoder(hidden_dim, d_sae).to(DEVICE)
        sae_state = torch.load(
            sae_path, map_location=DEVICE, weights_only=True)
        sae.load_state_dict(sae_state)
        sae.eval()

        # ═══════════════════════════════════════════════════════════════
        #  Phase 1: Decoder Column Analysis (no encoder model needed)
        # ═══════════════════════════════════════════════════════════════
        _log("Phase 1: Analyzing decoder columns...")

        with torch.no_grad():
            dec_weight = sae.decoder.weight.float()  # [d_model, d_sae]

            # Column norms
            col_norms = dec_weight.norm(dim=0).cpu()  # [d_sae]

            # Top dimensions per feature (which model dimensions each feature uses)
            top_dims_per_feat = {}
            for feat_idx in range(d_sae):
                col = dec_weight[:, feat_idx].abs()
                top_vals, top_ids = col.topk(min(5, hidden_dim))
                top_dims_per_feat[feat_idx] = {
                    "dims": top_ids.cpu().tolist(),
                    "weights": [round(v.item(), 4) for v in top_vals],
                }

            # Pairwise cosine similarity — sample random pairs for scalability
            _log(f"  Checking {n_redundancy_samples} random feature pairs...")
            dec_normed = F.normalize(dec_weight, dim=0)  # [d_model, d_sae]

            import random
            rng = random.Random(42)
            pair_cosines = []
            top_redundant = []

            if d_sae < 1000:
                # Small enough to do full pairwise
                cos_matrix = dec_normed.T @ dec_normed  # [d_sae, d_sae]
                cos_matrix.fill_diagonal_(0)
                # Find top redundant pairs
                flat_idx = cos_matrix.abs().reshape(-1).topk(20).indices
                for fi in flat_idx:
                    r = fi.item() // d_sae
                    c = fi.item() % d_sae
                    if r < c:  # avoid duplicates
                        top_redundant.append(
                            (r, c, round(cos_matrix[r, c].item(), 4)))
                mean_pairwise = cos_matrix.abs().mean().item()
            else:
                # Sample random pairs
                sampled_pairs = set()
                while len(sampled_pairs) < n_redundancy_samples:
                    a, b = rng.sample(range(d_sae), 2)
                    if a > b:
                        a, b = b, a
                    sampled_pairs.add((a, b))

                for a, b in sampled_pairs:
                    cos_val = F.cosine_similarity(
                        dec_normed[:, a].unsqueeze(0),
                        dec_normed[:, b].unsqueeze(0),
                    ).item()
                    pair_cosines.append((a, b, cos_val))

                pair_cosines.sort(key=lambda x: -abs(x[2]))
                top_redundant = [
                    (a, b, round(c, 4))
                    for a, b, c in pair_cosines[:15]
                    if abs(c) > 0.3
                ]
                mean_pairwise = (
                    sum(abs(c) for _, _, c in pair_cosines) /
                    max(len(pair_cosines), 1)
                )

        # ═══════════════════════════════════════════════════════════════
        #  Phase 2: Firing patterns via probe prompts
        # ═══════════════════════════════════════════════════════════════
        _log("Phase 2: Loading encoder and probing features...")

        model, tokenizer = load_qwen_encoder(encoder_path)

        prompts = list(PROBE_PROMPTS)
        n_extra = max(0, n_probe_prompts - len(prompts))
        if n_extra > 0:
            prompts.extend(generate_extra_prompts(n_extra))
        prompts = prompts[:n_probe_prompts]
        _log(f"  Probing with {len(prompts)} prompts...")

        # Collect activations — per-token (matching SAE training), not mean-pooled
        collected_tokens: list[torch.Tensor] = []  # list of [n_tokens, hidden_dim]

        def _hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            collected_tokens.append(h.detach().cpu().float())

        handle = model.layers[layer].register_forward_hook(_hook)

        for i, text in enumerate(prompts):
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=64
            ).to(DEVICE)
            with torch.no_grad():
                model(inputs.input_ids, attention_mask=inputs.attention_mask)
            # Extract non-padding token activations (match training collection)
            if collected_tokens:
                act = collected_tokens[-1]  # [1, seq_len, hidden]
                mask = inputs.attention_mask.cpu()  # [1, seq_len]
                valid_act = act[0, mask[0].bool()]  # [n_valid, hidden]
                collected_tokens[-1] = valid_act
            if (i + 1) % 50 == 0:
                _log(f"  {i+1}/{len(prompts)} prompts encoded")

        handle.remove()

        # Stack all per-token activations → [total_tokens, hidden_dim]
        all_token_acts = torch.cat(collected_tokens, dim=0)
        n_total_tokens = all_token_acts.shape[0]
        _log(f"  Collected {n_total_tokens:,} token activations from {len(prompts)} prompts")
        del collected_tokens

        # ── Reconstruction quality (on per-token vectors, matching SAE training) ──
        _log("  Computing reconstruction quality...")
        with torch.no_grad():
            sample = all_token_acts[:min(500, n_total_tokens)].to(DEVICE)
            recon, z_sample = sae(sample)
            recon_cos = F.cosine_similarity(sample, recon, dim=-1).mean().item()
            recon_mse = (sample - recon).pow(2).mean().item()
            _log(f"  Reconstruction: cos={recon_cos:.4f}, MSE={recon_mse:.6f}")
            _log(f"  Activation norm: {sample.norm(dim=-1).mean().item():.1f}")

        # ── Encode all tokens through SAE ──
        _log(f"  Encoding {n_total_tokens:,} token vectors through SAE...")
        with torch.no_grad():
            # Batch to avoid OOM
            batch_sz = 2048
            all_features_list = []
            for b in range(0, n_total_tokens, batch_sz):
                chunk = all_token_acts[b:b + batch_sz].to(DEVICE)
                all_features_list.append(sae.encode(chunk).cpu())
            all_features = torch.cat(all_features_list, dim=0)
            # [total_tokens, d_sae]
        del all_token_acts

        # ── Per-feature statistics ──
        # Start with probe-based stats; will be replaced by cached acts if available
        firing_rates = (all_features > 0).float().mean(dim=0)  # [d_sae]
        mean_acts = torch.zeros(d_sae)
        for fi in range(d_sae):
            active = all_features[:, fi]
            active_vals = active[active > 0]
            if len(active_vals) > 0:
                mean_acts[fi] = active_vals.mean()

        # ── Override with cached activations if provided ──
        cached_path = cached_activations_path.strip() if cached_activations_path else ""
        stats_source = "probes"
        n_stats_vectors = all_features.shape[0]  # total token activations

        if cached_path:
            if not os.path.isabs(cached_path):
                cached_path = str(_PACKAGE_ROOT / cached_path)
            if os.path.isfile(cached_path):
                _log(f"  Loading cached activations from {os.path.basename(cached_path)}...")
                cached_acts = torch.load(cached_path, map_location="cpu", weights_only=True)
                n_total = cached_acts.shape[0]
                _log(f"  Cached: {n_total:,} vectors, using up to {max_cached_vectors:,}")

                # Subsample if needed
                if n_total > max_cached_vectors:
                    perm = torch.randperm(n_total)[:max_cached_vectors]
                    cached_acts = cached_acts[perm]
                n_used = cached_acts.shape[0]

                # Encode through SAE in batches to avoid OOM
                _log(f"  Encoding {n_used:,} cached vectors through SAE...")
                batch_size = 2048
                cached_firing = torch.zeros(d_sae)
                cached_act_sum = torch.zeros(d_sae)
                cached_act_count = torch.zeros(d_sae)

                with torch.no_grad():
                    for b_start in range(0, n_used, batch_size):
                        b_end = min(b_start + batch_size, n_used)
                        batch = cached_acts[b_start:b_end].to(DEVICE)
                        z = sae.encode(batch).cpu()
                        active_mask = z > 0
                        cached_firing += active_mask.float().sum(dim=0)
                        cached_act_sum += (z * active_mask.float()).sum(dim=0)
                        cached_act_count += active_mask.float().sum(dim=0)
                        if (b_start // batch_size + 1) % 10 == 0:
                            _log(f"    {b_end:,}/{n_used:,} encoded")

                del cached_acts

                # Replace firing stats with cached-derived ones
                firing_rates = cached_firing / n_used
                mean_acts = torch.where(
                    cached_act_count > 0,
                    cached_act_sum / cached_act_count,
                    torch.zeros_like(cached_act_sum),
                )
                stats_source = "cached"
                n_stats_vectors = n_used
                _log(f"  Firing stats from {n_used:,} real vectors (much more accurate)")
            else:
                _log(f"  WARNING: Cached activations not found at {cached_path}, using probes only")

        dead_mask = firing_rates == 0
        dead_ratio = dead_mask.float().mean().item()
        n_dead = dead_mask.sum().item()

        _log(f"  Dead features: {n_dead}/{d_sae} ({dead_ratio:.1%}) [source: {stats_source}, n={n_stats_vectors:,}]")

        # ═══════════════════════════════════════════════════════════════
        #  Phase 3: Co-occurrence clustering
        # ═══════════════════════════════════════════════════════════════
        _log("Phase 3: Computing co-occurrence clusters...")

        # Binary activation matrix from probe tokens → co-occurrence
        binary = (all_features > 0).float()  # [n_tokens, d_sae]
        # Only consider features that actually fire
        active_feat_mask = firing_rates > 0
        active_indices = active_feat_mask.nonzero(as_tuple=True)[0].tolist()

        # Focus on informative features: fire 1-80% (skip always-on and dead)
        # These are the features that actually discriminate between concepts
        informative_mask = (firing_rates > 0.01) & (firing_rates < 0.8)
        informative_indices = informative_mask.nonzero(as_tuple=True)[0].tolist()
        _log(f"  {len(informative_indices)} informative features (1-80% firing rate)")

        cooccurrence_clusters: list[list[int]] = []
        if len(informative_indices) > 2:
            # If too many informative features, subsample for tractability
            import random as _rng
            _rng_inst = _rng.Random(42)
            co_indices = informative_indices
            if len(co_indices) > 2000:
                co_indices = _rng_inst.sample(co_indices, 2000)
                _log(f"  Subsampled to {len(co_indices)} for co-occurrence")

            # Compute co-occurrence on probe binary activations
            freq_binary = binary[:, co_indices]  # [n_tokens, n_selected]
            # Jaccard similarity between features
            intersection = freq_binary.T @ freq_binary  # [n_sel, n_sel]
            union = (freq_binary.sum(0).unsqueeze(0) +
                     freq_binary.sum(0).unsqueeze(1) - intersection)
            jaccard = intersection / union.clamp(min=1)
            jaccard.fill_diagonal_(0)

            # Simple greedy clustering: features with Jaccard > 0.4
            visited = set()
            for i, fi in enumerate(co_indices):
                if fi in visited:
                    continue
                cluster = [fi]
                visited.add(fi)
                for j, fj in enumerate(co_indices):
                    if fj not in visited and jaccard[i, j] > 0.4:
                        cluster.append(fj)
                        visited.add(fj)
                if len(cluster) >= 2:
                    cooccurrence_clusters.append(cluster)

            _log(f"  Found {len(cooccurrence_clusters)} co-occurrence clusters")

        # ═══════════════════════════════════════════════════════════════
        #  Phase 4: Merge with existing dictionary labels
        # ═══════════════════════════════════════════════════════════════
        existing_labels = {}
        dict_path_str = dict_path.strip() if dict_path else ""
        if dict_path_str:
            if not os.path.isabs(dict_path_str):
                dict_path_str = str(_PACKAGE_ROOT / dict_path_str)
            if os.path.isfile(dict_path_str):
                with open(dict_path_str) as f:
                    dict_data = json.load(f)
                existing_labels = {
                    k: v.get("label", "")
                    for k, v in dict_data.get("features", {}).items()
                }
                _log(f"  Merged {len(existing_labels)} labels from dictionary")

        # ═══════════════════════════════════════════════════════════════
        #  Phase 5: Build atlas JSON
        # ═══════════════════════════════════════════════════════════════
        _log("Phase 5: Building atlas JSON...")

        features_atlas = {}
        for fi in active_indices:
            entry = {
                "col_norm": round(col_norms[fi].item(), 4),
                "firing_rate": round(firing_rates[fi].item(), 4),
                "mean_activation": round(mean_acts[fi].item(), 4),
                "top_dims": top_dims_per_feat[fi]["dims"],
                "top_dim_weights": top_dims_per_feat[fi]["weights"],
            }

            # Add label if available
            label = existing_labels.get(str(fi), "")
            if label:
                entry["label"] = label

            # Categorize
            fr = firing_rates[fi].item()
            if fr > 0.8:
                entry["category"] = "general"
            elif fr > 0.3:
                entry["category"] = "common"
            elif fr > 0.05:
                entry["category"] = "selective"
            elif fr > 0.0:
                entry["category"] = "rare"
            else:
                entry["category"] = "dead"

            features_atlas[str(fi)] = entry

        atlas = {
            "sae_path": sae_path,
            "layer": layer,
            "expansion": sae_expansion,
            "d_model": hidden_dim,
            "d_sae": d_sae,
            "n_prompts_probed": len(prompts),
            "firing_stats_source": stats_source,
            "firing_stats_n_vectors": n_stats_vectors,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "diagnostics": {
                "reconstruction_cosine": round(recon_cos, 4),
                "reconstruction_mse": round(recon_mse, 6),
                "dead_features": n_dead,
                "dead_ratio": round(dead_ratio, 4),
                "mean_col_norm": round(col_norms.mean().item(), 4),
                "std_col_norm": round(col_norms.std().item(), 4),
                "mean_pairwise_cosine": round(mean_pairwise, 4),
            },
            "redundant_pairs": [
                {"a": a, "b": b, "cosine": c}
                for a, b, c in top_redundant
            ],
            "cooccurrence_clusters": cooccurrence_clusters,
            "feature_categories": {
                "general": sum(1 for v in features_atlas.values()
                               if v.get("category") == "general"),
                "common": sum(1 for v in features_atlas.values()
                              if v.get("category") == "common"),
                "selective": sum(1 for v in features_atlas.values()
                                if v.get("category") == "selective"),
                "rare": sum(1 for v in features_atlas.values()
                            if v.get("category") == "rare"),
                "dead": n_dead,
            },
            "features": features_atlas,
        }

        # ── Save ──
        save_obj = Path(save_path)
        save_obj.parent.mkdir(parents=True, exist_ok=True)
        if protect_existing and save_obj.exists():
            v = 2
            while True:
                candidate = save_obj.parent / f"{save_obj.stem}_v{v}{save_obj.suffix}"
                if not candidate.exists():
                    _log(f"File exists — saving as '{candidate.name}' (protect_existing=True)")
                    save_obj = candidate
                    break
                v += 1

        with open(save_obj, "w") as f:
            json.dump(atlas, f, indent=2)

        elapsed = time.time() - t0
        file_size = save_obj.stat().st_size / 1024

        # ── Render diagnostics ──
        diagnostics_img = _render_diagnostics(
            col_norms=col_norms,
            firing_rates=firing_rates,
            recon_cos=recon_cos,
            recon_mse=recon_mse,
            dead_ratio=dead_ratio,
            mean_pairwise_cos=mean_pairwise,
            top_redundant=top_redundant,
            d_sae=d_sae,
            d_model=hidden_dim,
        )

        # ── Summary ──
        cats = atlas["feature_categories"]
        stats_note = (
            f"Firing stats from {n_stats_vectors:,} cached FineWeb vectors"
            if stats_source == "cached"
            else f"Firing stats from {len(prompts)} probe prompts only"
        )
        summary_lines = [
            f"Feature Atlas Built ({elapsed:.0f}s)",
            f"SAE: {d_sae:,} features, {len(features_atlas):,} active, "
            f"{n_dead:,} dead ({dead_ratio:.1%})",
            f"{stats_note}",
            f"Saved to: {save_path} ({file_size:.0f} KB)",
            f"",
            f"── SAE Health ──",
            f"Reconstruction:      cos={recon_cos:.4f}  "
            f"{'✓ Good' if recon_cos > 0.95 else '⚠ Moderate' if recon_cos > 0.85 else '✗ Poor'}",
            f"Dead features:       {dead_ratio:.1%}  "
            f"{'✓' if dead_ratio < 0.2 else '⚠ Normal for high expansion' if dead_ratio < 0.5 else '✗ Many dead — retrain with lower L1 or more data'}",
            f"Mean pairwise cos:   {mean_pairwise:.4f}  "
            f"{'✓ Independent' if mean_pairwise < 0.1 else '⚠ Some redundancy' if mean_pairwise < 0.2 else '✗ Redundant'}",
            f"Decoder col norm σ:  {col_norms.std().item():.4f}",
            f"",
            f"── Feature Categories ──",
            f"General (fire >80%): {cats['general']:,}",
            f"Common  (30−80%):    {cats['common']:,}",
            f"Selective (5−30%):   {cats['selective']:,}",
            f"Rare (<5%):          {cats['rare']:,}",
            f"Dead (never fire):   {cats['dead']:,}",
            f"",
            f"Co-occurrence clusters: {len(cooccurrence_clusters)}",
            f"Redundant pairs (cos>0.3): {len(top_redundant)}",
        ]

        if cooccurrence_clusters:
            summary_lines.extend(["", "── Top Co-occurrence Clusters ──"])
            for i, cluster in enumerate(cooccurrence_clusters[:5]):
                cluster_labels = []
                for fi in cluster[:5]:
                    lbl = existing_labels.get(str(fi), f"F{fi}")
                    cluster_labels.append(lbl if lbl else f"F{fi}")
                dots = "..." if len(cluster) > 5 else ""
                summary_lines.append(
                    f"  Cluster {i+1} ({len(cluster)} features): "
                    f"{', '.join(cluster_labels)}{dots}"
                )

        summary = "\n".join(summary_lines)
        _log(summary)

        return io.NodeOutput(str(save_obj), summary, diagnostics_img)
