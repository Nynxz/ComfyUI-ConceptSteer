"""
ConceptFeatureMap — Visualize SAE feature activations in conditioning.

Decomposes CLIP/text-encoder conditioning through a trained SAE to reveal
which interpretable features are active. Outputs a bar chart visualization
and a text listing of the top-K features.

This lets you inspect what "ingredients" make up your prompt's conditioning
before selectively suppressing or amplifying them with the Feature Gate node.

Usage in ComfyUI:
  [CLIP Text Encode] → conditioning → [Feature Map] → IMAGE + STRING
                                                        ↓
                                         (read feature indices, then wire
                                          them into Feature Gate)

Note: The SAE was trained on intermediate layer activations but its decoder
columns define meaningful directions in the shared 2560d hidden space.
Decomposition of the final conditioning is approximate but the semantic
meaning of features transfers well.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from io import BytesIO
from pathlib import Path
from comfy_api.latest import io, ui

# ── Resolve imports ──────────────────────────────────────────────────────────
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_DIR = _PACKAGE_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


def _log(msg: str):
    print(f"[Concept Steer] {msg}")


# ── SAE cache (avoid reloading on every execution) ──────────────────────────

_sae_cache: dict[str, object] = {}


def _load_sae(sae_path: str, d_model: int, expansion: int):
    """Load SAE or transcoder from path, auto-detecting format.

    Uses the universal loader from lens_factory which handles both native
    SparseAutoencoder state dicts (.pt) and pretrained transcoder safetensors
    (W_enc/W_dec format). Expansion is auto-detected from weight shapes.
    """
    cache_key = sae_path
    if cache_key in _sae_cache:
        return _sae_cache[cache_key]

    from lens_factory import load_sae_or_transcoder

    _log(f"Loading SAE/transcoder: {os.path.basename(sae_path)}")
    sae = load_sae_or_transcoder(
        sae_path, d_model=d_model, expected_expansion=expansion)
    _sae_cache[cache_key] = sae
    return sae


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
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, 3]
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


def _generate_feature_chart(
    feature_indices: list[int],
    feature_activations: list[float],
    title: str = "SAE Feature Activations",
    max_display: int = 40,
    labels: list[str] | None = None,
) -> torch.Tensor:
    """Generate a horizontal bar chart of feature activations."""
    plt = _setup_dark_style()

    n = min(len(feature_indices), max_display)
    indices = feature_indices[:n]
    values = feature_activations[:n]

    fig_height = max(3, n * 0.35 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Build y-tick labels with optional semantic label
    if labels:
        tick_labels = []
        for i, idx in enumerate(indices):
            lbl = labels[i] if i < len(labels) and labels[i] else ""
            if lbl:
                tick_labels.append(f"F{idx}  {lbl}")
            else:
                tick_labels.append(f"F{idx}")
    else:
        tick_labels = [f"F{idx}" for idx in indices]

    y_pos = range(n - 1, -1, -1)  # reversed so highest is at top

    colors = ["#58a6ff" if v >= 0 else "#f85149" for v in values]
    ax.barh(list(y_pos), values, color=colors, height=0.7, alpha=0.85)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(tick_labels, fontfamily="monospace", fontsize=8)
    ax.set_xlabel("Activation")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    max_val = max(values) if values else 1
    for i, (y, v) in enumerate(zip(y_pos, values)):
        ax.text(v + max_val * 0.02, y, f"{v:.3f}",
                va="center", fontsize=8, color="#cccccc")

    plt.tight_layout()
    tensor = _render_figure_to_tensor(fig)
    plt.close(fig)
    return tensor


# ── SAE file discovery ──────────────────────────────────────────────────────

def _discover_sae_files() -> list[str]:
    """Find SAE/transcoder weight files in common locations."""
    files = []
    search_dirs = [
        _PACKAGE_ROOT / "sae",
        _PACKAGE_ROOT / "sae" / "transcoders",
        _PACKAGE_ROOT / "lenses",
        _PACKAGE_ROOT,
    ]
    for d in search_dirs:
        if not d.is_dir():
            continue
        # Find .pt SAE files
        for f in d.rglob("*.pt"):
            name = f.name.lower()
            if "sae" in name and f.stat().st_size > 1_000_000:
                rel = str(f.relative_to(_PACKAGE_ROOT))
                if rel not in files:
                    files.append(rel)
        # Find .safetensors transcoder files
        for f in d.rglob("*.safetensors"):
            if f.stat().st_size > 1_000_000:
                rel = str(f.relative_to(_PACKAGE_ROOT))
                if rel not in files:
                    files.append(rel)
    files.sort()
    return files


# ── The Node ─────────────────────────────────────────────────────────────────

class ConceptFeatureMapNode(io.ComfyNode):
    """Visualize SAE feature activations in conditioning."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="conceptsteer.FeatureMap",
            display_name="Feature Map",
            description=(
                "Decompose conditioning through a Sparse Autoencoder to see "
                "which interpretable features are active. Use the feature "
                "indices shown here to drive the Feature Gate node."
            ),
            category="Concept Steer/Features",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.String.Input(
                    "sae_path",
                    default="",
                    tooltip=(
                        "Absolute path to SAE weights (.pt file). "
                        "Train one with 'Train Lens (SAE)' using --sae-save, "
                        "or from the lens_factory CLI."
                    ),
                ),
                io.Int.Input(
                    "top_k",
                    default=30,
                    min=5,
                    max=100,
                    tooltip="Number of top features to display",
                ),
                io.Int.Input(
                    "sae_expansion",
                    default=8,
                    min=2,
                    max=128,
                    tooltip=(
                        "SAE expansion factor (auto-detected for transcoders). "
                        "8x for trained SAEs, 64x for pretrained transcoders."
                    ),
                ),
                io.String.Input(
                    "transcoder_repo",
                    default="",
                    tooltip=(
                        "HuggingFace repo for pretrained transcoders "
                        "(e.g. 'mwhanna/qwen3-4b-transcoders'). "
                        "Auto-downloads ~1.7GB layer file on first use. "
                        "Leave empty to use sae_path instead."
                    ),
                ),
                io.Combo.Input(
                    "pool_mode",
                    default="per_token",
                    options=["per_token", "mean", "max"],
                    tooltip=(
                        "How to handle token activations before SAE encoding. "
                        "'per_token' = encode each token individually (matches SAE training), "
                        "'mean' = average tokens then encode (fast but less accurate), "
                        "'max' = max activation per feature across tokens"
                    ),
                ),
                io.String.Input(
                    "dict_path",
                    default="",
                    tooltip=(
                        "Path to feature dictionary JSON (from Feature Dictionary node). "
                        "When provided, features are labeled with what they respond to."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("VISUALIZATION"),
                io.String.Output("FEATURES"),
            ],
        )

    @classmethod
    def execute(
        cls,
        conditioning,
        sae_path: str = "",
        top_k: int = 30,
        sae_expansion: int = 8,
        pool_mode: str = "mean",
        transcoder_repo: str = "",
        dict_path: str = "",
    ):
        # ── Resolve SAE / transcoder path ──
        sae_path = sae_path.strip()
        transcoder_repo = transcoder_repo.strip()

        if transcoder_repo and not sae_path:
            # Auto-download transcoder layer
            try:
                from lens_factory import download_transcoder_layer
                tc_path = download_transcoder_layer(
                    layer=22, repo_id=transcoder_repo)
                sae_path = str(tc_path)
                _log(f"Using transcoder from {transcoder_repo}")
            except Exception as e:
                _log(f"Failed to download transcoder: {e}")
                return io.NodeOutput(
                    torch.zeros(1, 64, 64, 3),
                    f"Error downloading transcoder: {e}",
                )

        if not sae_path:
            _log("No SAE/transcoder path provided — cannot decompose features")
            return io.NodeOutput(
                torch.zeros(1, 64, 64, 3),
                "Error: SAE path or transcoder_repo required",
            )

        if not os.path.isabs(sae_path):
            sae_path = str(_PACKAGE_ROOT / sae_path)

        if not os.path.isfile(sae_path):
            _log(f"SAE/transcoder file not found: {sae_path}")
            return io.NodeOutput(
                torch.zeros(1, 64, 64, 3),
                f"Error: file not found at {sae_path}",
            )

        # ── Extract conditioning tensor ──
        cond_tensor = conditioning[0][0]  # [B, tokens, dim]
        cond_dim = cond_tensor.shape[-1]
        n_tokens = cond_tensor.shape[1]
        device = cond_tensor.device

        _log(
            f"Conditioning: {cond_tensor.shape} (dim={cond_dim}, tokens={n_tokens})")

        # ── Load SAE / transcoder ──
        sae = _load_sae(sae_path, cond_dim, sae_expansion)
        sae_device = next(sae.parameters()).device

        # Auto-detect actual expansion from loaded weights
        actual_d_sae = sae.encoder.weight.shape[0]
        actual_expansion = actual_d_sae // cond_dim
        is_transcoder = sae_path.endswith(
            ".safetensors") or actual_expansion > 16

        # ── Encode through SAE ──
        with torch.no_grad():
            # [B, tokens, dim] → flatten for SAE
            c = cond_tensor.float().to(sae_device)

            if pool_mode == "per_token":
                # Encode each token position
                c_flat = c.reshape(-1, cond_dim)  # [B*tokens, dim]
                features = sae.encode(c_flat)  # [B*tokens, d_sae]
                features = features.reshape(
                    cond_tensor.shape[0], n_tokens, -1)  # [B, tokens, d_sae]

                # For the top-K chart, mean-pool across tokens
                pooled = features.mean(dim=1)[0]  # [d_sae]
            elif pool_mode == "max":
                c_flat = c.reshape(-1, cond_dim)
                features_all = sae.encode(c_flat)
                features_all = features_all.reshape(
                    cond_tensor.shape[0], n_tokens, -1)
                pooled = features_all.max(dim=1).values[0]  # [d_sae]
            else:  # mean
                c_mean = c.mean(dim=1)  # [B, dim]
                pooled = sae.encode(c_mean)[0]  # [d_sae]

        pooled = pooled.cpu()

        # ── Find top-K features ──
        active_mask = pooled > 0
        active_count = active_mask.sum().item()
        d_sae = pooled.shape[0]

        top_vals, top_idx = pooled.topk(min(top_k, active_count))
        top_indices = top_idx.tolist()
        top_values = top_vals.tolist()

        _log(f"SAE decomposition: {active_count}/{d_sae} features active, "
             f"top activation={top_values[0]:.3f}" if top_values else "no active features")

        # ── Load feature dictionary if available ──
        feat_dict = {}
        dict_path = dict_path.strip() if dict_path else ""
        if dict_path:
            # Manual dictionary file provided
            if not os.path.isabs(dict_path):
                dict_path = str(_PACKAGE_ROOT / dict_path)
            if os.path.isfile(dict_path):
                import json
                with open(dict_path) as f:
                    dict_data = json.load(f)
                feat_dict = dict_data.get("features", {})
                _log(f"Loaded feature dictionary: {len(feat_dict)} entries")
            else:
                _log(f"Feature dictionary not found: {dict_path}")
        elif is_transcoder and transcoder_repo:
            # Auto-load labels from transcoder repo's feature dictionary
            try:
                from lens_factory import load_transcoder_feature_labels
                tc_labels = load_transcoder_feature_labels(
                    feature_indices=top_indices,
                    layer=22,
                    repo_id=transcoder_repo,
                )
                if tc_labels:
                    feat_dict = {
                        str(idx): {"label": label}
                        for idx, label in tc_labels.items()
                    }
                    _log(
                        f"Auto-loaded {len(feat_dict)} feature labels from transcoder dictionary")
            except Exception as e:
                _log(f"Could not load transcoder feature dictionary: {e}")

        # ── Build text output ──
        has_labels = bool(feat_dict)
        tc_label = "Transcoder" if is_transcoder else "SAE"
        lines = [
            f"{tc_label} Feature Decomposition",
            f"Conditioning: {cond_dim}d × {n_tokens} tokens",
            f"{tc_label}: {d_sae:,} features ({actual_expansion}×)",
            f"Active features: {active_count:,} / {d_sae:,}",
            f"Pool mode: {pool_mode}",
        ]
        if has_labels:
            lines.append(f"Dictionary: {len(feat_dict)} labeled features")
        lines.extend([
            f"",
            f"Top {len(top_indices)} features by activation:",
        ])

        if has_labels:
            lines.append(f"{'Index':>8s}  {'Activation':>10s}  Label")
            lines.append(f"{'─'*8}  {'─'*10}  {'─'*30}")
        else:
            lines.append(f"{'Index':>8s}  {'Activation':>10s}")
            lines.append(f"{'─'*8}  {'─'*10}")

        for idx, val in zip(top_indices, top_values):
            label = feat_dict.get(str(idx), {}).get("label", "")
            if has_labels and label:
                lines.append(f"F{idx:>6d}  {val:>10.4f}  {label}")
            else:
                lines.append(f"F{idx:>6d}  {val:>10.4f}")

        # ── SAE reconstruction sanity check (use per-token vectors to match training) ──
        with torch.no_grad():
            # Always test on per-token activations for accurate quality metric
            test_in = c.reshape(-1, cond_dim)  # [B*tokens, D]
            # Sample up to 500 tokens to keep it fast
            if test_in.shape[0] > 500:
                test_in = test_in[:500]
            test_recon, _ = sae(test_in.to(sae_device))
            sanity_cos = F.cosine_similarity(
                test_in.to(sae_device), test_recon, dim=-1
            ).mean().item()
            sanity_mse = (
                test_in.to(sae_device) - test_recon
            ).pow(2).mean().item()

        if sanity_cos > 0.95:
            quality_note = "✓ Faithful — features are reliable"
        elif sanity_cos > 0.85:
            quality_note = "⚠ Approximate — features are directionally correct but lossy"
        else:
            quality_note = "✗ Poor — SAE may need retraining, features are unreliable"

        lines.extend([
            "",
            "── SAE Reconstruction Quality ──",
            f"Cosine similarity: {sanity_cos:.4f}  {quality_note}",
            f"MSE: {sanity_mse:.6f}",
        ])

        # Suppression helper
        lines.extend([
            "",
            "── Copy into Feature Gate ──",
            "Suppress: " + ",".join(str(i) for i in top_indices[:10]),
        ])

        text_output = "\n".join(lines)

        # ── Generate chart ──
        if pool_mode == "per_token" and cond_tensor.shape[0] > 0:
            # Generate a heatmap showing per-token feature activations
            chart = _generate_token_heatmap(
                features[0].cpu(), top_indices[:min(top_k, 30)], n_tokens)
        else:
            # Build labels for chart
            chart_labels = None
            if feat_dict:
                chart_labels = [
                    feat_dict.get(str(idx), {}).get("label", "")
                    for idx in top_indices
                ]
            chart = _generate_feature_chart(
                top_indices, top_values,
                title=f"{tc_label} Feature Activations ({active_count:,} active)",
                max_display=min(top_k, 40),
                labels=chart_labels,
            )

        return io.NodeOutput(chart, text_output)


def _generate_token_heatmap(
    features: torch.Tensor,
    top_indices: list[int],
    n_tokens: int,
) -> torch.Tensor:
    """Generate a heatmap of feature activations per token position.

    Args:
        features: [tokens, d_sae] feature activations
        top_indices: which feature indices to show
        n_tokens: number of tokens
    """
    plt = _setup_dark_style()

    n_feats = len(top_indices)
    data = features[:n_tokens, top_indices].numpy()  # [tokens, n_feats]

    fig, ax = plt.subplots(
        figsize=(max(6, n_tokens * 0.4), max(3, n_feats * 0.3)))

    im = ax.imshow(
        data.T, aspect="auto", cmap="inferno",
        interpolation="nearest",
    )
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Feature")
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels([f"F{idx}" for idx in top_indices],
                       fontfamily="monospace", fontsize=8)
    ax.set_title("Per-Token Feature Activations", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Activation")
    plt.tight_layout()

    tensor = _render_figure_to_tensor(fig)
    plt.close(fig)
    return tensor
