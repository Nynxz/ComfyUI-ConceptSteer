"""
ConceptLensInspect — Inspect and visualize the internals of a concept lens.

Loads a lens file and generates a visual report showing:
  - Direction vector statistics (norm, dimensionality, sparsity)
  - Top SAE feature weights (if SAE lens) with bar chart
  - Training metadata (method, accuracy, margins)
  - Feature histogram showing weight distribution

Outputs a preview image of the visualization and a text summary string.

Usage in ComfyUI:
  [Concept Lens Inspect] → IMAGE (chart) + STRING (summary)
"""

import os
import json
import math
import torch
import numpy as np
from io import BytesIO
from pathlib import Path
from comfy_api.latest import io, ui


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
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, 3]
    buf.close()
    return tensor


def _setup_dark_style():
    """Apply a dark theme to matplotlib for ComfyUI consistency."""
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


# ── Lens loading (lightweight, just for inspection) ─────────────────────────

def _find_lens_root() -> str:
    """Find the lenses directory relative to this package."""
    node_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(node_dir, "..", "lenses"),
        os.path.join(node_dir, "..", "..", "..", "lenses"),
    ]
    for c in candidates:
        p = os.path.normpath(c)
        if os.path.isdir(p):
            return p
    return os.path.normpath(candidates[0])


def _discover_lenses() -> list[str]:
    """Find all lens files recursively."""
    lenses = []
    root = _find_lens_root()
    if not os.path.isdir(root):
        return lenses
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith((".pt", ".safetensors")):
                rel = os.path.relpath(os.path.join(dirpath, fname), root)
                lenses.append(rel)
    lenses.sort()
    return lenses


def _resolve_lens_path(filename: str) -> str | None:
    """Resolve a lens filename to its full path."""
    root = _find_lens_root()
    full = os.path.normpath(os.path.join(root, filename))
    if os.path.isfile(full):
        return full
    basename = os.path.basename(filename)
    for dirpath, _dirnames, filenames in os.walk(root):
        if basename in filenames:
            return os.path.join(dirpath, basename)
    return None


# ── Chart generation ────────────────────────────────────────────────────────

def _generate_inspect_chart(data: dict, lens_name: str) -> torch.Tensor:
    """Generate a multi-panel inspection chart for a concept lens."""
    plt = _setup_dark_style()

    direction = None
    if isinstance(data, torch.Tensor):
        direction = data.float().squeeze()
    elif isinstance(data, dict) and "direction" in data:
        direction = data["direction"].float().squeeze()

    if direction is None:
        # Fallback: find first tensor
        if isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, torch.Tensor) and val.dim() <= 2:
                    direction = val.float().reshape(-1)
                    break

    if direction is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.text(0.5, 0.5, "Could not extract direction from lens",
                ha="center", va="center", fontsize=14, color="#e94560")
        ax.set_axis_off()
        tensor = _render_figure_to_tensor(fig)
        plt.close(fig)
        return tensor

    d_np = direction.cpu().numpy()
    dim = len(d_np)

    # Check for SAE features
    has_sae = isinstance(data, dict) and "sae_feature_indices" in data
    sae_features = None
    sae_weights = None
    if has_sae:
        sae_features = data.get("sae_feature_indices", [])
        sae_weights = data.get("sae_feature_weights", [])
        if isinstance(sae_features, torch.Tensor):
            sae_features = sae_features.cpu().tolist()
        if isinstance(sae_weights, torch.Tensor):
            sae_weights = sae_weights.cpu().tolist()

    n_panels = 3 if has_sae else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(f"Concept Lens: {lens_name}", fontsize=16, fontweight="bold",
                 color="#e94560", y=1.02)

    # ── Panel 1: Direction weight distribution ──
    ax1 = axes[0]
    abs_vals = np.abs(d_np)
    ax1.hist(d_np, bins=60, color="#0f3460", edgecolor="#e94560", alpha=0.85)
    ax1.axvline(x=0, color="#e94560", linestyle="--", alpha=0.7)
    ax1.set_title("Direction Weight Distribution")
    ax1.set_xlabel("Weight Value")
    ax1.set_ylabel("Count")

    # Stats annotation
    nonzero = np.count_nonzero(abs_vals > 1e-6)
    sparsity = 1.0 - (nonzero / dim)
    stats_text = (
        f"dim: {dim}\n"
        f"norm: {np.linalg.norm(d_np):.4f}\n"
        f"mean: {d_np.mean():.4f}\n"
        f"std: {d_np.std():.4f}\n"
        f"max: {d_np.max():.4f}\n"
        f"min: {d_np.min():.4f}\n"
        f"sparsity: {sparsity:.1%}"
    )
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=8, va="top", ha="right", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a0a1a", alpha=0.8))

    # ── Panel 2: Top-K largest weights ──
    ax2 = axes[1]
    top_k = min(30, dim)
    top_indices = np.argsort(np.abs(d_np))[-top_k:][::-1]
    top_vals = d_np[top_indices]

    colors = ["#e94560" if v > 0 else "#0f3460" for v in top_vals]
    bars = ax2.barh(range(top_k), top_vals, color=colors, edgecolor="#2a2a4a")
    ax2.set_yticks(range(top_k))
    ax2.set_yticklabels([f"d[{i}]" for i in top_indices], fontsize=7)
    ax2.set_title(f"Top {top_k} Direction Weights")
    ax2.set_xlabel("Weight")
    ax2.invert_yaxis()
    ax2.axvline(x=0, color="#aaaaaa", linestyle="-", alpha=0.3)

    # ── Panel 3: SAE feature weights (if available) ──
    if has_sae and sae_features and sae_weights:
        ax3 = axes[2]
        n_feats = len(sae_features)
        feat_labels = [f"F{int(f)}" for f in sae_features[:30]]
        feat_vals = sae_weights[:30]

        colors_sae = ["#e94560" if v > 0 else "#533483" for v in feat_vals]
        ax3.barh(range(len(feat_vals)), feat_vals, color=colors_sae,
                 edgecolor="#2a2a4a")
        ax3.set_yticks(range(len(feat_vals)))
        ax3.set_yticklabels(feat_labels, fontsize=8)
        ax3.set_title(f"SAE Feature Activations (top {n_feats})")
        ax3.set_xlabel("Differential Activation")
        ax3.invert_yaxis()
        ax3.axvline(x=0, color="#aaaaaa", linestyle="-", alpha=0.3)

        # Annotate interpretability
        ax3.text(0.98, 0.02, f"{n_feats} active features\nout of SAE dictionary",
                 transform=ax3.transAxes, fontsize=8, va="bottom", ha="right",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a1a", alpha=0.8))

    plt.tight_layout()
    tensor = _render_figure_to_tensor(fig)
    plt.close(fig)
    return tensor


def _build_summary(data: dict, lens_name: str, lens_path: str) -> str:
    """Build a text summary of a lens's properties."""
    lines = [f"=== Lens: {lens_name} ==="]

    if isinstance(data, dict):
        # Basic info
        if "concept" in data:
            lines.append(f"Concept: {data['concept']}")
        if "training_mode" in data:
            lines.append(f"Training Mode: {data['training_mode']}")
        if "target_model" in data:
            lines.append(f"Target Model: {data['target_model']}")

        # Direction stats
        if "direction" in data:
            d = data["direction"].float().squeeze()
            lines.append(f"Direction: {d.shape[0]}d, norm={d.norm():.4f}")

        if "direction_dim" in data:
            lines.append(f"Direction Dim: {data['direction_dim']}")

        # DPO stats
        if "dpo_accuracy" in data:
            lines.append(f"DPO Accuracy: {data['dpo_accuracy']:.1%}")
        if "dpo_beta" in data:
            lines.append(f"DPO Beta: {data['dpo_beta']}")
        if "dpo_mean_margin" in data:
            lines.append(f"DPO Mean Margin: {data['dpo_mean_margin']:+.4f}")
        if "dpo_min_margin" in data:
            lines.append(f"DPO Min Margin: {data['dpo_min_margin']:+.4f}")

        # SAE stats
        if "sae_feature_indices" in data:
            feats = data["sae_feature_indices"]
            n = len(feats) if isinstance(
                feats, (list, tuple)) else feats.shape[0]
            lines.append(f"SAE Features: {n} active")
        if "sae_layer" in data:
            lines.append(f"SAE Layer: {data['sae_layer']}")
        if "sae_expansion" in data:
            lines.append(f"SAE Expansion: {data['sae_expansion']}x")
        if "sae_direction_weight" in data:
            lines.append(
                f"SAE/DPO Blend: {data['sae_direction_weight']:.0%} SAE")

        # Cross-modal
        if "siglip_dim" in data:
            lines.append(f"SigLIP Dim: {data['siglip_dim']}")
        if "cross_modal_direction_acc" in data:
            lines.append(
                f"Cross-Modal Acc: {data['cross_modal_direction_acc']:.1%}")

        # Training info
        if "n_training_pairs" in data:
            lines.append(f"Training Pairs: {data['n_training_pairs']}")
        if "encoder_name" in data:
            lines.append(f"Encoder: {data['encoder_name']}")

    lines.append(f"File: {lens_path}")
    file_size = os.path.getsize(lens_path) if os.path.isfile(lens_path) else 0
    lines.append(f"Size: {file_size / 1e6:.1f} MB")

    return "\n".join(lines)


# ── The Node ────────────────────────────────────────────────────────────────

class ConceptLensInspectNode(io.ComfyNode):
    """Inspect a concept lens and visualize its internal structure."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        available_lenses = _discover_lenses()
        lens_options = ["None"] + available_lenses

        return io.Schema(
            node_id="conceptsteer.LensInspect",
            display_name="Lens Inspect",
            description=(
                "Load a concept lens and visualize its internal structure: "
                "weight distribution, top direction components, SAE features "
                "breakdown, and training metadata."
            ),
            category="Concept Steer/Interpret",
            is_output_node=True,
            inputs=[
                io.Combo.Input(
                    "lens",
                    default="None",
                    options=lens_options,
                    tooltip="Select a concept lens to inspect",
                ),
                io.String.Input(
                    "custom_lens_path",
                    default="",
                    tooltip="Override: absolute path to a .pt lens file",
                ),
            ],
            outputs=[
                io.Image.Output("chart"),
                io.String.Output("summary"),
            ],
        )

    @classmethod
    def execute(
        cls,
        lens: str = "None",
        custom_lens_path: str = "",
    ):
        # ── Resolve lens path ──
        if custom_lens_path and custom_lens_path.strip():
            lens_path = custom_lens_path.strip()
        elif lens and lens != "None":
            lens_path = _resolve_lens_path(lens)
        else:
            _log("No lens selected for inspection")
            return io.NodeOutput(torch.zeros(1, 64, 64, 3), "No lens selected")

        if not lens_path or not os.path.isfile(lens_path):
            _log(f"Lens file not found: {lens_path}")
            return io.NodeOutput(torch.zeros(1, 64, 64, 3), f"File not found: {lens_path}")

        # ── Load lens ──
        lens_name = os.path.basename(lens_path)
        _log(f"Inspecting lens: {lens_name}")

        try:
            data = torch.load(lens_path, map_location="cpu",
                              weights_only=False)
        except Exception as e:
            _log(f"Failed to load lens: {e}")
            return io.NodeOutput(torch.zeros(1, 64, 64, 3), f"Load error: {e}")

        # ── Generate visualization ──
        chart_tensor = _generate_inspect_chart(data, lens_name)
        summary = _build_summary(data, lens_name, lens_path)

        _log(f"Inspection complete: {lens_name}")
        print(summary)

        return io.NodeOutput(chart_tensor, summary, ui=ui.PreviewImage(chart_tensor))
