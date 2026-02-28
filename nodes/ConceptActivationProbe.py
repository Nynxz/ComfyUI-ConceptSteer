"""
ConceptActivationProbe — Probe how a concept lens affects conditioning in detail.

Takes CONDITIONING before AND after steering, and analyzes the differences:
  - Per-token cosine similarity (how much each token changed)
  - Norm change per token (magnitude of intervention)
  - Direction alignment per token (how much each token aligns with the lens)
  - Global statistics (average shift, max shift, etc.)

Outputs a multi-panel visualization image and text summary.

Usage in ComfyUI:
  [CLIP Text Encode] → conditioning →─┬─→ [Concept Steer] → steered
                                       │                        │
                                       └──→ [Activation Probe] ←┘
                                                    ↓
                                            IMAGE (charts) + STRING (analysis)
"""

import os
import torch
import numpy as np
from io import BytesIO
from comfy_api.latest import io, ui


def _log(msg: str):
    print(f"[Concept Steer] {msg}")


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


class ConceptActivationProbeNode(io.ComfyNode):
    """Probe how a concept lens affects conditioning — visualize the intervention."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="conceptsteer.ActivationProbe",
            display_name="Activation Probe",
            description=(
                "Analyze the difference between original and steered conditioning. "
                "Shows per-token cosine similarity, norm changes, and direction "
                "alignment to help understand how the lens affects generation."
            ),
            category="Concept Steer/Interpret",
            is_output_node=True,
            inputs=[
                io.Conditioning.Input(
                    "original",
                    tooltip="Original conditioning BEFORE steering",
                ),
                io.Conditioning.Input(
                    "steered",
                    tooltip="Conditioning AFTER applying a concept lens",
                ),
                io.String.Input(
                    "label",
                    default="",
                    tooltip="Optional label for the comparison (e.g. concept name)",
                ),
            ],
            outputs=[
                io.Image.Output("chart"),
                io.String.Output("analysis"),
            ],
        )

    @classmethod
    def execute(
        cls,
        original,
        steered,
        label: str = "",
    ):
        if not original or not steered:
            _log("Missing conditioning input(s)")
            return io.NodeOutput(
                torch.zeros(1, 64, 64, 3),
                "Error: Missing conditioning"
            )

        # Extract first conditioning entry
        orig_tensor = original[0][0].float()   # [B, tokens, dim]
        steer_tensor = steered[0][0].float()    # [B, tokens, dim]

        if orig_tensor.shape != steer_tensor.shape:
            _log(
                f"Shape mismatch: original {orig_tensor.shape} vs steered {steer_tensor.shape}")
            return io.NodeOutput(
                torch.zeros(1, 64, 64, 3),
                f"Shape mismatch: {orig_tensor.shape} vs {steer_tensor.shape}"
            )

        B, T, D = orig_tensor.shape
        _log(f"Probing: {B} batch × {T} tokens × {D}d")

        # ── Compute metrics ──────────────────────────────────────────────
        delta = steer_tensor - orig_tensor  # [B, T, D]

        # Per-token norms
        orig_norms = orig_tensor.norm(dim=-1)   # [B, T]
        steer_norms = steer_tensor.norm(dim=-1)  # [B, T]
        delta_norms = delta.norm(dim=-1)         # [B, T]

        # Per-token cosine similarity (before vs after)
        cos_sim = torch.nn.functional.cosine_similarity(
            orig_tensor, steer_tensor, dim=-1
        )  # [B, T]

        # Active token mask (non-padding)
        active_mask = orig_norms > 0.01  # [B, T]

        # Average over batch dim
        cos_per_token = cos_sim.mean(dim=0).cpu().numpy()      # [T]
        delta_per_token = delta_norms.mean(dim=0).cpu().numpy()  # [T]
        orig_norm_per_token = orig_norms.mean(dim=0).cpu().numpy()
        steer_norm_per_token = steer_norms.mean(dim=0).cpu().numpy()
        active_per_token = active_mask.float().mean(dim=0).cpu().numpy()

        # Global stats
        active_cos = cos_sim[active_mask]
        active_delta = delta_norms[active_mask]
        global_stats = {
            "tokens": T,
            "dim": D,
            "batch": B,
            "active_tokens": int(active_mask.sum().item() / B),
            "mean_cosine": float(active_cos.mean().item()) if active_cos.numel() > 0 else 0.0,
            "min_cosine": float(active_cos.min().item()) if active_cos.numel() > 0 else 0.0,
            "mean_delta_norm": float(active_delta.mean().item()) if active_delta.numel() > 0 else 0.0,
            "max_delta_norm": float(active_delta.max().item()) if active_delta.numel() > 0 else 0.0,
            "mean_orig_norm": float(orig_norms[active_mask].mean().item()) if active_cos.numel() > 0 else 0.0,
            "relative_perturbation": float(
                (active_delta.mean() / orig_norms[active_mask].mean()).item()
            ) if active_cos.numel() > 0 and orig_norms[active_mask].mean() > 0 else 0.0,
        }

        # ── Generate visualization ──
        plt = _setup_dark_style()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        title = f"Activation Probe: {label}" if label else "Activation Probe"
        fig.suptitle(title, fontsize=16, fontweight="bold",
                     color="#e94560", y=1.01)

        # ── Panel 1: Per-token cosine similarity ──
        ax1 = axes[0, 0]
        token_indices = range(T)
        colors_cos = ["#e94560" if a >
                      0.5 else "#333355" for a in active_per_token]
        ax1.bar(token_indices, cos_per_token,
                color=colors_cos, alpha=0.85, width=1.0)
        ax1.axhline(y=1.0, color="#aaaaaa", linestyle="--", alpha=0.4)
        ax1.axhline(y=global_stats["mean_cosine"], color="#533483",
                    linestyle="-", alpha=0.7, label=f"mean: {global_stats['mean_cosine']:.4f}")
        ax1.set_title("Per-Token Cosine Similarity (original ↔ steered)")
        ax1.set_xlabel("Token Position")
        ax1.set_ylabel("Cosine Similarity")
        ax1.set_ylim(min(0.9, cos_per_token.min() - 0.01)
                     if cos_per_token.min() > 0.5 else 0, 1.02)
        ax1.legend(fontsize=8)

        # ── Panel 2: Per-token delta norm ──
        ax2 = axes[0, 1]
        ax2.bar(token_indices, delta_per_token,
                color="#0f3460", alpha=0.85, width=1.0)
        ax2.axhline(y=global_stats["mean_delta_norm"], color="#e94560",
                    linestyle="-", alpha=0.7,
                    label=f"mean: {global_stats['mean_delta_norm']:.4f}")
        ax2.set_title("Per-Token Perturbation Magnitude")
        ax2.set_xlabel("Token Position")
        ax2.set_ylabel("‖delta‖")
        ax2.legend(fontsize=8)

        # ── Panel 3: Norm comparison (original vs steered) ──
        ax3 = axes[1, 0]
        width = 0.4
        x = np.arange(T)
        ax3.bar(x - width / 2, orig_norm_per_token, width,
                label="Original", color="#0f3460", alpha=0.7)
        ax3.bar(x + width / 2, steer_norm_per_token, width,
                label="Steered", color="#e94560", alpha=0.7)
        ax3.set_title("Token Norm Comparison")
        ax3.set_xlabel("Token Position")
        ax3.set_ylabel("‖token‖")
        ax3.legend(fontsize=8)

        # ── Panel 4: Summary statistics ──
        ax4 = axes[1, 1]
        ax4.set_axis_off()

        stats_lines = [
            f"Conditioning: {B} batch × {T} tokens × {D}d",
            f"Active tokens: {global_stats['active_tokens']} / {T}",
            "",
            "─── Cosine Similarity ───",
            f"  Mean:  {global_stats['mean_cosine']:.6f}",
            f"  Min:   {global_stats['min_cosine']:.6f}",
            f"  Deviation: {1 - global_stats['mean_cosine']:.6f}",
            "",
            "─── Perturbation ───",
            f"  Mean ‖Δ‖: {global_stats['mean_delta_norm']:.4f}",
            f"  Max ‖Δ‖:  {global_stats['max_delta_norm']:.4f}",
            f"  Mean ‖orig‖: {global_stats['mean_orig_norm']:.4f}",
            f"  Relative: {global_stats['relative_perturbation']:.2%}",
        ]

        if global_stats["relative_perturbation"] < 0.01:
            stats_lines.append("\n⚡ Very subtle steering")
        elif global_stats["relative_perturbation"] < 0.05:
            stats_lines.append("\n✓ Moderate steering")
        elif global_stats["relative_perturbation"] < 0.15:
            stats_lines.append("\n⚠ Strong steering")
        else:
            stats_lines.append("\n🔥 Very aggressive steering")

        stats_text = "\n".join(stats_lines)
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                 fontsize=10, va="top", ha="left", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="#0a0a1a", alpha=0.9))

        plt.tight_layout()
        chart_tensor = _render_figure_to_tensor(fig)
        plt.close(fig)

        # ── Build analysis text ──
        analysis_lines = [
            f"=== Activation Probe {'(' + label + ')' if label else ''} ===",
            f"Shape: {B}×{T}×{D}",
            f"Active: {global_stats['active_tokens']}/{T} tokens",
            f"Mean cos(orig,steered): {global_stats['mean_cosine']:.6f}",
            f"Min cos: {global_stats['min_cosine']:.6f}",
            f"Mean ‖Δ‖: {global_stats['mean_delta_norm']:.4f}",
            f"Max ‖Δ‖: {global_stats['max_delta_norm']:.4f}",
            f"Relative perturbation: {global_stats['relative_perturbation']:.2%}",
        ]
        analysis = "\n".join(analysis_lines)

        _log(f"Probe complete: mean_cos={global_stats['mean_cosine']:.4f}, "
             f"rel_pert={global_stats['relative_perturbation']:.2%}")

        return io.NodeOutput(chart_tensor, analysis, ui=ui.PreviewImage(chart_tensor))
