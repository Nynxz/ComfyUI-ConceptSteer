"""
ConceptLensCompare — Compare two concept lenses side by side.

Loads two lens files and visualizes their relationship:
  - Cosine similarity between direction vectors
  - Shared vs unique top-weight components
  - Side-by-side weight distribution overlays
  - SAE feature overlap (if both are SAE lenses)
  - Orthogonality analysis (are they composable?)

Outputs a comparison chart image and text analysis.

Usage in ComfyUI:
  [Lens Compare] → IMAGE (chart) + STRING (analysis)
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


# ── Lens loading helpers ────────────────────────────────────────────────────

def _find_lens_root() -> str:
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
    root = _find_lens_root()
    full = os.path.normpath(os.path.join(root, filename))
    if os.path.isfile(full):
        return full
    basename = os.path.basename(filename)
    for dirpath, _dirnames, filenames in os.walk(root):
        if basename in filenames:
            return os.path.join(dirpath, basename)
    return None


def _extract_direction(data) -> torch.Tensor | None:
    """Extract direction vector from lens data."""
    if isinstance(data, torch.Tensor):
        return data.float().squeeze()
    if isinstance(data, dict):
        for key in ["direction", "d_in_siglip", "d_shared"]:
            if key in data:
                return data[key].float().squeeze()
        for val in data.values():
            if isinstance(val, torch.Tensor) and val.dim() <= 2 and 256 <= val.numel() <= 8192:
                return val.float().reshape(-1)
    return None


def _get_concept_name(data, fallback: str) -> str:
    if isinstance(data, dict) and "concept" in data:
        return data["concept"]
    return os.path.splitext(os.path.basename(fallback))[0]


class ConceptLensCompareNode(io.ComfyNode):
    """Compare two concept lenses — analyze overlap, orthogonality, and features."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        available_lenses = _discover_lenses()
        lens_options = ["None"] + available_lenses

        return io.Schema(
            node_id="conceptsteer.LensCompare",
            display_name="Lens Compare",
            description=(
                "Compare two concept lenses side by side. Shows cosine similarity, "
                "shared weight components, SAE feature overlap, and composability "
                "analysis (orthogonality)."
            ),
            category="Concept Steer/Interpret",
            is_output_node=True,
            inputs=[
                io.Combo.Input(
                    "lens_a",
                    default="None",
                    options=lens_options,
                    tooltip="First lens to compare",
                ),
                io.Combo.Input(
                    "lens_b",
                    default="None",
                    options=lens_options,
                    tooltip="Second lens to compare",
                ),
                io.String.Input(
                    "custom_path_a",
                    default="",
                    tooltip="Override: absolute path to first lens",
                ),
                io.String.Input(
                    "custom_path_b",
                    default="",
                    tooltip="Override: absolute path to second lens",
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
        lens_a: str = "None",
        lens_b: str = "None",
        custom_path_a: str = "",
        custom_path_b: str = "",
    ):
        # ── Resolve paths ──
        path_a = custom_path_a.strip() if custom_path_a.strip() else (
            _resolve_lens_path(lens_a) if lens_a != "None" else None
        )
        path_b = custom_path_b.strip() if custom_path_b.strip() else (
            _resolve_lens_path(lens_b) if lens_b != "None" else None
        )

        if not path_a or not os.path.isfile(path_a):
            _log(f"Lens A not found: {path_a}")
            return io.NodeOutput(torch.zeros(1, 64, 64, 3), "Lens A not found")
        if not path_b or not os.path.isfile(path_b):
            _log(f"Lens B not found: {path_b}")
            return io.NodeOutput(torch.zeros(1, 64, 64, 3), "Lens B not found")

        # ── Load both lenses ──
        try:
            data_a = torch.load(path_a, map_location="cpu", weights_only=False)
            data_b = torch.load(path_b, map_location="cpu", weights_only=False)
        except Exception as e:
            _log(f"Failed to load lenses: {e}")
            return io.NodeOutput(torch.zeros(1, 64, 64, 3), f"Load error: {e}")

        dir_a = _extract_direction(data_a)
        dir_b = _extract_direction(data_b)

        if dir_a is None or dir_b is None:
            return io.NodeOutput(
                torch.zeros(1, 64, 64, 3),
                "Could not extract direction from one or both lenses"
            )

        name_a = _get_concept_name(data_a, path_a)
        name_b = _get_concept_name(data_b, path_b)

        _log(
            f"Comparing: {name_a} ({dir_a.shape[0]}d) vs {name_b} ({dir_b.shape[0]}d)")

        # ── Align dimensions if needed ──
        if dir_a.shape[0] != dir_b.shape[0]:
            max_dim = max(dir_a.shape[0], dir_b.shape[0])
            if dir_a.shape[0] < max_dim:
                padded = torch.zeros(max_dim)
                padded[:dir_a.shape[0]] = dir_a
                dir_a = padded
            if dir_b.shape[0] < max_dim:
                padded = torch.zeros(max_dim)
                padded[:dir_b.shape[0]] = dir_b
                dir_b = padded

        # ── Compute metrics ──
        import torch.nn.functional as F
        cos_sim = F.cosine_similarity(
            dir_a.unsqueeze(0), dir_b.unsqueeze(0)).item()
        dot_product = (dir_a @ dir_b).item()

        # Project a onto b and vice versa
        proj_a_on_b = (dir_a @ F.normalize(dir_b, dim=0)).item()
        proj_b_on_a = (dir_b @ F.normalize(dir_a, dim=0)).item()

        # Orthogonal component of B relative to A (composability)
        unit_a = F.normalize(dir_a, dim=0)
        b_parallel = (dir_b @ unit_a) * unit_a
        b_orthogonal = dir_b - b_parallel
        orthogonal_ratio = b_orthogonal.norm().item() / (dir_b.norm().item() + 1e-8)

        # SAE feature overlap
        sae_overlap_info = None
        if isinstance(data_a, dict) and isinstance(data_b, dict):
            feats_a = data_a.get("sae_feature_indices")
            feats_b = data_b.get("sae_feature_indices")
            if feats_a is not None and feats_b is not None:
                if isinstance(feats_a, torch.Tensor):
                    feats_a = set(feats_a.cpu().tolist())
                else:
                    feats_a = set(feats_a)
                if isinstance(feats_b, torch.Tensor):
                    feats_b = set(feats_b.cpu().tolist())
                else:
                    feats_b = set(feats_b)
                shared = feats_a & feats_b
                only_a = feats_a - feats_b
                only_b = feats_b - feats_a
                sae_overlap_info = {
                    "shared": len(shared),
                    "only_a": len(only_a),
                    "only_b": len(only_b),
                    "total_a": len(feats_a),
                    "total_b": len(feats_b),
                    "jaccard": len(shared) / len(feats_a | feats_b) if (feats_a | feats_b) else 0,
                    "shared_ids": sorted(shared)[:20],
                }

        # ── Generate visualization ──
        plt = _setup_dark_style()
        n_panels = 3 if sae_overlap_info else 2
        fig_width = 6 * n_panels
        fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 6))
        if n_panels == 1:
            axes = [axes]

        fig.suptitle(f"Lens Compare: {name_a} vs {name_b}",
                     fontsize=16, fontweight="bold", color="#e94560", y=1.02)

        # ── Panel 1: Weight distribution overlay ──
        ax1 = axes[0]
        d_a_np = dir_a.cpu().numpy()
        d_b_np = dir_b.cpu().numpy()
        ax1.hist(d_a_np, bins=50, alpha=0.6, color="#e94560",
                 label=name_a, edgecolor="none")
        ax1.hist(d_b_np, bins=50, alpha=0.6, color="#0f3460",
                 label=name_b, edgecolor="none")
        ax1.axvline(x=0, color="#aaaaaa", linestyle="--", alpha=0.4)
        ax1.set_title("Weight Distribution Overlay")
        ax1.set_xlabel("Weight Value")
        ax1.set_ylabel("Count")
        ax1.legend(fontsize=9)

        # Stats box
        stats_text = (
            f"cos(A,B): {cos_sim:+.4f}\n"
            f"dot(A,B): {dot_product:+.4f}\n"
            f"proj(A→B): {proj_a_on_b:+.4f}\n"
            f"proj(B→A): {proj_b_on_a:+.4f}\n"
            f"orthogonal: {orthogonal_ratio:.1%}\n"
            f"{'─' * 20}\n"
            f"composable: {'✓ yes' if abs(cos_sim) < 0.3 else '~ partial' if abs(cos_sim) < 0.6 else '✗ aligned'}"
        )
        ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
                 fontsize=8, va="top", ha="right", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a0a1a", alpha=0.8))

        # ── Panel 2: Top-K largest weights side by side ──
        ax2 = axes[1]
        top_k = 20
        dim = len(d_a_np)

        # Get top components from each
        top_a = np.argsort(np.abs(d_a_np))[-top_k:][::-1]
        top_b = np.argsort(np.abs(d_b_np))[-top_k:][::-1]

        # Merge unique indices
        all_top = list(dict.fromkeys(list(top_a) + list(top_b)))[:top_k]

        y_pos = np.arange(len(all_top))
        width = 0.35
        vals_a = [d_a_np[i] for i in all_top]
        vals_b = [d_b_np[i] for i in all_top]

        ax2.barh(y_pos - width / 2, vals_a, width, label=name_a,
                 color="#e94560", alpha=0.8)
        ax2.barh(y_pos + width / 2, vals_b, width, label=name_b,
                 color="#0f3460", alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"d[{i}]" for i in all_top], fontsize=7)
        ax2.set_title(f"Top-{len(all_top)} Weight Components")
        ax2.set_xlabel("Weight")
        ax2.invert_yaxis()
        ax2.axvline(x=0, color="#aaaaaa", linestyle="-", alpha=0.3)
        ax2.legend(fontsize=8)

        # ── Panel 3: SAE feature overlap (if available) ──
        if sae_overlap_info and n_panels > 2:
            ax3 = axes[2]

            # Venn-like bar chart
            categories = [f"Only {name_a}", "Shared", f"Only {name_b}"]
            values = [sae_overlap_info["only_a"], sae_overlap_info["shared"],
                      sae_overlap_info["only_b"]]
            colors = ["#e94560", "#533483", "#0f3460"]

            bars = ax3.bar(categories, values, color=colors,
                           edgecolor="#2a2a4a")
            ax3.set_title("SAE Feature Overlap")
            ax3.set_ylabel("Feature Count")

            # Add value labels
            for bar, val in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         str(val), ha="center", va="bottom", fontsize=10,
                         fontweight="bold")

            # Jaccard annotation
            ax3.text(0.98, 0.98,
                     f"Jaccard: {sae_overlap_info['jaccard']:.2%}\n"
                     f"A: {sae_overlap_info['total_a']} features\n"
                     f"B: {sae_overlap_info['total_b']} features",
                     transform=ax3.transAxes, fontsize=9, va="top", ha="right",
                     fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a1a", alpha=0.8))

        plt.tight_layout()
        chart_tensor = _render_figure_to_tensor(fig)
        plt.close(fig)

        # ── Build analysis text ──
        lines = [
            f"=== Lens Compare: {name_a} vs {name_b} ===",
            f"Dimensions: {dir_a.shape[0]}d vs {dir_b.shape[0]}d",
            f"Cosine similarity: {cos_sim:+.4f}",
            f"Dot product: {dot_product:+.4f}",
            f"Projection A→B: {proj_a_on_b:+.4f}",
            f"Projection B→A: {proj_b_on_a:+.4f}",
            f"Orthogonal ratio: {orthogonal_ratio:.1%}",
            "",
        ]

        if abs(cos_sim) < 0.1:
            lines.append(
                "Assessment: Nearly orthogonal — excellent composability")
        elif abs(cos_sim) < 0.3:
            lines.append("Assessment: Low overlap — good composability")
        elif abs(cos_sim) < 0.6:
            lines.append(
                "Assessment: Moderate overlap — partial composability")
        elif cos_sim > 0.6:
            lines.append("Assessment: Highly aligned — may be redundant")
        else:
            lines.append(
                "Assessment: Opposing directions — anti-correlated concepts")

        if sae_overlap_info:
            lines.extend([
                "",
                "SAE Feature Overlap:",
                f"  Shared: {sae_overlap_info['shared']}",
                f"  Only {name_a}: {sae_overlap_info['only_a']}",
                f"  Only {name_b}: {sae_overlap_info['only_b']}",
                f"  Jaccard index: {sae_overlap_info['jaccard']:.2%}",
            ])
            if sae_overlap_info['shared_ids']:
                lines.append(
                    f"  Shared feature IDs: {sae_overlap_info['shared_ids']}")

        analysis = "\n".join(lines)
        _log(
            f"Compare complete: cos={cos_sim:+.4f}, ortho={orthogonal_ratio:.1%}")

        return io.NodeOutput(chart_tensor, analysis, ui=ui.PreviewImage(chart_tensor))
