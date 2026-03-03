"""
ConceptFeatureDashboard — Browse and explore an SAE feature atlas visually.

Takes a pre-built atlas JSON (from the Feature Atlas node) and renders an
interactive dashboard visualization showing:
  - Feature distribution by category (general / common / selective / rare / dead)
  - Top features ranked by selectivity (most interpretable first)
  - Co-occurrence cluster map
  - Feature search: find features by label keyword
  - Ready-to-paste feature indices for Gate/Probe nodes

This is the "browse and discover" node — use it after building an atlas
to explore what your SAE has learned before doing targeted experiments.

Workflow:
  [Feature Atlas] → atlas_path → [Feature Dashboard] → IMAGE + STRING

  Then take feature indices from the dashboard into Feature Gate or Feature Probe.

Modes:
  - overview:   Category distribution + top selective features
  - search:     Filter features by keyword in labels
  - cluster:    Visualize co-occurrence clusters
  - top:        Ranked list of most selective/interesting features
"""

import os
import sys
import json
import torch
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


# ── Dashboard rendering modes ───────────────────────────────────────────────

def _render_overview(atlas: dict) -> tuple[torch.Tensor, str]:
    """Overview: category distribution + top selective features + health summary."""
    plt = _setup_dark_style()

    features = atlas.get("features", {})
    diag = atlas.get("diagnostics", {})
    cats = atlas.get("feature_categories", {})

    fig, axes = plt.subplots(1, 3, figsize=(18, 7),
                             gridspec_kw={"width_ratios": [1, 1.5, 1.5]})

    # ── Panel 1: Category pie chart ──
    ax = axes[0]
    cat_labels = []
    cat_sizes = []
    cat_colors = ["#f85149", "#d29922", "#3fb950", "#58a6ff", "#333333"]
    for cat_name, color in zip(
        ["general", "common", "selective", "rare", "dead"], cat_colors
    ):
        count = cats.get(cat_name, 0)
        if count > 0:
            cat_labels.append(f"{cat_name}\n({count})")
            cat_sizes.append(count)

    if cat_sizes:
        wedges, texts, autotexts = ax.pie(
            cat_sizes, labels=cat_labels, colors=cat_colors[:len(cat_sizes)],
            autopct="%1.0f%%", startangle=90,
            textprops={"color": "#eaeaea", "fontsize": 9},
        )
        for at in autotexts:
            at.set_fontsize(8)
    ax.set_title("Feature Categories", fontweight="bold")

    # ── Panel 2: Top selective features ──
    ax = axes[1]
    # Sort by firing rate to find selective features (0.05-0.3 range)
    selective = []
    for feat_id, info in features.items():
        fr = info.get("firing_rate", 0)
        if 0.01 < fr < 0.5:  # not dead, not general
            selective.append((
                int(feat_id),
                fr,
                info.get("mean_activation", 0),
                info.get("label", ""),
                info.get("col_norm", 0),
            ))

    # Sort by ~selectivity: low firing rate but reasonable activation
    selective.sort(key=lambda x: x[1])
    n_show = min(20, len(selective))

    if n_show > 0:
        y_pos = range(n_show - 1, -1, -1)
        labels = []
        values = []
        colors = []

        for i in range(n_show):
            idx, fr, mean_act, label, norm = selective[i]
            lbl = f"F{idx}  {label}" if label else f"F{idx}"
            labels.append(lbl)
            values.append(fr)
            # Color by firing rate
            if fr < 0.05:
                colors.append("#58a6ff")  # rare = blue
            elif fr < 0.15:
                colors.append("#3fb950")  # selective = green
            else:
                colors.append("#d29922")  # common = yellow

        ax.barh(list(y_pos), values, color=colors, height=0.7, alpha=0.85)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, fontfamily="monospace", fontsize=7)
        ax.set_xlabel("Firing Rate")
        ax.set_title("Most Selective Features", fontweight="bold")

        for y, v in zip(y_pos, values):
            ax.text(v + 0.005, y, f"{v:.2%}", va="center", fontsize=7,
                    color="#cccccc")
    else:
        ax.text(0.5, 0.5, "No selective features found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="#8b949e")
        ax.set_title("Most Selective Features", fontweight="bold")

    # ── Panel 3: Health summary text ──
    ax = axes[2]
    ax.axis("off")

    d_sae = atlas.get("d_sae", 0)
    d_model = atlas.get("d_model", 0)
    recon_cos = diag.get("reconstruction_cosine", 0)

    health_lines = [
        "SAE Dashboard",
        "═" * 40,
        f"Dimensions: {d_model}d → {d_sae}d ({atlas.get('expansion', '?')}×)",
        f"Probed with: {atlas.get('n_prompts_probed', '?')} prompts",
        f"Created: {atlas.get('created_at', '?')}",
        "",
        "Health Metrics",
        "─" * 40,
        f"Reconstruction cos:  {recon_cos:.4f}  "
        f"{'✓' if recon_cos > 0.95 else '⚠' if recon_cos > 0.85 else '✗'}",
        f"Dead features:       {diag.get('dead_ratio', 0):.1%}  "
        f"{'✓' if diag.get('dead_ratio', 1) < 0.1 else '⚠'}",
        f"Mean pairwise cos:   {diag.get('mean_pairwise_cosine', 0):.4f}",
        f"Col norm μ±σ:        {diag.get('mean_col_norm', 0):.3f} ± "
        f"{diag.get('std_col_norm', 0):.3f}",
        "",
        "Feature Counts",
        "─" * 40,
        f"Total:     {d_sae:,}",
        f"Active:    {len(features):,}",
        f"Selective: {cats.get('selective', 0):,}  (most interpretable)",
        f"General:   {cats.get('general', 0):,}  (fire on everything)",
        f"Dead:      {cats.get('dead', 0):,}",
        "",
        f"Clusters:  {len(atlas.get('cooccurrence_clusters', [])):,}",
        f"Redundant: {len(atlas.get('redundant_pairs', [])):,} pairs",
    ]

    ax.text(0.05, 0.95, "\n".join(health_lines),
            transform=ax.transAxes, fontfamily="monospace", fontsize=9,
            verticalalignment="top", color="#eaeaea")

    plt.tight_layout()
    img = _render_figure_to_tensor(fig)
    plt.close(fig)

    # Text output: copy-paste feature lists
    text_lines = ["Feature Dashboard — Overview", ""]
    text_lines.append("Top selective features (paste into Feature Gate/Probe):")
    if selective:
        top_selective_ids = [str(s[0]) for s in selective[:15]]
        text_lines.append(f"  {','.join(top_selective_ids)}")
        text_lines.append("")
        for idx, fr, mean_act, label, norm in selective[:15]:
            lbl = f"  {label}" if label else ""
            text_lines.append(
                f"  F{idx:>6d}: fires {fr:.1%}, "
                f"mean_act={mean_act:.3f}, norm={norm:.3f}{lbl}"
            )

    return img, "\n".join(text_lines)


def _render_search(atlas: dict, keyword: str) -> tuple[torch.Tensor, str]:
    """Search mode: find features by keyword in labels."""
    plt = _setup_dark_style()
    features = atlas.get("features", {})

    keyword_lower = keyword.lower().strip()
    matches = []
    for feat_id, info in features.items():
        label = info.get("label", "").lower()
        if keyword_lower in label:
            matches.append((
                int(feat_id),
                info.get("firing_rate", 0),
                info.get("mean_activation", 0),
                info.get("label", ""),
                info.get("col_norm", 0),
            ))

    matches.sort(key=lambda x: -x[2])  # sort by activation strength
    n_show = min(30, len(matches))

    if n_show == 0:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5,
                f"No features matching '{keyword}'\n\n"
                f"Try broader terms or run Feature Dictionary with more prompts.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="#f85149")
        ax.axis("off")
        plt.tight_layout()
        img = _render_figure_to_tensor(fig)
        plt.close(fig)
        return img, f"No features matching '{keyword}'"

    fig_height = max(3, n_show * 0.35 + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_pos = range(n_show - 1, -1, -1)
    labels = []
    values = []

    for i in range(n_show):
        idx, fr, mean_act, label, norm = matches[i]
        lbl = f"F{idx}  {label}" if label else f"F{idx}"
        labels.append(lbl)
        values.append(mean_act)

    ax.barh(list(y_pos), values, color="#58a6ff", height=0.7, alpha=0.85)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontfamily="monospace", fontsize=8)
    ax.set_xlabel("Mean Activation")
    ax.set_title(f"Features matching '{keyword}' ({len(matches)} found)",
                 fontweight="bold")

    for y, v in zip(y_pos, values):
        ax.text(v + max(values) * 0.02, y, f"{v:.3f}",
                va="center", fontsize=7, color="#cccccc")

    plt.tight_layout()
    img = _render_figure_to_tensor(fig)
    plt.close(fig)

    text_lines = [
        f"Feature Search: '{keyword}' ({len(matches)} matches)",
        "",
        "Feature indices (paste into Gate/Probe):",
        ",".join(str(m[0]) for m in matches[:15]),
        "",
    ]
    for idx, fr, mean_act, label, norm in matches[:20]:
        text_lines.append(
            f"  F{idx:>6d}: fires {fr:.1%}, act={mean_act:.3f}  {label}")

    return img, "\n".join(text_lines)


def _render_clusters(atlas: dict) -> tuple[torch.Tensor, str]:
    """Cluster mode: visualize co-occurrence clusters."""
    plt = _setup_dark_style()
    features = atlas.get("features", {})
    clusters = atlas.get("cooccurrence_clusters", [])

    if not clusters:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5,
                "No co-occurrence clusters found\n\n"
                "Try running Feature Atlas with more prompts.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="#8b949e")
        ax.axis("off")
        plt.tight_layout()
        img = _render_figure_to_tensor(fig)
        plt.close(fig)
        return img, "No co-occurrence clusters found"

    n_clusters = min(8, len(clusters))
    fig, ax = plt.subplots(figsize=(14, max(4, n_clusters * 1.2)))

    cluster_colors = [
        "#58a6ff", "#3fb950", "#d29922", "#f85149",
        "#bc8cff", "#f0883e", "#a5d6ff", "#ffa657",
    ]

    y = 0
    y_positions = []
    bar_widths = []
    bar_colors = []
    bar_labels = []

    text_lines = [
        f"Co-occurrence Clusters ({len(clusters)} total)",
        "",
    ]

    for ci, cluster in enumerate(clusters[:n_clusters]):
        color = cluster_colors[ci % len(cluster_colors)]
        cluster_feat_labels = []

        for fi in cluster:
            label = features.get(str(fi), {}).get("label", "")
            fr = features.get(str(fi), {}).get("firing_rate", 0)
            lbl = f"F{fi} {label}" if label else f"F{fi}"
            cluster_feat_labels.append(lbl)

            y_positions.append(y)
            bar_widths.append(fr)
            bar_colors.append(color)
            bar_labels.append(lbl)
            y += 1

        text_lines.append(
            f"Cluster {ci+1} ({len(cluster)} features, color: "
            f"{['blue','green','yellow','red','purple','orange','cyan','amber'][ci % 8]}):"
        )
        feat_ids = [str(fi) for fi in cluster]
        text_lines.append(f"  Indices: {','.join(feat_ids)}")
        for lbl in cluster_feat_labels[:5]:
            text_lines.append(f"    {lbl}")
        if len(cluster) > 5:
            text_lines.append(f"    ... and {len(cluster)-5} more")
        text_lines.append("")

        # Add gap between clusters
        y += 0.5

    ax.barh(y_positions, bar_widths, color=bar_colors, height=0.8, alpha=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(bar_labels, fontfamily="monospace", fontsize=7)
    ax.set_xlabel("Firing Rate")
    ax.set_title(f"Co-occurrence Clusters (features that fire together)",
                 fontweight="bold")
    ax.invert_yaxis()

    plt.tight_layout()
    img = _render_figure_to_tensor(fig)
    plt.close(fig)

    return img, "\n".join(text_lines)


def _render_top(atlas: dict, sort_by: str) -> tuple[torch.Tensor, str]:
    """Top mode: ranked list of features sorted by specified metric."""
    plt = _setup_dark_style()
    features = atlas.get("features", {})

    all_feats = []
    for feat_id, info in features.items():
        all_feats.append((
            int(feat_id),
            info.get("firing_rate", 0),
            info.get("mean_activation", 0),
            info.get("col_norm", 0),
            info.get("label", ""),
            info.get("category", ""),
        ))

    if sort_by == "activation":
        all_feats.sort(key=lambda x: -x[2])
        metric_label = "Mean Activation"
        metric_idx = 2
    elif sort_by == "norm":
        all_feats.sort(key=lambda x: -x[3])
        metric_label = "Column Norm"
        metric_idx = 3
    elif sort_by == "rare":
        # Filter to only features that fire, sort by rarity
        all_feats = [f for f in all_feats if f[1] > 0]
        all_feats.sort(key=lambda x: x[1])
        metric_label = "Firing Rate (rarest first)"
        metric_idx = 1
    else:  # selectivity — features that fire 5-30%
        all_feats = [f for f in all_feats if 0.01 < f[1] < 0.5]
        all_feats.sort(key=lambda x: x[1])
        metric_label = "Firing Rate (most selective)"
        metric_idx = 1

    n_show = min(35, len(all_feats))

    fig_height = max(4, n_show * 0.35 + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    if n_show > 0:
        y_pos = range(n_show - 1, -1, -1)
        labels = []
        values = []
        colors = []

        for i in range(n_show):
            idx, fr, mean_act, norm, label, cat = all_feats[i]
            lbl = f"F{idx}  {label}" if label else f"F{idx}"
            labels.append(lbl)
            values.append(all_feats[i][metric_idx])
            cat_colors = {
                "general": "#f85149",
                "common": "#d29922",
                "selective": "#3fb950",
                "rare": "#58a6ff",
            }
            colors.append(cat_colors.get(cat, "#8b949e"))

        ax.barh(list(y_pos), values, color=colors, height=0.7, alpha=0.85)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, fontfamily="monospace", fontsize=7)
        ax.set_xlabel(metric_label)
        ax.set_title(f"Top Features by {metric_label}", fontweight="bold")

        max_val = max(values) if values else 1
        for y, v in zip(y_pos, values):
            fmt = f"{v:.3f}" if v > 0.01 else f"{v:.1%}"
            ax.text(v + max_val * 0.02, y, fmt,
                    va="center", fontsize=7, color="#cccccc")

    plt.tight_layout()
    img = _render_figure_to_tensor(fig)
    plt.close(fig)

    text_lines = [
        f"Top Features by {metric_label}",
        "",
        f"Feature indices: {','.join(str(f[0]) for f in all_feats[:15])}",
        "",
    ]
    for idx, fr, mean_act, norm, label, cat in all_feats[:20]:
        lbl = f"  {label}" if label else ""
        text_lines.append(
            f"  F{idx:>6d} [{cat:>9s}]: fires={fr:.1%}, "
            f"act={mean_act:.3f}, norm={norm:.3f}{lbl}"
        )

    return img, "\n".join(text_lines)


# ── The Node ─────────────────────────────────────────────────────────────────

class ConceptFeatureDashboardNode(io.ComfyNode):
    """Browse and explore an SAE feature atlas visually."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        default_atlas = str(_PACKAGE_ROOT / "sae" / "feature_atlas.json")

        return io.Schema(
            node_id="conceptsteer.FeatureDashboard",
            display_name="Feature Dashboard",
            description=(
                "Browse a pre-built feature atlas to explore what your SAE "
                "has learned. View feature distributions, search by keyword, "
                "explore co-occurrence clusters, and rank features by "
                "selectivity. Outputs feature indices ready to paste into "
                "Feature Gate or Feature Probe."
            ),
            category="Concept Steer/Research",
            inputs=[
                io.String.Input(
                    "atlas_path",
                    default=default_atlas,
                    tooltip=(
                        "Path to feature atlas JSON (from Feature Atlas node). "
                        "Also accepts feature dictionary JSON from Feature Dictionary."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    default="overview",
                    options=["overview", "search", "clusters", "top_selective",
                             "top_activation", "top_rare", "top_norm"],
                    tooltip=(
                        "'overview' = category distribution + health summary. "
                        "'search' = find features by keyword in labels. "
                        "'clusters' = co-occurrence cluster visualization. "
                        "'top_selective' = features that fire 5-30% (most interesting). "
                        "'top_activation' = highest mean activation. "
                        "'top_rare' = rarest features. "
                        "'top_norm' = strongest decoder columns."
                    ),
                ),
                io.String.Input(
                    "search_keyword",
                    default="",
                    tooltip=(
                        "Keyword to search for in feature labels. "
                        "Only used in 'search' mode. "
                        "Try: 'lighting', 'color', 'texture', 'dark', etc."
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
        atlas_path: str = "",
        mode: str = "overview",
        search_keyword: str = "",
    ):
        atlas_path = atlas_path.strip()

        if not atlas_path:
            _log("No atlas path provided")
            return io.NodeOutput(
                torch.zeros(1, 64, 64, 3),
                "Error: atlas_path required. Run Feature Atlas first.",
            )

        if not os.path.isabs(atlas_path):
            atlas_path = str(_PACKAGE_ROOT / atlas_path)

        if not os.path.isfile(atlas_path):
            _log(f"Atlas file not found: {atlas_path}")
            return io.NodeOutput(
                torch.zeros(1, 64, 64, 3),
                f"Error: file not found at {atlas_path}. Run Feature Atlas first.",
            )

        # ── Load atlas ──
        with open(atlas_path) as f:
            atlas = json.load(f)

        _log(f"Loaded atlas: {len(atlas.get('features', {}))} features, "
             f"mode={mode}")

        # ── Render based on mode ──
        if mode == "search":
            if not search_keyword.strip():
                return io.NodeOutput(
                    torch.zeros(1, 64, 64, 3),
                    "Error: search_keyword required in search mode. "
                    "Try 'lighting', 'color', 'texture', etc.",
                )
            img, text = _render_search(atlas, search_keyword)
        elif mode == "clusters":
            img, text = _render_clusters(atlas)
        elif mode.startswith("top_"):
            sort_key = mode.replace("top_", "")
            sort_map = {
                "selective": "selectivity",
                "activation": "activation",
                "rare": "rare",
                "norm": "norm",
            }
            img, text = _render_top(atlas, sort_map.get(sort_key, "selectivity"))
        else:  # overview
            img, text = _render_overview(atlas)

        return io.NodeOutput(img, text)
