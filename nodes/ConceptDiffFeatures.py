"""
ConceptDiffFeatures — Discover what SAE features differentiate two conditionings.

Takes two CONDITIONING inputs (e.g. "cinematic photo" vs "snapshot photo") and
decomposes both through the same SAE to find which features are:
  - Uniquely active in A but not B
  - Uniquely active in B but not A
  - Shared but at different magnitudes

This is the primary *discovery* tool for mechanistic interpretability research
on text encoders used in diffusion models.  Instead of guessing feature indices,
you compare two prompts and the differential tells you exactly which features
encode the semantic difference.

Outputs:
  - A bar chart showing the top differential features
  - A text listing you can paste directly into Feature Gate or Feature Probe
  - A lens direction (.pt) you can save and use with Concept Steer

Workflow:
  [CLIP Text Encode "cinematic photo"]  → conditioning_a ─┐
  [CLIP Text Encode "snapshot photo"]   → conditioning_b ─┤
                                                           ↓
                                                  [Diff Features]
                                                     ↓       ↓
                                                  IMAGE    STRING
                                                  (chart)  (feature list)

Why this is powerful:
  Anthropic's mechanistic interpretability work shows that the most
  meaningful features are found by *contrast*, not by looking at absolute
  activations.  A feature that fires strongly on "cinematic" AND "snapshot"
  is not what makes cinematic special — it's the features that fire
  differentially that encode the actual concept difference.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from io import BytesIO
from pathlib import Path
from comfy_api.latest import io

# ── Resolve imports ──────────────────────────────────────────────────────────
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_DIR = _PACKAGE_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


def _log(msg: str):
    print(f"[Concept Steer] {msg}")


# ── SAE cache ────────────────────────────────────────────────────────────────

_sae_cache: dict[str, object] = {}


def _load_sae(sae_path: str, d_model: int, expansion: int):
    """Load SAE or transcoder from path, auto-detecting format."""
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
        for f in d.rglob("*.pt"):
            name = f.name.lower()
            if "sae" in name and f.stat().st_size > 1_000_000:
                rel = str(f.relative_to(_PACKAGE_ROOT))
                if rel not in files:
                    files.append(rel)
        for f in d.rglob("*.safetensors"):
            if f.stat().st_size > 1_000_000:
                rel = str(f.relative_to(_PACKAGE_ROOT))
                if rel not in files:
                    files.append(rel)
    files.sort()
    return files


# ── Diff chart rendering ────────────────────────────────────────────────────

def _generate_diff_chart(
    a_stronger: list[tuple[int, float, float, float]],
    b_stronger: list[tuple[int, float, float, float]],
    shared_top: list[tuple[int, float, float]],
    title: str = "Differential Feature Analysis",
    labels_a: list[str] | None = None,
    labels_b: list[str] | None = None,
    max_display: int = 15,
) -> torch.Tensor:
    """Generate a differential feature comparison chart.

    Args:
        a_stronger: [(feat_idx, diff, a_val, b_val), ...]
        b_stronger: [(feat_idx, diff, a_val, b_val), ...]
        shared_top: [(feat_idx, a_val, b_val), ...] features strong in both
        title: chart title
        labels_a/b: optional semantic labels
        max_display: max features per section
    """
    plt = _setup_dark_style()

    n_a = min(len(a_stronger), max_display)
    n_b = min(len(b_stronger), max_display)
    n_shared = min(len(shared_top), 8)
    n_total = n_a + n_b + (n_shared + 1 if n_shared > 0 else 0) + 2  # +2 for dividers

    fig_height = max(4, n_total * 0.38 + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_pos = []
    bar_vals = []
    bar_colors = []
    tick_labels = []
    y = n_total - 1

    # ── Section: Stronger in A (positive diff → blue) ──
    for i in range(n_a):
        idx, diff, a_val, b_val = a_stronger[i]
        label = labels_a[i] if labels_a and i < len(labels_a) else ""
        tag = f"  {label}" if label else ""
        tick_labels.append(f"F{idx}{tag}")
        bar_vals.append(diff)
        bar_colors.append("#58a6ff")
        y_pos.append(y)
        y -= 1

    # Divider
    if n_a > 0 and n_b > 0:
        tick_labels.append("─── ▲A  ▼B ───")
        bar_vals.append(0)
        bar_colors.append("#333333")
        y_pos.append(y)
        y -= 1

    # ── Section: Stronger in B (negative diff → red) ──
    for i in range(n_b):
        idx, diff, a_val, b_val = b_stronger[i]
        label = labels_b[i] if labels_b and i < len(labels_b) else ""
        tag = f"  {label}" if label else ""
        tick_labels.append(f"F{idx}{tag}")
        bar_vals.append(-diff)  # flip to negative for visual
        bar_colors.append("#f85149")
        y_pos.append(y)
        y -= 1

    # ── Section: Shared strong features ──
    if n_shared > 0:
        tick_labels.append("── shared ──")
        bar_vals.append(0)
        bar_colors.append("#333333")
        y_pos.append(y)
        y -= 1

        for i in range(n_shared):
            idx, a_val, b_val = shared_top[i]
            tick_labels.append(f"F{idx} (both)")
            bar_vals.append((a_val + b_val) / 2)
            bar_colors.append("#bc8cff")
            y_pos.append(y)
            y -= 1

    ax.barh(y_pos, bar_vals, color=bar_colors, height=0.7, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tick_labels, fontfamily="monospace", fontsize=8)
    ax.set_xlabel("Differential Activation (A − B)")
    ax.set_title(title, fontweight="bold")
    ax.axvline(x=0, color="#555555", linewidth=0.8, linestyle="--")
    ax.grid(axis="x", alpha=0.3)

    # Value annotations
    max_abs = max(abs(v) for v in bar_vals) if bar_vals else 1
    for yp, v in zip(y_pos, bar_vals):
        if abs(v) > 1e-4:
            offset = max_abs * 0.02
            ax.text(v + offset if v >= 0 else v - offset, yp,
                    f"{v:+.3f}",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=7, color="#cccccc")

    plt.tight_layout()
    tensor = _render_figure_to_tensor(fig)
    plt.close(fig)
    return tensor


# ── The Node ─────────────────────────────────────────────────────────────────

class ConceptDiffFeaturesNode(io.ComfyNode):
    """Discover which SAE features differentiate two conditionings."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="conceptsteer.DiffFeatures",
            display_name="Diff Features",
            description=(
                "Compare two conditionings through an SAE to discover which "
                "features encode the semantic difference between them. "
                "The most powerful research tool: instead of guessing features, "
                "let the diff tell you what makes 'cinematic' different from "
                "'snapshot', or 'oil painting' from 'watercolor'. "
                "Outputs a chart and a ready-to-paste feature list for "
                "Feature Gate or Feature Probe."
            ),
            category="Concept Steer/Research",
            inputs=[
                io.Conditioning.Input(
                    "conditioning_a",
                    tooltip="First conditioning (e.g. 'cinematic photo')",
                ),
                io.Conditioning.Input(
                    "conditioning_b",
                    tooltip="Second conditioning (e.g. 'snapshot photo')",
                ),
                io.String.Input(
                    "sae_path",
                    default="",
                    tooltip=(
                        "Absolute path to SAE weights (.pt file). "
                        "Must be the same SAE used in Feature Map/Gate."
                    ),
                ),
                io.Int.Input(
                    "top_k",
                    default=15,
                    min=3,
                    max=50,
                    tooltip="Number of top differential features to show per side",
                ),
                io.Float.Input(
                    "min_diff",
                    default=0.01,
                    min=0.0,
                    max=1.0,
                    step=0.005,
                    tooltip=(
                        "Minimum absolute difference to include a feature. "
                        "Raise this to filter noise. Lower to see subtle differences."
                    ),
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
                        "Auto-downloads on first use. "
                        "Leave empty to use sae_path instead."
                    ),
                ),
                io.Combo.Input(
                    "pool_mode",
                    default="mean",
                    options=["mean", "max"],
                    tooltip=(
                        "'mean' = average across tokens (recommended). "
                        "'max' = max activation per feature across tokens."
                    ),
                ),
                io.Boolean.Input(
                    "save_lens",
                    default=False,
                    tooltip=(
                        "Save the differential direction as a .pt lens file "
                        "in the lenses/ directory. Use with Concept Steer to "
                        "apply the discovered concept difference."
                    ),
                ),
                io.String.Input(
                    "lens_name",
                    default="diff_discovery",
                    tooltip=(
                        "Name for the saved lens file (only used when "
                        "save_lens is enabled)."
                    ),
                ),
                io.String.Input(
                    "dict_path",
                    default="",
                    tooltip=(
                        "Path to feature dictionary JSON (from Feature Dictionary node). "
                        "When provided, differential features are labeled."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("VISUALIZATION"),
                io.String.Output("DIFF_REPORT"),
                io.String.Output("A_FEATURES"),
                io.String.Output("B_FEATURES"),
            ],
        )

    @classmethod
    def execute(
        cls,
        conditioning_a,
        conditioning_b,
        sae_path: str = "",
        top_k: int = 15,
        min_diff: float = 0.01,
        sae_expansion: int = 8,
        transcoder_repo: str = "",
        pool_mode: str = "mean",
        save_lens: bool = False,
        lens_name: str = "diff_discovery",
        dict_path: str = "",
    ):
        import json as _json

        sae_path = sae_path.strip()
        transcoder_repo = transcoder_repo.strip()

        # ── Resolve SAE / transcoder path ──
        if transcoder_repo and not sae_path:
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
                    f"Error: {e}", "", "",
                )

        if not sae_path:
            _log("No SAE/transcoder path provided")
            return io.NodeOutput(
                torch.zeros(1, 64, 64, 3),
                "Error: SAE path or transcoder_repo required",
                "", "",
            )

        if not os.path.isabs(sae_path):
            sae_path = str(_PACKAGE_ROOT / sae_path)

        if not os.path.isfile(sae_path):
            _log(f"SAE/transcoder file not found: {sae_path}")
            return io.NodeOutput(
                torch.zeros(1, 64, 64, 3),
                f"Error: file not found at {sae_path}",
                "", "",
            )

        # ── Extract conditioning tensors ──
        cond_a = conditioning_a[0][0]  # [B, tokens, dim]
        cond_b = conditioning_b[0][0]
        cond_dim = cond_a.shape[-1]

        # ── Load SAE ──
        sae = _load_sae(sae_path, cond_dim, sae_expansion)
        sae_device = next(sae.parameters()).device

        d_sae = sae.encoder.weight.shape[0]
        actual_expansion = d_sae // cond_dim
        is_transcoder = actual_expansion > 16

        # ── Encode both through SAE ──
        with torch.no_grad():
            ca = cond_a.float().to(sae_device)
            cb = cond_b.float().to(sae_device)

            if pool_mode == "max":
                ca_flat = ca.reshape(-1, cond_dim)
                cb_flat = cb.reshape(-1, cond_dim)
                feats_a_all = sae.encode(ca_flat)
                feats_b_all = sae.encode(cb_flat)
                B_a, T_a = cond_a.shape[0], cond_a.shape[1]
                B_b, T_b = cond_b.shape[0], cond_b.shape[1]
                feats_a = feats_a_all.reshape(B_a, T_a, -1).max(dim=1).values[0]
                feats_b = feats_b_all.reshape(B_b, T_b, -1).max(dim=1).values[0]
            else:  # mean
                ca_mean = ca.mean(dim=1)
                cb_mean = cb.mean(dim=1)
                feats_a = sae.encode(ca_mean)[0]  # [d_sae]
                feats_b = sae.encode(cb_mean)[0]

            feats_a = feats_a.cpu()
            feats_b = feats_b.cpu()

        # ── Compute differential ──
        diff = feats_a - feats_b  # positive = stronger in A

        # ── Sanity check: reconstruction quality ──
        with torch.no_grad():
            if pool_mode == "max":
                test_input = ca.reshape(-1, cond_dim).mean(dim=0, keepdim=True)
            else:
                test_input = ca.mean(dim=1)  # [B, D]
            recon, z = sae(test_input)
            recon_cos = F.cosine_similarity(test_input, recon, dim=-1).mean().item()
            recon_mse = (test_input - recon).pow(2).mean().item()

        # ── Find differential features ──
        # Stronger in A (positive diff)
        a_mask = diff > min_diff
        a_diffs = diff[a_mask]
        a_indices = a_mask.nonzero(as_tuple=True)[0]
        sorted_a = torch.argsort(-a_diffs)
        a_stronger = []
        for i in sorted_a[:top_k]:
            idx = a_indices[i].item()
            a_stronger.append((
                idx,
                diff[idx].item(),
                feats_a[idx].item(),
                feats_b[idx].item(),
            ))

        # Stronger in B (negative diff)
        b_mask = diff < -min_diff
        b_diffs = (-diff)[b_mask]
        b_indices = b_mask.nonzero(as_tuple=True)[0]
        sorted_b = torch.argsort(-b_diffs)
        b_stronger = []
        for i in sorted_b[:top_k]:
            idx = b_indices[i].item()
            b_stronger.append((
                idx,
                -diff[idx].item(),  # store as positive magnitude
                feats_a[idx].item(),
                feats_b[idx].item(),
            ))

        # Shared strong features (active in both, small diff)
        both_active = (feats_a > 0.01) & (feats_b > 0.01) & (diff.abs() < min_diff * 2)
        shared_strength = (feats_a + feats_b) * both_active.float()
        shared_sorted = torch.argsort(-shared_strength)
        shared_top = []
        for idx in shared_sorted[:10]:
            idx_val = idx.item()
            if both_active[idx_val]:
                shared_top.append((
                    idx_val,
                    feats_a[idx_val].item(),
                    feats_b[idx_val].item(),
                ))

        # ── Load feature dictionary for labels ──
        feat_dict = {}
        dict_path_str = dict_path.strip() if dict_path else ""
        if dict_path_str:
            if not os.path.isabs(dict_path_str):
                dict_path_str = str(_PACKAGE_ROOT / dict_path_str)
            if os.path.isfile(dict_path_str):
                with open(dict_path_str) as f:
                    dict_data = _json.load(f)
                feat_dict = dict_data.get("features", {})
                _log(f"Loaded feature dictionary: {len(feat_dict)} entries")
        elif is_transcoder and transcoder_repo:
            try:
                from lens_factory import load_transcoder_feature_labels
                all_diff_indices = (
                    [t[0] for t in a_stronger] +
                    [t[0] for t in b_stronger]
                )
                tc_labels = load_transcoder_feature_labels(
                    feature_indices=all_diff_indices,
                    layer=22, repo_id=transcoder_repo,
                )
                if tc_labels:
                    feat_dict = {
                        str(idx): {"label": label}
                        for idx, label in tc_labels.items()
                    }
            except Exception:
                pass

        # ── Build labels for chart ──
        labels_a = [
            feat_dict.get(str(t[0]), {}).get("label", "")
            for t in a_stronger
        ]
        labels_b = [
            feat_dict.get(str(t[0]), {}).get("label", "")
            for t in b_stronger
        ]

        # ── Stats ──
        active_a = (feats_a > 0).sum().item()
        active_b = (feats_b > 0).sum().item()
        cond_cos = F.cosine_similarity(
            cond_a.float().reshape(1, -1),
            cond_b.float().reshape(1, -1),
        ).item()
        feat_cos = F.cosine_similarity(
            feats_a.unsqueeze(0), feats_b.unsqueeze(0),
        ).item()

        tc_label = "Transcoder" if is_transcoder else "SAE"

        # ── Build text report ──
        lines = [
            f"Differential Feature Analysis — {tc_label}",
            f"SAE: {d_sae:,} features ({actual_expansion}×)",
            f"",
            f"Conditioning cosine similarity: {cond_cos:.4f}",
            f"Feature-space cosine similarity: {feat_cos:.4f}",
            f"SAE reconstruction quality: cos={recon_cos:.4f}, MSE={recon_mse:.6f}",
            f"Active features: A={active_a:,}, B={active_b:,}",
            f"",
        ]

        # Features stronger in A
        lines.append(f"── Stronger in A ({len(a_stronger)} features) ──")
        a_feat_strs = []
        for idx, d, av, bv in a_stronger:
            label = feat_dict.get(str(idx), {}).get("label", "")
            label_str = f"  {label}" if label else ""
            lines.append(
                f"  F{idx:>6d}: diff={d:+.4f}  A={av:.4f}  B={bv:.4f}{label_str}")
            a_feat_strs.append(str(idx))

        lines.append(f"")

        # Features stronger in B
        lines.append(f"── Stronger in B ({len(b_stronger)} features) ──")
        b_feat_strs = []
        for idx, d, av, bv in b_stronger:
            label = feat_dict.get(str(idx), {}).get("label", "")
            label_str = f"  {label}" if label else ""
            lines.append(
                f"  F{idx:>6d}: diff={d:+.4f}  A={av:.4f}  B={bv:.4f}{label_str}")
            b_feat_strs.append(str(idx))

        # Shared
        if shared_top:
            lines.extend(["", f"── Shared strong features ──"])
            for idx, av, bv in shared_top:
                label = feat_dict.get(str(idx), {}).get("label", "")
                label_str = f"  {label}" if label else ""
                lines.append(
                    f"  F{idx:>6d}: A={av:.4f}  B={bv:.4f}{label_str}")

        # Copy-paste helpers
        lines.extend([
            "",
            "── Copy into Feature Gate / Probe ──",
            f"A-unique features: {','.join(a_feat_strs[:10])}",
            f"B-unique features: {','.join(b_feat_strs[:10])}",
        ])

        text_output = "\n".join(lines)
        a_features_str = ",".join(a_feat_strs)
        b_features_str = ",".join(b_feat_strs)

        _log(f"Diff: {len(a_stronger)} stronger in A, "
             f"{len(b_stronger)} stronger in B, "
             f"{len(shared_top)} shared")

        # ── Save lens if requested ──
        if save_lens and (a_stronger or b_stronger):
            # Build direction from differential features (A minus B)
            # This gives a steering direction: adding it pushes toward A,
            # subtracting pushes toward B
            sparse_diff = torch.zeros(d_sae, device=sae_device)
            for idx, d, _, _ in a_stronger:
                sparse_diff[idx] = d
            for idx, d, _, _ in b_stronger:
                sparse_diff[idx] = -d

            with torch.no_grad():
                direction = sae.decode_sparse(
                    sparse_diff.unsqueeze(0))[0].cpu()
                unit_dir = F.normalize(direction, dim=0)

            lens_data = {
                "direction": unit_dir,
                "concept": lens_name,
                "method": "differential_features",
                "norm": direction.norm().item(),
                "sae_path": os.path.basename(sae_path),
                "a_features": [t[0] for t in a_stronger],
                "b_features": [t[0] for t in b_stronger],
                "a_diffs": [t[1] for t in a_stronger],
                "b_diffs": [t[1] for t in b_stronger],
                "conditioning_cosine": cond_cos,
                "feature_cosine": feat_cos,
            }

            lens_dir = _PACKAGE_ROOT / "lenses" / "discoveries"
            lens_dir.mkdir(parents=True, exist_ok=True)
            lens_path = lens_dir / f"{lens_name}.pt"
            torch.save(lens_data, lens_path)
            _log(f"Saved differential lens to {lens_path}")
            text_output += f"\n\nLens saved: {lens_path}"

        # ── Generate chart ──
        chart = _generate_diff_chart(
            a_stronger, b_stronger, shared_top,
            title=f"Differential Features — {len(a_stronger)} vs {len(b_stronger)}",
            labels_a=labels_a,
            labels_b=labels_b,
            max_display=top_k,
        )

        return io.NodeOutput(chart, text_output, a_features_str, b_features_str)
