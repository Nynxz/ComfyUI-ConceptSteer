"""
ConceptFeatureProbe — Isolate & visualize what individual features *do*.

Unlike Feature Gate (which modifies features already present in a prompt),
Feature Probe injects a single feature's decoder direction into conditioning
*regardless* of whether that feature is currently active.  This forces the
sampler to produce imagery dominated by that feature's concept, revealing
what the decoder column truly encodes — far richer than token labels.

Modes
-----
- **single**: Inject one feature, output one conditioning.
- **sweep**: Inject each feature from a list one-at-a-time, output a
  batch of conditionings.  Connect to a batch-aware KSampler to get a
  grid showing what each feature does side-by-side.

Why this works
--------------
Each feature's decoder column is a unit direction in the 2560-d hidden
space.  When you add that direction (scaled by `probe_strength ×
conditioning_magnitude`) to every token, you steer the denoiser toward
the visual concept that direction encodes.  The concept is whatever the
SAE/transcoder learned lives along that axis — one feature might be
"cinematic warm lighting", another "fingers/hands", another "watercolour
edges".  You won't know until you see it.

Workflow
--------
  [CLIP Text Encode] → [Feature Probe] → [KSampler] → IMAGE
                         ↑
              (pick feature_index from Feature Map output)
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from comfy_api.latest import io

# ── Resolve imports ──────────────────────────────────────────────────────────
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_DIR = _PACKAGE_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


def _log(msg: str):
    print(f"[Concept Steer] {msg}")


# ── SAE cache (shared with other feature nodes) ─────────────────────────────
_sae_cache: dict[str, object] = {}


def _load_sae(sae_path: str, d_model: int, expansion: int):
    cache_key = sae_path
    if cache_key in _sae_cache:
        return _sae_cache[cache_key]

    from lens_factory import load_sae_or_transcoder

    _log(f"Loading SAE/transcoder: {os.path.basename(sae_path)}")
    sae = load_sae_or_transcoder(
        sae_path, d_model=d_model, expected_expansion=expansion)
    _sae_cache[cache_key] = sae
    return sae


# ── The Node ─────────────────────────────────────────────────────────────────

class ConceptFeatureProbeNode(io.ComfyNode):
    """Inject a single feature direction into conditioning to see what it does."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="conceptsteer.FeatureProbe",
            display_name="Feature Probe",
            description=(
                "Inject a single SAE/transcoder feature direction into "
                "conditioning to visualize what it encodes. Each feature's "
                "decoder column is a direction in the hidden space — this node "
                "adds that direction to your conditioning so the generated "
                "image reveals the concept. Use 'sweep' mode to batch-test "
                "multiple features at once."
            ),
            category="Concept Steer/Features",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.String.Input(
                    "sae_path",
                    default="",
                    tooltip=(
                        "Absolute path to SAE weights (.pt file). "
                        "Must match the SAE used in Feature Map."
                    ),
                ),
                io.String.Input(
                    "feature_indices",
                    default="0",
                    tooltip=(
                        "Feature index or comma-separated list of indices. "
                        "Single index → one output conditioning. "
                        "Multiple indices → batch output (one per feature). "
                        "Get indices from the Feature Map node."
                    ),
                ),
                io.Float.Input(
                    "probe_strength",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.1,
                    tooltip=(
                        "How strongly to inject the feature direction. "
                        "1.0 = moderate effect (~30% of conditioning norm). "
                        "2.0-5.0 = strong effect, dominates the image. "
                        "Start at 1.0 and increase until the concept is visible."
                    ),
                ),
                io.Combo.Input(
                    "inject_mode",
                    default="add",
                    options=["add", "replace"],
                    tooltip=(
                        "'add' = adds feature direction to existing conditioning. "
                        "The prompt still matters, feature concept is layered on. "
                        "'replace' = replaces conditioning with pure feature direction. "
                        "Shows the raw concept with no prompt influence."
                    ),
                ),
                io.Float.Input(
                    "replace_blend",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip=(
                        "Only used in 'replace' mode. Blend between pure feature "
                        "direction (0.0) and original conditioning (1.0). "
                        "0.0 = pure feature, 0.5 = half-and-half, 1.0 = no change."
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
            ],
            outputs=[
                io.Conditioning.Output("CONDITIONING"),
                io.String.Output("PROBE_INFO"),
            ],
        )

    @classmethod
    def execute(
        cls,
        conditioning,
        sae_path: str = "",
        feature_indices: str = "0",
        probe_strength: float = 1.0,
        inject_mode: str = "add",
        replace_blend: float = 0.0,
        sae_expansion: int = 8,
        transcoder_repo: str = "",
    ):
        # ── Parse feature indices ──
        sae_path = sae_path.strip()
        transcoder_repo = transcoder_repo.strip()
        indices = _parse_indices(feature_indices)

        if not indices:
            _log("No valid feature indices — passing through")
            return io.NodeOutput(conditioning, "No features specified")

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
                return io.NodeOutput(conditioning, f"Error: {e}")

        if not sae_path:
            _log("No SAE/transcoder path — passing through")
            return io.NodeOutput(conditioning, "No SAE path")

        if not os.path.isabs(sae_path):
            sae_path = str(_PACKAGE_ROOT / sae_path)

        if not os.path.isfile(sae_path):
            _log(f"SAE not found: {sae_path}")
            return io.NodeOutput(conditioning, f"SAE not found: {sae_path}")

        # ── Load SAE / transcoder ──
        cond_dim = conditioning[0][0].shape[-1]
        sae = _load_sae(sae_path, cond_dim, sae_expansion)
        sae_device = next(sae.parameters()).device

        d_sae = sae.encoder.weight.shape[0]
        is_transcoder = d_sae > cond_dim * 16

        # Validate indices
        valid_indices = [i for i in indices if 0 <= i < d_sae]
        invalid = [i for i in indices if i not in valid_indices]
        if invalid:
            _log(f"Ignoring out-of-range indices: {invalid}")

        if not valid_indices:
            _log("No valid feature indices after validation")
            return io.NodeOutput(conditioning, "All indices out of range")

        tc_label = "Transcoder" if is_transcoder else "SAE"
        _log(f"Feature Probe: {len(valid_indices)} features, "
             f"strength={probe_strength:.1f}, mode={inject_mode}")

        # ── Load feature labels if available ──
        feature_labels: dict[int, str] = {}
        if is_transcoder and transcoder_repo:
            try:
                from lens_factory import load_transcoder_feature_labels
                feature_labels = load_transcoder_feature_labels(
                    feature_indices=valid_indices,
                    layer=22,
                    repo_id=transcoder_repo,
                )
            except Exception:
                pass  # Labels are optional

        # ── Extract decoder columns (the feature directions) ──
        # decoder.weight shape: [d_model, d_sae] — columns are feature dirs
        decoder_weight = sae.decoder.weight.float()  # [d_model, d_sae]

        # ── Build probed conditionings ──
        NORM_SCALE = 0.3
        info_lines = [
            f"Feature Probe — {tc_label}",
            f"Features: {len(valid_indices)}, "
            f"Strength: {probe_strength:.1f}, Mode: {inject_mode}",
            "",
        ]

        new_cond = []
        base_cond = conditioning[0]  # use first conditioning entry
        cond_tensor = base_cond[0]   # [B, T, D]
        extra = base_cond[1] if len(base_cond) > 1 else {}

        B, T, D = cond_tensor.shape

        with torch.no_grad():
            c = cond_tensor.float().to(sae_device)
            avg_norm = c.norm(dim=-1).mean().item()

            for feat_idx in valid_indices:
                # Get the decoder column for this feature
                feat_dir = decoder_weight[:, feat_idx]  # [d_model]
                feat_dir_unit = feat_dir / feat_dir.norm().clamp(min=1e-8)

                # Scale the direction relative to conditioning magnitude
                scaled_dir = feat_dir_unit * (probe_strength * NORM_SCALE
                                              * avg_norm)

                if inject_mode == "replace":
                    # Replace: blend between pure feature direction and original
                    feat_cond = scaled_dir.unsqueeze(0).unsqueeze(0).expand(
                        B, T, -1)
                    result = (feat_cond * (1.0 - replace_blend)
                              + c * replace_blend)
                else:
                    # Add: layer feature direction on top of existing cond
                    delta = scaled_dir.unsqueeze(0).unsqueeze(0).expand(
                        B, T, -1)
                    result = c + delta

                result = result.to(
                    device=cond_tensor.device, dtype=cond_tensor.dtype)
                new_cond.append([result, extra])

                # Compute diagnostics
                cos_sim = F.cosine_similarity(
                    cond_tensor.float().reshape(-1, D),
                    result.float().reshape(-1, D),
                ).mean().item()

                label = feature_labels.get(feat_idx, "")
                label_str = f"  {label}" if label else ""
                info_lines.append(
                    f"F{feat_idx:>6d}: cos={cos_sim:.4f}{label_str}"
                )
                _log(f"  F{feat_idx}: injected, cos(orig, probed)={cos_sim:.4f}"
                     f"{label_str}")

        # Add helpful footer
        info_lines.extend([
            "",
            f"Conditioning norm: {avg_norm:.2f}",
            f"Feature direction scale: {probe_strength * NORM_SCALE * avg_norm:.2f}",
        ])
        if len(valid_indices) > 1:
            info_lines.append(
                f"Output: batch of {len(valid_indices)} conditionings "
                f"(one per feature)"
            )

        info_text = "\n".join(info_lines)

        return io.NodeOutput(new_cond, info_text)


def _parse_indices(spec: str) -> list[int]:
    """Parse comma-separated feature indices, supporting ranges.

    Examples:
        "4821"           → [4821]
        "4821,12033,100" → [4821, 12033, 100]
        "100-105"        → [100, 101, 102, 103, 104, 105]
        "100-105,200"    → [100, 101, 102, 103, 104, 105, 200]
    """
    result = []
    if not spec.strip():
        return result

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        # Strip any scale suffix (e.g. "4821:2.0" → "4821")
        if ":" in part:
            part = part.split(":")[0].strip()
        if "-" in part and not part.startswith("-"):
            try:
                start_s, end_s = part.split("-", 1)
                start, end = int(start_s.strip()), int(end_s.strip())
                result.extend(range(start, end + 1))
            except ValueError:
                _log(f"Skipping invalid range: '{part}'")
        else:
            try:
                result.append(int(part))
            except ValueError:
                _log(f"Skipping invalid index: '{part}'")

    return result
