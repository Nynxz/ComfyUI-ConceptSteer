"""
ConceptFeatureGate — Suppress or amplify SAE features in conditioning.

Decomposes conditioning through a trained SAE, selectively zeros out or scales
specific features, then subtracts (or adds) the difference — effectively
removing or boosting individual interpretable concepts from the conditioning
before it reaches the sampler.

This gives you surgical control over what concepts are present in the
conditioning, without retraining anything. Unlike ConceptSteer which adds
a pre-trained direction, Feature Gate lets you remove or tune individual
components of whatever prompt you've already encoded.

Workflow:
  [CLIP Text Encode] → [Feature Map]  (inspect which features are active)
  [CLIP Text Encode] → [Feature Gate] → [KSampler]
                         ↑
               (paste feature indices from Feature Map)

Algorithm:
  1. Encode conditioning through SAE → sparse feature activations z
  2. Build z_modified by zeroing suppressed features / scaling amplified ones
  3. Compute delta = decode_sparse(z - z_modified)  (bias-free to get clean direction)
  4. Subtract delta from original conditioning
  5. Output modified conditioning

This avoids a full SAE round-trip — we only compute the contribution of
the affected features and adjust the original conditioning accordingly.
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


# ── SAE cache ────────────────────────────────────────────────────────────────

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


# ── Feature spec parsing ────────────────────────────────────────────────────

def _parse_feature_spec(spec: str) -> dict[int, float]:
    """Parse a feature specification string into {index: scale} dict.

    Formats:
      "4821,12033,8192"           → suppress all (scale=0)
      "4821:0, 12033:0, 8192:0"  → explicit suppress
      "4821:0, 12033:2.5"        → suppress 4821, amplify 12033 to 2.5×
      "4821:0.5"                 → reduce feature to 50%

    Returns:
      dict mapping feature_index → target_scale
      (0.0 = fully suppress, 1.0 = unchanged, 2.0 = double, etc.)
    """
    result = {}
    if not spec.strip():
        return result

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        if ":" in part:
            idx_str, scale_str = part.split(":", 1)
            try:
                idx = int(idx_str.strip())
                scale = float(scale_str.strip())
                result[idx] = scale
            except ValueError:
                _log(f"Skipping invalid feature spec: '{part}'")
        else:
            try:
                idx = int(part.strip())
                result[idx] = 0.0  # bare index = suppress
            except ValueError:
                _log(f"Skipping invalid feature index: '{part}'")

    return result


# ── The Node ─────────────────────────────────────────────────────────────────

class ConceptFeatureGateNode(io.ComfyNode):
    """Suppress or amplify individual SAE features in conditioning."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="conceptsteer.FeatureGate",
            display_name="Feature Gate",
            description=(
                "Decompose conditioning through a trained SAE, then suppress "
                "or amplify individual features. This lets you surgically "
                "remove or boost specific visual concepts (lighting, mood, "
                "texture, etc.) from any prompt's conditioning."
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
                    "suppress_features",
                    default="",
                    tooltip=(
                        "Features to SUPPRESS (remove from conditioning). "
                        "Comma-separated indices: '4821,12033,8192'. "
                        "Get these from the Feature Map node output."
                    ),
                ),
                io.String.Input(
                    "amplify_features",
                    default="",
                    tooltip=(
                        "Features to AMPLIFY with custom scales. "
                        "Format: 'index:scale' pairs comma-separated. "
                        "Example: '4821:2.0,12033:3.0' — doubles feature 4821, "
                        "triples feature 12033. Scale=0.5 halves the feature."
                    ),
                ),
                io.Float.Input(
                    "gate_strength",
                    default=1.0,
                    min=0.0,
                    max=5.0,
                    step=0.1,
                    tooltip=(
                        "Strength of the gating effect, scaled relative to "
                        "conditioning magnitude (like the steer strength). "
                        "1.0 = visible effect (~30% of token norm). "
                        "2.0 = strong. 0.0 = no change. "
                        "Try 0.5–2.0 range first."
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
                io.Boolean.Input(
                    "per_token",
                    default=True,
                    tooltip=(
                        "Apply gating per-token (recommended). "
                        "When disabled, uses mean-pooled features which is "
                        "faster but less precise."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output("CONDITIONING"),
            ],
        )

    @classmethod
    def execute(
        cls,
        conditioning,
        sae_path: str = "",
        suppress_features: str = "",
        amplify_features: str = "",
        gate_strength: float = 1.0,
        sae_expansion: int = 8,
        transcoder_repo: str = "",
        per_token: bool = True,
    ):
        # ── Early exits ──
        sae_path = sae_path.strip()
        transcoder_repo = transcoder_repo.strip()
        suppress_spec = _parse_feature_spec(suppress_features)
        amplify_spec = _parse_feature_spec(amplify_features)

        all_mods = {**suppress_spec}
        for idx, scale in amplify_spec.items():
            all_mods[idx] = scale

        if (not sae_path and not transcoder_repo) or not all_mods or abs(gate_strength) < 1e-6:
            if not sae_path and not transcoder_repo:
                _log("No SAE/transcoder path — passing conditioning through")
            elif not all_mods:
                _log("No features specified — passing conditioning through")
            else:
                _log("Gate strength ≈ 0 — passing conditioning through")
            return io.NodeOutput(conditioning)

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
                return io.NodeOutput(conditioning)

        if not os.path.isabs(sae_path):
            sae_path = str(_PACKAGE_ROOT / sae_path)

        if not os.path.isfile(sae_path):
            _log(f"SAE not found: {sae_path} — passing through")
            return io.NodeOutput(conditioning)

        # ── Load SAE / transcoder ──
        cond_dim = conditioning[0][0].shape[-1]
        sae = _load_sae(sae_path, cond_dim, sae_expansion)
        sae_device = next(sae.parameters()).device

        # Auto-detect actual d_sae from loaded model weights
        d_sae = sae.encoder.weight.shape[0]

        # Validate feature indices
        valid_mods = {
            idx: scale for idx, scale in all_mods.items()
            if 0 <= idx < d_sae
        }
        invalid = set(all_mods.keys()) - set(valid_mods.keys())
        if invalid:
            _log(f"Ignoring out-of-range feature indices: {sorted(invalid)}")

        if not valid_mods:
            _log("No valid feature indices — passing through")
            return io.NodeOutput(conditioning)

        n_suppress = sum(1 for s in valid_mods.values() if abs(s) < 1e-6)
        n_amplify = sum(1 for s in valid_mods.values()
                        if abs(s) >= 1e-6 and abs(s - 1.0) > 1e-6)
        _log(f"Gating {len(valid_mods)} features: "
             f"{n_suppress} suppressed, {n_amplify} amplified "
             f"(strength={gate_strength:.2f})")

        # ── Apply to each conditioning entry ──
        new_cond = []
        for cond_entry in conditioning:
            cond_tensor = cond_entry[0]  # [B, tokens, dim]
            extra = cond_entry[1] if len(cond_entry) > 1 else {}

            device = cond_tensor.device
            dtype = cond_tensor.dtype
            B, T, D = cond_tensor.shape

            steered = cond_tensor.clone()

            with torch.no_grad():
                c = steered.float().to(sae_device)
                cond_norm = c.norm(dim=-1)  # [B, T]
                active_mask = cond_norm > 0.01
                avg_cond_norm = cond_norm[active_mask].mean().item() \
                    if active_mask.any() else cond_norm.mean().item()

                if per_token:
                    # Process each token position independently
                    c_flat = c.reshape(-1, D)  # [B*T, dim]
                    z = sae.encode(c_flat)  # [B*T, d_sae]

                    # Log feature activations for the features we're modifying
                    for idx, scale in sorted(valid_mods.items()):
                        feat_act = z[:, idx]
                        act_mean = feat_act.mean().item()
                        act_max = feat_act.max().item()
                        act_nonzero = (feat_act > 0).float().mean().item()
                        _log(f"  F{idx}: act_mean={act_mean:.4f}, "
                             f"act_max={act_max:.4f}, "
                             f"fires={act_nonzero:.0%}, "
                             f"scale→{scale}")

                    z_modified = z.clone()
                    for idx, scale in valid_mods.items():
                        z_modified[:, idx] = z[:, idx] * scale

                    # Compute the contribution delta (bias-free)
                    diff = z - z_modified  # what we're removing
                    delta = sae.decode_sparse(diff)  # [B*T, dim]

                    raw_delta_norm = delta.norm(dim=-1).mean().item()
                    _log(f"  Raw delta norm: {raw_delta_norm:.4f}, "
                         f"Cond norm: {avg_cond_norm:.4f}, "
                         f"Ratio: {raw_delta_norm / max(avg_cond_norm, 1e-8):.4%}")

                    # ── Norm-relative scaling ──
                    # The SAE was trained on layer-22 hidden states but is now
                    # applied to the final conditioning tensor (different
                    # distribution). Raw decode_sparse output may be either too
                    # large or too small relative to the conditioning. We
                    # normalize the delta direction and scale it proportional
                    # to the conditioning magnitude so gate_strength has a
                    # consistent, predictable effect.
                    #
                    # gate_strength=1.0 → delta magnitude = 30% of avg token
                    # norm per modified feature. Matches NORM_SCALE logic.
                    NORM_SCALE = 0.3
                    n_modified = len(valid_mods)
                    delta_norms = delta.norm(
                        dim=-1, keepdim=True).clamp(min=1e-8)
                    delta_unit = delta / delta_norms  # unit direction
                    delta = delta_unit * (gate_strength * NORM_SCALE *
                                          avg_cond_norm / max(n_modified, 1))
                    delta = delta.reshape(B, T, D)

                else:
                    # Mean-pool approach (faster but less precise)
                    c_mean = c.mean(dim=1)  # [B, dim]
                    z = sae.encode(c_mean)  # [B, d_sae]

                    for idx, scale in sorted(valid_mods.items()):
                        feat_act = z[:, idx]
                        _log(f"  F{idx}: act={feat_act.mean().item():.4f}, "
                             f"scale→{scale}")

                    z_modified = z.clone()
                    for idx, scale in valid_mods.items():
                        z_modified[:, idx] = z[:, idx] * scale

                    diff = z - z_modified
                    delta = sae.decode_sparse(diff)  # [B, dim]

                    raw_delta_norm = delta.norm(dim=-1).mean().item()
                    _log(f"  Raw delta norm: {raw_delta_norm:.4f}, "
                         f"Cond norm: {avg_cond_norm:.4f}")

                    NORM_SCALE = 0.3
                    n_modified = len(valid_mods)
                    delta_norms = delta.norm(
                        dim=-1, keepdim=True).clamp(min=1e-8)
                    delta_unit = delta / delta_norms
                    delta = delta_unit * (gate_strength * NORM_SCALE *
                                          avg_cond_norm / max(n_modified, 1))
                    delta = delta.unsqueeze(1).expand(-1, T, -1)  # broadcast

                # Subtract delta from conditioning
                # (removing the contribution of suppressed features,
                #  or adding extra for amplified features where scale > 1)
                steered = (c - delta).to(device=device, dtype=dtype)

            new_cond.append([steered, extra])

        # ── Log stats ──
        if len(new_cond) > 0:
            orig_norm = cond_tensor.norm(dim=-1).mean().item()
            new_norm = new_cond[0][0].norm(dim=-1).mean().item()
            cos_sim = F.cosine_similarity(
                cond_tensor.float().reshape(-1, D),
                new_cond[0][0].float().reshape(-1, D),
            ).mean().item()
            _log(f"Gated: norm {orig_norm:.2f} → {new_norm:.2f}, "
                 f"cos(orig, gated)={cos_sim:.4f}")

        return io.NodeOutput(new_cond)
