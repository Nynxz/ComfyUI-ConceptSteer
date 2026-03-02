"""
ConceptLayerSteer — Per-layer concept steering via cross-attention hooks.

While Concept Steer adds a direction vector to the final conditioning tensor
(which is the same for ALL transformer blocks), this node hooks into the model
at the cross-attention level and can inject different strengths at different
layers of the DiT/UNet. This enables:

  1. Interpretability — see which layers respond most to concept steering
  2. Precision — steer only at layers where the concept manifests (e.g.,
     high-level composition at early layers, texture at late layers)
  3. Reduced artifacts — avoid pushing concept directions in layers that
     don't "understand" that concept, which can cause incoherence

Architecture:
  Uses set_model_attn2_output_patch() to intercept the OUTPUT of each
  cross-attention block. For each block, the patch adds a scaled concept
  direction to the attention output before it continues through the residual
  stream. The per-block strength can be controlled via a layer mask.

  This is much more surgical than Concept Steer's approach of modifying
  the input conditioning:
    - Concept Steer:  modifies K,V (same modification for every layer)
    - Layer Steer:    modifies the output of each cross-attn independently

  Additionally, this node collects per-layer statistics (how much each layer's
  output changed) and prints them as a diagnostic, giving interpretability
  into where the concept is being injected.

Usage in ComfyUI:
  [Load Model] → MODEL → [Layer Steer] → MODEL → [KSampler]
                             ↑
                       lens: cinematic
                       layer_mask: "0-5:0.0, 6-18:1.0, 19-23:0.5"
                       (inject full strength in middle layers only)
"""

import os
import math
import torch
import torch.nn.functional as F
from comfy_api.latest import io


def _log(msg: str):
    print(f"[Concept Steer | Layer] {msg}")


# ── Lens utilities (shared) ─────────────────────────────────────────────────

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


_LENS_ROOT = _find_lens_root()


def _discover_lenses() -> list[str]:
    lenses = []
    root = _LENS_ROOT
    if not os.path.isdir(root):
        return lenses
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith((".pt", ".safetensors")):
                if fname.endswith("_metadata.json"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fname), root)
                lenses.append(rel)
    lenses.sort()
    return lenses


def _resolve_lens_path(filename: str) -> str | None:
    full = os.path.normpath(os.path.join(_LENS_ROOT, filename))
    if os.path.isfile(full):
        return full
    basename = os.path.basename(filename)
    for dirpath, _dirnames, filenames in os.walk(_LENS_ROOT):
        if basename in filenames:
            return os.path.join(dirpath, basename)
    return None


def _extract_direction(data) -> torch.Tensor:
    """Extract a 1-D direction vector from various lens formats."""
    if isinstance(data, torch.Tensor):
        if data.dim() == 1:
            return data
        elif data.dim() == 2 and data.shape[0] == 1:
            return data.squeeze(0)
        raise ValueError(
            f"Cannot interpret tensor shape {data.shape} as direction")

    if isinstance(data, dict):
        for key in ("direction", "d_in_siglip", "d_shared"):
            if key in data:
                return data[key].float().squeeze()
        for key, val in data.items():
            if isinstance(val, torch.Tensor) and val.dim() <= 2:
                total_elements = val.numel()
                if 256 <= total_elements <= 8192:
                    return val.float().reshape(-1)

    raise ValueError(f"Could not extract direction from lens data.")


def _load_direction(lens: str, custom_lens_path: str) -> torch.Tensor | None:
    """Resolve and load a direction vector from lens selection or custom path."""
    if custom_lens_path and custom_lens_path.strip():
        lens_path = custom_lens_path.strip()
    elif lens and lens != "None":
        lens_path = _resolve_lens_path(lens)
    else:
        return None

    if not lens_path or not os.path.isfile(lens_path):
        _log(f"Lens file not found: {lens_path}")
        return None

    try:
        data = torch.load(lens_path, map_location="cpu", weights_only=False)
    except Exception as e:
        _log(f"Failed to load lens: {e}")
        return None

    try:
        return _extract_direction(data)
    except ValueError as e:
        _log(f"{e}")
        return None


# ── Layer mask parsing ──────────────────────────────────────────────────────

def _parse_layer_mask(spec: str, num_layers: int = 48) -> dict[int, float]:
    """Parse a layer mask specification into a dict: layer_index → strength.

    Formats:
      - "all"            → all layers at 1.0
      - "6-18"           → layers 6-18 at 1.0, rest at 0.0
      - "0-5:0.2, 6-18:1.0, 19-23:0.5"  → ranges with explicit strengths
      - "6,7,8,9:1.0, 15:0.5"           → individual layers + ranges
      - "middle"         → middle third at 1.0
      - "early"          → first third at 1.0
      - "late"           → last third at 1.0

    Returns a dict mapping layer index to strength multiplier.
    Default is 0.0 for unspecified layers (no steering).
    """
    spec = spec.strip().lower()
    mask = {}

    if spec in ("", "all"):
        return {i: 1.0 for i in range(num_layers)}

    if spec == "early":
        third = max(1, num_layers // 3)
        return {i: 1.0 for i in range(third)}

    if spec == "middle":
        third = max(1, num_layers // 3)
        return {i: 1.0 for i in range(third, 2 * third)}

    if spec == "late":
        third = max(1, num_layers // 3)
        return {i: 1.0 for i in range(2 * third, num_layers)}

    # Parse comma-separated entries
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        # Split off optional :strength
        if ":" in part:
            range_spec, strength_str = part.rsplit(":", 1)
            try:
                strength = float(strength_str.strip())
            except ValueError:
                strength = 1.0
        else:
            range_spec = part
            strength = 1.0

        range_spec = range_spec.strip()

        # Parse range (e.g., "6-18") or single layer (e.g., "6")
        if "-" in range_spec:
            try:
                start, end = range_spec.split("-", 1)
                start, end = int(start.strip()), int(end.strip())
                for i in range(start, end + 1):
                    mask[i] = strength
            except ValueError:
                continue
        else:
            try:
                layer = int(range_spec)
                mask[layer] = strength
            except ValueError:
                continue

    return mask


# ── Cross-attention output patch ────────────────────────────────────────────

class _LayerSteerAttnPatch:
    """Callable installed via set_model_attn2_output_patch.

    Modifies the output of cross-attention blocks, adding a concept direction
    scaled per-layer and per-timestep. Collects per-layer statistics for
    interpretability.

    Signature: fn(out, extra_options) -> out
      - out: cross-attention output tensor [B, tokens, dim]
      - extra_options: dict with 'block', 'n_heads', 'sigmas', 'cond_or_uncond', etc.
    """

    def __init__(
        self,
        direction: torch.Tensor,
        strength: float,
        layer_mask: dict[int, float],
        normalize: bool,
        report_stats: bool,
        report_interval: int = 5,
    ):
        self.direction = direction
        self.strength = strength
        self.layer_mask = layer_mask
        self.normalize = normalize
        self.report_stats = report_stats
        self.report_interval = report_interval

        # Stats collection
        self._layer_stats: dict[str, list[float]] = {}
        self._call_count = 0
        self._step_count = 0
        self._last_sigma = -1.0

    def to(self, device):
        """ComfyUI calls .to(device) for device management."""
        self.direction = self.direction.to(device)
        return self

    def __call__(self, out: torch.Tensor, extra_options: dict) -> torch.Tensor:
        """Intercept cross-attention output and inject concept direction."""
        self._call_count += 1

        # ── Identify which block/layer we're in ──
        # extra_options structure varies by model type. Common keys:
        # DiT:  'block' = ('double', idx), ('single', idx), etc.
        # UNet: 'block' = ('input', idx), ('middle', idx), ('output', idx)
        block_info = extra_options.get("block", None)
        if block_info is None:
            block_name = "unknown"
            block_idx = -1
        elif isinstance(block_info, (list, tuple)) and len(block_info) >= 2:
            block_type, block_idx = block_info[0], block_info[1]
            block_name = f"{block_type}_{block_idx}"
        else:
            block_name = str(block_info)
            block_idx = -1

        # ── Check layer mask ──
        # Try matching by block index first, then by full block_name
        layer_strength = 0.0
        if isinstance(block_idx, int) and block_idx in self.layer_mask:
            layer_strength = self.layer_mask[block_idx]
        elif not self.layer_mask:
            # Empty mask = apply everywhere (shouldn't happen with parse)
            layer_strength = 1.0

        if abs(layer_strength) < 1e-6:
            return out

        effective_strength = self.strength * layer_strength

        # ── Track sigma for step counting ──
        sigmas = extra_options.get("sigmas", None)
        if sigmas is not None:
            if isinstance(sigmas, torch.Tensor):
                current_sigma = sigmas.max().detach().cpu().item()
            else:
                current_sigma = float(sigmas)
            if abs(current_sigma - self._last_sigma) > 0.01:
                self._step_count += 1
                self._last_sigma = current_sigma

        # ── Inject direction into cross-attention output ──
        device = out.device
        dtype = out.dtype
        out_dim = out.shape[-1]

        direction = self.direction.to(device=device, dtype=torch.float32)

        # Project to match output dimension
        if direction.shape[0] != out_dim:
            if direction.shape[0] < out_dim:
                padded = torch.zeros(
                    out_dim, device=device, dtype=torch.float32)
                padded[:direction.shape[0]] = direction
                direction = padded
            else:
                direction = direction[:out_dim]

        unit_dir = F.normalize(direction, dim=0)

        if self.normalize:
            NORM_SCALE = 0.3
            out_float = out.float()
            token_norms = out_float.norm(dim=-1)
            avg_norm = token_norms.mean()
            delta = effective_strength * NORM_SCALE * avg_norm * unit_dir
        else:
            delta = effective_strength * direction

        # Broadcast delta to match output shape
        while delta.dim() < out.dim():
            delta = delta.unsqueeze(0)

        steered_out = out.float() + delta
        steered_out = steered_out.to(dtype=dtype)

        # ── Collect statistics ──
        if self.report_stats:
            with torch.no_grad():
                # Measure how much we changed the output (relative L2)
                change = (steered_out.float() - out.float()
                          ).norm() / (out.float().norm() + 1e-8)
                change_val = change.item()

                if block_name not in self._layer_stats:
                    self._layer_stats[block_name] = []
                self._layer_stats[block_name].append(change_val)

            # Print report periodically
            if self._step_count > 0 and self._step_count % self.report_interval == 0:
                if self._call_count % max(len(self.layer_mask), 1) == 0:
                    self._print_report()

        return steered_out

    def _print_report(self):
        """Print per-layer steering statistics for interpretability."""
        if not self._layer_stats:
            return

        _log(f"── Layer Steering Report (step ~{self._step_count}) ──")
        sorted_layers = sorted(
            self._layer_stats.items(),
            key=lambda x: x[0],
        )
        for layer_name, changes in sorted_layers:
            avg_change = sum(changes[-10:]) / len(changes[-10:])
            max_change = max(changes[-10:])
            _log(
                f"  {layer_name:>20s}  avg_Δ={avg_change:.5f}  max_Δ={max_change:.5f}")
        _log(f"  {'─' * 50}")


# ── The Node ────────────────────────────────────────────────────────────────

class ConceptLayerSteerNode(io.ComfyNode):
    """Per-layer concept steering via cross-attention output hooks."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        available_lenses = _discover_lenses()
        lens_options = ["None"] + available_lenses

        return io.Schema(
            node_id="conceptsteer.LayerSteer",
            display_name="Layer Steer",
            description=(
                "Apply concept steering at specific transformer layers with per-layer "
                "strength control. Hooks into cross-attention outputs for surgical "
                "injection. Prints per-layer statistics for interpretability."
            ),
            category="Concept Steer/Advanced",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "lens",
                    default="None",
                    options=lens_options,
                    tooltip="Select a concept lens to apply",
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=-10.0,
                    max=10.0,
                    step=0.05,
                    tooltip="Global steering strength (multiplied by per-layer mask)",
                ),
                io.Combo.Input(
                    "layer_preset",
                    default="all",
                    options=["all", "early", "middle", "late", "custom"],
                    tooltip=(
                        "Which layers to steer:\n"
                        "• all — every layer at full strength\n"
                        "• early — first third of layers (composition/layout)\n"
                        "• middle — middle third (subject/object features)\n"
                        "• late — last third (texture/detail/lighting)\n"
                        "• custom — use layer_mask input for fine control"
                    ),
                ),
                io.String.Input(
                    "layer_mask",
                    default="0-5:0.2, 6-18:1.0, 19-23:0.5",
                    tooltip=(
                        "Fine-grained per-layer strength (only used when preset='custom'). "
                        "Format: 'range:strength, range:strength, ...'\n"
                        "Examples:\n"
                        "  '6-18'  → layers 6-18 at 1.0\n"
                        "  '0-5:0.2, 6-18:1.0, 19-23:0.5'  → graduated\n"
                        "  '10,11,12:1.5'  → individual layers\n"
                        "Layers not specified default to 0.0 (no steering)."
                    ),
                ),
                io.Int.Input(
                    "num_layers",
                    default=48,
                    min=1,
                    max=200,
                    tooltip=(
                        "Total number of transformer layers in the model. "
                        "Used for preset calculations. "
                        "48 for Lumina2/Z Image Turbo, 24 for SD 1.5, 70 for SDXL."
                    ),
                ),
                io.Boolean.Input(
                    "normalize",
                    default=True,
                    tooltip="Scale direction relative to cross-attention output norms",
                ),
                io.Boolean.Input(
                    "report_stats",
                    default=True,
                    tooltip=(
                        "Print per-layer steering statistics to console. "
                        "Shows which layers are most affected by the concept injection — "
                        "useful for understanding where concepts 'live' in the model."
                    ),
                ),
                io.Int.Input(
                    "report_interval",
                    default=5,
                    min=1,
                    max=100,
                    tooltip="Print layer report every N denoising steps",
                ),
                io.String.Input(
                    "custom_lens_path",
                    default="",
                    tooltip="Override: absolute path to a .pt lens file",
                ),
            ],
            outputs=[
                io.Model.Output("MODEL"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        lens: str = "None",
        strength: float = 1.0,
        layer_preset: str = "all",
        layer_mask: str = "",
        num_layers: int = 48,
        normalize: bool = True,
        report_stats: bool = True,
        report_interval: int = 5,
        custom_lens_path: str = "",
    ):
        # ── Load direction ──
        direction = _load_direction(lens, custom_lens_path)
        if direction is None or abs(strength) < 1e-6:
            _log("No lens or zero strength — passing model through unchanged")
            return io.NodeOutput(model)

        _log(f"Direction: {direction.shape[0]}d, norm={direction.norm():.4f}")

        # ── Resolve layer mask ──
        if layer_preset == "custom":
            parsed_mask = _parse_layer_mask(layer_mask, num_layers)
        else:
            parsed_mask = _parse_layer_mask(layer_preset, num_layers)

        active_layers = sum(1 for v in parsed_mask.values() if abs(v) > 1e-6)
        _log(f"Layer mask: {active_layers} active layers out of {num_layers}")

        if active_layers == 0:
            _log("No active layers in mask — passing model through unchanged")
            return io.NodeOutput(model)

        # ── Clone model and install attn2 output patch ──
        m = model.clone()

        patch = _LayerSteerAttnPatch(
            direction=direction,
            strength=strength,
            layer_mask=parsed_mask,
            normalize=normalize,
            report_stats=report_stats,
            report_interval=report_interval,
        )

        m.set_model_attn2_output_patch(patch)

        _log(f"Installed layer steering (preset={layer_preset}, "
             f"strength={strength:.2f}, {active_layers} active layers)")

        return io.NodeOutput(m)
