"""
ConceptTimestepSteer — Timestep-aware concept steering via sampler hooks.

Motivated by "The Geometry of Noise" (Sahraee-Ardakan et al., 2026):
  The effective influence of a conditioning perturbation varies with the noise
  level σ(t). At high noise, the perturbation is washed out by the stochastic
  component; at low noise, it dominates and can overshoot. Standard Concept Steer
  adds a constant delta to conditioning, giving the same magnitude at every step.

  This node instead hooks into the diffusion model's forward pass and modulates
  the concept direction's strength per-timestep, using a user-defined schedule.
  This gives finer control — e.g., push hard in the mid-range where the model is
  making compositional decisions, and back off near t=0 where details crystallize.

Schedule Modes:
  - constant     : Same as vanilla Concept Steer (flat across all σ)
  - cosine       : Bell-shaped — peaks at mid-noise, tapers at extremes
  - linear_decay : Full strength at high noise, zero at low noise
  - linear_ramp  : Zero at high noise, full strength at low noise
  - front_loaded : Strong at high noise, rapid decay (composition steering)
  - back_loaded  : Weak at high noise, strong at low noise (detail steering)
  - custom       : User-provided comma-separated keyframes (sigma:strength pairs)

Architecture:
  Takes MODEL input, clones it, installs a model_function_wrapper that intercepts
  every forward pass, reads the current sigma, computes a schedule-modulated
  strength, and adds the concept direction to the cross-attention conditioning
  tensor (c_crossattn) before it reaches the transformer blocks.

  This is fundamentally different from the original Concept Steer node:
    - Original: modifies CONDITIONING once, before sampling begins
    - This node: modifies conditioning at each denoising step with σ-dependent strength

Usage in ComfyUI:
  [Load Model] → MODEL → [Timestep Steer] → MODEL → [KSampler]
                                ↑
                          lens: cinematic
                          schedule: cosine
                          strength: 1.5
"""

import os
import math
import torch
import torch.nn.functional as F
from comfy_api.latest import io


def _log(msg: str):
    print(f"[Concept Steer | Timestep] {msg}")


# ── Lens utilities (shared with ConceptSteer.py) ────────────────────────────

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


# ── Schedule functions ──────────────────────────────────────────────────────
# All schedules map a normalized noise level t ∈ [0, 1] to a multiplier ∈ [0, 1].
# t=1 means highest noise (pure noise), t=0 means no noise (clean data).

def _schedule_constant(t: float) -> float:
    """Flat schedule — same strength at all timesteps."""
    return 1.0


def _schedule_cosine(t: float) -> float:
    """Bell-shaped — peaks at mid-noise (t≈0.5), tapers at both extremes.

    Motivated by the observation that compositional decisions happen in the
    mid-range of σ, where the model is neither overwhelmed by noise nor
    locked into fine details.
    """
    return math.sin(math.pi * t)


def _schedule_linear_decay(t: float) -> float:
    """Full strength at high noise, linearly decaying to zero at clean.

    Useful for broad compositional steering — push the concept during
    the coarse structure phase, then let the model refine freely.
    """
    return t


def _schedule_linear_ramp(t: float) -> float:
    """Zero at high noise, linearly ramping to full at clean.

    Useful for detail-level steering — don't interfere with composition,
    but push concept aesthetics during the detail-crystallization phase.
    """
    return 1.0 - t


def _schedule_front_loaded(t: float) -> float:
    """Strong at high noise, rapid exponential decay.

    Concentrates the steering in the first ~30% of denoising steps
    where the model decides global composition and structure.
    """
    return math.exp(-3.0 * (1.0 - t))


def _schedule_back_loaded(t: float) -> float:
    """Weak at high noise, strong ramp at low noise.

    Concentrates steering in the last ~30% of steps where the model
    is refining textures, lighting, and fine-grained style.
    """
    return math.exp(-3.0 * t)


def _parse_custom_schedule(spec: str) -> list[tuple[float, float]]:
    """Parse custom keyframes: 'sigma:strength,sigma:strength,...'

    Example: '1.0:0.0, 0.5:1.5, 0.2:1.0, 0.0:0.3'
    Linearly interpolates between keyframes.
    """
    keyframes = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            s, v = part.split(":", 1)
            try:
                keyframes.append((float(s.strip()), float(v.strip())))
            except ValueError:
                continue
    keyframes.sort(key=lambda kf: kf[0], reverse=True)  # high sigma first
    return keyframes


def _eval_custom_schedule(keyframes: list[tuple[float, float]], t: float) -> float:
    """Evaluate custom schedule via linear interpolation of keyframes."""
    if not keyframes:
        return 1.0
    if t >= keyframes[0][0]:
        return keyframes[0][1]
    if t <= keyframes[-1][0]:
        return keyframes[-1][1]
    for i in range(len(keyframes) - 1):
        t_hi, v_hi = keyframes[i]
        t_lo, v_lo = keyframes[i + 1]
        if t_lo <= t <= t_hi:
            alpha = (t - t_lo) / (t_hi - t_lo) if t_hi != t_lo else 0.5
            return v_lo + alpha * (v_hi - v_lo)
    return 1.0


_SCHEDULES = {
    "constant": _schedule_constant,
    "cosine": _schedule_cosine,
    "linear_decay": _schedule_linear_decay,
    "linear_ramp": _schedule_linear_ramp,
    "front_loaded": _schedule_front_loaded,
    "back_loaded": _schedule_back_loaded,
}


# ── Model wrapper ───────────────────────────────────────────────────────────

class _TimestepSteerWrapper:
    """Callable wrapper installed via set_model_unet_function_wrapper.

    Intercepts every forward pass, reads sigma from the timestep, computes
    the schedule-modulated strength, and injects the concept direction into
    the cross-attention conditioning before calling the original model.
    """

    def __init__(
        self,
        direction: torch.Tensor,
        strength: float,
        schedule_fn,
        custom_keyframes: list[tuple[float, float]] | None,
        sigma_max: float,
        normalize: bool,
        active_tokens_only: bool,
        old_wrapper=None,
    ):
        self.direction = direction
        self.strength = strength
        self.schedule_fn = schedule_fn
        self.custom_keyframes = custom_keyframes
        self.sigma_max = sigma_max
        self.normalize = normalize
        self.active_tokens_only = active_tokens_only
        self.old_wrapper = old_wrapper
        self._step_count = 0

    def to(self, device):
        """ComfyUI calls .to(device) on wrapper objects for device management."""
        self.direction = self.direction.to(device)
        return self

    def _get_schedule_multiplier(self, sigma: float) -> float:
        """Convert raw sigma to normalized t ∈ [0,1] and evaluate schedule."""
        # Normalize: t=1 at sigma_max (pure noise), t=0 at sigma=0 (clean)
        t = min(sigma / max(self.sigma_max, 1e-6), 1.0)

        if self.custom_keyframes is not None:
            return _eval_custom_schedule(self.custom_keyframes, t)
        return self.schedule_fn(t)

    def _steer_conditioning(self, cond_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        """Add schedule-modulated concept direction to conditioning tensor."""
        schedule_mult = self._get_schedule_multiplier(sigma)
        effective_strength = self.strength * schedule_mult

        if abs(effective_strength) < 1e-6:
            return cond_tensor

        device = cond_tensor.device
        dtype = cond_tensor.dtype
        cond_dim = cond_tensor.shape[-1]

        # Project direction to match conditioning dim
        direction = self.direction.to(device=device, dtype=torch.float32)
        if direction.shape[0] != cond_dim:
            if direction.shape[0] < cond_dim:
                padded = torch.zeros(
                    cond_dim, device=device, dtype=torch.float32)
                padded[:direction.shape[0]] = direction
                direction = padded
            else:
                direction = direction[:cond_dim]

        unit_dir = F.normalize(direction, dim=0)

        # Scale relative to conditioning magnitude
        NORM_SCALE = 0.3
        token_norms = cond_tensor.float().norm(dim=-1)
        active_mask = token_norms > 0.01
        if active_mask.any():
            avg_norm = token_norms[active_mask].mean()
        else:
            avg_norm = token_norms.mean()

        if self.normalize:
            delta = effective_strength * NORM_SCALE * avg_norm * unit_dir
        else:
            delta = effective_strength * direction

        # Broadcast [dim] → shape compatible with cond_tensor
        while delta.dim() < cond_tensor.dim():
            delta = delta.unsqueeze(0)

        steered = cond_tensor.clone().float()
        if self.active_tokens_only:
            mask = (token_norms > 0.01).float()
            while mask.dim() < steered.dim():
                mask = mask.unsqueeze(-1)
            steered = steered + delta * mask
        else:
            steered = steered + delta

        return steered.to(dtype=dtype)

    def __call__(self, apply_model, args):
        """Intercept forward pass, inject concept direction into conditioning."""
        # Read current sigma from timestep
        sigma = args["timestep"]
        if isinstance(sigma, torch.Tensor):
            sigma_val = sigma.max().detach().cpu().item()
        else:
            sigma_val = float(sigma)

        self._step_count += 1

        # Modify cross-attention conditioning
        c = args["c"].copy()
        if "c_crossattn" in c:
            c_crossattn = c["c_crossattn"]
            if isinstance(c_crossattn, torch.Tensor):
                c["c_crossattn"] = self._steer_conditioning(
                    c_crossattn, sigma_val)

        # Log occasionally for debugging
        schedule_mult = self._get_schedule_multiplier(sigma_val)
        if self._step_count <= 3 or self._step_count % 10 == 0:
            _log(f"Step {self._step_count}: σ={sigma_val:.4f}, "
                 f"schedule={schedule_mult:.3f}, "
                 f"effective={self.strength * schedule_mult:.3f}")

        args_modified = {**args, "c": c}

        if self.old_wrapper is not None:
            return self.old_wrapper(apply_model, args_modified)
        return apply_model(
            args_modified["input"],
            args_modified["timestep"],
            **args_modified["c"],
        )


# ── The Node ────────────────────────────────────────────────────────────────

class ConceptTimestepSteerNode(io.ComfyNode):
    """Timestep-aware concept steering via model forward pass hooks."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        available_lenses = _discover_lenses()
        lens_options = ["None"] + available_lenses

        return io.Schema(
            node_id="conceptsteer.TimestepSteer",
            display_name="Timestep Steer",
            description=(
                "Apply concept steering with timestep-dependent strength. "
                "Hooks into the model's forward pass and modulates the concept "
                "direction at each denoising step based on the current noise level σ. "
                "Different schedules emphasize composition (high noise) vs detail (low noise)."
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
                    tooltip="Peak steering strength (modulated by schedule)",
                ),
                io.Combo.Input(
                    "schedule",
                    default="cosine",
                    options=[
                        "constant", "cosine", "linear_decay", "linear_ramp",
                        "front_loaded", "back_loaded", "custom",
                    ],
                    tooltip=(
                        "How strength varies with noise level σ:\n"
                        "• constant — flat (same as vanilla Concept Steer)\n"
                        "• cosine — bell-shaped, peaks at mid-noise\n"
                        "• linear_decay — full at high noise → zero at clean\n"
                        "• linear_ramp — zero at high noise → full at clean\n"
                        "• front_loaded — strong early (composition), rapid decay\n"
                        "• back_loaded — weak early, strong late (detail/texture)\n"
                        "• custom — user keyframes (see custom_schedule)"
                    ),
                ),
                io.String.Input(
                    "custom_schedule",
                    default="1.0:0.0, 0.7:1.0, 0.3:1.5, 0.1:0.5, 0.0:0.0",
                    tooltip=(
                        "Custom keyframes for 'custom' schedule mode. "
                        "Format: 'sigma_norm:strength, ...' where sigma_norm is "
                        "normalized noise level (1.0=max noise, 0.0=clean). "
                        "Linearly interpolated between keyframes."
                    ),
                ),
                io.Float.Input(
                    "sigma_max",
                    default=14.6,
                    min=0.1,
                    max=1000.0,
                    step=0.1,
                    tooltip=(
                        "Maximum sigma value for normalization. "
                        "14.6 is typical for SDXL/flow models. "
                        "Used to map raw sigma → normalized t ∈ [0, 1]."
                    ),
                ),
                io.Boolean.Input(
                    "normalize",
                    default=True,
                    tooltip="Scale direction relative to conditioning token norms",
                ),
                io.Boolean.Input(
                    "active_tokens_only",
                    default=True,
                    tooltip="Only modify non-padding token positions",
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
        schedule: str = "cosine",
        custom_schedule: str = "",
        sigma_max: float = 14.6,
        normalize: bool = True,
        active_tokens_only: bool = True,
        custom_lens_path: str = "",
    ):
        # ── Load direction ──
        direction = _load_direction(lens, custom_lens_path)
        if direction is None or abs(strength) < 1e-6:
            _log("No lens or zero strength — passing model through unchanged")
            return io.NodeOutput(model)

        _log(f"Direction: {direction.shape[0]}d, norm={direction.norm():.4f}")

        # ── Resolve schedule ──
        custom_keyframes = None
        if schedule == "custom":
            custom_keyframes = _parse_custom_schedule(custom_schedule)
            schedule_fn = None
            _log(f"Custom schedule: {len(custom_keyframes)} keyframes")
        else:
            schedule_fn = _SCHEDULES.get(schedule, _schedule_cosine)
            _log(f"Schedule: {schedule}")

        # ── Clone model and install wrapper ──
        m = model.clone()

        # Preserve any existing wrapper
        old_wrapper = m.model_options.get("model_function_wrapper", None)

        wrapper = _TimestepSteerWrapper(
            direction=direction,
            strength=strength,
            schedule_fn=schedule_fn,
            custom_keyframes=custom_keyframes,
            sigma_max=sigma_max,
            normalize=normalize,
            active_tokens_only=active_tokens_only,
            old_wrapper=old_wrapper,
        )

        m.set_model_unet_function_wrapper(wrapper)

        _log(f"Installed timestep steering (schedule={schedule}, "
             f"strength={strength:.2f}, σ_max={sigma_max})")

        return io.NodeOutput(m)
