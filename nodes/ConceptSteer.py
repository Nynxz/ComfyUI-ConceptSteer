"""
ConceptSteer — Apply concept directions to conditioning for steered image generation.

Loads a concept lens (.pt file containing a direction vector trained via DPO/SAE)
and adds it to the CLIP/text-encoder conditioning tensor, steering image generation
towards (or away from) a learned concept.

The node operates on ComfyUI's CONDITIONING type, which is a list of
(cond_tensor, pooled_dict) tuples.  cond_tensor is [B, tokens, dim].

Supported lens formats:
  1. DPO concept lens (.pt) — contains 'direction' key (2560d for Qwen, 768d for CLIP)
  2. Cross-modal lens (.pt) — contains 'd_in_siglip' (768d SigLIP direction)
  3. SAE concept lens (.safetensors/.pt) — contains 'direction' key
  4. Raw direction tensor (.pt) — a single tensor

Dimension handling:
  - If direction dim == conditioning dim → direct injection
  - If direction dim != conditioning dim → zero-padded/truncated with warning

Usage in ComfyUI:
  [CLIP Text Encode] → CONDITIONING → [Concept Steer] → [KSampler]
"""

import os
import torch
import torch.nn.functional as F
from comfy_api.latest import io


def _log(msg: str):
    print(f"[Concept Steer] {msg}")


# ── Lens discovery ──────────────────────────────────────────────────────────

def _find_lens_root() -> str:
    """Find the lenses directory relative to this package.

    Searches:
      1. Package lenses dir: comfyui-conceptsteer/lenses/
      2. User lenses dir: ComfyUI/lenses/ (for user-created lenses)
    """
    node_dir = os.path.dirname(__file__)
    candidates = [
        # Package-bundled lenses
        os.path.join(node_dir, "..", "lenses"),
        # User lenses in ComfyUI root
        os.path.join(node_dir, "..", "..", "..", "lenses"),
    ]
    for c in candidates:
        p = os.path.normpath(c)
        if os.path.isdir(p):
            return p
    # Return the package path even if it doesn't exist yet
    return os.path.normpath(candidates[0])


_LENS_ROOT = _find_lens_root()


def _discover_lenses() -> list[str]:
    """Find all .pt and .safetensors lens files recursively.

    Returns relative paths like 'cinematic_zimage_dpo.pt' or
    'zimage/cinematic_zimage_dpo.pt' so users see which subdirectory a lens
    belongs to.
    """
    lenses = []
    root = _LENS_ROOT
    if not os.path.isdir(root):
        return lenses
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith((".pt", ".safetensors")):
                # Skip metadata JSON companions
                if fname.endswith("_metadata.json"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fname), root)
                lenses.append(rel)
    lenses.sort()
    return lenses


def _resolve_lens_path(filename: str) -> str | None:
    """Resolve a lens filename (possibly with subdir prefix) to its full path."""
    full = os.path.normpath(os.path.join(_LENS_ROOT, filename))
    if os.path.isfile(full):
        return full
    # Fallback: search all subdirs for just the basename
    basename = os.path.basename(filename)
    for dirpath, _dirnames, filenames in os.walk(_LENS_ROOT):
        if basename in filenames:
            return os.path.join(dirpath, basename)
    return None


# ── Direction extraction ────────────────────────────────────────────────────

def _extract_direction(data) -> torch.Tensor:
    """Extract a 1-D direction vector from various lens formats.

    Supported keys (checked in order):
      - 'direction'    → DPO/SAE concept lens (primary format)
      - 'd_in_siglip'  → cross-modal lens, SigLIP 768d direction
      - 'd_shared'     → shared-space direction
      - raw tensor     → use directly
    """
    if isinstance(data, torch.Tensor):
        if data.dim() == 1:
            return data
        elif data.dim() == 2 and data.shape[0] == 1:
            return data.squeeze(0)
        raise ValueError(
            f"Cannot interpret tensor shape {data.shape} as direction")

    if isinstance(data, dict):
        # DPO/SAE concept lens — primary format
        if "direction" in data:
            _log(f"Using 'direction' key ({data['direction'].shape[0]}d)")
            return data["direction"].float().squeeze()

        # Cross-modal lens — SigLIP-space direction
        if "d_in_siglip" in data:
            _log("Using d_in_siglip (SigLIP-space direction, 768d)")
            return data["d_in_siglip"].float().squeeze()

        # Shared-space direction (fallback)
        if "d_shared" in data:
            _log("Using d_shared (shared-space direction)")
            return data["d_shared"].float().squeeze()

        # Try first tensor-valued key that looks like a direction
        for key, val in data.items():
            if isinstance(val, torch.Tensor) and val.dim() <= 2:
                total_elements = val.numel()
                if 256 <= total_elements <= 8192:
                    _log(
                        f"Using key '{key}' as direction ({total_elements}-dim)")
                    return val.float().reshape(-1)

    raise ValueError(
        f"Could not extract direction from lens data. "
        f"Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
    )


# ── Projection / alignment ─────────────────────────────────────────────────

def _project_direction(direction: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Project direction to match conditioning dimension.

    If dims match, return as-is. Otherwise zero-pad or truncate.
    """
    src_dim = direction.shape[0]

    if src_dim == target_dim:
        return direction

    if src_dim < target_dim:
        _log(
            f"Direction {src_dim}d < conditioning {target_dim}d — zero-padding")
        padded = torch.zeros(target_dim, dtype=direction.dtype)
        padded[:src_dim] = direction
        return padded
    else:
        _log(f"Direction {src_dim}d > conditioning {target_dim}d — truncating")
        return direction[:target_dim]


# ── The Node ────────────────────────────────────────────────────────────────

class ConceptSteerNode(io.ComfyNode):
    """Apply a concept direction to CLIP conditioning for steered generation."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        available_lenses = _discover_lenses()
        lens_options = ["None"] + available_lenses

        return io.Schema(
            node_id="conceptsteer.Steer",
            display_name="Concept Steer",
            description=(
                "Apply a DPO/SAE-trained concept direction to conditioning. "
                "Steers image generation towards (positive strength) or away from "
                "(negative strength) a learned concept like 'cinematic', 'ethereal', etc."
            ),
            category="Concept Steer",
            inputs=[
                io.Conditioning.Input("conditioning"),
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
                    tooltip=(
                        "Steering strength. 1.0 = moderate, clearly visible effect. "
                        "Negative values steer AWAY from the concept."
                    ),
                ),
                io.Boolean.Input(
                    "normalize",
                    default=True,
                    tooltip=(
                        "Scale direction relative to conditioning norm. "
                        "When enabled, strength=1.0 means the perturbation "
                        "magnitude equals the average active token norm."
                    ),
                ),
                io.Boolean.Input(
                    "active_tokens_only",
                    default=True,
                    tooltip=(
                        "Only modify active (non-padding) token positions. "
                        "Disable to steer ALL token positions including padding."
                    ),
                ),
                io.String.Input(
                    "custom_lens_path",
                    default="",
                    tooltip=(
                        "Optional: absolute path to a .pt lens file. "
                        "Overrides the dropdown selection if provided."
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
        lens: str = "None",
        strength: float = 1.0,
        normalize: bool = True,
        active_tokens_only: bool = True,
        custom_lens_path: str = "",
    ):
        # ── Resolve lens path ──
        if custom_lens_path and custom_lens_path.strip():
            lens_path = custom_lens_path.strip()
        elif lens and lens != "None":
            lens_path = _resolve_lens_path(lens)
        else:
            _log("No lens selected — passing conditioning through unchanged")
            return io.NodeOutput(conditioning)

        if not lens_path or not os.path.isfile(lens_path):
            _log(f"Lens file not found: {lens_path} — passing through")
            return io.NodeOutput(conditioning)

        if abs(strength) < 1e-6:
            _log("Strength ~ 0 — passing conditioning through unchanged")
            return io.NodeOutput(conditioning)

        # ── Load lens ──
        _log(f"Loading lens: {os.path.basename(lens_path)}")
        try:
            data = torch.load(lens_path, map_location="cpu",
                              weights_only=False)
        except Exception as e:
            _log(f"Failed to load lens: {e}")
            return io.NodeOutput(conditioning)

        try:
            direction = _extract_direction(data)
        except ValueError as e:
            _log(f"{e}")
            return io.NodeOutput(conditioning)

        _log(
            f"Direction: {direction.shape[0]}-dim, norm={direction.norm():.4f}")

        # ── Apply to each conditioning entry ──
        new_cond = []
        for cond_entry in conditioning:
            cond_tensor = cond_entry[0]  # [B, tokens, dim]
            extra = cond_entry[1] if len(cond_entry) > 1 else {}

            cond_dim = cond_tensor.shape[-1]
            device = cond_tensor.device
            dtype = cond_tensor.dtype

            # Project direction to match conditioning dimension
            proj_dir = _project_direction(direction, cond_dim)
            proj_dir = proj_dir.to(device=device, dtype=dtype)

            # Unit-normalize the direction
            unit_dir = F.normalize(proj_dir, dim=0)

            steered = cond_tensor.clone()

            if normalize:
                # Scale relative to average active token norm.
                # The scaling factor (0.3) is chosen so that strength=1.0
                # produces a visible but non-destructive effect. Without it,
                # strength=1.0 adds a perturbation equal to the full average
                # token norm at every position, which overwhelms the signal.
                NORM_SCALE = 0.3
                token_norms = steered.norm(dim=-1)  # [B, tokens]
                active_mask = token_norms > 0.01
                if active_mask.any():
                    avg_norm = token_norms[active_mask].mean()
                else:
                    avg_norm = token_norms.mean()

                delta = strength * NORM_SCALE * avg_norm * unit_dir  # [dim]
            else:
                delta = strength * proj_dir  # [dim]

            # Broadcast: [dim] → [1, 1, dim]
            delta = delta.unsqueeze(0).unsqueeze(0)

            if active_tokens_only:
                token_norms = steered.norm(
                    dim=-1, keepdim=True)  # [B, tokens, 1]
                mask = (token_norms > 0.01).float()
                steered = steered + delta * mask
            else:
                steered = steered + delta

            new_cond.append([steered, extra])

        _log(f"Applied '{os.path.basename(lens_path)}' "
             f"(strength={strength:.2f}, dim={cond_dim}d)")

        return io.NodeOutput(new_cond)
