"""
ConceptTrainFewShot — Train a concept lens from example images within ComfyUI.

Computes mean SigLIP embedding of positive images, subtracts the mean of negative
images (or origin) to get a direction vector. Optionally projects through a
SigLIP→Qwen bridge for Z Image compatibility.

Usage in ComfyUI:
  [Concept Train FewShot] → lens_path → [Concept Steer (custom_lens_path)]
"""

import os
import sys
import time
from pathlib import Path
from comfy_api.latest import io

# ── Resolve lens_factory import ──────────────────────────────────────────────
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_DIR = _PACKAGE_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


def _log(msg: str):
    print(f"[Concept Steer] {msg}")


def _get_output_dir(target: str) -> Path:
    """Get the default lens output directory for a target model."""
    lenses_dir = _PACKAGE_ROOT / "lenses"
    out = lenses_dir / target
    out.mkdir(parents=True, exist_ok=True)
    return out


class ConceptTrainFewShotNode(io.ComfyNode):
    """Train a concept lens from example images (few-shot learning)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="conceptsteer.TrainFewShot",
            display_name="Train Lens (Few-Shot)",
            description=(
                "Train a concept lens from example images. Point to a directory "
                "of images that embody the concept, and optionally a directory of "
                "images WITHOUT the concept. Uses SigLIP embeddings."
            ),
            category="Concept Steer",
            inputs=[
                io.String.Input(
                    "concept_name",
                    default="my_concept",
                    tooltip="Name for the concept (used in filename)",
                ),
                io.String.Input(
                    "positive_dir",
                    default="",
                    tooltip=(
                        "Absolute path to directory of images embodying the concept. "
                        "Minimum 2 images required. Supports jpg, png, webp."
                    ),
                ),
                io.String.Input(
                    "negative_dir",
                    default="",
                    tooltip=(
                        "Optional: directory of images WITHOUT the concept. "
                        "Leave empty to use origin as contrast (works fine for most cases)."
                    ),
                ),
                io.Combo.Input(
                    "target",
                    default="zimage",
                    options=["zimage", "sd15"],
                    tooltip=(
                        "Target model. 'zimage' = project through SigLIP→Qwen bridge "
                        "(if available), 'sd15' = use raw SigLIP 768d direction."
                    ),
                ),
                io.String.Input(
                    "output_dir",
                    default="",
                    tooltip="Override output directory for the lens file",
                ),
            ],
            outputs=[
                io.String.Output("lens_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
        concept_name: str = "my_concept",
        positive_dir: str = "",
        negative_dir: str = "",
        target: str = "zimage",
        output_dir: str = "",
    ):
        # ── Validate inputs ──
        pos_path = positive_dir.strip()
        if not pos_path:
            _log("ERROR: No positive image directory provided")
            return io.NodeOutput("")

        if not os.path.isdir(pos_path):
            _log(f"ERROR: Positive directory does not exist: {pos_path}")
            return io.NodeOutput("")

        neg_path = negative_dir.strip() if negative_dir.strip() else None
        if neg_path and not os.path.isdir(neg_path):
            _log(f"ERROR: Negative directory does not exist: {neg_path}")
            return io.NodeOutput("")

        _log(f"Training few-shot lens: '{concept_name}' ({target})")
        _log(f"  Positive dir: {pos_path}")
        if neg_path:
            _log(f"  Negative dir: {neg_path}")

        # ── Import factory ──
        try:
            import lens_factory
        except ImportError as e:
            _log(f"ERROR: Could not import lens_factory: {e}")
            _log(f"  Expected at: {_TOOLS_DIR / 'lens_factory.py'}")
            return io.NodeOutput("")

        # ── Determine output directory ──
        out = Path(output_dir.strip()) if output_dir.strip(
        ) else _get_output_dir(target)

        # ── Run training ──
        try:
            t0 = time.time()
            lens_path = lens_factory.generate_lens_from_images(
                concept=concept_name,
                positive_dir=pos_path,
                negative_dir=neg_path,
                target=target,
                output_dir=out,
            )
            elapsed = time.time() - t0
            _log(f"Few-shot lens trained in {elapsed:.1f}s → {lens_path}")
            return io.NodeOutput(str(lens_path))

        except Exception as e:
            _log(f"ERROR during few-shot training: {e}")
            import traceback
            traceback.print_exc()
            return io.NodeOutput("")
