"""
ConceptTrainFewShot — Train a concept lens from example images within ComfyUI.

Three modes (from weakest to strongest):

  1. Centroid: Mean SigLIP embedding difference (fast, weak)
  2. Contrastive: Paired margin optimization on SigLIP embeddings (recommended)
  3. VL Caption: A VL model captions each image, then contrastive training runs
     in native Qwen text-encoder space — no lossy SigLIP→Qwen bridge (best quality)

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
                "Train a concept lens from example images. Three modes:\n"
                "• Contrastive (default): SigLIP embeddings + contrastive optimization\n"
                "• VL Caption: A VL model captions images → native text-encoder training (best)\n"
                "• Centroid: Simple mean difference (fast but weak)"
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
                        "Strongly recommended for contrastive mode. "
                        "Leave empty to use origin as contrast."
                    ),
                ),
                io.Combo.Input(
                    "target",
                    default="zimage",
                    options=["zimage", "sd15"],
                    tooltip=(
                        "Target model. 'zimage' = Qwen 3.4B (2560d), "
                        "'sd15' = SigLIP (768d)."
                    ),
                ),
                io.Combo.Input(
                    "method",
                    default="contrastive",
                    options=["contrastive", "vl_caption", "centroid"],
                    tooltip=(
                        "Training method:\n"
                        "• contrastive: SigLIP embeddings + paired margin optimization (recommended)\n"
                        "• vl_caption: VL model captions → native text-encoder training (best quality, needs VL model)\n"
                        "• centroid: Simple mean difference (fast but weak)"
                    ),
                ),
                io.Int.Input(
                    "contrastive_steps",
                    default=500,
                    min=100,
                    max=5000,
                    step=100,
                    tooltip="Contrastive optimization steps (more = better but slower)",
                ),
                io.String.Input(
                    "vl_model",
                    default="",
                    tooltip=(
                        "VL model for captioning (only used in vl_caption mode). "
                        "Leave empty to auto-detect. Examples:\n"
                        "• Qwen/Qwen2.5-VL-7B-Instruct (best, ~8GB VRAM)\n"
                        "• Qwen/Qwen2.5-VL-3B-Instruct (good, ~4GB VRAM)\n"
                        "• microsoft/Florence-2-large (light, ~1.5GB)"
                    ),
                ),
                io.String.Input(
                    "encoder_path",
                    default="",
                    tooltip=(
                        "Path to Qwen 3.4B safetensors file (for zimage target). "
                        "Leave empty to auto-detect from ComfyUI model paths."
                    ),
                ),
                io.String.Input(
                    "transcoder_repo",
                    default="",
                    tooltip=(
                        "HuggingFace repo for pretrained transcoders "
                        "(e.g. 'mwhanna/qwen3-4b-transcoders'). "
                        "When set with VL caption mode, captions are decomposed through "
                        "163,840 monosemantic transcoder features for much better concept "
                        "isolation. Only supported for zimage target."
                    ),
                ),
                io.String.Input(
                    "output_dir",
                    default="",
                    tooltip="Override output directory for the lens file",
                ),
                io.Boolean.Input(
                    "protect_existing",
                    default=True,
                    tooltip=(
                        "If the output lens file already exists, save as _v2, _v3, … "
                        "instead of overwriting. Disable only when intentionally replacing."
                    ),
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
        method: str = "contrastive",
        contrastive_steps: int = 500,
        vl_model: str = "",
        encoder_path: str = "",
        transcoder_repo: str = "",
        output_dir: str = "",
        protect_existing: bool = True,
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

        # Set encoder path env var if provided
        if encoder_path.strip():
            os.environ["QWEN_ENCODER_PATH"] = encoder_path.strip()

        _log(f"Training few-shot lens: '{concept_name}' ({target}, {method})")
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
                use_contrastive=(method in ("contrastive", "vl_caption")),
                contrastive_steps=contrastive_steps,
                use_vl_captions=(method == "vl_caption"),
                vl_model=vl_model.strip() or None,
                output_dir=out,
                transcoder_repo=transcoder_repo.strip() or None,
                overwrite=(not protect_existing),
            )
            elapsed = time.time() - t0
            _log(f"Few-shot lens trained in {elapsed:.1f}s → {lens_path}")
            return io.NodeOutput(str(lens_path))

        except Exception as e:
            _log(f"ERROR during few-shot training: {e}")
            import traceback
            traceback.print_exc()
            return io.NodeOutput("")
