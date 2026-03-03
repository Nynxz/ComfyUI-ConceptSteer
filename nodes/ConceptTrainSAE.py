"""
ConceptTrainSAE — Train an SAE-based concept lens from text pairs within ComfyUI.

Hooks into the text encoder's residual stream, trains a Sparse Autoencoder to
decompose activations into interpretable features, then identifies which features
define the concept. Optionally blends with contrastive optimization for robust separation.

This produces an interpretable lens where you can see exactly which SAE features
define your concept.

Usage in ComfyUI:
  [Concept Train SAE] → lens_path → [Concept Steer (custom_lens_path)]
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


def _parse_multiline(text: str) -> list[str]:
    """Split multiline text into a list of non-empty lines."""
    return [line.strip() for line in text.strip().splitlines() if line.strip()]


def _get_output_dir(target: str) -> Path:
    """Get the default lens output directory for a target model."""
    lenses_dir = _PACKAGE_ROOT / "lenses"
    out = lenses_dir / target
    out.mkdir(parents=True, exist_ok=True)
    return out


class ConceptTrainSAENode(io.ComfyNode):
    """Train a SAE-decomposed concept lens from text descriptions."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="conceptsteer.TrainSAE",
            display_name="Train Lens (SAE)",
            description=(
                "Train a concept direction via Sparse Autoencoder decomposition "
                "of the text encoder's residual stream. Produces an interpretable "
                "lens showing which features define the concept. "
                "Slower (~5min) but more informative than contrastive alone."
            ),
            category="Concept Steer",
            inputs=[
                io.String.Input(
                    "concept_name",
                    default="my_concept",
                    tooltip="Name for the concept (used in filename)",
                ),
                io.String.Input(
                    "positive_texts",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Texts that EMBODY the concept (one per line). "
                        "10+ pairs recommended for strong results."
                    ),
                ),
                io.String.Input(
                    "negative_texts",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Neutral/opposite texts (one per line, same count as positive). "
                        "Should describe SIMILAR scenes WITHOUT the concept."
                    ),
                ),
                io.Int.Input(
                    "layer",
                    default=22,
                    min=1,
                    max=36,
                    step=1,
                    tooltip=(
                        "Which transformer layer to hook into. "
                        "Default 22 (~60% depth) is the sweet spot for concepts. "
                        "Earlier layers = lower-level features, later = more abstract."
                    ),
                ),
                io.Int.Input(
                    "sae_expansion",
                    default=8,
                    min=2,
                    max=16,
                    step=1,
                    tooltip=(
                        "SAE hidden dimension multiplier. "
                        "8x (default) = 20,480 features for 2560d Qwen. "
                        "Higher = more features but slower training."
                    ),
                ),
                io.Int.Input(
                    "sae_epochs",
                    default=200,
                    min=50,
                    max=1000,
                    step=50,
                    tooltip="SAE training epochs (default 200, more = better features)",
                ),
                io.Int.Input(
                    "sae_features",
                    default=30,
                    min=5,
                    max=100,
                    step=5,
                    tooltip=(
                        "Number of top SAE features to keep in the concept direction. "
                        "More features = richer concept but noisier."
                    ),
                ),
                io.Int.Input(
                    "n_prompts",
                    default=500,
                    min=100,
                    max=2000,
                    step=100,
                    tooltip="Number of diverse prompts for SAE activation collection",
                ),
                io.Boolean.Input(
                    "refine_contrastive",
                    default=True,
                    tooltip=(
                        "Also run contrastive optimization on output embeddings and blend with SAE direction. "
                        "Recommended for best results."
                    ),
                ),
                io.Int.Input(
                    "contrastive_steps",
                    default=500,
                    min=500,
                    max=20000,
                    step=500,
                    tooltip="Contrastive optimization steps (only used if refine_contrastive is enabled)",
                ),
                io.String.Input(
                    "sae_save_path",
                    default="",
                    tooltip=(
                        "Save the trained SAE to this path for reuse across concepts. "
                        "Saves ~5min per subsequent concept."
                    ),
                ),
                io.String.Input(
                    "sae_load_path",
                    default="",
                    tooltip=(
                        "Load a pre-trained SAE/transcoder instead of training a new one. "
                        "Supports both native .pt SAE and .safetensors transcoder formats."
                    ),
                ),
                io.String.Input(
                    "transcoder_repo",
                    default="",
                    tooltip=(
                        "HuggingFace repo for pretrained transcoders "
                        "(e.g. 'mwhanna/qwen3-4b-transcoders'). "
                        "Downloads a 64x expansion transcoder with 163,840 features "
                        "trained on ~1B tokens. Much better than training a small SAE. "
                        "Leave empty to train your own SAE."
                    ),
                ),
                io.String.Input(
                    "encoder_path",
                    default="",
                    tooltip=(
                        "Path to Qwen 3.4B safetensors file. "
                        "Leave empty to use QWEN_ENCODER_PATH env var."
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
        positive_texts: str = "",
        negative_texts: str = "",
        layer: int = 22,
        sae_expansion: int = 8,
        sae_epochs: int = 200,
        sae_features: int = 30,
        n_prompts: int = 500,
        refine_contrastive: bool = True,
        contrastive_steps: int = 5000,
        sae_save_path: str = "",
        sae_load_path: str = "",
        transcoder_repo: str = "",
        encoder_path: str = "",
        output_dir: str = "",
        protect_existing: bool = True,
    ):
        # ── Validate inputs ──
        pos_lines = _parse_multiline(positive_texts)
        neg_lines = _parse_multiline(negative_texts)

        if not pos_lines:
            _log("ERROR: No positive texts provided")
            return io.NodeOutput("")

        if not neg_lines:
            _log("ERROR: No negative texts provided")
            return io.NodeOutput("")

        if len(pos_lines) != len(neg_lines):
            _log(
                f"ERROR: Unequal text counts — {len(pos_lines)} positive vs "
                f"{len(neg_lines)} negative. Must be the same."
            )
            return io.NodeOutput("")

        _log(f"Training SAE lens: '{concept_name}'")
        _log(f"  {len(pos_lines)} text pairs, layer {layer}, "
             f"{sae_expansion}x expansion, top-{sae_features} features")

        # ── Set encoder path if provided ──
        if encoder_path.strip():
            os.environ["QWEN_ENCODER_PATH"] = encoder_path.strip()

        # ── Import factory ──
        try:
            import lens_factory
        except ImportError as e:
            _log(f"ERROR: Could not import lens_factory: {e}")
            _log(f"  Expected at: {_TOOLS_DIR / 'lens_factory.py'}")
            return io.NodeOutput("")

        # ── Determine output directory ──
        out = Path(output_dir.strip()) if output_dir.strip(
        ) else _get_output_dir("zimage")

        # ── Resolve SAE paths ──
        sae_save = Path(sae_save_path.strip()
                        ) if sae_save_path.strip() else None
        sae_load = Path(sae_load_path.strip()
                        ) if sae_load_path.strip() else None

        # ── Run training ──
        try:
            t0 = time.time()
            lens_path = lens_factory.generate_lens_sae(
                concept=concept_name,
                positive_texts=pos_lines,
                negative_texts=neg_lines,
                target="zimage",
                layer=layer,
                sae_expansion=sae_expansion,
                sae_epochs=sae_epochs,
                n_activation_prompts=n_prompts,
                top_k=sae_features,
                refine_contrastive=refine_contrastive,
                contrastive_steps=contrastive_steps,
                include_bridge=True,
                output_dir=out,
                sae_save_path=sae_save,
                sae_load_path=sae_load,
                transcoder_repo=transcoder_repo.strip() or None,
                overwrite=(not protect_existing),
            )
            elapsed = time.time() - t0
            _log(f"SAE lens trained in {elapsed:.1f}s → {lens_path}")
            return io.NodeOutput(str(lens_path))

        except Exception as e:
            _log(f"ERROR during SAE training: {e}")
            import traceback
            traceback.print_exc()
            return io.NodeOutput("")
