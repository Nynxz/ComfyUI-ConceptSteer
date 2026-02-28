"""
ConceptTrainDPO — Train a DPO concept lens from text pairs within ComfyUI.

Takes positive and negative text descriptions (one per line), trains a DPO
direction vector that separates the concept from its absence, and saves
the resulting lens as a .pt file that can be loaded by Concept Steer.

Usage in ComfyUI:
  [Concept Train DPO] → lens_path → [Concept Steer (custom_lens_path)]
"""

import os
import sys
import time
import json
import torch
from pathlib import Path
from comfy_api.latest import io

# ── Resolve lens_factory import ──────────────────────────────────────────────
# The factory lives in tools/ relative to the package root.
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


class ConceptTrainDPONode(io.ComfyNode):
    """Train a DPO concept lens from positive/negative text descriptions."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="conceptsteer.TrainDPO",
            display_name="Train Lens (DPO)",
            description=(
                "Train a concept direction via Direct Preference Optimization. "
                "Provide positive texts embodying the concept and negative texts "
                "describing similar scenes WITHOUT the concept. Fast (~30s)."
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
                        "E.g. for 'cinematic': 'A dramatic chiaroscuro portrait "
                        "with deep shadows and golden rim lighting'"
                    ),
                ),
                io.String.Input(
                    "negative_texts",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Neutral/opposite texts (one per line, same count as positive). "
                        "E.g. 'A portrait with standard studio lighting and even exposure'"
                    ),
                ),
                io.Combo.Input(
                    "target",
                    default="zimage",
                    options=["zimage", "sd15"],
                    tooltip=(
                        "Target model architecture. "
                        "'zimage' = Qwen 3.4B (2560d), 'sd15' = SigLIP (768d)"
                    ),
                ),
                io.Int.Input(
                    "dpo_steps",
                    default=5000,
                    min=500,
                    max=20000,
                    step=500,
                    tooltip="DPO optimization steps (more = more refined)",
                ),
                io.String.Input(
                    "encoder_path",
                    default="",
                    tooltip=(
                        "Path to Qwen 3.4B safetensors file (for zimage target). "
                        "Leave empty to use QWEN_ENCODER_PATH env var."
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
        positive_texts: str = "",
        negative_texts: str = "",
        target: str = "zimage",
        dpo_steps: int = 5000,
        encoder_path: str = "",
        output_dir: str = "",
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

        _log(f"Training DPO lens: '{concept_name}' ({target})")
        _log(f"  {len(pos_lines)} text pairs, {dpo_steps} steps")

        # ── Set encoder path if provided ──
        if encoder_path.strip():
            os.environ["QWEN_ENCODER_PATH"] = encoder_path.strip()

        # ── Import factory (deferred to avoid loading models at node registration) ──
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
            lens_path = lens_factory.generate_lens_from_text_pairs(
                concept=concept_name,
                positive_texts=pos_lines,
                negative_texts=neg_lines,
                target=target,
                include_bridge=(target == "zimage"),
                output_dir=out,
                dpo_steps=dpo_steps,
            )
            elapsed = time.time() - t0
            _log(f"DPO lens trained in {elapsed:.1f}s → {lens_path}")
            return io.NodeOutput(str(lens_path))

        except Exception as e:
            _log(f"ERROR during DPO training: {e}")
            import traceback
            traceback.print_exc()
            return io.NodeOutput("")
