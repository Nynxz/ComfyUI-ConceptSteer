"""
ConceptTrainSAEOnly — Train a standalone SAE on the text encoder's activations.

Trains a Sparse Autoencoder on diverse activations from the text encoder's
residual stream. The resulting SAE can then be loaded by the Feature Map
and Feature Gate nodes for interpretability and surgical feature control.

The SAE is concept-agnostic — it learns to decompose the general activation
space into interpretable features. Train it once, reuse it across all
concepts and prompts.

Usage in ComfyUI:
  [Train SAE] → sae_path → use in [Feature Map] and [Feature Gate]

Typical training: ~2-5 minutes on GPU, produces a ~200MB file.
"""

import os
import gc
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


class ConceptTrainSAEOnlyNode(io.ComfyNode):
    """Train a standalone SAE for Feature Map and Feature Gate nodes."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        default_save = str(_PACKAGE_ROOT / "sae" / "sae_layer22_8x.pt")

        return io.Schema(
            node_id="conceptsteer.TrainSAEOnly",
            display_name="Train SAE",
            description=(
                "Train a Sparse Autoencoder on diverse text encoder activations. "
                "The SAE learns to decompose the activation space into ~20K "
                "interpretable features. Train once, then use the saved file "
                "with Feature Map and Feature Gate nodes. Takes ~2-5 min on GPU."
            ),
            category="Concept Steer/Features",
            inputs=[
                io.String.Input(
                    "save_path",
                    default=default_save,
                    tooltip=(
                        "Where to save the trained SAE weights. "
                        "Use this path in Feature Map and Feature Gate nodes."
                    ),
                ),
                io.Int.Input(
                    "layer",
                    default=22,
                    min=1,
                    max=36,
                    step=1,
                    tooltip=(
                        "Which transformer layer to decompose. "
                        "22 (~60% depth) is the sweet spot for style/aesthetic "
                        "concepts — earlier layers capture syntax, later ones "
                        "are too abstract."
                    ),
                ),
                io.Int.Input(
                    "sae_expansion",
                    default=8,
                    min=2,
                    max=16,
                    step=1,
                    tooltip=(
                        "Feature multiplier. 8× on 2560d Qwen = 20,480 features. "
                        "Higher = more fine-grained features but slower training "
                        "and larger file."
                    ),
                ),
                io.Int.Input(
                    "epochs",
                    default=200,
                    min=50,
                    max=1000,
                    step=50,
                    tooltip=(
                        "Training epochs. 200 is usually sufficient. "
                        "More epochs improve feature quality at diminishing returns."
                    ),
                ),
                io.Int.Input(
                    "n_prompts",
                    default=500,
                    min=100,
                    max=2000,
                    step=100,
                    tooltip=(
                        "Diverse prompts for activation collection. "
                        "500 gives ~15K activation vectors which is plenty. "
                        "More prompts = better coverage but slower collection."
                    ),
                ),
                io.Float.Input(
                    "l1_coeff",
                    default=0.01,
                    min=0.001,
                    max=0.1,
                    step=0.001,
                    tooltip=(
                        "Sparsity penalty. Higher = fewer active features per "
                        "input (more selective). 0.01 is a good default."
                    ),
                ),
                io.String.Input(
                    "encoder_path",
                    default="",
                    tooltip=(
                        "Path to Qwen 3.4B safetensors. "
                        "Leave empty to use QWEN_ENCODER_PATH env var."
                    ),
                ),
            ],
            outputs=[
                io.String.Output("sae_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
        save_path: str = "",
        layer: int = 22,
        sae_expansion: int = 8,
        epochs: int = 200,
        n_prompts: int = 500,
        l1_coeff: float = 0.01,
        encoder_path: str = "",
    ):
        import torch

        save_path = save_path.strip()
        if not save_path:
            save_path = str(_PACKAGE_ROOT / "sae" / "sae_layer22_8x.pt")

        _log(f"Training standalone SAE")
        _log(f"  Layer: {layer}, Expansion: {sae_expansion}×")
        _log(f"  Epochs: {epochs}, Prompts: {n_prompts}")
        _log(f"  Save to: {save_path}")

        # ── Set encoder path ──
        if encoder_path.strip():
            os.environ["QWEN_ENCODER_PATH"] = encoder_path.strip()

        # ── Import factory ──
        try:
            from lens_factory import (
                load_qwen_encoder,
                generate_diverse_prompts,
                collect_layer_activations,
                train_sae,
                SparseAutoencoder,
                QWEN_HIDDEN_DIM,
                DEVICE,
            )
        except ImportError as e:
            _log(f"ERROR: Could not import lens_factory: {e}")
            return io.NodeOutput("")

        t0 = time.time()
        hidden_dim = QWEN_HIDDEN_DIM
        d_sae = hidden_dim * sae_expansion

        # ── Step 1: Load encoder ──
        _log("Step 1/3: Loading Qwen encoder...")
        model, tokenizer = load_qwen_encoder(encoder_path)

        # ── Step 2: Collect activations ──
        _log(f"Step 2/3: Collecting activations from layer {layer}...")
        prompts = generate_diverse_prompts(n_prompts)
        _log(f"  Generated {len(prompts)} diverse prompts")

        all_acts = collect_layer_activations(
            prompts, model, tokenizer, layer, hidden_dim,
            pool="all", verbose=True,
        )
        _log(f"  Activation matrix: {all_acts.shape}")

        # ── Step 3: Train SAE ──
        _log(f"Step 3/3: Training SAE ({hidden_dim}d → {d_sae}d)...")
        sae_model = train_sae(
            all_acts, d_sae,
            l1_coeff=l1_coeff,
            epochs=epochs,
            lr=5e-4,
            batch_size=512,
            label=f"layer-{layer}",
            verbose=True,
        )

        # Free activation memory
        del all_acts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Save ──
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sae_model.state_dict(), save_path_obj)

        elapsed = time.time() - t0
        file_size = save_path_obj.stat().st_size / 1024 / 1024
        _log(f"SAE trained in {elapsed:.1f}s")
        _log(f"  Saved to: {save_path} ({file_size:.1f} MB)")
        _log(f"  Use this path in Feature Map and Feature Gate nodes")

        return io.NodeOutput(str(save_path_obj))
