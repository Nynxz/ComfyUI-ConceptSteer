"""
ConceptTrainSAEOnly — Train a standalone SAE on the text encoder's activations.

Trains a Sparse Autoencoder on diverse activations from the text encoder's
residual stream. The resulting SAE can then be loaded by the Feature Map
and Feature Gate nodes for interpretability and surgical feature control.

The SAE is concept-agnostic — it learns to decompose the general activation
space into interpretable features. Train it once, reuse it across all
concepts and prompts.

Two data source modes:
  - "fineweb" (recommended): Stream real diverse text from HuggingFace FineWeb.
    Produces 500K+ activation vectors for high-quality features.
  - "synthetic": Generate prompts from combinatorial templates (fast but limited).
    Only ~15K vectors — ok for testing, bad for production SAEs.

Usage in ComfyUI:
  [Train SAE] → sae_path → use in [Feature Map] and [Feature Gate]

Typical training:
  - synthetic:  ~2–5 min on GPU (fast, but low quality)
  - fineweb:    ~15–30 min on GPU (slower, but high quality features)
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

# Must match lens_factory.QWEN_HIDDEN_DIM — duplicated here so we can
# compute d_sae for the log message before importing lens_factory (which
# pulls in torch and is slow).
QWEN_HIDDEN_DIM = 2560


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
                "with Feature Map and Feature Gate nodes.\n\n"
                "Data source:\n"
                "• 'fineweb' (recommended): Streams real web text from HuggingFace "
                "FineWeb dataset. Produces 500K+ diverse activation vectors for "
                "high-quality, well-separated features. ~15-30 min on GPU.\n"
                "• 'synthetic': Generates prompts from templates. Fast (~2-5 min) "
                "but only ~15K vectors — underdetermined for 20K features."
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
                io.Combo.Input(
                    "data_source",
                    options=["fineweb", "synthetic"],
                    default="fineweb",
                    tooltip=(
                        "Where to get training text.\n"
                        "• fineweb: Stream from HuggingFace FineWeb (recommended). "
                        "Real diverse web text → high-quality features.\n"
                        "• synthetic: Generate from templates (fast, lower quality)."
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
                    "n_vectors",
                    default=500_000,
                    min=10_000,
                    max=5_000_000,
                    step=50_000,
                    tooltip=(
                        "Target activation vectors for training (fineweb mode). "
                        "Rule of thumb: 25–50× your SAE feature count.\n"
                        "• 8× expansion (20K features): 500K–1M vectors\n"
                        "• 16× expansion (40K features): 1M–2M vectors\n"
                        "In synthetic mode, this is ignored (uses n_prompts)."
                    ),
                ),
                io.Int.Input(
                    "epochs",
                    default=8,
                    min=1,
                    max=100,
                    step=1,
                    tooltip=(
                        "Training epochs over the collected data.\n"
                        "• fineweb (500K+ vectors): 5–10 epochs recommended.\n"
                        "• synthetic (15K vectors): use 100–300 epochs.\n"
                        "More data + fewer epochs > less data + many epochs."
                    ),
                ),
                io.Int.Input(
                    "n_prompts",
                    default=500,
                    min=100,
                    max=2000,
                    step=100,
                    tooltip=(
                        "Diverse prompts for activation collection (synthetic mode). "
                        "500 gives ~15K activation vectors. "
                        "Ignored in fineweb mode."
                    ),
                ),
                io.Float.Input(
                    "l1_coeff",
                    default=0.008,
                    min=0.001,
                    max=0.1,
                    step=0.001,
                    tooltip=(
                        "Sparsity penalty. Higher = fewer active features per "
                        "input (more selective). 0.005–0.01 recommended.\n"
                        "Too high → dead features. Too low → dense, uninterpretable."
                    ),
                ),
                io.Float.Input(
                    "learning_rate",
                    default=3e-4,
                    min=1e-5,
                    max=1e-2,
                    step=1e-5,
                    tooltip=(
                        "Peak Adam learning rate (after warmup). "
                        "3e-4 is safe for fineweb. LR warms up linearly "
                        "over the first 5% of steps then cosine-decays."
                    ),
                ),
                io.Boolean.Input(
                    "cache_activations",
                    default=True,
                    tooltip=(
                        "Save collected activations to disk for reuse. "
                        "Avoids re-collecting when re-training with different "
                        "hyperparameters. Cache is saved next to the SAE file."
                    ),
                ),
                io.String.Input(
                    "hf_dataset",
                    default="HuggingFaceFW/fineweb",
                    tooltip=(
                        "HuggingFace dataset to stream from (fineweb mode). "
                        "Default is FineWeb. Any text dataset with a 'text' column works.\n"
                        "Other options: 'HuggingFaceFW/fineweb-edu', 'allenai/c4'"
                    ),
                ),
                io.String.Input(
                    "hf_subset",
                    default="sample-10BT",
                    tooltip=(
                        "Dataset config/subset. For FineWeb, 'sample-10BT' is a "
                        "10B-token sample that's fast to stream."
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
                io.Int.Input(
                    "seed",
                    default=-1,
                    min=-1,
                    max=0xFFFFFFFF,
                    tooltip=(
                        "Random seed for reproducible feature dictionaries.\n"
                        "-1 = random (different features each run).\n"
                        "Set a fixed value (e.g. 42) to get the same feature\n"
                        "indices every time you retrain with the same data and\n"
                        "hyperparameters. Write the seed down alongside your\n"
                        "saved feature indices!"
                    ),
                ),
                io.Boolean.Input(
                    "protect_existing",
                    default=True,
                    tooltip=(
                        "If the save path already exists, auto-rename to _v2, _v3, … "
                        "instead of overwriting. Disable only when you intentionally "
                        "want to replace the file."
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
        data_source: str = "fineweb",
        layer: int = 22,
        sae_expansion: int = 8,
        n_vectors: int = 500_000,
        epochs: int = 8,
        n_prompts: int = 500,
        l1_coeff: float = 0.008,
        learning_rate: float = 3e-4,
        cache_activations: bool = True,
        hf_dataset: str = "HuggingFaceFW/fineweb",
        hf_subset: str = "sample-10BT",
        encoder_path: str = "",
        seed: int = -1,
        protect_existing: bool = True,
    ):
        import torch

        # ── Seed ──
        if seed >= 0:
            torch.manual_seed(seed)
            try:
                import numpy as np
                np.random.seed(seed & 0xFFFFFFFF)
            except ImportError:
                pass
            import random
            random.seed(seed)
            _log(f"Random seed: {seed}")
        else:
            _log("No seed set — training will produce different features each run")

        save_path = save_path.strip()
        if not save_path:
            save_path = str(_PACKAGE_ROOT / "sae" / "sae_layer22_8x.pt")

        # ── Set encoder path ──
        if encoder_path.strip():
            os.environ["QWEN_ENCODER_PATH"] = encoder_path.strip()

        # ── Import factory ──
        try:
            from lens_factory import (
                load_qwen_encoder,
                generate_diverse_prompts,
                collect_layer_activations,
                collect_activations_from_dataset,
                train_sae,
                SparseAutoencoder,
                QWEN_HIDDEN_DIM as _HIDDEN_DIM,
                DEVICE,
            )
        except ImportError as e:
            _log(f"ERROR: Could not import lens_factory: {e}")
            return io.NodeOutput("")

        t0 = time.time()
        hidden_dim = _HIDDEN_DIM
        d_sae = hidden_dim * sae_expansion

        _log(f"Training standalone SAE ({data_source} mode)")
        _log(f"  Layer: {layer}, Expansion: {sae_expansion}× → {d_sae:,} features")
        if data_source == "fineweb":
            _log(f"  Dataset: {hf_dataset}/{hf_subset}")
            _log(f"  Target vectors: {n_vectors:,}, Epochs: {epochs}")
        else:
            _log(f"  Synthetic prompts: {n_prompts}, Epochs: {epochs}")
        _log(f"  L1: {l1_coeff}, LR: {learning_rate}")
        _log(f"  Save to: {save_path}")

        # ── Step 1: Load encoder ──
        _log("Step 1/3: Loading Qwen encoder...")
        model, tokenizer = load_qwen_encoder(encoder_path)

        # ── Step 2: Collect activations ──
        if data_source == "fineweb":
            _log(f"Step 2/3: Streaming activations from {hf_dataset}...")
            cache_path = None
            if cache_activations:
                cache_path = str(
                    Path(save_path).parent
                    / f"activations_layer{layer}_{n_vectors // 1000}k.pt"
                )

            all_acts = collect_activations_from_dataset(
                model=model,
                tokenizer=tokenizer,
                layer_idx=layer,
                hidden_dim=hidden_dim,
                n_vectors=n_vectors,
                dataset_name=hf_dataset,
                dataset_subset=hf_subset,
                max_length=128,
                batch_size=16,
                verbose=True,
                save_path=cache_path,
            )
        else:
            _log(f"Step 2/3: Collecting activations (synthetic, {n_prompts} prompts)...")
            prompts = generate_diverse_prompts(n_prompts)
            _log(f"  Generated {len(prompts)} diverse prompts")

            all_acts = collect_layer_activations(
                prompts, model, tokenizer, layer, hidden_dim,
                pool="all", verbose=True,
            )

        _log(f"  Activation matrix: {all_acts.shape}")

        # ── Step 3: Train SAE ──
        _log(f"Step 3/3: Training SAE ({hidden_dim}d → {d_sae}d) for {epochs} epochs...")
        sae_model = train_sae(
            all_acts, d_sae,
            l1_coeff=l1_coeff,
            epochs=epochs,
            lr=learning_rate,
            batch_size=2048 if data_source == "fineweb" else 512,
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
        if protect_existing and save_path_obj.exists():
            v = 2
            while True:
                candidate = save_path_obj.parent / f"{save_path_obj.stem}_v{v}{save_path_obj.suffix}"
                if not candidate.exists():
                    _log(f"File exists — saving as '{candidate.name}' (protect_existing=True)")
                    save_path_obj = candidate
                    break
                v += 1
        torch.save(sae_model.state_dict(), save_path_obj)

        elapsed = time.time() - t0
        file_size = save_path_obj.stat().st_size / 1024 / 1024
        _log(f"SAE trained in {elapsed:.1f}s")
        _log(f"  Saved to: {save_path} ({file_size:.1f} MB)")
        _log(f"  Use this path in Feature Map and Feature Gate nodes")

        return io.NodeOutput(str(save_path_obj))
