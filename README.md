# Concept Steer — Concept Steering for ComfyUI

Steer image generation toward (or away from) learned aesthetic concepts using **direction vectors** extracted via Sparse Autoencoder feature decomposition and DPO optimization. No LoRA, no fine-tuning, no extra model weights — just a single vector that nudges the conditioning toward your desired style.

## What Are Concept Lenses?

A concept lens is a direction vector in the text encoder's embedding space. Adding it to the conditioning tensor during image generation steers the output toward a concept like "cinematic," "ethereal," or "vintage film" — without changing your prompt.

```
[CLIP Text Encode] → [Concept Steer] → [KSampler]
                         ↑
                    lens: cinematic
                    strength: 1.0
```

**Think of it as an invisible prompt modifier** the model can feel but the user never has to type.

## Features

- **Instant application** — no model reload, just drop a node into your workflow
- **Tiny files** — each lens is ~10 KB of direction data (vs 50-300 MB LoRAs)
- **Bidirectional** — positive strength = more concept, negative = anti-concept
- **Composable** — chain multiple lenses for combined effects
- **Strength control** — from subtle nudge (0.1) to dominant (3.0+)
- **Works with Z Image Turbo** (Qwen 3.4B / 2560d) and SD 1.5 (SigLIP / 768d)

## Installation

### ComfyUI Manager

Search for "Concept Steer" in the ComfyUI Manager and install.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nynxz/comfyui-conceptsteer.git
# Restart ComfyUI
```

## Quick Start — Generate Your First Lens

No pre-trained lenses are shipped. Generate them locally in ~30 seconds:

```bash
cd ComfyUI/custom_nodes/comfyui-conceptsteer

# Generate a single lens
python tools/lens_factory.py auto cinematic --target zimage

# Generate all 6 presets at once
python tools/lens_factory.py batch-all --target zimage
```

Or use the **Train Lens** nodes directly inside ComfyUI — no terminal needed.

### Available Presets

| Preset | Style |
|--------|-------|
| cinematic | Hollywood dramatic lighting & composition |
| ethereal | Dreamy, soft, luminous otherworldly quality |
| dark_moody | High contrast, deep shadows, emotional intensity |
| vintage_film | Analog film grain, warm tones, light leaks |
| minimalist | Clean, sparse, negative space, restrained palette |
| vibrant_pop | Highly saturated, bold colors, graphic punch |

Generated lenses are saved to `lenses/` and automatically appear in the Concept Steer node dropdown.

## Node: Concept Steer

### Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| conditioning | CONDITIONING | — | From CLIP Text Encode |
| lens | Dropdown | None | Select a concept lens |
| strength | Float | 1.0 | -10.0 to 10.0. Negative = anti-concept |
| normalize | Boolean | True | Scale relative to token norms |
| active_tokens_only | Boolean | True | Only modify non-padding tokens |
| custom_lens_path | String | "" | Override: absolute path to a lens file |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| CONDITIONING | CONDITIONING | Steered conditioning for KSampler |

### Strength Guide

| Strength | Effect |
|----------|--------|
| 0.1 – 0.3 | Subtle nudge |
| 0.5 – 1.0 | Clearly visible |
| 1.5 – 3.0 | Strong push |
| 3.0+ | Dominant |
| Negative | Steers away |

## Node: Train Lens (DPO)

Train a concept direction via Direct Preference Optimization from text descriptions.

```
[Train Lens (DPO)] → lens_path → [Concept Steer (custom_lens_path)]
```

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| concept_name | String | "my_concept" | Name for the concept |
| positive_texts | String (multiline) | — | Texts embodying the concept (one per line) |
| negative_texts | String (multiline) | — | Neutral texts without the concept (one per line) |
| target | Combo | zimage | zimage (2560d) or sd15 (768d) |
| dpo_steps | Int | 5000 | Optimization steps |
| encoder_path | String | "" | Path to Qwen 3.4B safetensors |
| output_dir | String | "" | Override output directory |

**Output**: `lens_path` (String) — absolute path to the saved lens file.

## Node: Train Lens (SAE)

Train an interpretable concept lens via Sparse Autoencoder decomposition of the text encoder's residual stream. Shows exactly which features define your concept.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| concept_name | String | "my_concept" | Name for the concept |
| positive_texts | String (multiline) | — | Concept texts (one per line) |
| negative_texts | String (multiline) | — | Neutral texts (one per line) |
| layer | Int | 22 | Transformer layer to hook (1-36) |
| sae_expansion | Int | 8 | Hidden dimension multiplier |
| sae_epochs | Int | 200 | SAE training epochs |
| sae_features | Int | 30 | Top-K features to keep |
| n_prompts | Int | 500 | Diverse prompts for activation collection |
| refine_dpo | Boolean | True | Blend with DPO direction |
| dpo_steps | Int | 5000 | DPO optimization steps |
| sae_save_path | String | "" | Save SAE for reuse |
| sae_load_path | String | "" | Load pre-trained SAE |
| encoder_path | String | "" | Path to Qwen 3.4B |
| output_dir | String | "" | Override output directory |

**Output**: `lens_path` (String) — absolute path to the saved lens file.

## Node: Train Lens (Few-Shot)

Train a concept lens from example images using SigLIP embeddings.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| concept_name | String | "my_concept" | Name for the concept |
| positive_dir | String | — | Directory of concept images (min 2) |
| negative_dir | String | "" | Optional: directory of non-concept images |
| target | Combo | zimage | zimage or sd15 |
| output_dir | String | "" | Override output directory |

**Output**: `lens_path` (String) — absolute path to the saved lens file.

## Interpretability Nodes

These nodes help you understand what's happening inside the steering process — which concepts are being pushed, how much each token changes, and how lenses relate to each other.

### Node: Lens Inspect

Visualize a lens's internal structure: weight distributions, top direction components, SAE feature breakdown, and training metadata.

```
[Lens Inspect] → IMAGE (multi-panel chart) + STRING (summary)
```

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| lens | Dropdown | None | Select a lens to inspect |
| custom_lens_path | String | "" | Override: absolute path |

**Outputs**: `chart` (IMAGE) — multi-panel visualization, `summary` (STRING) — text report of lens properties.

**What you'll see:**
- Weight distribution histogram (how sparse/dense the direction is)
- Top-30 largest weight components with their indices
- SAE feature activations (if SAE lens) — which interpretable features define the concept
- Stats: dimensionality, norm, sparsity, training accuracy

### Node: Activation Probe

Compare conditioning BEFORE and AFTER steering to see exactly how the lens affects each token.

```
[CLIP Text Encode] → original ──→ [Activation Probe] ←── steered ← [Concept Steer]
                                         ↓
                                  IMAGE + STRING
```

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| original | CONDITIONING | — | Conditioning before steering |
| steered | CONDITIONING | — | Conditioning after steering |
| label | String | "" | Optional label for the chart |

**Outputs**: `chart` (IMAGE) — 4-panel analysis, `analysis` (STRING) — metrics.

**What you'll see:**
- Per-token cosine similarity (how much each token changed direction)
- Per-token perturbation magnitude (how much each token was pushed)
- Token norm comparison (original vs steered side by side)
- Summary statistics with steering intensity assessment

### Node: Lens Compare

Side-by-side comparison of two concept lenses to understand their relationship.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| lens_a | Dropdown | None | First lens |
| lens_b | Dropdown | None | Second lens |
| custom_path_a | String | "" | Override for first lens |
| custom_path_b | String | "" | Override for second lens |

**Outputs**: `chart` (IMAGE) — comparison panels, `analysis` (STRING) — relationship report.

**What you'll see:**
- Weight distribution overlays (how the two lenses differ)
- Top weight components side by side
- SAE feature overlap (if both are SAE lenses) — Venn-style chart
- Composability assessment (cosine similarity, orthogonality ratio)
- Whether chaining the two lenses will produce combined or redundant effects

## Creating Custom Lenses

Use the included `lens_factory.py` tool to create your own lenses:

```bash
# SAE + DPO (recommended — interpretable & robust)
python tools/lens_factory.py sae cinematic --target zimage

# SAE only (no DPO refinement)
python tools/lens_factory.py sae cinematic --no-refine-dpo

# DPO only (fast, ~30s per lens)
python tools/lens_factory.py auto cinematic --target zimage

# From custom text pairs
python tools/lens_factory.py text-pairs pairs.json --concept mystyle --target zimage

# From example images (few-shot)
python tools/lens_factory.py few-shot ./my_style_images/ --concept my_style

# Generate all presets with SAE + DPO
python tools/lens_factory.py batch-all --target zimage --method sae

# Reuse a trained SAE across concepts (saves time)
python tools/lens_factory.py sae cinematic --sae-save ./my_sae.pt
python tools/lens_factory.py sae ethereal --sae-load ./my_sae.pt
```

### SAE vs DPO

| Method | Speed | What You Get |
|--------|-------|-------------|
| `auto` (DPO) | ~30s/lens | Opaque but effective direction |
| `sae` | ~5min/lens | Interpretable features + optional DPO blend |
| `sae --sae-load` | ~30s/lens | Reuse cached SAE, fast feature extraction |

### Text Pairs Format

```json
[
    {
        "positive": "A dramatic chiaroscuro portrait with deep shadows...",
        "negative": "A portrait with standard studio lighting..."
    }
]
```

10 pairs is sufficient for strong results. The positive text should richly embody the concept; the negative should describe a **similar scene** without the concept.

## How It Works

See [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for the full technical write-up.

**Short version:**

Two lens extraction methods are available:

### SAE Mode (Interpretable)

1. Hook into the text encoder's **residual stream** at layer 22 (~60% depth)
2. Collect token-level activations from 500 diverse prompts
3. Train a **Sparse Autoencoder** (8x expansion → 20,480 features) to decompose activations into interpretable features
4. Run contrastive concept texts through the model + SAE
5. Find which features fire differentially (top-K by magnitude)
6. Reconstruct a clean direction from those features via `decode_sparse()` (bias-free)
7. Optionally blend with a DPO-refined direction for robust separation

SAE lenses include metadata showing exactly which features define the concept — making them interpretable and debuggable.

### DPO Mode (Fast)

1. Encode positive/negative text pairs through the text encoder
2. Optimize a unit vector in embedding space using Direct Preference Optimization to maximally separate the concept from its absence
3. At inference: add the direction to every active token position, scaled by strength × average token norm

Both work because modern text encoders represent concepts as linear directions in their hidden states (the Linear Representation Hypothesis), and the diffusion model's conditioning mechanism is additive by design.

## Requirements

- ComfyUI (latest)
- PyTorch (comes with ComfyUI)
- For lens creation: `transformers`, `safetensors` (via `pip install transformers safetensors`)

## License

MIT
