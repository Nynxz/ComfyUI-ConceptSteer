# Concept Steer — Conditioning Vectors for ComfyUI

Add direction vectors to text encoder conditioning to bias image generation toward aesthetic concepts. Based on the [Linear Representation Hypothesis](https://arxiv.org/abs/2310.15154) — the observation that concepts are encoded as linear directions in transformer hidden states.

This is an engineering application of existing research (steering vectors, SAEs, contrastive probing) packaged as ComfyUI nodes. Nothing here is novel — see [Attribution](#attribution) for the actual research this builds on.

<table>
<tr>
<td align="center"><b>Without Concept Steer</b></td>
<td align="center"><b>With Concept Steer</b></td>
</tr>
<tr>
<td><img src="assets/z-image_01190_.png" width="400" alt="Without concept steering"></td>
<td><img src="assets/z-image_01191_.png" width="400" alt="With concept steering applied"></td>
</tr>
<tr>
<td align="center"><em>Same prompt, no steering</em></td>
<td align="center"><em>Same prompt + <a href="docs/EXAMPLES.md#vintage-film">vintage film</a> lens (strength 0.42)</em></td>
</tr>
</table>

## What Is This?

A "concept lens" is a single direction vector (2560d for Qwen, 768d for SigLIP) trained to separate concept-embodying text embeddings from neutral ones. At inference, this vector is added to the conditioning tensor before it reaches the diffusion model — biasing generation toward (or away from) the concept.

```
[CLIP Text Encode] → [Concept Steer] → [KSampler]
                         ↑
                    lens: cinematic
                    strength: 1.0
```

It works because text-to-image conditioning is injected via cross-attention, and adding a direction to the embedding before projection is equivalent to adding a bias term across all cross-attention layers. The effect is a consistent stylistic push without altering the prompt.

**This is not a replacement for LoRAs or fine-tuning.** It's a lightweight, composable alternative for broad aesthetic nudges. For fine-grained style transfer or specific subject fidelity, LoRAs remain more effective.

## Features

- **No model reload** — drop a node into your workflow
- **Small files** — each lens is ~10 KB of direction data (~14 MB with cross-modal bridge weights)
- **Bidirectional** — positive strength adds concept, negative steers away
- **Composable** — chain multiple lenses (though results from stacking are not always predictable)
- **Strength control** — from subtle (0.1) to dominant (3.0+)
- **Currently supports** Z Image Turbo (Qwen 3.4B / 2560d) and SD 1.5 (SigLIP / 768d)

## Limitations

- **Broad concepts only** — works well for general aesthetics (cinematic, vintage, moody) but not for specific subjects, characters, or fine-grained styles. It's adding a single vector to every token, so it can't encode spatially-varying or compositionally-complex concepts.
- **Prompt interference** — the direction is added regardless of prompt content. If your prompt already describes the concept, stacking a lens on top can overshoot or create artifacts.
- **Training data sensitivity** — with only 10 text pairs, the direction can overfit to incidental correlations in the training texts rather than the intended concept. Carefully crafted contrastive pairs matter a lot.
- **Not interpretable by default** — DPO lenses are opaque direction vectors. SAE lenses record which sparse features they use, but without a feature labeling pipeline those indices are just numbers.
- **Limited model support** — currently only tested with Z Image Turbo and SD 1.5. Other architectures may use non-linear conditioning injection where adding a direction doesn't transfer cleanly.

## Installation

### ComfyUI Manager

Search for "Concept Steer" in the ComfyUI Manager and install.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nynxz/comfyui-conceptsteer.git
# Restart ComfyUI
```

## Quick Start

No pre-trained lenses are shipped. Generate them locally (~30 seconds for DPO, ~5 minutes for SAE):

```bash
cd ComfyUI/custom_nodes/comfyui-conceptsteer

# Generate a single lens (DPO — fast)
python tools/lens_factory.py auto cinematic --target zimage

# Generate all 6 presets at once
python tools/lens_factory.py batch-all --target zimage
```

Or use the **Train Lens** nodes directly inside ComfyUI — no terminal needed.

### Available Presets

| Preset | Description |
|--------|-------------|
| cinematic | Dramatic lighting, shallow depth of field, film-like composition |
| ethereal | Soft focus, luminous quality, dreamy atmosphere |
| dark_moody | High contrast, deep shadows, low-key lighting |
| vintage_film | Warm color cast, grain texture, analog film characteristics |
| minimalist | Negative space, sparse composition, restrained palette |
| vibrant_pop | High saturation, bold colors, graphic contrast |

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

## Debugging / Inspection Nodes

Nodes for examining what the steering actually does to your conditioning. Useful for verifying lenses work as expected and understanding failure modes.

### Node: Lens Inspect

Dumps a lens's internal structure: weight distribution, top components, SAE feature breakdown (if applicable), and training metadata.

```
[Lens Inspect] → IMAGE (multi-panel chart) + STRING (summary)
```

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| lens | Dropdown | None | Select a lens to inspect |
| custom_lens_path | String | "" | Override: absolute path |

**Outputs**: `chart` (IMAGE) — multi-panel visualization, `summary` (STRING) — text summary.

### Node: Activation Probe

Compare conditioning before and after steering to see per-token changes in magnitude and direction.

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

**Outputs**: `chart` (IMAGE) — 4-panel analysis (cosine similarity, perturbation magnitude, norm comparison), `analysis` (STRING) — metrics.

### Node: Lens Compare

Cosine similarity and weight overlap between two lenses. Helps determine whether chaining them will produce combined or redundant effects.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| lens_a | Dropdown | None | First lens |
| lens_b | Dropdown | None | Second lens |
| custom_path_a | String | "" | Override for first lens |
| custom_path_b | String | "" | Override for second lens |

**Outputs**: `chart` (IMAGE) — comparison panels, `analysis` (STRING) — relationship report.

## Creating Custom Lenses

Use the included `lens_factory.py` CLI to create lenses:

```bash
# DPO only (fast, ~30s per lens)
python tools/lens_factory.py auto cinematic --target zimage

# SAE + DPO (slower, records which sparse features define the concept)
python tools/lens_factory.py sae cinematic --target zimage

# SAE without DPO refinement
python tools/lens_factory.py sae cinematic --no-refine-dpo

# From custom text pairs
python tools/lens_factory.py text-pairs pairs.json --concept mystyle --target zimage

# From example images (few-shot, SigLIP-based)
python tools/lens_factory.py few-shot ./my_style_images/ --concept my_style

# Reuse a trained SAE across concepts (skips the expensive training step)
python tools/lens_factory.py sae cinematic --sae-save ./my_sae.pt
python tools/lens_factory.py sae ethereal --sae-load ./my_sae.pt
```

### DPO vs SAE

| Method | Speed | Notes |
|--------|-------|-------|
| `auto` (DPO) | ~30s/lens | Trains a separating hyperplane on output embeddings. Fast, opaque. |
| `sae` | ~5min/lens | Hooks into residual stream, decomposes into sparse features, finds differential ones. Records feature indices in metadata. |
| `sae --sae-load` | ~30s/lens | Reuses a cached SAE — only the feature extraction step runs. |

SAE mode is slower because it collects ~15k activation vectors from 500 diverse prompts and trains an autoencoder with 8x expansion before doing feature extraction. The interpretability payoff is that you get a list of specific SAE feature indices that define the concept, though without a feature labeling pipeline those indices aren't human-readable.

### Text Pairs Format

```json
[
    {
        "positive": "A dramatic chiaroscuro portrait with deep shadows...",
        "negative": "A portrait with standard studio lighting..."
    }
]
```

10 pairs works for broad concepts. The negative text should describe a **similar scene** without the concept — this forces the direction to capture the style, not the content. Bad pairs (e.g., "cinematic mountain" vs "cat on couch") will learn scene differences instead.

## How It Works

See [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for the full technical details.

**Short version:**

### DPO Mode

1. Encode positive/negative text pairs through the text encoder (Qwen 3.4B or SigLIP)
2. Optimize a unit vector using a DPO-adapted loss to maximally separate positive from negative embeddings (sweep beta, pick best min-margin)
3. At inference: add the direction to every active token position, scaled by `strength × average_token_norm`

This is essentially training a linear classifier. With 10 well-crafted contrastive pairs, DPO typically achieves 100% separation.

### SAE Mode

1. Hook into the text encoder's residual stream at layer 22 (~60% depth)
2. Collect token-level activations from 500 diverse prompts
3. Train a Sparse Autoencoder (8x expansion → 20,480 features) on those activations
4. Run contrastive concept texts through the model + SAE, find features that fire differentially
5. Reconstruct a direction from the top-K features via bias-free decoding
6. Optionally blend with a DPO direction for better separation

The SAE approach follows [Bricken et al. (2023)](https://transformer-circuits.pub/2023/monosemantic-features/index.html). The resulting direction is built from identifiable sparse features rather than an opaque optimization target.

**Why this works at all:** text encoders in image generation models represent concepts as approximately linear directions in their hidden states. The diffusion model's conditioning is injected via cross-attention — adding a direction to the embedding adds a consistent bias to the attention computation. Z Image Turbo's `cap_embedder` is a linear layer (2560→3840), so the direction passes through without distortion.

## Attribution

This is an integration of existing research into a practical tool. The underlying ideas belong to:

- **Linear Representation Hypothesis**: Nanda et al. (2023), Park et al. (2024)
- **Steering Vectors / Activation Addition**: Turner et al. (2023)
- **Concept Activation Vectors (TCAVs)**: Kim et al. (ICML 2018)
- **DPO Loss**: Rafailov et al. (2023)
- **Sparse Autoencoders**: Bricken et al. (Anthropic, 2023)
- **Representation Engineering**: Zou et al. (2023)

Built by [nynxz](https://github.com/nynxz) with significant help from Claude (Anthropic).

## Requirements

- ComfyUI (latest)
- PyTorch (comes with ComfyUI)
- For lens creation: `transformers`, `safetensors` (`pip install transformers safetensors`)
- For SAE/DPO training with Qwen: ~10 GB VRAM

## License

MIT
