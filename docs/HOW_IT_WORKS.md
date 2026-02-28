# Concept Lenses — How They Work

> **TL;DR:** A concept lens is a single direction vector in a text encoder's embedding space. Adding it to the conditioning tensor during image generation steers the output toward (or away from) an aesthetic concept — no fine-tuning, no LoRA, no extra model weights. Just a 2560-dimensional arrow that says "this way is cinematic."

---

## Table of Contents

1. [The Core Idea](#the-core-idea)
2. [Why It Works](#why-it-works)
3. [How Lenses Are Trained — DPO](#how-lenses-are-trained)
4. [How Lenses Are Trained — SAE](#sae-mode)
5. [The Cross-Modal Bridge](#the-cross-modal-bridge)
6. [How Lenses Steer Generation](#how-lenses-steer-generation)
7. [Architecture Details](#architecture-details)
8. [Lens Formats](#lens-formats)
9. [Training Modes](#training-modes)
10. [FAQ](#faq)

---

## The Core Idea

Modern text-to-image models work by encoding a text prompt into a high-dimensional embedding, then using a diffusion model to generate an image conditioned on that embedding. The embedding lives in a continuous vector space where **directions** correspond to semantic concepts.

A concept lens exploits this: we find the **direction** in the embedding space that separates "cinematic" from "not cinematic" (or any concept from its absence), then add that direction to the conditioning tensor at inference time.

```
[Your Prompt] → Text Encoder → embedding
                                    ↓
                              embedding + (strength × lens_direction)
                                    ↓
                              Diffusion Model → Image
```

This is equivalent to nudging every token's representation a little bit in the direction of your concept — without changing the prompt itself. It's like adding an invisible modifier that the model "feels" but the user never has to type.

---

## Why It Works

### Linear Representation Hypothesis

Large language models and vision-language models encode concepts as **linear directions** in their hidden states. This has been demonstrated extensively in mechanistic interpretability research:

- "King - Man + Woman ≈ Queen" (word2vec, 2013)
- Probing classifiers find linear features for sentiment, topic, style
- Sparse Autoencoders (SAEs) recover interpretable features as linear directions

The same principle applies to text encoders used in image generation. The Qwen 3.4B encoder (used by Z Image Turbo) represents "cinematic lighting" as a direction in its 2560-dimensional hidden state. If we can find that direction, we can amplify it.

### Why Addition Works

The diffusion model's conditioning mechanism is additive by design. The text embedding is injected via cross-attention, where it's multiplied by learned projection matrices. Adding a direction to the embedding before this projection is equivalent to adding a bias term to every cross-attention operation. The model was trained on a continuous distribution of embeddings — a small perturbation along a semantically meaningful direction produces a correspondingly meaningful shift in the output.

### The Norm Trick

We normalize the lens direction relative to the average token norm in the conditioning tensor. This means `strength=1.0` adds a perturbation roughly equal in magnitude to a typical token embedding. This makes the strength parameter intuitive and consistent across different prompts:

- `strength=0.3` → subtle influence
- `strength=1.0` → clearly present
- `strength=2.0` → dominant
- `strength=-1.0` → steer **away** from the concept

---

## How Lenses Are Trained

### DPO (Direct Preference Optimization) Direction Training

We adapt DPO — originally a reinforcement learning technique for LLM alignment — to find concept directions. Instead of training a policy, we train a **direction vector**:

**Input:** N text pairs. Each pair has a "positive" text (embodies the concept) and a "negative" text (neutral/opposite).

**Example pair for "cinematic":**
- **Positive:** "A sweeping aerial shot of a misty mountain range at golden hour, with dramatic lens flare cutting through the clouds and warm amber light painting the peaks in cinematic glory"
- **Negative:** "A mountain range photograph taken during the day showing peaks and clouds in the sky with normal lighting conditions"

**Training procedure:**

1. Encode all positive texts → `h_positive` (N × 2560)
2. Encode all negative texts → `h_negative` (N × 2560)
3. Initialize random unit direction `d` (2560-dim)
4. For each optimization step:
   - Compute margins: `margin_i = (h_positive_i · d) - (h_negative_i · d)`
   - DPO loss: `L = -mean(log(σ(β × margins)))`
   - Update `d` via Adam optimizer
   - Re-normalize `d` to unit length
5. Sweep β ∈ {0.1, 0.3, 0.5, 1.0, 2.0} and select the β that maximizes the minimum margin (most robust separation)

**What β controls:** The temperature parameter β determines how sharply the loss penalizes negative margins. Lower β is more forgiving (smoother gradient), higher β demands every single pair be clearly separated. We sweep and pick the best.

**What we get:** A single unit vector `d` in 2560-dimensional space such that:
- Positive concept texts have **high** dot products with `d`
- Negative/neutral texts have **low** dot products with `d`
- The minimum margin across all pairs is maximized

Typical results: 100% accuracy (all positive texts score higher than their negative counterpart), minimum margins of +15 to +24.

### Why DPO Over Simpler Methods?

We could compute `d = mean(positive) - mean(negative)` (centroid difference). This works but is fragile — outliers can skew the direction, and it doesn't guarantee that every pair is correctly separated. DPO explicitly optimizes for **worst-case separation**, yielding more robust directions.

---

## SAE Mode

### Sparse Autoencoder Feature Decomposition

The SAE mode goes deeper than DPO. Instead of operating only on the text encoder's **output** embeddings, it hooks into the **residual stream** — the intermediate hidden states flowing through the transformer layers — and decomposes them into interpretable features using a Sparse Autoencoder.

**Why this matters:** DPO gives you a direction that separates concepts, but you can't inspect *what* it learned. SAE gives you a direction built from specific, identifiable features — you know exactly which sparse features fire for "cinematic" and by how much.

### Architecture

```
Sparse Autoencoder (SAE)
  Input:  x ∈ ℝ^{2560}     (residual stream activation)
  Encode: z = ReLU(W_enc · (x - b_dec))   (pre-center, then sparse expansion)
  Decode: x̂ = W_dec · z + b_dec            (reconstruction)

  Key properties:
  - d_sae = 8 × d_input = 20,480 features (8x expansion)
  - Unit-norm decoder columns (prevents L1 cheating)
  - Pre-encoder centering (subtract b_dec = learned data mean)
  - decode_sparse(): W_dec · z (NO bias) → bias-free concept directions
```

### Full Pipeline

```
1. COLLECT: Run 500 diverse prompts through Qwen 3.4B
   Hook into layer 22 (60% depth) → collect all token-level activations
   Result: ~15,000 activation vectors in ℝ^{2560}

2. TRAIN SAE: Decompose activations into 20,480 sparse features
   - L1 sparsity with 20% warmup (prevents premature feature death)
   - Unit-norm decoder columns (each column = one feature direction)
   - 200 epochs, cosine LR schedule
   - Monitor L0 (active features per input), dead features, cosine similarity

3. EXTRACT FEATURES: Run contrastive concept texts through model + SAE
   - Positive texts → layer 22 activations → SAE encode → sparse features
   - Negative texts → layer 22 activations → SAE encode → sparse features  
   - Compute diff = mean(pos_features) - mean(neg_features)
   - Select top-K features by |diff| (default K=30)

4. RECONSTRUCT DIRECTION: Build concept direction from selected features
   - Create sparse vector with only top-K features
   - decode_sparse() → bias-free direction in activation space
   - Scale to match raw activation magnitude

5. OPTIONAL DPO REFINEMENT: Run DPO on output embeddings, blend
   - SAE direction: interpretable, decomposed into known features
   - DPO direction: optimized for robust separation
   - Blend: normalize(SAE + DPO) → best of both worlds
```

### What Layer 22 Captures

In a 36-layer transformer, layer 22 (~60% depth) is where the model transitions from syntactic processing to semantic representation. Features at this layer correspond to abstract concepts rather than surface-level patterns. The SAE decomposes these activations into ~20,000 sparse features, each representing a distinct "concept neuron."

### Interpreting SAE Features

Each lens trained with SAE mode includes metadata showing which features it uses:

```json
{
  "sae_top_k": 30,
  "sae_features": [4821, 12033, 7944, 18201, ...],
  "cos_raw_sae": 0.847,
  "cos_sae_dpo": 0.912
}
```

- `sae_features`: The specific SAE feature indices that define this concept
- `cos_raw_sae`: How well the SAE direction matches raw centroid difference (>0.8 = good)
- `cos_sae_dpo`: Alignment between SAE and DPO directions (>0.7 = good agreement)

### SAE vs DPO: When to Use Which

| Criterion | DPO | SAE | SAE + DPO |
|---|---|---|---|
| Speed | Fast (~30s) | Slow (~5min) | Slowest (~6min) |
| Interpretability | Opaque | Full feature list | Features + robust separation |
| Robustness | High | Moderate | Highest |
| Feature reuse | No | SAE cached across concepts | SAE cached + DPO |
| Best for | Quick lenses | Research, debugging | Production quality |

### Reusing a Trained SAE

Training the SAE takes the most time (collecting activations + 200 epochs). Once trained, it can be reused across all concepts:

```bash
# Train SAE once, save it
python lens_factory.py sae cinematic --sae-save ./my_sae.pt

# Reuse for other concepts (skips SAE training)
python lens_factory.py sae ethereal --sae-load ./my_sae.pt
python lens_factory.py sae dark_moody --sae-load ./my_sae.pt
```

The `batch-all --method sae` command automatically caches the SAE in memory across concepts.

---

## The Cross-Modal Bridge

### The Problem

Z Image Turbo uses Qwen 3.4B as its text encoder (2560-dimensional embeddings). But for **image-based** lens creation (few-shot mode), we need to encode images through a vision model. We use SigLIP (768-dimensional output).

Different embedding spaces can't be directly compared or combined — 768d SigLIP vectors and 2560d Qwen vectors live in completely different universes.

### The Solution: Contrastive Projection

We train a small projection network (768 → 1024 → 2560) that maps SigLIP embeddings into Qwen space:

```
SigLIP embedding (768d) → Linear(768, 1024) → GELU → LayerNorm → Linear(1024, 2560) → L2 normalize
```

**Training:** We encode the same set of texts through both SigLIP and Qwen, then train the projection with InfoNCE contrastive loss:

```
z_projected = project(siglip_embed)        # (N, 2560)
z_target = normalize(qwen_embed)           # (N, 2560)
logits = (z_projected @ z_target.T) / τ    # (N, N) similarity matrix
loss = CrossEntropy(logits, identity)       # Each row should match its diagonal
```

Temperature τ = 0.07 makes the loss sensitive to fine-grained alignment.

**Result:** After 300 epochs, the projection achieves 100% top-1 retrieval — for any SigLIP embedding, its projection is closest to the correct Qwen embedding. This means directions transfer cleanly across the bridge.

### Why This Matters

With the bridge trained, a lens direction found via DPO in Qwen space can be verified in SigLIP space (and vice versa). Image embeddings from SigLIP can be projected into Qwen space to create lenses from visual examples, not just text.

---

## How Lenses Steer Generation

The ComfyUI node (`ConceptSteer`) operates as a simple conditioning modifier:

```python
# Pseudocode for lens application
direction = load_lens("cinematic.pt")["direction"]  # (2560,)
direction = normalize(direction)                      # unit vector

for each conditioning entry:
    cond_tensor = entry[0]                           # (B, tokens, 2560)
    avg_norm = mean_norm(active_tokens(cond_tensor)) # scalar

    # Scale direction to match conditioning magnitude
    delta = strength * avg_norm * direction           # (2560,)

    # Add to all active token positions
    cond_tensor += delta                              # broadcast over (B, tokens)
```

### Step by Step

1. **Load** the lens `.pt` file → extract the direction vector
2. **Project** the direction if dimensions don't match (e.g., 768d lens on 2560d conditioning → zero-pad with warning)
3. **Normalize** the direction to unit length
4. **Scale** by `strength × average_active_token_norm` — this makes strength=1.0 mean "perturbation equals typical token magnitude"
5. **Add** the scaled direction to every active (non-padding) token position in the conditioning tensor
6. **Pass** the modified conditioning to the KSampler as usual

### What "Active Tokens Only" Means

CLIP/text encoders pad to a fixed sequence length (77 tokens for CLIP, variable for Qwen). Padding tokens are near-zero vectors. By default, we only modify tokens with norm > 0.01, avoiding polluting the padding positions. This can be toggled off if desired.

---

## Architecture Details

### Z Image Turbo Pipeline

```
Text → Qwen 3.4B → (2560d embeddings) → cap_embedder(Linear 2560→3840) → Lumina2 DiT
                          ↑
                    [Concept Steer adds direction HERE]
```

Key insight: Z Image's `cap_embedder` is a **linear** layer (2560 → 3840, no activation). This means a direction in 2560d space passes through to the 3840d space with no information loss. The steering signal reaches the diffusion model cleanly.

### Qwen 3.4B Encoder Specs

| Parameter | Value |
|---|---|
| Hidden size | 2560 |
| Layers | 36 |
| Attention heads | 32 |
| KV heads | 8 (GQA) |
| Head dimension | 128 |
| Intermediate size | 9728 |
| Vocab size | 151,936 |
| Parameters | ~4.02B |
| VRAM usage | ~8.6 GB (bf16) |

### SigLIP Encoder Specs

| Parameter | Value |
|---|---|
| Model | google/siglip-base-patch16-224 |
| Output dimension | 768 |
| Used for | Image encoding (few-shot mode), cross-modal bridge |

---

## Lens Formats

A lens is a `.pt` file (PyTorch serialized dict) containing:

### Z Image DPO Lens (Primary Format)

```python
{
    "direction": Tensor(2560),        # The concept direction (unit vector)
    "direction_dim": 2560,
    "dpo_beta": 0.5,                  # DPO temperature used
    "dpo_accuracy": 1.0,              # Classification accuracy on training pairs
    "dpo_mean_margin": 17.77,         # Average margin (higher = better separation)
    "dpo_min_margin": 15.08,          # Worst-case margin
    "encoder_name": "qwen_3_4b",
    "encoder_hidden_dim": 2560,
    "encoder_type": "Qwen3Model",
    "target_model": "z_image_turbo",
    "concept": "cinematic",
    "n_training_pairs": 10,
    "training_mode": "text_pairs",

    # Optional: cross-modal bridge weights
    "proj_sig2qwen_state": dict,      # ProjectionHead state_dict
    "proj_sig2qwen_config": {...},    # Architecture config
    "cross_modal_direction_acc": 1.0, # Bridge validation accuracy
    "siglip_model": "google/siglip-base-patch16-224",
    "siglip_dim": 768,
}
```

### SAE + DPO Lens

```python
{
    "direction": Tensor(2560),        # Blended SAE+DPO direction
    "direction_dim": 2560,
    "training_mode": "sae_dpo",
    "concept": "cinematic",

    # SAE metadata (interpretability)
    "sae_layer": 22,                  # Which transformer layer was hooked
    "sae_expansion": 8,               # SAE expansion factor (8x)
    "sae_d_sae": 20480,               # SAE hidden dimension
    "sae_top_k": 30,                  # Number of features used
    "sae_feature_indices": Tensor(30), # Which SAE features define this concept
    "sae_feature_weights": Tensor(30), # Differential weights per feature
    "sae_direction": Tensor(2560),     # Pure SAE direction (pre-blend)
    "cos_raw_sae": 0.847,             # SAE vs raw centroid alignment

    # DPO refinement
    "dpo_direction": Tensor(2560),     # Pure DPO direction (pre-blend)
    "dpo_beta": 0.5,
    "dpo_accuracy": 1.0,
    "cos_sae_dpo": 0.912,             # SAE vs DPO alignment
    "blend_margin": 18.5,             # Blended direction's separation margin

    # Cross-modal bridge (same as DPO lens)
    "proj_sig2qwen_state": dict,
    ...
}
```

### Few-Shot Image Lens

```python
{
    "direction": Tensor(2560 or 768), # Concept direction
    "d_in_siglip": Tensor(768),       # Direction in SigLIP space (if projected)
    "direction_dim": 2560,
    "training_mode": "few_shot_images",
    "n_positive_images": 15,
    "n_negative_images": 0,
    "accuracy": 0.93,
}
```

### Legacy SigLIP Lens (768d)

```python
{
    "d_in_siglip": Tensor(768),       # SigLIP-space direction
    "direction_dim": 768,
}
```

Each lens has a companion `*_metadata.json` with human-readable training stats.

---

## Training Modes

### 1. Text Pairs (Most Robust)

Provide 10+ pairs of positive/negative texts. The DPO optimizer finds the direction that maximally separates them.

```bash
python lens_factory.py text-pairs pairs.json --concept mystyle --target zimage
```

**pairs.json format:**
```json
[
    {"positive": "A dramatic...", "negative": "A regular..."},
    {"positive": "An epic...", "negative": "A standard..."}
]
```

### 2. Preset (Quick Start)

Use one of the built-in concept presets (cinematic, ethereal, dark_moody, vintage_film, minimalist, vibrant_pop):

```bash
python lens_factory.py auto cinematic --target zimage
```

### 3. Few-Shot Images

Point at a directory of example images. The factory computes a mean SigLIP embedding and derives a direction:

```bash
python lens_factory.py few-shot ./my_style_images/ --concept my_style --target zimage
```

Optionally provide negative examples:

```bash
python lens_factory.py few-shot ./positive/ --negative ./negative/ --concept my_style
```

### 4. SAE Mode (Interpretable)

Generate a lens via Sparse Autoencoder decomposition of the residual stream:

```bash
# SAE + DPO (recommended)
python lens_factory.py sae cinematic --target zimage

# SAE only (no DPO refinement)
python lens_factory.py sae cinematic --no-refine-dpo

# Custom settings
python lens_factory.py sae cinematic --sae-features 50 --layer 25 --sae-epochs 300

# Save/load SAE for reuse
python lens_factory.py sae cinematic --sae-save ./sae_layer22.pt
python lens_factory.py sae ethereal --sae-load ./sae_layer22.pt
```

### 5. Batch All

Generate all built-in presets at once:

```bash
# DPO (fast)
python lens_factory.py batch-all --target zimage --steps 5000

# SAE + DPO (interpretable)
python lens_factory.py batch-all --target zimage --method sae
```

---

## FAQ

### How is this different from a LoRA?

A LoRA modifies the model's weights (thousands to millions of parameters). A lens is a **single vector** (~10 KB of actual data, ~14 MB with the bridge weights). Lenses are:
- Instant to apply (no model reload)
- Composable (add multiple lenses)
- Bidirectional (negative strength = anti-concept)
- Tiny (vs 50-300 MB LoRAs)

### Can I combine multiple lenses?

Yes. Apply multiple ConceptSteer nodes in sequence, each with its own lens and strength. The directions are additive.

### Does this work with other models?

The concept is universal. You need:
1. A direction trained in the same embedding space as the model's text encoder
2. A ConceptSteer node that adds the direction to conditioning

Currently supported: Z Image Turbo (Qwen 3.4B / 2560d). SD 1.5 (SigLIP / 768d) lenses also work.

### How many text pairs do I need?

10 pairs is sufficient for strong results. More pairs can improve robustness but we've seen 100% accuracy with just 10 well-crafted pairs.

### What makes a good training pair?

The positive text should richly embody the concept across multiple dimensions (lighting, mood, composition, color). The negative text should describe a **similar scene** but without the concept — this forces the direction to capture the concept itself, not the scene content.

**Good pair:**
- ✅ Positive: "A dramatic chiaroscuro portrait, deep shadows, a single shaft of warm light..."
- ✅ Negative: "A portrait with standard studio lighting, even illumination, neutral background..."

**Bad pair:**
- ❌ Positive: "A cinematic mountain landscape at sunset"
- ❌ Negative: "A cat sitting on a couch"

The bad pair would learn "mountain vs cat" rather than "cinematic vs not cinematic."

### What strength should I use?

| Strength | Effect |
|---|---|
| 0.1 – 0.3 | Subtle nudge, mostly preserves original prompt |
| 0.5 – 1.0 | Clearly visible influence |
| 1.5 – 3.0 | Strong concept push, may override prompt details |
| 3.0+ | Extreme, concept dominates |
| Negative | Steers **away** from concept |

### Is this related to SAEs (Sparse Autoencoders)?

Yes — deeply. The `sae` training mode hooks into the text encoder's residual stream (intermediate hidden states), trains a Sparse Autoencoder with 8x expansion to decompose activations into ~20,000 interpretable features, then finds which features fire differentially for the concept. The result is a direction built from specific, identifiable features — unlike pure DPO which gives an opaque direction.

The DPO mode is a faster alternative that operates only on output embeddings. The recommended approach is `sae` mode with DPO refinement (`--method sae`), which gives you interpretable features AND robust separation.

See [SAE Mode](#sae-mode) for the full pipeline.

---

## Attribution & Prior Art

Concept Steer is a direct application of well-established ideas from mechanistic interpretability and representation engineering. This implementation was built by [nynxz](https://github.com/nynxz) with significant help from Claude (Anthropic). Nothing here is novel — it's an engineering integration of existing research into a practical ComfyUI tool.

Core ideas this builds on:
- **Linear Representation Hypothesis**: Neel Nanda et al., "Actually, Othello-GPT Has a Linear Emergent World Representation" (2023) — the foundational insight that concepts are linear directions in hidden states
- **Steering Vectors / Activation Addition**: Turner et al., "Activation Addition: Steering Language Models Without Optimization" (2023) — the technique of adding direction vectors to steer model behavior, which is exactly what Concept Steer does
- **Concept Activation Vectors (TCAVs)**: Kim et al., "Interpretability Beyond Feature Attribution" (ICML 2018) — the original idea of learning concept directions from contrastive examples
- **DPO for Alignment**: Rafailov et al., "Direct Preference Optimization" (2023) — the loss function we adapted for direction training
- **Sparse Autoencoders for Interpretability**: Bricken et al., "Towards Monosemanticity" (Anthropic, 2023) — the SAE architecture and training methodology we use for feature decomposition
- **Representation Engineering**: Zou et al., "Representation Engineering" (2023) — reading and writing to model representations for controllable behavior

The only thing specific to this project is applying these known techniques to text encoder conditioning in image generation pipelines (Qwen 3.4B → diffusion model), with a cross-modal SigLIP bridge for image-based lens creation. The underlying science is entirely the work of the researchers cited above.
