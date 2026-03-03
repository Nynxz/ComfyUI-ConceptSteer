#!/usr/bin/env python3
"""
Concept Steer — Standalone Training Script
==========================================

Run this as a VS Code interactive Python file (cells separated by # %%)
or as a Jupyter notebook. No ComfyUI required.

Requirements:
  pip install torch safetensors transformers numpy

Setup:
  1. Set QWEN_ENCODER_PATH below to your qwen_3_4b.safetensors path
  2. Run cells in order, or jump to the section you need
"""

# %% [markdown]
# # Concept Steer — Lens Training
#
# Train concept steering lenses without ComfyUI.
# Each section is independent — run whichever training mode you need.

# %% ── Setup & Imports ──────────────────────────────────────────────────────

import torch.nn.functional as F
import json
import torch
import lens_factory
import sys
import os
from pathlib import Path

# Point to the comfyui-conceptsteer package root
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = PACKAGE_ROOT / "tools"

# Add tools dir to path so we can import lens_factory
sys.path.insert(0, str(TOOLS_DIR))

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  CONFIGURE THESE                                                        │
# └──────────────────────────────────────────────────────────────────────────┘

# Path to your Qwen 3.4B text encoder weights
# Set this to the location of your qwen_3_4b.safetensors file.
# If left empty, the factory will auto-discover from ComfyUI/models/text_encoders/
os.environ["QWEN_ENCODER_PATH"] = (
    ""  # e.g. "/path/to/models/text_encoders/qwen_3_4b.safetensors"
)

# Where trained lenses go (default: comfyui-conceptsteer/lenses/)
OUTPUT_DIR = PACKAGE_ROOT / "lenses" / "zimage"

print(f"Package root: {PACKAGE_ROOT}")
print(f"Output dir:   {OUTPUT_DIR}")
print(f"Encoder:      {os.environ['QWEN_ENCODER_PATH']}")

# %% ── Import Lens Factory ──────────────────────────────────────────────────


# Quick check: list available presets
lens_factory.list_presets()

# %% [markdown]
# ---
# ## 1. DPO Training — From Preset
#
# Fastest method (~2-5 min). Uses built-in text pairs.
# Available presets: cinematic, ethereal, dark_moody, vintage_film, minimalist, vibrant_pop

# %% ── DPO from Preset ──────────────────────────────────────────────────────

# Pick a preset
CONCEPT = "cinematic"

lens_path = lens_factory.generate_lens_from_preset(
    concept=CONCEPT,
    target="zimage",
    dpo_steps=5000,
    output_dir=OUTPUT_DIR,
)

print(f"\n✓ Lens saved to: {lens_path}")

# %% [markdown]
# ---
# ## 2. DPO Training — Custom Text Pairs
#
# Provide your own positive/negative text pairs.
# Must have equal counts. 10+ pairs recommended.

# %% ── DPO from Custom Pairs ────────────────────────────────────────────────

CONCEPT = "cyberpunk"

positive_texts = [
    "A rain-soaked neon megacity with holographic advertisements towering over crowded streets, chrome and glass reflecting electric pink and cyan light",
    "A cyborg woman with glowing circuitry visible beneath translucent skin, standing in a data-stream of floating kanji characters and wireframe geometry",
    "An underground hacker den lit by dozens of monitors casting blue light on walls covered in cables, circuit boards, and graffiti of digital skulls",
    "A massive corporate arcology piercing smog-filled clouds, its surface alive with crawling LED patterns, flying vehicles streaming between towers",
    "A street-level ramen stall beneath a tangle of power lines and holographic signs, steam rising into air thick with digital particles and neon glow",
    "A midnight chase through a market district, protagonist's cybernetic arm glowing blue as they vault over stalls draped in holographic fabric",
    "A rooftop view of endless neon towers stretching to the horizon under an orange-polluted sky, drones weaving between antenna arrays and satellite dishes",
    "A VR lounge where patrons float in sensory deprivation pods, their neural interfaces casting soft green light, cables trailing like cybernetic tentacles",
    "A back-alley cyberdoc's workshop cramped with surgical robots, biometric scanners, and jars of replacement organs bathed in sterile ultraviolet light",
    "A megastructure bridge spanning a toxic river, its surface covered in LED graffiti and makeshift shelters, acid rain creating prismatic puddles below",
]

negative_texts = [
    "A city at night showing buildings with some lights and signs, people walking on sidewalks, normal urban photography",
    "A person wearing electronic accessories standing in front of a computer screen, standard portrait lighting",
    "A room with multiple computer monitors and cables, typical office or workspace lighting",
    "A tall modern building photographed from below against a cloudy sky, standard architectural photography",
    "A food stall on a street with overhead wiring, normal evening lighting and some steam from cooking",
    "A person running through a market at night, standard action photography with typical lighting",
    "A cityscape view from a rooftop showing buildings and sky, standard landscape photography",
    "A spa or lounge with ambient lighting, people relaxing in standard wellness environment",
    "A medical clinic interior with equipment and standard fluorescent overhead lighting",
    "A bridge over a river in an urban area, photographed in rainy conditions with standard exposure",
]

lens_path = lens_factory.generate_lens_from_text_pairs(
    concept=CONCEPT,
    positive_texts=positive_texts,
    negative_texts=negative_texts,
    target="zimage",
    dpo_steps=5000,
    output_dir=OUTPUT_DIR,
)

print(f"\n✓ Lens saved to: {lens_path}")

# %% [markdown]
# ---
# ## 3. SAE Training — From Preset (Interpretable)
#
# Trains a Sparse Autoencoder on the residual stream, then finds which
# features fire differentially for the concept. More interpretable than DPO.
# Takes ~10-20 min. Requires more VRAM (~10GB).

# %% ── SAE from Preset ──────────────────────────────────────────────────────

CONCEPT = "cinematic"

lens_path = lens_factory.generate_lens_sae_from_preset(
    concept=CONCEPT,
    target="zimage",
    layer=None,              # default: 60% depth (layer 22)
    sae_expansion=8,         # 8x expansion → 20480 features
    sae_epochs=200,          # SAE training epochs
    top_k=30,                # keep top 30 concept features
    refine_dpo=True,         # also run DPO refinement
    dpo_steps=5000,
    sae_save_path=None,      # set to save SAE weights for reuse
    sae_load_path=None,      # set to reuse a previously trained SAE
)

print(f"\n✓ SAE lens saved to: {lens_path}")

# %% [markdown]
# ---
# ## 4. SAE Training — Custom Text Pairs
#
# Same as above but with your own pairs.

# %% ── SAE from Custom Pairs ────────────────────────────────────────────────

CONCEPT = "watercolor"

positive_texts = [
    "A landscape where colors bleed and bloom like wet watercolor on textured paper, soft edges dissolving mountains into sky with granulating pigments",
    "A portrait rendered in translucent washes of cerulean and raw sienna, the white of the paper glowing through skin tones, edges lost and found",
    "A garden scene with flowers painted in loose, flowing strokes, colors running into each other at wet boundaries, creating beautiful unpredictable bleeds",
    "An architectural study where a stone bridge emerges from fog in graded washes, the water beneath captured in flowing cobalt with salt-texture effects",
    "A still life of fruit where each piece is a study in transparent glazing, light seeming to pass through the paint itself, edges softly feathered",
    "A seascape in wet-on-wet technique where wave crests dissolve into spray, the horizon a soft lost edge between ultramarine water and cerulean sky",
    "A forest path in autumn rendered with dropping wet pigment into damp paper, leaves becoming soft explosions of cadmium orange and burnt sienna",
    "A Venice canal scene where reflections bloom unpredictably in the wet wash, buildings shimmer between representation and pure color abstraction",
    "A rainy street where figures dissolve into their reflections, the entire scene rendered in flowing graded washes of indigo, payne's grey, and yellow ochre",
    "A mountain dawn where the sky is a single graduated wash from rose to gold, peaks rendered in a few confident wet brushstrokes with granulation",
]

negative_texts = [
    "A landscape photograph with sharp detail throughout, clear boundaries between land and sky, standard digital camera output",
    "A portrait with precise skin detail, sharp focus on features, standard studio photography with accurate colors",
    "A garden photographed with standard settings, clear detail on each flower, typical digital nature photography",
    "A bridge photographed with standard lens showing clear architectural detail, normal landscape composition",
    "A photo of fruit on a table with standard still life lighting, sharp focus and accurate color reproduction",
    "An ocean photograph showing waves with sharp detail, clear horizon line, standard seascape photography",
    "A forest path photograph in autumn showing detailed leaves and trees, standard nature photography with clear textures",
    "A Venice canal photographed with standard settings, clear building reflections, typical travel photography",
    "A rainy street photograph with standard exposure, people and reflections visible with typical urban photography settings",
    "A mountain sunrise photograph with accurate gradient sky, sharp peaks, standard landscape photography",
]

# Save the SAE so we can reuse it for other concepts
SAE_SAVE_PATH = PACKAGE_ROOT / "examples" / "sae_qwen_layer22_8x.pt"

lens_path = lens_factory.generate_lens_sae(
    concept=CONCEPT,
    positive_texts=positive_texts,
    negative_texts=negative_texts,
    target="zimage",
    layer=22,
    sae_expansion=8,
    sae_epochs=200,
    n_activation_prompts=500,
    top_k=30,
    refine_dpo=True,
    dpo_steps=5000,
    output_dir=OUTPUT_DIR,
    sae_save_path=SAE_SAVE_PATH,   # save for reuse
    sae_load_path=None,            # or load: SAE_SAVE_PATH
)

print(f"\n✓ SAE lens saved to: {lens_path}")
print(f"  SAE weights saved to: {SAE_SAVE_PATH}")

# %% [markdown]
# ---
# ## 5. Few-Shot Training — From Images
#
# Train a lens from example images using SigLIP embeddings.
# Place positive images in one folder, negative in another.
# Works best with SD1.5 target (768d SigLIP output).

# %% ── Few-Shot from Images ─────────────────────────────────────────────────

CONCEPT = "my_style"
POSITIVE_DIR = "/path/to/positive/images"    # ← edit this
# ← edit this (optional, set to None)
NEGATIVE_DIR = "/path/to/negative/images"

lens_path = lens_factory.generate_lens_from_images(
    concept=CONCEPT,
    positive_dir=POSITIVE_DIR,
    negative_dir=NEGATIVE_DIR,    # set to None to use origin as contrast
    target="sd15",                # "sd15" for SigLIP-native 768d
    output_dir=OUTPUT_DIR,
)

print(f"\n✓ Few-shot lens saved to: {lens_path}")

# %% [markdown]
# ---
# ## 6. Batch Training — All Presets
#
# Train lenses for every built-in preset at once.

# %% ── Batch All Presets (DPO) ───────────────────────────────────────────────

results = []
for concept in sorted(lens_factory.CONCEPT_PRESETS.keys()):
    try:
        path = lens_factory.generate_lens_from_preset(
            concept=concept,
            target="zimage",
            dpo_steps=5000,
            output_dir=OUTPUT_DIR,
        )
        results.append((concept, "OK", str(path)))
    except Exception as e:
        results.append((concept, "FAIL", str(e)))
        print(f"\nFailed: {concept}: {e}\n")

print(f"\n{'='*60}")
print("  Batch Results")
print(f"{'='*60}")
for concept, status, info in results:
    print(f"  [{status:4s}] {concept:20s} → {info}")

# %% [markdown]
# ---
# ## 7. Batch Training — All Presets (SAE)
#
# Same as above but using SAE decomposition.
# The SAE is trained once and reused across all concepts.

# %% ── Batch All Presets (SAE) ───────────────────────────────────────────────

SAE_SAVE_PATH = PACKAGE_ROOT / "examples" / "sae_qwen_layer22_8x.pt"

results = []
for concept in sorted(lens_factory.CONCEPT_PRESETS.keys()):
    try:
        path = lens_factory.generate_lens_sae_from_preset(
            concept=concept,
            target="zimage",
            top_k=30,
            refine_dpo=True,
            dpo_steps=5000,
            sae_save_path=SAE_SAVE_PATH,
            # After 1st concept trains the SAE, load it for the rest:
            sae_load_path=SAE_SAVE_PATH if SAE_SAVE_PATH.exists() else None,
        )
        results.append((concept, "OK", str(path)))
    except Exception as e:
        results.append((concept, "FAIL", str(e)))
        print(f"\nFailed: {concept}: {e}\n")

print(f"\n{'='*60}")
print("  Batch SAE Results")
print(f"{'='*60}")
for concept, status, info in results:
    print(f"  [{status:4s}] {concept:20s} → {info}")

# %% [markdown]
# ---
# ## 8. Inspect a Trained Lens
#
# Load and examine a lens file's contents.

# %% ── Inspect Lens ─────────────────────────────────────────────────────────


LENS_PATH = OUTPUT_DIR / "cinematic.pt"   # ← edit this

data = torch.load(LENS_PATH, map_location="cpu", weights_only=False)

print(f"Lens: {LENS_PATH.name}")
print(f"Keys: {list(data.keys())}")

if "direction" in data:
    d = data["direction"]
    print(f"Direction shape: {d.shape}")
    print(f"Direction norm:  {d.norm().item():.4f}")
    print(f"Direction dtype: {d.dtype}")

if "metadata" in data:
    meta = data["metadata"]
    print(f"\nMetadata:")
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            print(f"  {k}: {v}")

if "sae_feature_indices" in data:
    indices = data["sae_feature_indices"]
    weights = data.get("sae_feature_weights", None)
    print(f"\nSAE Features: {len(indices)} active features")
    if weights is not None:
        top5 = sorted(zip(indices, weights.tolist()),
                      key=lambda x: -abs(x[1]))[:5]
        print("  Top 5 by weight:")
        for idx, w in top5:
            print(f"    Feature {idx}: weight {w:.4f}")

# Check for SigLIP bridge
if "bridge_weight" in data:
    bw = data["bridge_weight"]
    print(f"\nSigLIP→Qwen bridge: {bw.shape} ({bw.dtype})")

# %% ── List All Installed Lenses ─────────────────────────────────────────────

lens_factory.list_lenses()

# %% [markdown]
# ---
# ## 9. Compare Two Lenses
#
# Quick cosine similarity between two trained directions.

# %% ── Compare Lenses ───────────────────────────────────────────────────────


LENS_A = OUTPUT_DIR / "cinematic.pt"
LENS_B = OUTPUT_DIR / "ethereal.pt"

a = torch.load(LENS_A, map_location="cpu", weights_only=False)
b = torch.load(LENS_B, map_location="cpu", weights_only=False)

dir_a = a["direction"].float()
dir_b = b["direction"].float()

# Match dimensions if needed
min_dim = min(dir_a.shape[-1], dir_b.shape[-1])
dir_a = dir_a[..., :min_dim]
dir_b = dir_b[..., :min_dim]

cosine_sim = F.cosine_similarity(
    dir_a.reshape(1, -1),
    dir_b.reshape(1, -1),
).item()

print(f"Lens A: {LENS_A.name} ({dir_a.shape})")
print(f"Lens B: {LENS_B.name} ({dir_b.shape})")
print(f"Cosine similarity: {cosine_sim:.4f}")
print()

if abs(cosine_sim) < 0.1:
    print("→ Nearly orthogonal — these concepts are independent. Great for composing!")
elif abs(cosine_sim) < 0.3:
    print("→ Low overlap — mostly independent concepts. Safe to combine.")
elif abs(cosine_sim) < 0.6:
    print("→ Moderate overlap — some shared features. Combining may amplify shared aspects.")
else:
    print("→ High overlap — these concepts share significant direction. Careful when stacking.")

# %% [markdown]
# ---
# ## 10. Export Pairs to JSON
#
# Save custom pairs to a JSON file for CLI usage.

# %% ── Export Pairs to JSON ──────────────────────────────────────────────────


pairs = [
    {
        "positive": pos,
        "negative": neg,
    }
    for pos, neg in zip(positive_texts, negative_texts)
]

output_file = PACKAGE_ROOT / "examples" / "watercolor_pairs.json"
output_file.write_text(json.dumps(pairs, indent=2))
print(f"Saved {len(pairs)} pairs to {output_file}")
print(f"\nCLI usage:")
print(
    f"  python tools/lens_factory.py text-pairs {output_file} --concept watercolor --target zimage")
