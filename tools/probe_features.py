#!/usr/bin/env python3
"""
SAE Feature Probe — Discover what individual SAE features respond to.

For each feature in a trained SAE, finds which prompts maximally activate it,
giving you a rough semantic label. Can also probe specific features from a
trained lens to understand what concept components it uses.

Usage:
  # Probe all active features with diverse prompts
  python tools/probe_features.py --sae-path ./sae_layer22_8x.pt

  # Probe specific features from a lens file
  python tools/probe_features.py --lens-path ./lenses/zimage/cinematic_zimage_sae_contrastive.pt

  # Probe with more prompts for better coverage
  python tools/probe_features.py --sae-path ./sae.pt --n-prompts 1000 --top-k 10

  # Save results to JSON
  python tools/probe_features.py --sae-path ./sae.pt --output features.json
"""

from lens_factory import (
    SparseAutoencoder,
    load_qwen_encoder,
    DEVICE,
    QWEN_HIDDEN_DIM,
)
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# Add tools dir to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


# ── Diverse probing prompts ─────────────────────────────────────────────────
# These are designed to cover a wide range of visual and semantic concepts
# so that different SAE features will activate for different ones.

PROBE_PROMPTS = [
    # Colors
    "A deep red rose on a dark background",
    "A bright blue sky over a calm ocean",
    "A field of golden wheat under warm sunlight",
    "A neon green sign glowing in the dark",
    "A purple sunset with pink clouds",
    "Everything rendered in monochrome black and white",
    "Pastel pink and lavender soft tones everywhere",
    "Deep orange and amber autumn colors",
    "Cool blue and teal underwater tones",
    "Vibrant rainbow of saturated colors",

    # Lighting
    "Dramatic chiaroscuro lighting with deep shadows",
    "Soft diffused light coming through curtains",
    "Harsh midday sun creating strong sharp shadows",
    "Golden hour warm backlit scene",
    "Cold blue moonlight illuminating a scene",
    "Neon lights reflecting on wet surfaces",
    "Candlelight flickering in a dark room",
    "Overcast flat grey lighting on a cloudy day",
    "Rim lighting silhouetting a figure from behind",
    "Studio lighting with soft boxes and fill light",
    "Volumetric light rays through fog",
    "Fluorescent harsh overhead lighting",

    # Mood / Atmosphere
    "A dark moody atmospheric scene with tension",
    "A bright cheerful happy scene full of joy",
    "A melancholic lonely scene in the rain",
    "A peaceful serene calm tranquil scene",
    "A chaotic energetic dynamic action scene",
    "A mysterious foggy scene with hidden details",
    "A nostalgic warm memory of childhood",
    "A cold sterile clinical environment",
    "An eerie unsettling uncanny atmosphere",
    "A romantic dreamy soft focus scene",

    # Composition / Style
    "Extreme close-up macro detail of a surface",
    "Wide angle panoramic landscape vista",
    "Shallow depth of field with blurred background",
    "Everything in sharp focus deep depth of field",
    "Bird's eye view from directly above",
    "Low angle looking up at a tall structure",
    "Symmetrical perfectly balanced composition",
    "Rule of thirds off-center subject placement",
    "Dutch angle tilted frame disorienting",
    "Long exposure smooth water and streaked clouds",

    # Texture / Material
    "Rough weathered rustic wood grain texture",
    "Smooth polished reflective chrome metal",
    "Soft fluffy fuzzy fabric texture",
    "Wet glistening dewy surface with droplets",
    "Cracked dry desert earth texture",
    "Translucent glass with light passing through",
    "Matte paper-like flat surface",
    "Organic natural stone with veins and patterns",

    # Camera / Film
    "Shot on 35mm film with visible grain",
    "Clean digital photography with no noise",
    "Polaroid instant camera with characteristic border",
    "Medium format camera with creamy bokeh",
    "Lens flare and optical aberrations",
    "Infrared photography with surreal colors",
    "Tilt shift miniature effect",
    "Fish eye lens extreme distortion",

    # Art Styles
    "Oil painting with visible brushstrokes",
    "Watercolor painting with soft bleeding edges",
    "Pencil sketch with crosshatching",
    "Digital art clean vector illustration",
    "Japanese ukiyo-e woodblock print style",
    "Art nouveau with flowing organic curves",
    "Brutalist concrete geometric architecture",
    "Psychedelic swirling patterns and colors",
    "Pixel art retro 8-bit style",
    "Photorealistic hyperrealistic rendering",

    # Subjects (to see if features are content vs style)
    "A portrait of a person looking at camera",
    "A landscape with mountains and a lake",
    "A still life arrangement on a table",
    "An architectural interior of a building",
    "A street scene with people and vehicles",
    "Animals in their natural habitat",
    "Food plated beautifully on a dish",
    "Abstract shapes and geometric forms",
    "A forest with trees and foliage",
    "An ocean scene with waves",

    # Specific aesthetic concepts
    "Cinematic movie still with dramatic framing",
    "Ethereal otherworldly dreamy floating quality",
    "Vintage retro nostalgic old fashioned look",
    "Minimalist clean simple sparse composition",
    "Cyberpunk neon futuristic dystopian city",
    "Steampunk Victorian industrial with gears",
    "Gothic dark ornate Victorian architecture",
    "Kawaii cute pastel soft rounded shapes",
    "Wabi-sabi imperfect weathered aged beauty",
    "Vaporwave glitch aesthetic retro digital",

    # Technical / Rendering
    "High contrast with crushed blacks",
    "Low contrast flat muted tones",
    "High saturation vivid intense colors",
    "Desaturated muted subdued palette",
    "Warm color temperature orange tinted",
    "Cool color temperature blue tinted",
    "High key bright overexposed luminous",
    "Low key dark underexposed shadowy",
    "Sharp crisp detailed high resolution",
    "Soft blurry out of focus dreamy",

    # Emotion / Narrative
    "A sense of vast scale and grandeur",
    "Intimate personal close quiet moment",
    "Motion and speed dynamic movement",
    "Stillness and frozen time",
    "Decay entropy and the passage of time",
    "Growth bloom and new beginnings",
    "Isolation solitude empty vast space",
    "Crowd density and urban chaos",
]


def generate_extra_prompts(n: int = 500, seed: int = 123) -> list[str]:
    """Generate additional diverse prompts for broader coverage.

    Uses multiple template structures to avoid formulaic patterns that
    would cause the same modifier words to dominate feature labels.
    """
    import random as _rng
    rng = _rng.Random(seed)

    # ── Vocabulary pools ────────────────────────────────────────────────
    subjects = [
        "a bottle", "a chair", "a tree", "a car", "a house", "a dog",
        "a bridge", "a clock", "a book", "a guitar", "a ship", "a lamp",
        "a sword", "shoes", "a robot", "a skull", "a castle", "a mushroom",
        "a crystal", "a candle", "a teapot", "a statue", "a mask", "a cage",
        "a mirror", "a feather", "a key", "a lantern", "a compass", "a crown",
        "a butterfly", "a piano", "a telescope", "a bicycle", "a train",
        "a horseshoe", "a lighthouse", "a windmill", "a violin", "a hammer",
        "a pocket watch", "a globe", "a staircase", "a doorway", "a window",
        "a typewriter", "a camera", "a telescope", "a gramophone", "a bell",
        "a chalice", "a ring", "a pearl", "a coin", "a dice", "a chess piece",
        "a leaf", "a seashell", "a pinecone", "a coral", "a bone", "a horn",
    ]

    environments = [
        "at sunset", "in the rain", "covered in frost", "on fire",
        "in a dark room", "with fog around it", "reflected in water",
        "in a field of flowers", "on a city street at night", "in a forest",
        "in a studio with white background", "in a cluttered workshop",
        "in a desert", "on a mountaintop", "in a library",
        "on a beach at dawn", "in an abandoned building", "in a greenhouse",
        "on a rooftop at night", "in a subway tunnel", "on a frozen lake",
        "beside a campfire", "in a cathedral", "in a marketplace",
        "on a train platform", "in a garden", "in a cave", "on a pier",
        "in a parking lot", "in a museum gallery", "floating in clouds",
        "on a chessboard", "in a dollhouse", "on cracked earth",
    ]

    materials = [
        "made of glass", "made of wood", "made of metal", "made of stone",
        "made of ice", "made of clay", "made of paper", "made of silk",
        "made of copper", "made of obsidian", "made of porcelain",
        "made of concrete", "made of driftwood", "made of leather",
        "made of wax", "made of coral", "made of bone", "made of amber",
        "made of smoke", "made of light", "made of shadow", "made of vines",
    ]

    lighting = [
        "in soft pink light", "in harsh fluorescent light", "in candlelight",
        "in neon light", "at golden hour", "in blue moonlight",
        "in warm amber light", "in cool teal light", "backlit with lens flare",
        "in silhouette", "in dramatic spotlight", "in dappled forest light",
        "in firelight", "in overcast diffused light", "in infrared",
        "in bioluminescent glow", "in stark white light", "in twilight",
        "lit from below", "in starlight", "in stained glass light",
    ]

    conditions = [
        "covered in moss", "rusted and old", "brand new and shiny",
        "crumbling and ancient", "wrapped in thorns", "dusted with snow",
        "dripping with honey", "tangled in rope", "half-buried in sand",
        "overgrown with ivy", "charred and burnt", "polished to a mirror",
        "cracked and repaired with gold", "peeling paint", "barnacle-covered",
        "frost-covered and sparkling", "sun-bleached and faded",
        "splattered with paint", "wrapped in cloth", "melting slowly",
    ]

    viewpoints = [
        "extreme close up", "from far away", "from above looking down",
        "from a low angle looking up", "through a doorway", "through a window",
        "reflected in a mirror", "seen through frosted glass",
        "shot with a macro lens", "with shallow depth of field",
        "with tilt-shift miniature effect", "in forced perspective",
        "from behind", "from the side", "as a cross-section",
    ]

    render_styles = [
        "highly detailed", "minimalist and simple", "abstract and surreal",
        "photorealistic", "painted in oil", "drawn in pencil", "in watercolor",
        "as a woodcut print", "as a cyanotype", "as a charcoal sketch",
        "as stained glass", "as a mosaic", "rendered in 3D",
        "as a blueprint", "as an engraving", "as a linocut",
        "pointillist dots", "in gouache", "as a collage",
    ]

    adjectives = [
        "ancient", "futuristic", "tiny", "enormous", "broken", "ornate",
        "simple", "twisted", "luminous", "dark", "translucent", "heavy",
        "delicate", "geometric", "organic", "symmetrical", "chaotic",
        "pristine", "decayed", "ethereal", "grotesque", "elegant", "raw",
        "hollow", "solid", "floating", "buried", "tangled", "smooth", "rough",
    ]

    scenes = [
        "a quiet morning in a coastal village",
        "a busy intersection at rush hour",
        "an empty classroom after school",
        "a crowded festival with lanterns",
        "a single tree on a vast plain",
        "a narrow alley between tall buildings",
        "footprints disappearing into snow",
        "a table set for dinner with nobody there",
        "a child's drawing pinned to a wall",
        "raindrops on a window pane",
        "laundry hanging on a clothesline",
        "a row of mailboxes at a crossroad",
        "a broken fence beside a field",
        "a stack of old books with dust",
        "tire tracks in fresh mud",
        "a jar of fireflies in darkness",
        "autumn leaves in a puddle",
        "a shadow cast on a blank wall",
        "a crack of light under a closed door",
        "mist rising from a river at dawn",
        "a cobblestone street glistening after rain",
        "shelves of apothecary bottles",
        "a chess game mid-play on a park bench",
        "tangled headphones on a desk",
        "a half-eaten meal on a kitchen counter",
        "wilting flowers in a vase",
        "a telescope pointed at the night sky",
        "graffiti on a concrete wall",
        "a paper crane on a windowsill",
        "rusted tools hanging in a shed",
    ]

    abstract_concepts = [
        "the feeling of nostalgia",
        "silence made visible",
        "the weight of time",
        "controlled entropy",
        "frozen music",
        "visible gravity",
        "compressed space",
        "slow motion impact",
        "recursive patterns",
        "tension and release",
        "balance and imbalance",
        "negative space as subject",
        "the boundary between order and chaos",
        "emergence from simplicity",
        "the texture of sound",
    ]

    # ── Template functions ──────────────────────────────────────────────
    templates = [
        # Type 1: subject + environment
        lambda: f"{rng.choice(subjects)} {rng.choice(environments)}",
        # Type 2: subject + material
        lambda: f"{rng.choice(subjects)} {rng.choice(materials)}",
        # Type 3: subject + lighting
        lambda: f"{rng.choice(subjects)} {rng.choice(lighting)}",
        # Type 4: subject + condition
        lambda: f"{rng.choice(subjects)} {rng.choice(conditions)}",
        # Type 5: subject + viewpoint
        lambda: f"{rng.choice(subjects)} {rng.choice(viewpoints)}",
        # Type 6: subject + render style
        lambda: f"{rng.choice(subjects)} {rng.choice(render_styles)}",
        # Type 7: adjective + subject + environment
        lambda: f"{rng.choice(adjectives)} {rng.choice(subjects).lstrip('a ')} {rng.choice(environments)}",
        # Type 8: subject + condition + lighting
        lambda: f"{rng.choice(subjects)} {rng.choice(conditions)}, {rng.choice(lighting)}",
        # Type 9: full scene descriptions
        lambda: rng.choice(scenes),
        # Type 10: abstract concepts
        lambda: rng.choice(abstract_concepts),
        # Type 11: material + subject + environment
        lambda: f"{rng.choice(subjects)} {rng.choice(materials)} {rng.choice(environments)}",
        # Type 12: just an adjective-heavy description
        lambda: f"something {rng.choice(adjectives)} and {rng.choice(adjectives)}",
        # Type 13: viewpoint + subject + lighting
        lambda: f"{rng.choice(subjects)} {rng.choice(viewpoints)}, {rng.choice(lighting)}",
    ]

    extra = set()
    while len(extra) < n:
        template_fn = rng.choice(templates)
        prompt = template_fn()
        extra.add(prompt)

    return sorted(extra)[:n]


def probe_features(
    sae_model: SparseAutoencoder,
    model,
    tokenizer,
    layer_idx: int,
    hidden_dim: int,
    prompts: list[str],
    feature_indices: Optional[list[int]] = None,
    top_prompts: int = 5,
    verbose: bool = True,
) -> dict[int, dict]:
    """Probe SAE features to find what they respond to.

    For each feature, finds the prompts that maximally activate it.

    Args:
        sae_model: trained SAE
        model: language model
        tokenizer: model tokenizer
        layer_idx: layer to hook
        hidden_dim: model hidden dim
        prompts: list of diverse probing prompts
        feature_indices: specific features to probe (None = all active)
        top_prompts: number of top-activating prompts per feature
        verbose: print progress

    Returns:
        dict mapping feature_idx -> {
            'top_prompts': [(prompt, activation), ...],
            'mean_activation': float,
            'max_activation': float,
            'firing_rate': float,  # fraction of prompts that activate > 0
        }
    """
    if verbose:
        print(f"Probing SAE features with {len(prompts)} prompts...")
        print(f"  Layer: {layer_idx}, Hidden dim: {hidden_dim}")

    # Collect activations
    collected = []
    hook_handle = None

    def _hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        collected.append(h.detach().cpu().float())

    hook_handle = model.layers[layer_idx].register_forward_hook(_hook)

    prompt_activations = []  # mean-pooled per prompt
    for i, text in enumerate(prompts):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=64
        ).to(DEVICE)
        with torch.no_grad():
            model(inputs.input_ids, attention_mask=inputs.attention_mask)

        h = collected[-1].squeeze(0)  # [tokens, hidden_dim]
        prompt_activations.append(h.mean(0))  # mean-pool

        if verbose and (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(prompts)} prompts processed")

    hook_handle.remove()

    # Stack: [n_prompts, hidden_dim]
    all_acts = torch.stack(prompt_activations)

    # Encode through SAE
    _dev = next(sae_model.parameters()).device
    with torch.no_grad():
        all_features = sae_model.encode(all_acts.to(_dev)).cpu()
        # [n_prompts, d_sae]

    d_sae = all_features.shape[1]

    # Determine which features to probe
    if feature_indices is not None:
        probe_indices = feature_indices
    else:
        # Find all features that fire for at least one prompt
        ever_active = (all_features > 0).any(dim=0).nonzero(as_tuple=True)[0]
        probe_indices = ever_active.tolist()
        if verbose:
            print(f"  {len(probe_indices)} / {d_sae} features are active")

    results = {}
    for feat_idx in probe_indices:
        feat_activations = all_features[:, feat_idx]  # [n_prompts]

        # Top activating prompts
        top_k = min(top_prompts, len(prompts))
        top_vals, top_ids = feat_activations.topk(top_k)

        top_prompt_list = [
            (prompts[idx.item()], val.item())
            for idx, val in zip(top_ids, top_vals)
            if val.item() > 0
        ]

        firing_rate = (feat_activations > 0).float().mean().item()
        mean_act = feat_activations[feat_activations >
                                    0].mean().item() if firing_rate > 0 else 0
        max_act = feat_activations.max().item()

        results[int(feat_idx)] = {
            "top_prompts": top_prompt_list,
            "mean_activation": round(mean_act, 4),
            "max_activation": round(max_act, 4),
            "firing_rate": round(firing_rate, 4),
        }

    if verbose:
        print(f"  Probed {len(results)} features")

    return results


def probe_lens_features(
    lens_path: str,
    sae_path: Optional[str] = None,
    n_extra_prompts: int = 200,
    top_prompts: int = 5,
    verbose: bool = True,
) -> dict:
    """Probe the SAE features used by a specific lens.

    Loads the lens, finds which feature indices it uses,
    then probes those features to discover what they respond to.

    Args:
        lens_path: path to a .pt lens file (must be SAE-trained)
        sae_path: path to SAE weights (if not embedded in lens)
        n_extra_prompts: additional generated prompts beyond built-in set
        top_prompts: top-activating prompts per feature

    Returns:
        dict with 'features' mapping and 'lens_metadata'
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Feature Probe: {Path(lens_path).name}")
        print(f"{'='*60}\n")

    # Load lens
    data = torch.load(lens_path, map_location="cpu", weights_only=False)

    if not isinstance(data, dict) or "sae_feature_indices" not in data:
        raise ValueError(
            f"Lens at {lens_path} is not an SAE lens — no feature indices found. "
            f"Only SAE-trained lenses have inspectable features."
        )

    feature_indices = data["sae_feature_indices"]
    if isinstance(feature_indices, torch.Tensor):
        feature_indices = feature_indices.tolist()
    feature_indices = [int(f) for f in feature_indices]

    feature_weights = data.get("sae_feature_weights", None)
    if isinstance(feature_weights, torch.Tensor):
        feature_weights = feature_weights.tolist()

    concept = data.get("concept", "unknown")
    sae_layer = data.get("sae_layer", 22)
    sae_expansion = data.get("sae_expansion", 8)
    d_sae = QWEN_HIDDEN_DIM * sae_expansion

    if verbose:
        print(f"  Concept: {concept}")
        print(f"  SAE layer: {sae_layer}")
        print(f"  Features to probe: {len(feature_indices)}")
        print(
            f"  Feature indices: {feature_indices[:10]}{'...' if len(feature_indices) > 10 else ''}")

    # Load SAE
    if sae_path and Path(sae_path).exists():
        if verbose:
            print(f"\n  Loading SAE from {sae_path}...")
        sae_model = SparseAutoencoder(QWEN_HIDDEN_DIM, d_sae).to(DEVICE)
        sae_state = torch.load(
            sae_path, map_location=DEVICE, weights_only=True)
        sae_model.load_state_dict(sae_state)
        sae_model.eval()
    else:
        raise FileNotFoundError(
            "SAE weights required for feature probing. "
            "Pass --sae-path or train with --sae-save to keep the SAE."
        )

    # Load encoder
    if verbose:
        print("  Loading Qwen encoder...")
    model, tokenizer = load_qwen_encoder()

    # Build prompt set
    prompts = list(PROBE_PROMPTS)
    if n_extra_prompts > 0:
        prompts.extend(generate_extra_prompts(n_extra_prompts))
    if verbose:
        print(f"  Total probing prompts: {len(prompts)}")

    # Probe
    feature_results = probe_features(
        sae_model, model, tokenizer,
        layer_idx=sae_layer,
        hidden_dim=QWEN_HIDDEN_DIM,
        prompts=prompts,
        feature_indices=feature_indices,
        top_prompts=top_prompts,
        verbose=verbose,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"  Feature Analysis: {concept}")
    print(f"{'='*60}\n")

    for i, feat_idx in enumerate(feature_indices):
        if feat_idx not in feature_results:
            continue
        info = feature_results[feat_idx]
        weight = feature_weights[i] if feature_weights else None

        weight_str = f"  weight={weight:+.3f}" if weight is not None else ""
        sign = "+" if (weight and weight >
                       0) else "-" if (weight and weight < 0) else "?"

        print(f"  [{sign}] Feature {feat_idx}{weight_str}  "
              f"(fires {info['firing_rate']:.0%} of prompts, max={info['max_activation']:.2f})")

        for prompt, act in info["top_prompts"][:top_prompts]:
            print(f"      {act:6.2f}  {prompt}")
        print()

    return {
        "concept": concept,
        "lens_path": lens_path,
        "sae_layer": sae_layer,
        "feature_count": len(feature_indices),
        "features": feature_results,
        "feature_indices": feature_indices,
        "feature_weights": feature_weights,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Probe SAE features to discover what they respond to",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Probe features from a trained lens
  python tools/probe_features.py --lens-path ./lenses/zimage/cinematic_zimage_sae_contrastive.pt --sae-path ./sae.pt

  # Probe all active features in an SAE
  python tools/probe_features.py --sae-path ./sae.pt --all-features

  # Save results to JSON
  python tools/probe_features.py --lens-path ./lens.pt --sae-path ./sae.pt --output features.json
        """,
    )

    parser.add_argument("--lens-path", type=str, default=None,
                        help="Path to an SAE-trained lens file")
    parser.add_argument("--sae-path", type=str, required=True,
                        help="Path to saved SAE weights (.pt)")
    parser.add_argument("--all-features", action="store_true",
                        help="Probe all active features (not just lens features)")
    parser.add_argument("--layer", type=int, default=22,
                        help="Transformer layer (default: 22)")
    parser.add_argument("--sae-expansion", type=int, default=8,
                        help="SAE expansion factor (default: 8)")
    parser.add_argument("--n-prompts", type=int, default=200,
                        help="Extra generated prompts beyond built-in set (default: 200)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top prompts per feature (default: 5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--encoder-path", type=str, default="",
                        help="Path to Qwen encoder (overrides env)")

    args = parser.parse_args()

    if args.encoder_path:
        os.environ["QWEN_ENCODER_PATH"] = args.encoder_path

    if args.lens_path:
        results = probe_lens_features(
            lens_path=args.lens_path,
            sae_path=args.sae_path,
            n_extra_prompts=args.n_prompts,
            top_prompts=args.top_k,
        )
    elif args.all_features:
        # Load SAE and probe all active features
        d_sae = QWEN_HIDDEN_DIM * args.sae_expansion
        print(f"Loading SAE from {args.sae_path}...")
        sae_model = SparseAutoencoder(QWEN_HIDDEN_DIM, d_sae).to(DEVICE)
        sae_state = torch.load(
            args.sae_path, map_location=DEVICE, weights_only=True)
        sae_model.load_state_dict(sae_state)
        sae_model.eval()

        print("Loading Qwen encoder...")
        model, tokenizer = load_qwen_encoder()

        prompts = list(PROBE_PROMPTS)
        if args.n_prompts > 0:
            prompts.extend(generate_extra_prompts(args.n_prompts))
        print(f"Total probing prompts: {len(prompts)}")

        feature_results = probe_features(
            sae_model, model, tokenizer,
            layer_idx=args.layer,
            hidden_dim=QWEN_HIDDEN_DIM,
            prompts=prompts,
            feature_indices=None,  # all active
            top_prompts=args.top_k,
        )

        # Print top features by max activation
        sorted_feats = sorted(
            feature_results.items(),
            key=lambda x: x[1]["max_activation"],
            reverse=True,
        )

        print(f"\n{'='*60}")
        print(f"  Top Features by Max Activation")
        print(f"{'='*60}\n")

        for feat_idx, info in sorted_feats[:50]:
            print(f"  Feature {feat_idx}  "
                  f"(fires {info['firing_rate']:.0%}, max={info['max_activation']:.2f})")
            for prompt, act in info["top_prompts"][:3]:
                print(f"      {act:6.2f}  {prompt}")
            print()

        results = {
            "mode": "all_features",
            "sae_path": args.sae_path,
            "layer": args.layer,
            "features": feature_results,
        }
    else:
        parser.error("Either --lens-path or --all-features is required")

    # Save to JSON
    if args.output:
        # Convert for JSON serialization
        out = {}
        for k, v in results.items():
            if k == "features":
                out[k] = {
                    str(fk): {
                        "top_prompts": fv["top_prompts"],
                        "mean_activation": fv["mean_activation"],
                        "max_activation": fv["max_activation"],
                        "firing_rate": fv["firing_rate"],
                    }
                    for fk, fv in v.items()
                }
            else:
                out[k] = v

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
