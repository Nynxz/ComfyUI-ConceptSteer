"""
ConceptFeatureDict — Build a feature dictionary that labels SAE features.

Runs a diverse bank of ~220 prompts through the text encoder + SAE and
records which prompts maximally activate each feature. The result is saved
as a JSON dictionary mapping feature indices to their top-activating prompts,
giving you human-readable labels for every active feature.

Run this once per SAE. Then the Feature Map node can display labels
alongside feature indices instead of raw numbers.

Usage in ComfyUI:
  [Feature Dictionary] → dict_path (JSON file)
  
  Then in Feature Map, set dict_path to auto-label features.

The dictionary file is small (~1MB) and reusable across all workflows.
"""

import os
import sys
import json
import time
from pathlib import Path
from comfy_api.latest import io

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_DIR = _PACKAGE_ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


def _log(msg: str):
    print(f"[Concept Steer] {msg}")


class ConceptFeatureDictNode(io.ComfyNode):
    """Build a feature dictionary that labels SAE features with meanings."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        default_save = str(_PACKAGE_ROOT / "sae" / "feature_dict.json")

        return io.Schema(
            node_id="conceptsteer.FeatureDict",
            display_name="Feature Dictionary",
            description=(
                "Run ~220 diverse prompts through the text encoder + SAE to "
                "discover what each feature responds to. Saves a JSON dictionary "
                "mapping feature indices to top-activating prompts. "
                "Run once per SAE, takes ~1–2 minutes."
            ),
            category="Concept Steer/Features",
            inputs=[
                io.String.Input(
                    "sae_path",
                    default="",
                    tooltip="Path to the trained SAE weights (.pt)",
                ),
                io.String.Input(
                    "save_path",
                    default=default_save,
                    tooltip=(
                        "Where to save the feature dictionary JSON. "
                        "Pass this path to Feature Map's dict_path input."
                    ),
                ),
                io.Int.Input(
                    "layer",
                    default=22,
                    min=1,
                    max=36,
                    tooltip="Transformer layer (must match SAE training)",
                ),
                io.Int.Input(
                    "sae_expansion",
                    default=8,
                    min=2,
                    max=16,
                    tooltip="SAE expansion factor (must match SAE training)",
                ),
                io.Int.Input(
                    "top_prompts",
                    default=5,
                    min=1,
                    max=20,
                    tooltip="How many top-activating prompts to store per feature",
                ),
                io.Int.Input(
                    "n_extra_prompts",
                    default=300,
                    min=0,
                    max=1000,
                    step=50,
                    tooltip=(
                        "Extra generated prompts beyond the built-in ~120. "
                        "More = better coverage and more discriminative labels. "
                        "Uses diverse templates to avoid formulaic patterns."
                    ),
                ),
                io.String.Input(
                    "encoder_path",
                    default="",
                    tooltip="Path to Qwen encoder safetensors (or set env var)",
                ),
            ],
            outputs=[
                io.String.Output("dict_path"),
                io.String.Output("summary"),
            ],
        )

    @classmethod
    def execute(
        cls,
        sae_path: str = "",
        save_path: str = "",
        layer: int = 22,
        sae_expansion: int = 8,
        top_prompts: int = 5,
        n_extra_prompts: int = 100,
        encoder_path: str = "",
    ):
        import torch

        sae_path = sae_path.strip()
        save_path = save_path.strip()

        if not sae_path:
            _log("ERROR: SAE path required")
            return io.NodeOutput("", "Error: SAE path required")

        if not os.path.isabs(sae_path):
            sae_path = str(_PACKAGE_ROOT / sae_path)

        if not os.path.isfile(sae_path):
            _log(f"ERROR: SAE not found at {sae_path}")
            return io.NodeOutput("", f"Error: SAE not found at {sae_path}")

        if not save_path:
            save_path = str(_PACKAGE_ROOT / "sae" / "feature_dict.json")

        if encoder_path.strip():
            os.environ["QWEN_ENCODER_PATH"] = encoder_path.strip()

        # ── Imports ──
        try:
            from lens_factory import (
                SparseAutoencoder,
                load_qwen_encoder,
                DEVICE,
                QWEN_HIDDEN_DIM,
            )
            from probe_features import PROBE_PROMPTS, generate_extra_prompts
        except ImportError as e:
            _log(f"ERROR: Import failed: {e}")
            return io.NodeOutput("", f"Error: {e}")

        t0 = time.time()
        hidden_dim = QWEN_HIDDEN_DIM
        d_sae = hidden_dim * sae_expansion

        # ── Load SAE ──
        _log(f"Loading SAE from {os.path.basename(sae_path)}...")
        sae = SparseAutoencoder(hidden_dim, d_sae).to(DEVICE)
        sae_state = torch.load(
            sae_path, map_location=DEVICE, weights_only=True)
        sae.load_state_dict(sae_state)
        sae.eval()

        # ── Load encoder ──
        _log("Loading Qwen encoder...")
        model, tokenizer = load_qwen_encoder(encoder_path)

        # ── Build prompt bank ──
        prompts = list(PROBE_PROMPTS)
        if n_extra_prompts > 0:
            prompts.extend(generate_extra_prompts(n_extra_prompts))
        _log(f"Probing with {len(prompts)} prompts...")

        # ── Collect per-token layer activations for all prompts ──
        # We collect per-token vectors (matching SAE training) then aggregate
        # to per-prompt max activations for labeling.
        prompt_token_acts: list[torch.Tensor] = []  # list of [n_tokens, hidden_dim]

        def _hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            prompt_token_acts.append(h.detach().cpu().float())

        handle = model.layers[layer].register_forward_hook(_hook)

        for i, text in enumerate(prompts):
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=64
            ).to(DEVICE)
            with torch.no_grad():
                model(inputs.input_ids, attention_mask=inputs.attention_mask)
            # Keep only non-padding tokens
            if prompt_token_acts:
                act = prompt_token_acts[-1]  # [1, seq_len, hidden]
                mask = inputs.attention_mask.cpu()  # [1, seq_len]
                prompt_token_acts[-1] = act[0, mask[0].bool()]  # [n_valid, hidden]
            if (i + 1) % 50 == 0:
                _log(f"  {i+1}/{len(prompts)} prompts encoded")

        handle.remove()

        # ── Encode per-token through SAE, aggregate to per-prompt ──
        _log("Encoding through SAE...")
        sae_dev = next(sae.parameters()).device

        # For each prompt: encode its tokens → take max activation per feature
        # This answers "did ANY token in this prompt strongly activate feature F?"
        prompt_max_acts = []  # [n_prompts, d_sae]
        with torch.no_grad():
            for token_acts in prompt_token_acts:
                z = sae.encode(token_acts.to(sae_dev))  # [n_tokens, d_sae]
                max_per_feat = z.max(dim=0).values  # [d_sae] — max across tokens
                prompt_max_acts.append(max_per_feat.cpu())

        all_features = torch.stack(prompt_max_acts)  # [n_prompts, d_sae]
        n_total_tokens = sum(t.shape[0] for t in prompt_token_acts)
        del prompt_token_acts
        _log(f"  {n_total_tokens:,} tokens from {len(prompts)} prompts")

        # ── Build dictionary: for each feature, find top-activating prompts ──
        _log("Building feature dictionary...")
        ever_active = (all_features > 0).any(dim=0).nonzero(as_tuple=True)[0]
        _log(f"  {len(ever_active)}/{d_sae} features fired at least once")

        # ── Z-score normalization ──
        # Raw activations don't work when features aren't sparse: every
        # feature fires on every prompt, so top-K by raw value returns the
        # same prompts for all features. Instead, z-score each feature
        # across prompts so we find which prompts activate it *unusually
        # strongly* relative to that feature's own baseline.
        active_features = all_features[:, ever_active]  # [n_prompts, n_active]
        feat_mean = active_features.mean(dim=0, keepdim=True)
        feat_std = active_features.std(dim=0, keepdim=True).clamp(min=1e-8)
        z_scored = (active_features - feat_mean) / feat_std
        # [n_prompts, n_active] — positive = unusually strong for this feature

        # First pass: collect top prompts per feature using z-scores
        raw_entries: dict[str, dict] = {}
        for local_idx, feat_idx in enumerate(ever_active.tolist()):
            feat_acts = all_features[:, feat_idx]
            feat_z = z_scored[:, local_idx]

            # Use z-score to rank prompts — finds what's distinctive
            top_k = min(top_prompts, len(prompts))
            top_z_vals, top_z_ids = feat_z.topk(top_k)

            top_list = []
            for idx_t, z_t in zip(top_z_ids, top_z_vals):
                raw_act = feat_acts[idx_t].item()
                if raw_act > 0:
                    top_list.append({
                        "prompt": prompts[idx_t.item()],
                        "activation": round(raw_act, 4),
                        "z_score": round(z_t.item(), 4),
                    })

            if not top_list:
                continue

            firing_rate = (feat_acts > 0).float().mean().item()
            mean_act = feat_acts[feat_acts > 0].mean(
            ).item() if firing_rate > 0 else 0

            # Selectivity: high when feature fires strongly on few prompts
            # but weakly/not-at-all on most. Penalize features that fire on
            # <1% or >80% of prompts — too rare = noise, too common = useless.
            if 0.005 < firing_rate < 0.8 and feat_mean[0, local_idx].item() > 0:
                selectivity = (feat_std[0, local_idx] / feat_mean[0, local_idx]).item()
                # Bonus for features in the sweet spot (2-30% firing)
                if 0.02 <= firing_rate <= 0.3:
                    selectivity *= 1.5
            else:
                selectivity = 0.0  # don't label noise or always-on features

            raw_entries[str(feat_idx)] = {
                "top_list": top_list,
                "firing_rate": round(firing_rate, 4),
                "mean_activation": round(mean_act, 4),
                "max_activation": round(top_list[0]["activation"], 4),
                "selectivity": round(selectivity, 4),
            }

        # Second pass: compute IDF — how many features each word appears in
        word_to_feature_count: dict[str, int] = {}
        for entry in raw_entries.values():
            words_in_feature = set()
            for p in entry["top_list"]:
                for w in _tokenize_prompt(p["prompt"]):
                    words_in_feature.add(w)
            for w in words_in_feature:
                word_to_feature_count[w] = word_to_feature_count.get(w, 0) + 1

        n_features = max(len(raw_entries), 1)
        _log(f"  IDF computed over {len(word_to_feature_count)} unique words, "
             f"{n_features} active features")

        # Third pass: label with IDF-weighted scoring
        import math
        word_idf = {
            w: math.log(n_features / count)
            for w, count in word_to_feature_count.items()
        }

        dictionary = {}
        for feat_id, entry in raw_entries.items():
            label = _auto_label(entry["top_list"], word_idf=word_idf)

            # Mark low-selectivity features (fires uniformly on everything)
            if entry["selectivity"] < 0.05:
                label = f"[general] {label}" if label != "unknown" else "[general]"

            dictionary[feat_id] = {
                "label": label,
                "firing_rate": entry["firing_rate"],
                "mean_activation": entry["mean_activation"],
                "max_activation": entry["max_activation"],
                "selectivity": entry["selectivity"],
                "top_prompts": entry["top_list"],
            }

        # ── Save ──
        save_obj = Path(save_path)
        save_obj.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "sae_path": sae_path,
            "layer": layer,
            "expansion": sae_expansion,
            "d_sae": d_sae,
            "total_features": d_sae,
            "active_features": len(dictionary),
            "n_prompts": len(prompts),
            "features": dictionary,
        }

        with open(save_obj, "w") as f:
            json.dump(output, f, indent=2)

        elapsed = time.time() - t0
        file_size = save_obj.stat().st_size / 1024

        summary_lines = [
            f"Feature Dictionary Built ({elapsed:.0f}s)",
            f"SAE: {d_sae} features, {len(dictionary)} active",
            f"Probed with {len(prompts)} prompts",
            f"Saved to: {save_path} ({file_size:.0f} KB)",
            "",
        ]

        # Count selective vs general features
        n_general = sum(1 for v in dictionary.values()
                        if v.get("selectivity", 0) < 0.05)
        n_selective = len(dictionary) - n_general
        summary_lines.append(
            f"Selective features: {n_selective}, General (non-discriminative): {n_general}")
        summary_lines.extend(["", "Most selective features:"])

        # Show features sorted by selectivity (most discriminative first)
        sorted_feats = sorted(
            dictionary.items(),
            key=lambda x: x[1].get("selectivity", 0),
            reverse=True,
        )
        for feat_id, info in sorted_feats[:10]:
            summary_lines.append(
                f"  F{feat_id}: \"{info['label']}\" "
                f"(sel={info.get('selectivity', 0):.3f}, "
                f"fires {info['firing_rate']:.0%})"
            )

        summary = "\n".join(summary_lines)
        _log(summary)

        return io.NodeOutput(str(save_obj), summary)


_STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "of", "with", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "to", "for",
    "from", "by", "as", "its", "it", "this", "that", "these", "those",
    "very", "too", "also", "just", "more", "most", "some", "all",
    "every", "each", "into", "through", "over", "under", "above",
    "scene", "looking", "type", "style", "something", "made",
    "rendered", "shot", "image", "photo", "picture", "view",
    "visible", "shown", "seen", "like", "has", "have", "had",
    "not", "but", "if", "then", "about", "after", "before",
    "between", "there", "here", "where", "when", "how", "what",
    "which", "who", "whom", "whose", "than", "both", "while",
}


def _tokenize_prompt(text: str) -> list[str]:
    """Extract meaningful words from a prompt."""
    text = text.lower()
    words = text.replace(",", " ").replace(".", " ").replace("-", " ").split()
    result = []
    for w in words:
        w = w.strip("'\"()[]{}")
        if len(w) >= 3 and w not in _STOP_WORDS:
            result.append(w)
    return result


def _auto_label(
    top_prompts: list[dict],
    max_words: int = 4,
    word_idf: dict[str, float] | None = None,
) -> str:
    """Generate a short auto-label from top-activating prompts.

    Uses TF-IDF scoring: words that appear in this feature's prompts
    but NOT in most other features' prompts get the highest scores.
    This prevents globally common words (e.g. 'underwater' when many
    prompts contain it) from dominating every label.

    Args:
        top_prompts: list of {"prompt": str, "activation": float}
        max_words: max words in the label
        word_idf: {word: idf_score} — higher = more discriminative.
                  If None, falls back to raw frequency (less accurate).
    """
    # TF: how many of this feature's top prompts contain each word
    word_tf: dict[str, int] = {}
    n_prompts = len(top_prompts)

    for entry in top_prompts:
        seen_in_prompt: set[str] = set()
        for w in _tokenize_prompt(entry["prompt"]):
            if w not in seen_in_prompt:
                word_tf[w] = word_tf.get(w, 0) + 1
                seen_in_prompt.add(w)

    if not word_tf:
        return "unknown"

    # Score each word
    scored: list[tuple[str, float]] = []
    for word, tf_count in word_tf.items():
        tf = tf_count / n_prompts  # normalize to [0, 1]

        if word_idf is not None:
            idf = word_idf.get(word, 0.0)
            # IDF of 0 means the word appears in every feature — useless
            if idf < 0.1:
                continue
            score = tf * idf
        else:
            # Fallback: just use frequency, prefer words in >1 prompt
            score = tf_count

        scored.append((word, score))

    if not scored:
        return "unknown"

    # Sort by score descending, then alphabetically for ties
    scored.sort(key=lambda x: (-x[1], x[0]))

    # Take top N words
    label_words = [w for w, _ in scored[:max_words]]

    return " / ".join(label_words)
