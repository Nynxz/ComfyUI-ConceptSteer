"""
LLM utility functions shared across all Concept Steer LLM nodes.

Discovery strategy:
  1. Known attribute paths — handles Gemma, LLaMA, Mistral, Qwen2, GPT-2, BLOOM
  2. Module introspection fallback — walks the model tree looking for the largest
     ModuleList whose elements contain attention-like children

Call diagnose_clip(clip) when discovery fails to get a printable tree of what
the clip object actually contains.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── Transformer layer discovery ──────────────────────────────────────────────

def get_transformer_layers(clip) -> list | None:
    """
    Return the list of transformer decoder layers from a ComfyUI clip.

    Tries multiple root paths and multiple layer attribute paths, then falls
    back to module introspection for unknown architectures.
    """
    # Root candidates (ordered by likelihood for ComfyUI textgen clips)
    root_fns = [
        lambda: clip.cond_stage_model,
        lambda: clip.patcher.model,
        lambda: clip.patcher.model.cond_stage_model,
        lambda: clip.model,
    ]

    # Layer attribute paths (relative to root)
    # Covers: Gemma/LLaMA/Mistral/Qwen2 (ForCausalLM → .model → .layers)
    #         GPT-2 (transformer.h), BLOOM (transformer.h), GPT-J (transformer.h)
    #         bare decoder-only (just .layers), T5/BART decoder
    layer_attrs = [
        "model.layers",                   # GemmaForCausalLM, LlamaForCausalLM, Qwen2ForCausalLM
        "model.model.layers",             # Double-wrapped (extra outer class)
        "transformer.h",                  # GPT-2, GPT-J, BLOOM
        "transformer.blocks",             # some OPT variants
        "decoder.layers",                 # T5 decoder
        "language_model.model.layers",    # LLaVA / multimodal wrappers
        "layers",                         # bare ModuleList at root
        "encoder.layers",                 # encoder-only (BERT-style)
    ]

    for root_fn in root_fns:
        try:
            root = root_fn()
        except (AttributeError, TypeError):
            continue

        for attr_path in layer_attrs:
            try:
                obj = root
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                # Accept nn.ModuleList or any sequence with >2 elements
                if len(obj) > 2:
                    return list(obj)
            except (AttributeError, TypeError, ValueError):
                continue

    # ── Introspection fallback ────────────────────────────────────────────────
    # Walk the model tree up to depth 5, find the largest ModuleList that
    # contains modules with attention-like sub-modules (looks like decoder layers).
    for root_fn in root_fns:
        try:
            root = root_fn()
        except (AttributeError, TypeError):
            continue

        best: list | None = None
        best_n = 0

        def _scan(module: nn.Module, depth: int = 0) -> None:
            nonlocal best, best_n
            if depth > 5:
                return
            for _name, child in module.named_children():
                if isinstance(child, nn.ModuleList) and len(child) > best_n and len(child) >= 4:
                    # Check if elements look like transformer decoder blocks
                    try:
                        child_names = [n for n, _ in child[0].named_children()]
                        looks_like_decoder = any(
                            any(k in n.lower() for k in ("attn", "attention", "mlp", "ffn", "feed"))
                            for n in child_names
                        )
                        if looks_like_decoder:
                            best = list(child)
                            best_n = len(child)
                    except Exception:
                        pass
                _scan(child, depth + 1)

        _scan(root)
        if best:
            return best

    return None


# ── Hidden size discovery ─────────────────────────────────────────────────────

def get_hidden_size(clip) -> int | None:
    """Try to discover the LLM's hidden dimension from config or embedding weights."""
    root_fns = [
        lambda: clip.cond_stage_model,
        lambda: clip.patcher.model,
        lambda: clip.model,
    ]
    embed_attrs = [
        "model.embed_tokens.weight",
        "model.model.embed_tokens.weight",
        "embed_tokens.weight",
        "wte.weight",
        "transformer.wte.weight",
    ]
    for fn in root_fns:
        try:
            m = fn()
        except (AttributeError, TypeError):
            continue
        # Try config first (fastest)
        if hasattr(m, "config") and hasattr(m.config, "hidden_size"):
            return m.config.hidden_size
        # Try embedding weight shape
        for attr in embed_attrs:
            try:
                obj = m
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return obj.shape[1]
            except AttributeError:
                pass
    return None


# ── Layer mask parsing ────────────────────────────────────────────────────────

def parse_layer_mask(spec: str, num_layers: int) -> dict[int, float]:
    """
    Parse a layer mask spec into {layer_idx: strength} dict.

    Formats:
      ""           → all layers at 1.0 (default)
      "all"        → all layers at 1.0
      "early"      → first third
      "middle"     → middle third
      "late"       → last third
      "10-20"      → layers 10-20 at 1.0
      "10-20:0.5"  → layers 10-20 at 0.5
      "10,12:1.5"  → layer 10 at 1.0, layer 12 at 1.5
    """
    spec = spec.strip().lower()
    if spec in ("", "all"):
        return {i: 1.0 for i in range(num_layers)}
    if spec == "early":
        t = max(1, num_layers // 3)
        return {i: 1.0 for i in range(t)}
    if spec == "middle":
        t = max(1, num_layers // 3)
        return {i: 1.0 for i in range(t, 2 * t)}
    if spec == "late":
        t = max(1, num_layers // 3)
        return {i: 1.0 for i in range(2 * t, num_layers)}

    mask: dict[int, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        rng, strength = (part.rsplit(":", 1) + [None])[:2]
        try:
            s = float(strength) if strength else 1.0
        except ValueError:
            s = 1.0
        rng = rng.strip()
        if "-" in rng:
            try:
                a, b = rng.split("-", 1)
                for i in range(int(a), int(b) + 1):
                    mask[i] = s
            except ValueError:
                pass
        else:
            try:
                mask[int(rng)] = s
            except ValueError:
                pass
    return mask


# ── Diagnostics ───────────────────────────────────────────────────────────────

def diagnose_clip(clip) -> str:
    """
    Return a diagnostic string showing the clip's top-level attribute structure.

    Call this when get_transformer_layers() returns None to understand what
    the clip object actually contains, then add the correct path.
    """
    lines = ["[LLM Utils] Clip structure:"]

    def _fmt(val: object) -> str:
        if isinstance(val, nn.ModuleList):
            return f"nn.ModuleList[{len(val)}]"
        if hasattr(val, "__len__") and not isinstance(val, str):
            try:
                return f"{type(val).__name__}[{len(val)}]"
            except Exception:
                pass
        return type(val).__name__

    for attr in ("cond_stage_model", "patcher", "model"):
        try:
            val = getattr(clip, attr)
            lines.append(f"  clip.{attr}: {_fmt(val)}")
        except AttributeError:
            lines.append(f"  clip.{attr}: (not found)")
            continue

        if attr == "patcher":
            try:
                pm = clip.patcher.model
                lines.append(f"    clip.patcher.model: {_fmt(pm)}")
                for sub in ("model", "cond_stage_model", "transformer", "layers"):
                    try:
                        sv = getattr(pm, sub)
                        lines.append(f"      .{sub}: {_fmt(sv)}")
                        if sub == "model":
                            for subsub in ("layers", "decoder", "blocks", "h"):
                                try:
                                    ssv = getattr(sv, subsub)
                                    lines.append(f"        .{subsub}: {_fmt(ssv)}")
                                except AttributeError:
                                    pass
                    except AttributeError:
                        pass
            except AttributeError:
                pass

        elif attr == "cond_stage_model":
            for sub in ("model", "transformer", "layers", "config"):
                try:
                    sv = getattr(val, sub)
                    lines.append(f"    .{sub}: {_fmt(sv)}")
                    if sub == "model":
                        for subsub in ("layers", "decoder", "blocks", "h", "embed_tokens"):
                            try:
                                ssv = getattr(sv, subsub)
                                lines.append(f"      .{subsub}: {_fmt(ssv)}")
                            except AttributeError:
                                pass
                except AttributeError:
                    pass

    return "\n".join(lines)
