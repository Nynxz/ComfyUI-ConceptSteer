"""
Concept Steer — Concept Steering for ComfyUI

Steer image generation toward learned concepts using DPO/SAE-trained direction
vectors. No LoRA, no fine-tuning — just a single vector that nudges conditioning
toward your desired aesthetic.

Nodes:
  Steering:
    - Concept Steer: Apply a concept lens to CLIP/text-encoder conditioning

  Training:
    - Train Lens (DPO): Train a concept lens from text pairs via DPO
    - Train Lens (SAE): Train a concept lens via SAE feature decomposition
    - Train Lens (Few-Shot): Train a concept lens from example images

  Interpretability:
    - Lens Inspect: Visualize a lens's internal structure and features
    - Activation Probe: Compare conditioning before/after steering
    - Lens Compare: Side-by-side comparison of two concept lenses

  Feature Surgery:
    - Train SAE: Train a standalone SAE (required for Feature Map/Gate)
    - Feature Dictionary: Build a lookup of what each SAE feature responds to
    - Feature Map: Visualize SAE feature activations in conditioning
    - Feature Gate: Suppress or amplify individual SAE features
"""

from comfy_api.latest import ComfyExtension, io
from .nodes.ConceptSteer import ConceptSteerNode
from .nodes.ConceptTrainDPO import ConceptTrainDPONode
from .nodes.ConceptTrainSAE import ConceptTrainSAENode
from .nodes.ConceptTrainFewShot import ConceptTrainFewShotNode
from .nodes.ConceptLensInspect import ConceptLensInspectNode
from .nodes.ConceptActivationProbe import ConceptActivationProbeNode
from .nodes.ConceptLensCompare import ConceptLensCompareNode
from .nodes.ConceptFeatureMap import ConceptFeatureMapNode
from .nodes.ConceptFeatureGate import ConceptFeatureGateNode
from .nodes.ConceptTrainSAEOnly import ConceptTrainSAEOnlyNode
from .nodes.ConceptFeatureDict import ConceptFeatureDictNode


class ConceptSteerExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            # Steering
            ConceptSteerNode,
            # Training
            ConceptTrainDPONode,
            ConceptTrainSAENode,
            ConceptTrainFewShotNode,
            # Interpretability
            ConceptLensInspectNode,
            ConceptActivationProbeNode,
            ConceptLensCompareNode,
            # Feature Surgery
            ConceptTrainSAEOnlyNode,
            ConceptFeatureDictNode,
            ConceptFeatureMapNode,
            ConceptFeatureGateNode,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    print("[Concept Steer] Initializing Concept Steer extension...")
    return ConceptSteerExtension()


WEB_DIRECTORY = "./js"
__all__ = ["ConceptSteerExtension", "comfy_entrypoint", "WEB_DIRECTORY"]
