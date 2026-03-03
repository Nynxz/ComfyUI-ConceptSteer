"""
SAE Score Adapter — preference steering in SAE feature space.

Two modes selected automatically by dataset size:

  Empirical (< NEURAL_THRESHOLD samples, default):
    No training.  Stores reference feature vectors directly and computes
    score as cosine-similarity-weighted mean-shift toward them.
      score(z) = (Σ_i w_i · z_i  −  z) / σ²
      w_i  = softmax(cos_sim(z, z_i) / τ)
    Works correctly with 10–100 samples.  Adaptive: the target shifts
    toward reference examples that are already similar to the current z.

  Neural (≥ NEURAL_THRESHOLD samples):
    Denoising score matching on a residual MLP.
    Requires ~200+ samples for reliable convergence.

Why empirical for small data:
    A neural network needs enough samples to learn the score function of the
    data distribution.  For SAE features (d=20480), estimating a distribution
    from 30 points is statistically impossible — the network just fits noise,
    producing a diverging loss and no useful gradients at inference.
    The empirical estimator is the *exact* score of the Gaussian-mixture
    distribution defined by the training points: no estimation error.

Reference: "The Geometry of Noise" (Sahraee-Ardakan et al., 2026)
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Datasets below this size use empirical scoring.
NEURAL_THRESHOLD = 200


# ── Empirical score adapter ───────────────────────────────────────────────────

class EmpiricalScoreAdapter:
    """
    Exact score of the Gaussian mixture defined by reference features.

    score(z, σ) = (Σ_i w_i · z_i  −  z) / σ²
    w_i = softmax(cos_sim(z, z_i) / temperature)

    Cosine similarity weighting handles high-dimensional sparse SAE features
    better than L2 distance (avoids the curse-of-dimensionality bandwidth
    problem that makes L2-based Parzen windows degenerate in ≥1000 dims).

    Adaptive property: reference points most similar to the current z get
    more weight, so portrait prompts are steered toward portrait-like examples,
    landscape prompts toward landscape-like ones — even from the same adapter.
    """

    def __init__(self, z_data: torch.Tensor, temperature: float = 0.05):
        self.z_data      = z_data.float().cpu()          # [N, d_sae]
        self.temperature = temperature
        # Pre-normalise reference vectors once
        self._z_norm = F.normalize(self.z_data, dim=-1)  # [N, d_sae]

    # Same call signature as SAEScoreNet so run_score_ode works with both.
    def __call__(
        self,
        z:           torch.Tensor,   # [B, d_sae]
        log_sigma:   torch.Tensor,   # [B]
        cond_pooled: torch.Tensor | None = None,
    ) -> torch.Tensor:               # [B, d_sae]
        device   = z.device
        z_ref    = self.z_data.to(device)             # [N, d_sae]
        z_norm   = self._z_norm.to(device)            # [N, d_sae]
        sigma_sq = log_sigma.exp().pow(2)[:, None]    # [B, 1]

        # Cosine similarity between current z and each reference
        z_n      = F.normalize(z.float(), dim=-1)     # [B, d_sae]
        cos_sim  = torch.mm(z_n, z_norm.T)            # [B, N]

        # Soft-nearest-neighbour weights
        w = torch.softmax(cos_sim / self.temperature, dim=-1)   # [B, N]

        # Weighted mean reference: adaptive target
        z_target = torch.mm(w, z_ref)                 # [B, d_sae]

        # Score: direction toward target, scaled by σ²
        return (z_target - z) / sigma_sq

    def eval(self):
        return self  # no-op, mirrors nn.Module interface

    def to(self, device):
        return self  # tensors moved lazily in __call__

    @property
    def d_sae(self) -> int:
        return self.z_data.shape[1]


# ── Neural score network ──────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    def __init__(self, d: int, d_ctx: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.ff   = nn.Sequential(
            nn.Linear(d + d_ctx, d * 2),
            nn.SiLU(),
            nn.Linear(d * 2, d),
        )
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return x + self.ff(torch.cat([self.norm(x), ctx], dim=-1))


def _sinusoidal_embed(log_sigma: torch.Tensor, dim: int = 256) -> torch.Tensor:
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=log_sigma.device)
        / (half - 1)
    )
    args  = log_sigma.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class SAEScoreNet(nn.Module):
    """
    Neural score network for large datasets (≥ NEURAL_THRESHOLD samples).
    Predicts ∇_z log p(z_noisy | cond) via denoising score matching.
    """

    _SIGMA_DIM = 256

    def __init__(self, d_sae: int, d_cond: int, hidden: int = 512, depth: int = 4):
        super().__init__()
        self.d_sae  = d_sae
        self.d_cond = d_cond
        self.hidden = hidden
        self.depth  = depth

        bottleneck = min(d_sae, hidden * 2)
        self.sigma_proj = nn.Sequential(nn.Linear(self._SIGMA_DIM, hidden), nn.SiLU())
        self.cond_proj  = nn.Sequential(nn.Linear(d_cond, hidden), nn.SiLU())
        self.proj_in    = nn.Linear(d_sae, bottleneck)
        self.proj_out   = nn.Linear(bottleneck, d_sae)
        self.blocks     = nn.ModuleList([_ResBlock(bottleneck, hidden) for _ in range(depth)])
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, z_noisy, log_sigma, cond_pooled):
        s   = self.sigma_proj(_sinusoidal_embed(log_sigma, self._SIGMA_DIM))
        c   = self.cond_proj(cond_pooled.float())
        ctx = s + c
        x   = self.proj_in(z_noisy.float())
        for block in self.blocks:
            x = block(x, ctx)
        return self.proj_out(x)


# ── Build adapter (training entry point) ─────────────────────────────────────

def build_score_adapter(
    z_data:      torch.Tensor,
    c_data:      torch.Tensor,
    hidden:      int   = 512,
    depth:       int   = 4,
    sigma_min:   float = 0.03,
    sigma_max:   float = 5.0,
    epochs:      int   = 500,
    lr:          float = 3e-4,
    batch_size:  int   = 64,
    device:      str   = "cpu",
    temperature: float = 0.05,
    log_cb:      Callable[[str], None] | None = None,
) -> tuple[EmpiricalScoreAdapter | SAEScoreNet, dict]:
    """
    Build (or train) a score adapter from SAE feature vectors.

    Automatically selects empirical mode for small datasets and neural mode
    for large ones.  The empirical mode requires no training and is always
    correct; the neural mode converges reliably only with ≥200 samples.
    """
    def _log(msg: str):
        if log_cb:
            log_cb(msg)

    N = z_data.shape[0]

    if N < NEURAL_THRESHOLD:
        _log(f"Dataset has {N} samples (< {NEURAL_THRESHOLD}) → using empirical score adapter.")
        _log("No training required — storing reference features directly.")
        adapter  = EmpiricalScoreAdapter(z_data, temperature=temperature)
        metadata = {
            "adapter_type": "empirical",
            "d_sae":        z_data.shape[1],
            "d_cond":       c_data.shape[1],
            "n_samples":    N,
            "temperature":  temperature,
            "training_mode": "empirical_score",
        }
        return adapter, metadata

    # ── Neural path (large dataset) ───────────────────────────────────────────
    _log(f"Dataset has {N} samples (≥ {NEURAL_THRESHOLD}) → training neural score network.")

    N, d_sae  = z_data.shape
    _,  d_cond = c_data.shape

    log_sigma_min = math.log(sigma_min)
    log_sigma_max = math.log(sigma_max)
    B  = min(batch_size, N)
    t0 = time.time()
    losses: list[float] = []

    with torch.inference_mode(False), torch.enable_grad():
        score_net = SAEScoreNet(d_sae=d_sae, d_cond=d_cond, hidden=hidden, depth=depth)
        score_net.to(device)
        n_params = sum(p.numel() for p in score_net.parameters())
        _log(f"Score network: {n_params:,} parameters")

        optimizer = torch.optim.AdamW(score_net.parameters(), lr=lr, weight_decay=1e-4)
        score_net.train()

        for epoch in range(epochs):
            idx       = torch.randint(0, N, (B,))
            z_clean   = z_data[idx].to(device)
            c_batch   = c_data[idx].to(device)
            log_sigma = torch.empty(B, device=device).uniform_(log_sigma_min, log_sigma_max)
            sigma     = log_sigma.exp()
            eps       = torch.randn_like(z_clean)
            z_noisy   = z_clean + sigma[:, None] * eps
            target    = -eps / sigma[:, None]
            pred      = score_net(z_noisy, log_sigma, c_batch)
            loss      = (sigma[:, None] ** 2 * (pred - target) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

            report_every = max(1, epochs // 10)
            if (epoch + 1) % report_every == 0:
                avg = sum(losses[-report_every:]) / report_every
                _log(f"  Epoch {epoch + 1:4d}/{epochs}  loss={avg:.5f}  ({time.time()-t0:.1f}s)")

        score_net.eval()

    final_loss = sum(losses[-min(20, len(losses)):]) / min(20, len(losses))
    _log(f"Training complete. Final loss: {final_loss:.5f}  ({time.time()-t0:.1f}s)")

    metadata = {
        "adapter_type":  "neural",
        "d_sae":         d_sae,
        "d_cond":        d_cond,
        "hidden":        hidden,
        "depth":         depth,
        "sigma_min":     sigma_min,
        "sigma_max":     sigma_max,
        "n_samples":     N,
        "epochs":        epochs,
        "final_loss":    final_loss,
        "training_mode": "score_adapter",
    }
    return score_net, metadata


# Keep old name as alias for backwards compatibility
train_score_adapter = build_score_adapter


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_score_adapter(
    adapter:  EmpiricalScoreAdapter | SAEScoreNet,
    path:     Path | str,
    metadata: dict,
) -> None:
    """Save a score adapter as safetensors.

    Empirical adapters store z_data (the reference feature matrix).
    Neural adapters store the model state dict.
    In both cases, hyperparameters go into the safetensors metadata dict.
    """
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(adapter, EmpiricalScoreAdapter):
        tensors = {
            "z_data": adapter.z_data.contiguous().cpu(),
        }
    else:
        tensors = {k: v.contiguous().cpu() for k, v in adapter.state_dict().items()}

    meta_strs = {k: str(v) for k, v in metadata.items()}
    save_file(tensors, str(path), metadata=meta_strs)


def load_score_adapter(
    path:   str | Path,
    device: str = "cpu",
) -> tuple[EmpiricalScoreAdapter | SAEScoreNet, dict]:
    """Load a score adapter from safetensors, reconstructing the correct type."""
    from safetensors import safe_open

    path = Path(path)
    with safe_open(str(path), framework="pt", device=device) as f:
        tensors  = {k: f.get_tensor(k) for k in f.keys()}
        meta_raw = f.metadata() or {}

    meta          = {k: _coerce(v) for k, v in meta_raw.items()}
    adapter_type  = str(meta_raw.get("adapter_type", "neural"))

    if adapter_type == "empirical":
        temperature = float(meta_raw.get("temperature", 0.05))
        adapter     = EmpiricalScoreAdapter(tensors["z_data"], temperature=temperature)
    else:
        d_sae  = int(meta_raw["d_sae"])
        d_cond = int(meta_raw["d_cond"])
        hidden = int(meta_raw.get("hidden", 512))
        depth  = int(meta_raw.get("depth",  4))
        net    = SAEScoreNet(d_sae=d_sae, d_cond=d_cond, hidden=hidden, depth=depth)
        net.load_state_dict(tensors)
        net.to(device).eval()
        adapter = net

    return adapter, meta


def _coerce(s: str):
    try:
        return int(s)
    except (ValueError, TypeError):
        pass
    try:
        return float(s)
    except (ValueError, TypeError):
        return s


# ── Inference ODE ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_score_ode(
    z_init:      torch.Tensor,
    cond_pooled: torch.Tensor,
    score_net:   EmpiricalScoreAdapter | SAEScoreNet,
    steps:       int   = 20,
    sigma_start: float = 0.3,
    sigma_end:   float = 1e-3,
) -> torch.Tensor:
    """
    Probability flow ODE in SAE feature space.

    Works with both EmpiricalScoreAdapter and SAEScoreNet — both accept
    (z, log_sigma, cond_pooled) and return a score of the same shape.

    Anneals σ from sigma_start → sigma_end using Euler steps:
        z_{t+1} = z_t − σ_t · score(z_t, log σ_t, cond) · (σ_{t+1} − σ_t)
    """
    device      = z_init.device
    cond_pooled = cond_pooled.to(device)
    sigmas      = torch.linspace(sigma_start, sigma_end, steps + 1, device=device)

    # Initial noise densifies sparse z_init so first score call is valid
    z = z_init.float() + sigmas[0] * torch.randn_like(z_init.float())

    for i in range(steps):
        sigma      = sigmas[i]
        sigma_next = sigmas[i + 1]
        log_sig    = sigma.log().expand(z.shape[0])
        score      = score_net(z, log_sig, cond_pooled)
        z          = z - sigma * score * (sigma_next - sigma)

    return z
