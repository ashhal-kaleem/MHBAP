"""
Faithfulness.py — attribution methods and faithfulness checks for VideoTCMT.

Pure PyTorch (no captum needed), so the checks run anywhere the model runs.

Attribution methods (input x: (B, T, F) normalised features, baseline = 0,
which is the train-set mean after normalisation):
  * integrated_gradients : Sundararajan et al. (2017), Riemann (trapezoid) sum
  * saliency             : vanilla gradient |df/dx|
  * gradient_x_input     : grad * x, a cheaper baseline method
  * random_attribution   : uniform noise, the "no information" control

Faithfulness checks:
  * deletion / insertion curves (Petsiuk et al., 2018, RISE): remove (or add)
    input elements in order of attribution and track the probability of the
    originally predicted level. A faithful map makes the probability fall
    fast on deletion (low AUC) and rise fast on insertion (high AUC),
    compared with a random ordering.
  * model-parameter randomisation (Adebayo et al., 2018, "Sanity Checks for
    Saliency Maps"): re-initialise the trained weights and compare
    attributions. A method that still gives the same map on a random model
    is not explaining the model. Reported as Spearman rank correlation of
    |attribution| (lower = passes).
  * IG completeness: sum of attributions ~= f(x) - f(baseline).
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

AttrFn = Callable[[nn.Module, torch.Tensor, torch.Tensor, int, int], torch.Tensor]


def _target_prob(model: nn.Module, x: torch.Tensor, valid: torch.Tensor,
                 state: int, level: torch.Tensor) -> torch.Tensor:
    """Softmax probability of `level` (B,) for affective `state`. Returns (B,)."""
    logits = model(x, valid)[:, state]                     # (B, L)
    return torch.softmax(logits, -1).gather(1, level.view(-1, 1)).squeeze(1)


def predicted_level(model: nn.Module, x: torch.Tensor, valid: torch.Tensor,
                    state: int) -> torch.Tensor:
    with torch.no_grad():
        return model(x, valid)[:, state].argmax(-1)


def integrated_gradients(model: nn.Module, x: torch.Tensor, valid: torch.Tensor,
                         state: int, level: torch.Tensor, steps: int = 32,
                         baseline: Optional[torch.Tensor] = None) -> torch.Tensor:
    """IG w.r.t. the probability of `level`. Returns attributions shaped like x."""
    model.eval()
    base = torch.zeros_like(x) if baseline is None else baseline
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=x.device)
    w = torch.ones_like(alphas)
    w[0] = w[-1] = 0.5                                      # trapezoid rule
    total = torch.zeros_like(x)
    for a, wa in zip(alphas, w):
        xi = (base + a * (x - base)).detach().requires_grad_(True)
        p = _target_prob(model, xi, valid, state, level).sum()
        g, = torch.autograd.grad(p, xi)
        total += wa * g
    return (x - base) * total / steps


def saliency(model: nn.Module, x: torch.Tensor, valid: torch.Tensor,
             state: int, level: torch.Tensor) -> torch.Tensor:
    """Vanilla gradient (Simonyan et al., 2014): no multiplication by the input."""
    model.eval()
    xi = x.detach().clone().requires_grad_(True)
    p = _target_prob(model, xi, valid, state, level).sum()
    g, = torch.autograd.grad(p, xi)
    return g


def gradient_x_input(model: nn.Module, x: torch.Tensor, valid: torch.Tensor,
                     state: int, level: torch.Tensor) -> torch.Tensor:
    model.eval()
    xi = x.detach().clone().requires_grad_(True)
    p = _target_prob(model, xi, valid, state, level).sum()
    g, = torch.autograd.grad(p, xi)
    return g * x


def random_attribution(model: nn.Module, x: torch.Tensor, valid: torch.Tensor,
                       state: int, level: torch.Tensor, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(x.shape, generator=gen).to(x.device)


# ── deletion / insertion ────────────────────────────────────────────────────

@torch.no_grad()
def deletion_insertion(model: nn.Module, x: torch.Tensor, valid: torch.Tensor,
                       state: int, level: torch.Tensor, attr: torch.Tensor,
                       n_steps: int = 20) -> Dict[str, np.ndarray]:
    """
    Per-sample deletion and insertion curves. Elements of invalid (masked)
    frames are ignored: they never reach the model anyway.
    Returns {"deletion": (B, n_steps+1), "insertion": (B, n_steps+1)}.
    """
    model.eval()
    B = x.shape[0]
    flat_x = x.reshape(B, -1)
    elem_valid = valid.unsqueeze(-1).expand_as(x).reshape(B, -1)
    score = attr.abs().reshape(B, -1).masked_fill(~elem_valid, -1.0)
    order = score.argsort(dim=1, descending=True)           # most important first
    n_valid = elem_valid.sum(1)                              # (B,)
    fracs = torch.linspace(0, 1, n_steps + 1)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(order.shape[1]).expand(B, -1))
    dele, ins = [], []
    for f in fracs:
        k = (f * n_valid.float()).round().long().view(-1, 1)
        top = rank < k                                       # top-k important elements
        x_del = flat_x.masked_fill(top, 0.0).view_as(x)
        x_ins = torch.where(top, flat_x, torch.zeros_like(flat_x)).view_as(x)
        dele.append(_target_prob(model, x_del, valid, state, level).cpu().numpy())
        ins.append(_target_prob(model, x_ins, valid, state, level).cpu().numpy())
    return {"deletion": np.stack(dele, 1), "insertion": np.stack(ins, 1)}


def auc(curve: np.ndarray) -> np.ndarray:
    """Normalised area under a curve sampled on [0, 1] (trapezoid). (B, S) -> (B,)."""
    s = curve.shape[-1] - 1
    return (curve[..., :-1] + curve[..., 1:]).sum(-1) / (2 * s)


# ── sanity checks ───────────────────────────────────────────────────────────

def randomise_model(model: nn.Module, seed: int = 0) -> nn.Module:
    """Deep copy with every parameter re-initialised (full randomisation)."""
    torch.manual_seed(seed + 1_000_003)   # never the seed the model was built with
    m = copy.deepcopy(model)
    for mod in m.modules():
        if mod is m:
            continue
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
        elif hasattr(mod, "_reset_parameters"):        # nn.MultiheadAttention
            mod._reset_parameters()
    with torch.no_grad():
        for name, p in m.named_parameters(recurse=False):
            p.normal_(0, 0.02)
    return m


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def completeness_gap(model: nn.Module, x: torch.Tensor, valid: torch.Tensor,
                     state: int, level: torch.Tensor, attr: torch.Tensor) -> np.ndarray:
    """|sum(IG) - (f(x) - f(0))| per sample."""
    with torch.no_grad():
        fx = _target_prob(model, x, valid, state, level)
        f0 = _target_prob(model, torch.zeros_like(x), valid, state, level)
    return (attr.reshape(x.shape[0], -1).sum(1) - (fx - f0)).abs().cpu().numpy()


def modality_shares(attr: torch.Tensor, slices: Dict[str, tuple]) -> Dict[str, np.ndarray]:
    """Fraction of total |attribution| per modality, per sample."""
    a = attr.abs()
    tot = a.sum((1, 2)).clamp(min=1e-12)
    return {m: (a[:, :, s:e].sum((1, 2)) / tot).cpu().numpy() for m, (s, e) in slices.items()}
