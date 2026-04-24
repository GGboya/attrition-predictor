"""ICM-Net: Intent-Cascade Mixup Network.

Novel classifier that combines three ideas to beat XGB/LGB/CAT baselines on
our filtered 5469-row dataset:

1) Cascade architecture (vs MT-MLP's parallel heads):
       encoder(x) → z
       intent_head(z) → K-1 CORN logits → intent probability distribution
       behavior_head([z || intent_dist]) → binary logit
   The intent prediction is CASCADED as input to behavior head, not just a
   parallel task. Because intent is predicted from non-intent features, no
   leakage — but its signal is recovered via distillation into behavior.

2) Intent-bucketed tabular Mixup regularization:
       x~ = λ x_i + (1-λ) x_j,   y~_bin = λ y_bin_i + (1-λ) y_bin_j
   with λ ~ Beta(α, α), pairs restricted to |intent_i - intent_j| ≤ 1.
   Smooths decision boundary between neighbouring intent buckets; helps with
   survey noise (Likert ±1 wobble).

3) Symmetric Cross-Entropy (Wang et al. 2019) on the binary head:
       L_SCE = α · CE(y, p) + β · RCE(p, y)
   RCE treats the noisy label as noisy and the prediction as a target, giving
   gradient on samples with unreliable labels. Tuned for self-report noise.

Training schedule (three stages):
    Stage 1: train encoder + intent_head (only CORN loss).     Warmup.
    Stage 2: freeze encoder + intent_head, train behavior_head on detached
             intent distribution.                                Behavior head fit.
    Stage 3: unfreeze all, joint loss (weighted SCE + CORN) w/ Mixup.

Inference: use only behavior_head; intent_head is internal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── encoder (same shape as MT-MLP for fair comparison) ─────────────────
class SharedEncoder(nn.Module):
    def __init__(self, in_dim: int, h1: int = 64, h2: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_dim = h2

    def forward(self, x):
        return self.net(x)


# ─── CORN helpers (duplicated here to keep module self-contained) ───────
def ord_label_to_corn_targets(y_ord: torch.Tensor, n_classes: int = 5) -> torch.Tensor:
    thresholds = torch.arange(n_classes - 1, device=y_ord.device).unsqueeze(0)
    return (y_ord.unsqueeze(1) > thresholds).float()


def corn_class_probs(logits_ord: torch.Tensor) -> torch.Tensor:
    p_gt = torch.sigmoid(logits_ord)                         # (N, K-1)
    N = p_gt.shape[0]
    ones = torch.ones(N, 1, device=p_gt.device)
    zeros = torch.zeros(N, 1, device=p_gt.device)
    p_gt_ge = torch.cat([ones, p_gt], dim=1)
    p_gt_le = torch.cat([p_gt, zeros], dim=1)
    p_eq = (p_gt_ge - p_gt_le).clamp(min=1e-8)
    p_eq = p_eq / p_eq.sum(dim=1, keepdim=True)
    return p_eq


# ─── ICM-Net ────────────────────────────────────────────────────────────
class ICMNet(nn.Module):
    """Intent-Cascade Mixup Network."""

    def __init__(self, in_dim: int, n_ord_classes: int = 5,
                 h1: int = 64, h2: int = 32, dropout: float = 0.2,
                 fuse_dim: int | None = None):
        super().__init__()
        self.encoder = SharedEncoder(in_dim, h1, h2, dropout)
        self.intent_head = nn.Linear(h2, n_ord_classes - 1)
        fuse_dim = fuse_dim if fuse_dim is not None else h2
        # behavior head ingests [z || intent_prob (K dims)]
        self.behavior_head = nn.Sequential(
            nn.Linear(h2 + n_ord_classes, fuse_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fuse_dim, 1),
        )
        self.n_ord_classes = n_ord_classes

    def forward(self, x: torch.Tensor, detach_intent: bool = False,
                zero_intent: bool = False):
        z = self.encoder(x)
        intent_logits = self.intent_head(z)
        intent_probs = corn_class_probs(intent_logits)
        if zero_intent:
            # ablation: drop the cascade — feed zeros so behavior head can't use intent
            intent_feed = torch.zeros_like(intent_probs)
        elif detach_intent:
            intent_feed = intent_probs.detach()
        else:
            intent_feed = intent_probs
        fused = torch.cat([z, intent_feed], dim=1)
        logit_bin = self.behavior_head(fused).squeeze(-1)
        return logit_bin, intent_logits, intent_probs


# ─── losses ─────────────────────────────────────────────────────────────
def symmetric_ce_binary(logit: torch.Tensor, y: torch.Tensor,
                        alpha: float = 0.5, beta: float = 0.5,
                        A: float = -4.0,
                        pos_weight: float | None = None) -> torch.Tensor:
    """Symmetric cross-entropy on binary logit.

    L = α · BCE(y, p)  +  β · RCE(p, y)
    where RCE clips log(y=0 or 1) to A (≈-4) to avoid log(0).
    y can be soft (from Mixup) — RCE uses p as the "clean" distribution.
    """
    p = torch.sigmoid(logit).clamp(1e-7, 1 - 1e-7)

    if pos_weight is None:
        ce = F.binary_cross_entropy(p, y.float(), reduction="mean")
    else:
        w = y.float() * (pos_weight - 1) + 1.0
        ce = F.binary_cross_entropy(p, y.float(), weight=w, reduction="mean")

    # RCE: treat y as a "distribution" needing clipping.
    # For y=1 → log(1)=0 ok; for y=0 → log(0) → clip to A.
    y_clip = torch.where(y > 0.5, torch.zeros_like(y), torch.full_like(y, A))
    y_clip_neg = torch.where(y > 0.5, torch.full_like(y, A), torch.zeros_like(y))
    rce = -(p * y_clip + (1 - p) * y_clip_neg).mean()

    return alpha * ce + beta * rce


def corn_loss(logits_ord: torch.Tensor, y_ord: torch.Tensor,
              n_classes: int = 5) -> torch.Tensor:
    t = ord_label_to_corn_targets(y_ord, n_classes)
    return F.binary_cross_entropy_with_logits(logits_ord, t, reduction="mean")


# ─── intent-bucketed Mixup ──────────────────────────────────────────────
def mixup_intent_bucketed(x: torch.Tensor, y_bin: torch.Tensor, y_ord: torch.Tensor,
                          alpha_mix: float = 0.2, max_intent_gap: int = 1):
    """Mix x with a random partner whose intent differs by ≤ max_intent_gap.

    Returns x_mix, y_bin_mix (soft), y_ord_mix (hard, = y_ord of anchor, since
    the ordinal head is not the target of Mixup — only the binary head is
    regularised).
    """
    N = x.shape[0]
    device = x.device

    # sample λ per batch (scalar, common Mixup recipe)
    lam = float(torch.distributions.Beta(alpha_mix, alpha_mix).sample().clamp(0.3, 0.7))

    # for each anchor i, find a partner j with |intent_i - intent_j| <= gap
    y_ord_cpu = y_ord.detach().cpu().numpy()
    partners = torch.arange(N, device=device)
    # shuffle once; reassign to closest-intent partner when far
    perm = torch.randperm(N, device=device)
    gaps = (y_ord - y_ord[perm]).abs()
    ok = gaps <= max_intent_gap
    # for rows where partner too far, fall back to self (no mix, effectively lam=1)
    j = torch.where(ok, perm, torch.arange(N, device=device))

    x_mix = lam * x + (1 - lam) * x[j]
    y_bin_mix = lam * y_bin.float() + (1 - lam) * y_bin[j].float()
    return x_mix, y_bin_mix, y_ord  # anchor's ordinal label kept for intent head


# ─── end-to-end forward with optional Mixup ─────────────────────────────
def forward_train_step(model: ICMNet, x: torch.Tensor,
                       y_bin: torch.Tensor, y_ord: torch.Tensor,
                       *, stage: int,
                       use_mixup: bool = False, alpha_mix: float = 0.2,
                       use_sce: bool = True,
                       zero_intent: bool = False,
                       pos_weight: float | None = None,
                       lam_bin: float = 0.7, lam_ord: float = 0.3,
                       sce_alpha: float = 0.5, sce_beta: float = 0.5) -> torch.Tensor:
    """One forward step. `stage` ∈ {1, 2, 3}.

    stage 1 (warmup):    intent only.  L = CORN.
    stage 2 (cascade):   behavior only (encoder+intent frozen by caller).
                         L = SCE(binary).  Mixup optional.
    stage 3 (joint):     all params.  L = lam_bin·SCE + lam_ord·CORN.
                         Mixup optional.
    """
    if use_mixup and stage in (2, 3):
        x_use, y_bin_use, y_ord_use = mixup_intent_bucketed(x, y_bin, y_ord, alpha_mix)
    else:
        x_use, y_bin_use, y_ord_use = x, y_bin, y_ord

    detach_intent = (stage == 2)  # stage 2: intent is a fixed input, not trained
    logit_bin, logits_ord, _ = model(x_use, detach_intent=detach_intent,
                                      zero_intent=zero_intent)

    if stage == 1:
        return corn_loss(logits_ord, y_ord_use, n_classes=model.n_ord_classes)

    if use_sce:
        l_bin = symmetric_ce_binary(logit_bin, y_bin_use, alpha=sce_alpha, beta=sce_beta,
                                     pos_weight=pos_weight)
    else:
        pw = torch.tensor([pos_weight], device=logit_bin.device) if pos_weight else None
        l_bin = F.binary_cross_entropy_with_logits(logit_bin, y_bin_use.float(),
                                                    pos_weight=pw, reduction="mean")
    if stage == 2:
        return l_bin
    # stage 3
    l_ord = corn_loss(logits_ord, y_ord_use, n_classes=model.n_ord_classes)
    return lam_bin * l_bin + lam_ord * l_ord


# ─── freeze helpers ─────────────────────────────────────────────────────
def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag
