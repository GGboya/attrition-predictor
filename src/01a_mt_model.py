"""Multi-task MLP: shared encoder → {binary turnover head, ordinal intent head}.

Rationale. In the filtered 5469-row dataset, 离职意向 carries non-trivial signal
(intent-only AUC 0.661) but using it as an input feature causes circular
"predict the answer with the answer" evaluation. We instead use it as an
*auxiliary label*: the encoder is pushed to produce representations that
support both the turnover binary prediction and the 1-5 intent ordinal
prediction. At inference, only the binary head is used — the test-time
pipeline never reads 离职意向 as input.

Ordinal head uses the conditional CORN formulation (Cao et al. 2022):
for K=5 classes, emit K-1=4 logits z_k where sigmoid(z_k) = P(y > k | y > k-1).
Independent BCE on each rank threshold; rank consistency is not enforced
hard but empirically close.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MTMlp(nn.Module):
    """Shared encoder + binary turnover head + ordinal intent head (CORN)."""

    def __init__(self, in_dim: int, n_ord_classes: int = 5,
                 h1: int = 64, h2: int = 32, dropout: float = 0.2):
        super().__init__()
        self.encoder = SharedEncoder(in_dim, h1, h2, dropout)
        self.binary_head = nn.Linear(self.encoder.out_dim, 1)
        # K-1 rank-threshold logits for K ordinal classes
        self.ordinal_head = nn.Linear(self.encoder.out_dim, n_ord_classes - 1)
        self.n_ord_classes = n_ord_classes

    def forward(self, x):
        z = self.encoder(x)
        logit_bin = self.binary_head(z).squeeze(-1)       # (N,)
        logits_ord = self.ordinal_head(z)                  # (N, K-1)
        return logit_bin, logits_ord

    @staticmethod
    def ord_label_to_corn_targets(y_ord: torch.Tensor, n_classes: int = 5) -> torch.Tensor:
        """
        Convert class labels in {0, 1, ..., K-1} to rank-threshold targets.
        t_k = 1 iff y > k, for k in {0, 1, ..., K-2}. Returns (N, K-1).
        """
        N = y_ord.shape[0]
        K = n_classes
        thresholds = torch.arange(K - 1, device=y_ord.device).unsqueeze(0)  # (1, K-1)
        return (y_ord.unsqueeze(1) > thresholds).float()

    @staticmethod
    def corn_prob_gt_k(logits_ord: torch.Tensor) -> torch.Tensor:
        """P(y > k) for k in {0, ..., K-2}. Simple per-threshold sigmoid
        (non-rank-consistent). Shape (N, K-1)."""
        return torch.sigmoid(logits_ord)

    @staticmethod
    def corn_class_probs(logits_ord: torch.Tensor) -> torch.Tensor:
        """Convert K-1 threshold logits to K class probabilities via
        P(y=k) = P(y>k-1) - P(y>k). Forces non-negativity via clipping."""
        p_gt = torch.sigmoid(logits_ord)             # (N, K-1)
        K = p_gt.shape[1] + 1
        ones = torch.ones(p_gt.shape[0], 1, device=p_gt.device)
        zeros = torch.zeros(p_gt.shape[0], 1, device=p_gt.device)
        p_gt_ge = torch.cat([ones, p_gt], dim=1)     # P(y > -1)=1, P(y>0)..P(y>K-2)
        p_gt_le = torch.cat([p_gt, zeros], dim=1)    # P(y>0)..P(y>K-2), P(y>K-1)=0
        p_eq = (p_gt_ge - p_gt_le).clamp(min=1e-8)
        p_eq = p_eq / p_eq.sum(dim=1, keepdim=True)
        return p_eq


def joint_loss(logit_bin: torch.Tensor, logits_ord: torch.Tensor,
               y_bin: torch.Tensor, y_ord: torch.Tensor,
               lam: float = 0.5, pos_weight: float | None = None,
               n_ord_classes: int = 5) -> torch.Tensor:
    """
    lam * BCE(y_bin, logit_bin) + (1-lam) * mean_k BCE(corn_target_k, logit_ord_k)
    """
    pw = torch.tensor([pos_weight], device=logit_bin.device) if pos_weight else None
    l_bin = F.binary_cross_entropy_with_logits(logit_bin, y_bin.float(), pos_weight=pw)

    t_ord = MTMlp.ord_label_to_corn_targets(y_ord, n_ord_classes)
    l_ord = F.binary_cross_entropy_with_logits(logits_ord, t_ord, reduction="mean")

    return lam * l_bin + (1.0 - lam) * l_ord
