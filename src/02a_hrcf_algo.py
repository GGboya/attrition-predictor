"""HR-CF: Hard-constrained, cost-aware counterfactual generator.

Problem:
    For a high-risk employee x with f(x) >= τ_high, find x' minimising
    cost c(x, x') subject to:
      (a) immutable features fixed to x
      (b) monotonic features can only move in one direction
      (c) bounded features clipped to [lo, hi]
      (d) integer/step-size constraints on Likert responses
      (e) f(x') < τ_target   (prediction flipped)

Approach:
    Projected gradient on the scaled feature vector. After every Adam step
    we apply the projection cascade (immutable → monotone → bound → snap).
    Multiple random restarts + greedy L2-diversity selection gives the
    top-k counterfactual menu.

Guarantee:
    If the algorithm returns a counterfactual, all (a)-(d) hold exactly
    by construction. (e) is checked at return time; samples where no
    projection step achieves it are marked `success=False`.

This is the core algorithmic contribution. The DiCE-style soft-constraint
baseline is obtainable by constructing a generator with all project_* flags
turned off; we expose that as `HRCFGenerator(hard_project=False)`.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


# ─── feature roles ───────────────────────────────────────────────────
IMMUTABLE_COLS = [
    "性别", "高校类型", "专业类型",
    "家庭所在地", "工作单位性质", "工作区域",
]
DOWN_COLS = ["工作压力"]                   # integer 1..3
UP_LIKERT_INT_COLS = ["工作氛围"]          # integer 1..5
UP_LIKERT_QUART_COLS = ["工作匹配度", "工作满意度", "工作机会"]  # 1..5 by 0.25
UP_CONT_COLS = ["收入水平"]                # income, bounded to 1.5x
ALL_ACTIONABLE = DOWN_COLS + UP_LIKERT_INT_COLS + UP_LIKERT_QUART_COLS + UP_CONT_COLS


# ─── default HR cost weights ────────────────────────────────────────
DEFAULT_COSTS: dict[str, float] = {
    "收入水平":   1.0,       # per ¥
    "工作压力":   3000.0,    # per Likert level reduction
    "工作氛围":   2000.0,    # per level increase
    "工作匹配度": 1500.0,
    "工作满意度": 2500.0,
    "工作机会":   1000.0,
}


@dataclass
class HRCFConfig:
    costs: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_COSTS))
    alpha: float = 5e-5          # cost-term weight in loss (calibrated so it
                                  # competes with logit-term; tuned below)
    lr: float = 0.05
    max_iters: int = 400
    target_prob: float = 0.5      # want f(x') < target_prob
    income_max_mult: float = 1.5  # income bounded above at x0 * 1.5
    n_restarts: int = 8           # multi-start for diversity
    noise_sigma: float = 0.15     # scaled-space noise on restarts
    top_k: int = 5
    hard_project: bool = True     # False → DiCE-style soft baseline
    early_stop_margin: float = 0.02  # stop when prob < target - margin


class HRCFGenerator:
    """Projected-gradient counterfactual generator for a calibrated torch model.

    The underlying model must expose a callable producing (binary_logit, _)
    when given scaled inputs of shape (N, D). We reuse the trained MT-MLP.
    """

    def __init__(
        self,
        model,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        feat_cols: list[str],
        isotonic=None,
        config: HRCFConfig | None = None,
        device: str | torch.device = "cpu",
    ):
        self.model = model.eval().to(device)
        self.feat_cols = list(feat_cols)
        self.device = torch.device(device)
        self.mean = torch.tensor(scaler_mean, dtype=torch.float32, device=self.device)
        self.scale = torch.tensor(scaler_scale, dtype=torch.float32, device=self.device)
        self.iso = isotonic
        self.cfg = config or HRCFConfig()

        self.idx = {c: i for i, c in enumerate(self.feat_cols)}
        self.immutable_idx = [self.idx[c] for c in IMMUTABLE_COLS if c in self.idx]
        self.down_idx = [self.idx[c] for c in DOWN_COLS if c in self.idx]
        self.up_int_idx = [self.idx[c] for c in UP_LIKERT_INT_COLS if c in self.idx]
        self.up_quart_idx = [self.idx[c] for c in UP_LIKERT_QUART_COLS if c in self.idx]
        self.up_cont_idx = [self.idx[c] for c in UP_CONT_COLS if c in self.idx]
        # cost weights in feature order
        self.cost_w = torch.tensor(
            [self.cfg.costs.get(c, 0.0) for c in self.feat_cols],
            dtype=torch.float32, device=self.device,
        )

    # ------- scaling helpers -------
    def _to_scaled(self, x_raw: torch.Tensor) -> torch.Tensor:
        return (x_raw - self.mean) / self.scale

    def _to_raw(self, x_scl: torch.Tensor) -> torch.Tensor:
        return x_scl * self.scale + self.mean

    # ------- forward probability -------
    def _prob_from_scaled(self, x_scl: torch.Tensor) -> torch.Tensor:
        logit, _ = self.model(x_scl)
        return torch.sigmoid(logit)

    def predict_prob(self, x_raw: np.ndarray) -> np.ndarray:
        """Calibrated (isotonic-adjusted) probability on raw-space inputs."""
        with torch.no_grad():
            xr = torch.tensor(x_raw, dtype=torch.float32, device=self.device)
            if xr.ndim == 1:
                xr = xr.unsqueeze(0)
            p = self._prob_from_scaled(self._to_scaled(xr)).cpu().numpy()
        if self.iso is not None:
            p = self.iso.predict(p)
        return p

    # ------- projections (raw space) -------
    def _project(self, x_raw: torch.Tensor, x0_raw: torch.Tensor) -> torch.Tensor:
        if not self.cfg.hard_project:
            return x_raw
        x_new = x_raw.clone()
        # immutable
        for i in self.immutable_idx:
            x_new[..., i] = x0_raw[..., i]
        # down-only (pressure): x' <= x0, and >= 1
        for i in self.down_idx:
            x_new[..., i] = torch.clamp(x_new[..., i], min=1.0, max=x0_raw[..., i])
        # up-only integer Likert (atmosphere): x0 <= x' <= 5
        for i in self.up_int_idx:
            x_new[..., i] = torch.clamp(x_new[..., i], min=x0_raw[..., i], max=5.0)
        # up-only 0.25 Likert
        for i in self.up_quart_idx:
            x_new[..., i] = torch.clamp(x_new[..., i], min=x0_raw[..., i], max=5.0)
        # income: [x0, x0 * mult]
        for i in self.up_cont_idx:
            x_new[..., i] = torch.clamp(
                x_new[..., i],
                min=x0_raw[..., i],
                max=x0_raw[..., i] * self.cfg.income_max_mult,
            )
        return x_new

    def _snap(self, x_raw: torch.Tensor, x0_raw: torch.Tensor) -> torch.Tensor:
        """Discretise integer/step-sized features. Used only at final return."""
        x_new = x_raw.clone()
        # categorical/immutable already fixed
        # pressure & atmosphere → nearest integer (and respect direction)
        for i in self.down_idx:
            v = torch.round(x_new[..., i])
            x_new[..., i] = torch.clamp(v, min=1.0, max=x0_raw[..., i])
        for i in self.up_int_idx:
            v = torch.round(x_new[..., i])
            x_new[..., i] = torch.clamp(v, min=x0_raw[..., i], max=5.0)
        # quart Likert → nearest 0.25
        for i in self.up_quart_idx:
            v = torch.round(x_new[..., i] * 4.0) / 4.0
            x_new[..., i] = torch.clamp(v, min=x0_raw[..., i], max=5.0)
        # income → nearest integer
        for i in self.up_cont_idx:
            v = torch.round(x_new[..., i])
            x_new[..., i] = torch.clamp(
                v, min=x0_raw[..., i], max=x0_raw[..., i] * self.cfg.income_max_mult,
            )
        return x_new

    # ------- cost (raw space) -------
    def cost(self, x_raw: torch.Tensor, x0_raw: torch.Tensor) -> torch.Tensor:
        return (self.cost_w * (x_raw - x0_raw).abs()).sum(dim=-1)

    # ------- main single-start optimisation -------
    def _optimise(self, x0_raw: torch.Tensor, init_raw: torch.Tensor) -> tuple[torch.Tensor, bool, dict]:
        x_raw = init_raw.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([x_raw], lr=self.cfg.lr)
        target_logit = float(np.log(self.cfg.target_prob / (1 - self.cfg.target_prob)))

        best_x = x_raw.detach().clone()
        best_feasible = False
        best_cost = float("inf")
        trace = {"loss": [], "prob": [], "cost": []}

        for step in range(self.cfg.max_iters):
            opt.zero_grad()
            x_scl = self._to_scaled(x_raw)
            logit, _ = self.model(x_scl)
            logit = logit.view(-1)[0]
            prob = torch.sigmoid(logit)
            cst = self.cost(x_raw, x0_raw)
            # hinge on logit: push logit below target only if above
            logit_term = torch.clamp(logit - target_logit, min=0.0)
            loss = logit_term + self.cfg.alpha * cst
            loss.backward()
            opt.step()

            with torch.no_grad():
                # project back to feasible region
                proj = self._project(x_raw.data, x0_raw)
                x_raw.data.copy_(proj)

                # bookkeeping: evaluate post-project
                x_scl_p = self._to_scaled(x_raw.data)
                logit_p, _ = self.model(x_scl_p)
                prob_p = torch.sigmoid(logit_p).view(-1)[0]
                cst_p = self.cost(x_raw.data, x0_raw).view(-1)[0]
                trace["loss"].append(float(loss.item()))
                trace["prob"].append(float(prob_p.item()))
                trace["cost"].append(float(cst_p.item()))

                feasible = prob_p.item() < self.cfg.target_prob
                if feasible and cst_p.item() < best_cost:
                    best_cost = cst_p.item()
                    best_x = x_raw.data.clone()
                    best_feasible = True
                if feasible and prob_p.item() < self.cfg.target_prob - self.cfg.early_stop_margin:
                    # already well past threshold — give cost some more steps then can bail
                    pass

        return best_x, best_feasible, trace

    # ------- top-k with multi-start + greedy diversity -------
    def generate(self, x0_raw: np.ndarray) -> dict:
        """Return up to top_k counterfactuals for single sample.

        Output dict:
            original        (D,) raw
            original_prob   float (calibrated)
            candidates      list of dicts each with
                x, prob, cost, changes (dict feat→Δ), feasible
            n_feasible      int
            success         bool  (>=1 feasible)
        """
        x0 = torch.tensor(x0_raw, dtype=torch.float32, device=self.device)
        if x0.ndim > 1:
            x0 = x0.squeeze(0)
        cands = []
        for r in range(self.cfg.n_restarts):
            init = x0.clone()
            if r == 0:
                pass  # start at x0 (minimum change)
            elif r == 1:
                # max-relief init: push every actionable feature to its
                # most-helpful end of the feasible region. Gives the lowest
                # prob achievable under hard constraints — the optimizer
                # then relaxes to save cost.
                for i in self.down_idx:
                    init[i] = torch.tensor(1.0, device=self.device)
                for i in self.up_int_idx:
                    init[i] = torch.tensor(5.0, device=self.device)
                for i in self.up_quart_idx:
                    init[i] = torch.tensor(5.0, device=self.device)
                for i in self.up_cont_idx:
                    init[i] = x0[i] * self.cfg.income_max_mult
            elif self.cfg.noise_sigma > 0:
                noise = torch.randn_like(init) * self.cfg.noise_sigma * self.scale
                for i in self.immutable_idx:
                    noise[i] = 0.0
                for i in self.down_idx:
                    noise[i] = -noise[i].abs()
                for i in self.up_int_idx + self.up_quart_idx + self.up_cont_idx:
                    noise[i] = noise[i].abs()
                init = self._project(init + noise, x0)
            x_hat, feasible, trace = self._optimise(x0, init)
            if not feasible:
                continue
            x_snapped = self._snap(x_hat, x0)
            # re-evaluate after snap
            with torch.no_grad():
                p = self._prob_from_scaled(self._to_scaled(x_snapped.unsqueeze(0)))
                p_cal = float(p.cpu().numpy()[0])
                if self.iso is not None:
                    p_cal = float(self.iso.predict([p_cal])[0])
            if p_cal >= self.cfg.target_prob:
                continue
            c = float(self.cost(x_snapped, x0).item())
            cands.append({
                "x": x_snapped.cpu().numpy(),
                "prob": p_cal,
                "cost": c,
                "trace": trace,
            })

        # greedy diversity: start with lowest-cost candidate, add farthest in scaled L2
        def scaled(v):
            return ((v - self.mean.cpu().numpy()) / self.scale.cpu().numpy())
        cands_sorted = sorted(cands, key=lambda d: d["cost"])
        selected: list[dict] = []
        selected_ids: set[int] = set()
        if cands_sorted:
            selected.append(cands_sorted[0])
            selected_ids.add(id(cands_sorted[0]))
            while len(selected) < self.cfg.top_k and len(selected) < len(cands_sorted):
                remaining = [c for c in cands_sorted if id(c) not in selected_ids]
                if not remaining:
                    break
                best_far, best_d = None, -1.0
                for c in remaining:
                    dmin = min(
                        float(np.linalg.norm(scaled(c["x"]) - scaled(s["x"])))
                        for s in selected
                    )
                    if dmin > best_d:
                        best_d, best_far = dmin, c
                if best_far is None:
                    break
                selected.append(best_far)
                selected_ids.add(id(best_far))

        # enrich with change dict & validate constraints
        with torch.no_grad():
            x0_np = x0.cpu().numpy()
            p0 = self.predict_prob(x0_np).item() if self.predict_prob(x0_np).size == 1 \
                 else float(self.predict_prob(x0_np.reshape(1, -1))[0])
        for c in selected:
            changes = {}
            for f_idx, f_name in enumerate(self.feat_cols):
                delta = float(c["x"][f_idx] - x0_np[f_idx])
                if abs(delta) > 1e-6:
                    changes[f_name] = delta
            c["changes"] = changes
            c["feasible"], c["violations"] = self._check_feasibility(c["x"], x0_np)

        return {
            "original": x0_np,
            "original_prob": p0,
            "candidates": selected,
            "n_feasible": len(selected),
            "success": len(selected) > 0,
        }

    # ------- constraint checker (used to compare hard vs soft) -------
    def _check_feasibility(self, x: np.ndarray, x0: np.ndarray) -> tuple[bool, dict]:
        v = {}
        for i in self.immutable_idx:
            if abs(x[i] - x0[i]) > 1e-6:
                v[self.feat_cols[i]] = "immutable_violated"
        for i in self.down_idx:
            if x[i] > x0[i] + 1e-6:
                v[self.feat_cols[i]] = "down_only_violated"
            if x[i] < 1 - 1e-6:
                v[self.feat_cols[i]] = v.get(self.feat_cols[i], "") + "|below_1"
        for i in self.up_int_idx + self.up_quart_idx:
            if x[i] < x0[i] - 1e-6:
                v[self.feat_cols[i]] = "up_only_violated"
            if x[i] > 5 + 1e-6:
                v[self.feat_cols[i]] = v.get(self.feat_cols[i], "") + "|above_5"
            # step-size check
            if self.feat_cols[i] in UP_LIKERT_INT_COLS:
                if abs(x[i] - round(x[i])) > 1e-6:
                    v[self.feat_cols[i]] = v.get(self.feat_cols[i], "") + "|non_integer"
            if self.feat_cols[i] in UP_LIKERT_QUART_COLS:
                if abs(x[i] * 4 - round(x[i] * 4)) > 1e-6:
                    v[self.feat_cols[i]] = v.get(self.feat_cols[i], "") + "|non_quarter"
        for i in self.up_cont_idx:
            if x[i] < x0[i] - 1e-6:
                v[self.feat_cols[i]] = "up_only_violated"
            if x[i] > x0[i] * self.cfg.income_max_mult + 1e-6:
                v[self.feat_cols[i]] = v.get(self.feat_cols[i], "") + "|above_bound"
        return (len(v) == 0), v
