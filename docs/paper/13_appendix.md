# 附录

## 附录 A — MT-MLP 作为 HRCF 可微代理的一致性验证

### A.1 背景

Phase 6 冠军是 Stack(RF + NR-Boost + MT-MLP + SVM + ET) + L2-LR 元学习器 + Isotonic 校准。Isotonic 回归是分段常数函数、无梯度，导致整个堆叠不可端到端微分。

HRCF 需要对目标分类器 $f(x')$ 求梯度，因此我们选择**元系数排名第二、唯一可微的基学习器** MT-MLP（coef = 0.316，仅次于 RF 的 0.376 但 RF 是树基不可微）作为 HRCF 的目标分类器。

本附录验证 MT-MLP 作为堆叠冠军代理的合理性。

### A.2 验证方案与结果

**全测试集（N = 1 094）**：

| 指标 | 值 | p 值 |
|---|---|---|
| Pearson $r$ | **0.9218** | < 1e-300 |
| Spearman $\rho$ | **0.9322** | < 1e-300 |

**高风险区（stack $p \geq 0.30$，N = 160）**：

| 指标 | 值 | 95% CI | p 值 |
|---|---|---|---|
| Pearson $r$ | 0.6112 | [0.526, 0.688] | 9.2e-18 |
| Spearman $\rho$ | 0.6313 | — | 3.6e-19 |

**Top-k 排序一致性**：

| 分位 | Jaccard | Cohen's κ |
|---|---|---|
| Top-10% (k = 109) | 0.591 | 0.715 |
| Top-20% (k = 218) | 0.677 | 0.759 |
| Top-30% (k = 328) | 0.745 | **0.791** |

**决策一致性**：`stack @ τ = 0.135` 与 `MT-MLP @ τ = 0.21`：
- Raw agreement = 0.847
- Cohen's κ = 0.639

### A.3 解读

- 全样本 $r$ = 0.922 表明 MT-MLP 与堆叠在**整体上**高度一致。
- 高风险区 $r$ = 0.611 低于 0.85，主要源于 **range-restriction 效应**（当只看 stack 概率 ≥ 0.30 的子集时，stack 概率的取值区间被截断在 [0.30, ~0.60]，相关性因分母方差下降而相应下降）。Spearman $\rho$ 在这个区间仍保持 0.63。
- Top-30% 的 Cohen's κ = 0.79 表明 HRCF 关心的"谁被标为高风险"这一决策，MT-MLP 与堆叠有 79% 以上的一致性。
- 决策阈值一致性 κ = 0.64 说明在部署层面，两者的"被分为高风险"的员工集合基本重合。

**结论**：MT-MLP 是 HRCF 一个**合理但有偏**的代理：对"谁需要干预"这类决策问题一致性高（κ = 0.64–0.79），对"具体概率数值"的估计存在系统偏差。因此我们在 Section 10 的 Limitations 明确披露此代理性问题，并建议未来研究用可微堆叠替代 Isotonic 以消除偏差。

## 附录 B — 超参数与运行时详表

### B.1 超参数

| 组件 | 超参数 | 值 |
|---|---|---|
| RF | n\_estimators | 400 |
| RF | max\_depth | 10 |
| RF | min\_samples\_leaf | 5 |
| RF | class\_weight | balanced_subsample |
| NR-Boost (XGBoost) | n\_rounds / stage | 400 |
| NR-Boost | n\_stages | 2 |
| NR-Boost | drop\_frac | 0.10 |
| NR-Boost | quantile q | 0.7 |
| MT-MLP | hidden layers | [64, 32] |
| MT-MLP | λ_ordinal | 0.3 |
| MT-MLP | lr / epochs | 1e-3 / 200 |
| SVM-RBF | C | 1.0 |
| SVM-RBF | γ | scale |
| SVM-RBF | class\_weight | balanced |
| ExtraTrees | n\_estimators | 400 |
| ExtraTrees | max\_depth | 12 |
| ExtraTrees | min\_samples\_leaf | 3 |
| Meta | solver | lbfgs |
| Meta | C | 10 |
| HRCF | lr / max\_iters | 0.15 / 1 500 |
| HRCF | α (cost weight) | 1e-7 |
| HRCF | n\_restarts | 12 |
| HRCF | top\_k / noise σ | 5 / 0.3 |
| HRCF | target prob | 0.20 |

### B.2 运行时（M2 Max, 12-core CPU, 64 GB RAM）

| Phase | 耗时 |
|---|---|
| 0–1 preprocessing + MT-MLP base | ~2 min |
| 2a HRCF algo tuning | ~5 min |
| 2b HRCF vs soft-CF (60 samples × 2 algos) | ~4 min |
| 5 Stacking 5-fold | ~6 min |
| 6a features refresh | ~20 s |
| 6b Cleanlab v6 | ~1 min |
| 6c stack + CL v6 | ~6 min |
| 7 stack v5 + ablation | ~7 min |
| 8 threshold sweep | ~10 s |
| 9 bootstrap CI + calibration | ~1 min |
| 10 baselines panel | ~3 min |
| 11 SHAP + PDP | ~2 min |
| 12 subgroup | < 10 s |
| 13 HRCF surrogate validity | < 10 s |

### B.3 代码与数据可复现性

- 所有脚本在 `src/` 目录下，按阶段命名（`00_` ~ `13_`）。
- 冠军中间产物缓存在 `data/processed/phase6_meta_{oof,test}_probs.npy` 与 `data/processed/sample_weights_v6.npy`。
- 固定随机种子 `seed = 42`；切分文件 `train_idx.npy / test_idx.npy` 已提交。
- Conda 环境与 requirements.txt 一并提供，可在其他 macOS / Linux 机器上复现。
