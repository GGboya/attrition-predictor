# 论文目录

**标题（暂定）**：*An Actionable Attrition Prediction System for New-Graduate Employees: Noise-Robust Stacking with Hard-Constrained Counterfactual Interventions*

**中文暂定**：基于噪声稳健堆叠与硬约束反事实干预的应届毕业生离职预测系统

**作者**：<pending>
**目标刊物**：HR 管理 / 数据挖掘交叉类期刊（投稿方向待定）

---

## 章节一览

| # | 文件 | 主要内容 |
|---|---|---|
| — | [01_abstract.md](01_abstract.md) | 摘要 + 关键词 |
| 1 | [02_intro.md](02_intro.md) | 问题背景、研究目标、贡献、论文结构 |
| 2 | [03_related_work.md](03_related_work.md) | 相关工作：离职预测 / 标签噪声 / 反事实解释 |
| 3 | [04_data_and_preprocessing.md](04_data_and_preprocessing.md) | 数据来源、特征选择、样本流向 |
| 4 | [05_methods_prediction.md](05_methods_prediction.md) | 5 基堆叠 + Cleanlab + Isotonic |
| 5 | [06_methods_hrcf.md](06_methods_hrcf.md) | HRCF 算法、投影级联 |
| 6 | [07_experiments_setup.md](07_experiments_setup.md) | 切分、指标、基线配置 |
| 7 | [08_results_prediction.md](08_results_prediction.md) | 预测结果：冠军 vs 7 基线 + SHAP + Subgroup |
| 8 | [09_results_intervention.md](09_results_intervention.md) | HRCF vs soft-CF + 案例 + 代理一致性 |
| 9 | [10_ablation.md](10_ablation.md) | 消融：基数、CL、校准、阈值、元学习器 |
| 10 | [11_discussion.md](11_discussion.md) | 发现解读、为何堆叠优于单模型 |
| 11 | [12_limitations.md](12_limitations.md) | 数据 / 模型 / 干预局限 + 未来工作 |
| — | [13_appendix.md](13_appendix.md) | 附录 A：MT-MLP 一致性；附录 B：超参 + 运行时 |

## 素材对照（图表）

**表格**（`src/tables/`）：
- `table1_sample_flow.csv` — 样本流向（Section 3）
- `table13_stack_v5_panel.csv / _weights.csv / _delong.csv` — 堆叠消融（Section 9.1）
- `table14_threshold_sweep.csv` — 阈值搜索（Section 9.4）
- `table16_calibration.csv` — 校准 ECE/MCE/Brier（Section 7 & 9.3）
- `table17_baselines_panel.csv / _delong.csv / _calibration.csv` — 7 基线对比（Section 7.1–7.2）
- `table18_shap_importance.csv / _meta_weighted_importance.csv` — SHAP（Section 7.3）
- `table19_subgroup_performance.csv` — 子群（Section 7.5）
- `table20_hrcf_surrogate_validity.csv` — MT-MLP 代理一致性（Section 8.4，附录 A）
- `table21_champion_bootstrap_ci.csv` — 冠军 bootstrap CI（Section 7）
- `table5b_hrcf_vs_dice.csv / table5c_topk_costs.csv / table5c_per_cf.csv` — HRCF（Section 8）

**图**（`src/figures/`）：
- `fig15` — (已删除，BORF 对比已下线)
- `fig16_calibration_reliability.png` — 可靠性 + 分布（Section 7）
- `fig17_baselines_*.png` — 基线 ROC + forest（Section 7.1）
- `fig18a_shap_summary_rf.png / fig18b_shap_bar_rf.png` — SHAP（Section 7.3）
- `fig19_pdp_top5.png` — PDP（Section 7.4）
- `fig20_subgroup_auc_bars.png` — 子群（Section 7.5）
- `fig21_champion_ci.png` — 冠军 CI forest（Section 7）
- `fig22_stack_vs_mtmlp.png` — MT-MLP vs Stack 散点（附录 A）
- `fig3a_hrcf_radar.png / fig3b_* / fig3c_*` — HRCF 五维雷达 + 成本 + Pareto（Section 8）
