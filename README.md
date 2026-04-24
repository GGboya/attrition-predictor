# 离职预测（Attrition Predictor）

基于问卷与行为数据对 **离职行为**（二分类：0 = 未离职，1 = 离职）建模，并在此之上叠加 **HR 反事实解释** 与 **噪声鲁棒堆叠**。主指标是 **AUC**（阈值无关、对类别不平衡稳健），参考基线为 Liu et al. 2024 的 BORF（AUC 0.69）。

所有结果产物由 `src/` 下分阶段脚本生成，写入 `src/figures/`、`src/tables/`、`models/`。

---

## 数据

| 项目 | 说明 |
|------|------|
| **主数据文件** | `data/处理之后的离职数据-5000.xlsx` |
| **样本量** | 约 5000 条，正负类约 838 / 4162（≈ 1 : 5） |
| **标签** | `离职行为`（0/1）；另有 `离职意向`（1–5）等字段 |
| **说明** | 原始数据与问卷材料放在 **`data/`**（见 `.gitignore`，**不随仓库推送到远端**；克隆后需自备数据或向维护者索取） |

字段含义与整数编码见 **`docs/variable-labels.md`**（可与 `data/变量赋值说明.docx` 对照）。

---

## Pipeline 概览

六个阶段，顺序执行。每个阶段只读前一阶段的产物，产出落在 `src/tables/` 与 `src/figures/`；训练好的模型落在 `models/`。

| Phase | 脚本 | 目的 | 关键产物 |
|---|---|---|---|
| **0** 数据治理 | `00_prepare_dataset.py` | 14 条 pre-registered 过滤规则 (F1–F14) 清洗 5771→5469 样本，冻结 80/20 分层划分 | `data/processed/clean.csv`、`data/processed/{train,test}_idx.npy`、`tables/table1_sample_flow.csv` |
| **1** MT-MLP 基线 | `01a_mt_model.py`（模型）<br>`01b_mt_train.py`（训练） | 共享编码器 + 二分类头 + 序数意向头（CORN），在冻结划分上对比 单任务 MLP / 三树 / Voting | `models/mt_mlp_calibrated.pkl`、`tables/table4_base_classifier.csv`、`figures/fig2_base_roc_pr.png` |
| **2** HR 反事实 | `02a_hrcf_algo.py`（算法）<br>`02b_hrcf_run.py`（评估）<br>`02c_cost_rank.py`（成本排序） | 对高风险离职者生成硬约束反事实，对比 DiCE 软约束基线；再按 HR 成本 / Δp 排序，产出干预菜单 + Pareto 前沿 | `tables/table5b_hrcf_vs_dice.csv`、`tables/table5c_*.csv`、`figures/fig3a_hrcf_radar.png`、`figures/fig3b_cf_cases.png`、`figures/fig3c_pareto.png` |
| **3** ICM-Net | `03a_icmnet_model.py`（模型）<br>`03b_icmnet_train.py`（训练） | 三阶段训练的意向→行为级联网络 + Mixup + 对称交叉熵；完整 baseline panel 与消融；DeLong 显著性对照 | `models/icmnet_calibrated.pkl`、`tables/table7_*.csv`、`figures/fig7_icmnet_roc.png` |
| **4** 噪声鲁棒 | `04a_nrboost.py`（NR-Boost，GCE + self-paced 重加权）<br>`04b_nrforest.py`（NR-Forest，OOB-loss 加权） | 在 ICM-Net 之外的两条噪声鲁棒路线，再次与 RF/XGB/ICM-Net 对照 | `tables/table8_*.csv`、`tables/table9_nrforest_*.csv`、`figures/fig8_nrboost_roc.png`、`figures/fig9_nrforest_roc.png` |
| **5** Stacking | `05_stacking.py`（当前正式版）<br>`05_legacy_stack_v1.py`、`05_legacy_stack_v2.py`（archive） | 种子平均 MT-MLP + ExtraTrees 加入的 5-base 堆叠，L2-logistic / 凸组合两种 meta | `tables/table9c_stack_v3_*.csv`、`figures/fig9c_stack_v3_roc.png` |

**关于 legacy stack v1/v2**：v1 的 4 个基全为树（OOF 相关≥0.86）→ 堆叠退化为 RF；v2 加入 MT-MLP + SVM-RBF 引入正交信号；v3（= 当前 `05_stacking.py`）在 v2 基础上做种子平均 + 第五个基。两个 legacy 脚本保留以便复现早期结论，产物命名仍带 `9`/`9b` 前缀。

> 注：`src/figures/` 和 `src/tables/` 下的文件名保留了论文草稿 Table/Figure 编号（table1/4/5/7/8/9、fig2/3/7/8/9），与 Phase 编号不一一对应。

---

## 运行

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

cd src

python 00_prepare_dataset.py   # data cleanup + frozen split
python 01b_mt_train.py         # MT-MLP + base learners   (→ models/mt_mlp_calibrated.pkl)
python 02b_hrcf_run.py         # HR-CF vs DiCE
python 02c_cost_rank.py        # intervention menu
python 03b_icmnet_train.py     # ICM-Net                  (→ models/icmnet_calibrated.pkl)
python 04a_nrboost.py          # NR-Boost
python 04b_nrforest.py         # NR-Forest
python 05_stacking.py          # production stacking
```

`01a_mt_model.py`、`02a_hrcf_algo.py`、`03a_icmnet_model.py` 是模型/算法定义模块，不单独跑；它们会被同 Phase 的 `*b` / `*c` 脚本动态加载（`importlib.util`）。

---

## 方法论

1. **指标选择：AUC 为主，并报告不确定性与显著性**
   AUC 是**阈值无关**、对类别不平衡稳健、在不同采样策略间可比的度量，是少数类问题中最稳的排序能力指标；同时报告 **PR-AUC**（不平衡下更能体现少数类排序价值，随机基线就是正类率 ≈0.17）、**Brier**（概率校准）、**MCC**（全四格均衡利用）。每个 AUC / PR-AUC 带 **1000 次自助法 95% CI**；关键模型间做 **DeLong 双侧检验**（`src/_utils.py::delong_test`），明确报告哪些差距达到 α = 0.05 显著、哪些未达。`bal_acc / F1` 在本仓库统一使用在**训练集 CV out-of-fold 概率**上按 F1 选出的阈值，所有模型同口径。

2. **类别不平衡：原始分布 + 类别权重**
   采用 `scale_pos_weight`（XGBoost）、`is_unbalance=True`（LightGBM）、`auto_class_weights='Balanced'`（CatBoost）、`class_weight='balanced'`（sklearn）的内置权重方案。**不做 SMOTE / CTGAN 等合成过采样**，避免合成样本污染评估或在整数编码类别上产生不真实组合，评估始终在原始分布上进行。

3. **阈值**
   仅在**训练集的 CV out-of-fold 概率**上按 F1 选阈值（避免测试泄漏），在测试集复用。模型对外输出的是 isotonic 校准后概率。

4. **反事实解释（Phase 2）**
   HR-CF 在梯度每一步做硬约束投影（不可变列保留、单向列只改变允许方向），生成的反事实总是可执行的；soft-CF / DiCE 基线靠 L1 惩罚施加"软约束"，会产出违约但更近的点。5 维对比：actionability、sparsity、proximity、plausibility、diversity。

5. **噪声鲁棒（Phase 4）**
   - NR-Boost：自定义 XGBoost 损失为 Generalized Cross-Entropy（Zhang & Sabuncu 2018），再叠 self-paced 对难样本降权。
   - NR-Forest：两阶段 RF，用 OOB GCE 损失按类归一化得样本权重，第二阶段按权重训练后与第一阶段按 `blend` 融合。

6. **可复现**
   所有随机性固定在 `RANDOM_STATE = 42`。冻结划分 `data/processed/{train,test}_idx.npy` 由 Phase 0 一次性产出，所有后续阶段共享。

---

## 参考

- Liu et al. 2024, **BORF**：参考 AUC 0.69。`docs/research-paper-2024-borf-turnover-data.md` 给出完整背景。
- 变量与取值：`docs/variable-labels.md`。
