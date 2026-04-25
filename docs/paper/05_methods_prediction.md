# 预测方法

## 4.1 总体架构

预测端的核心是一套**两层堆叠集成**：底层由 5 个异质基学习器组成，顶层用带 L2 正则的 Logistic Regression 作为元学习器。集成输出的概率经 Isotonic 回归校准得到最终预测概率。训练时在底层基学习器上注入 Cleanlab v6 样本权重以降低标签噪声的影响。整体流程见 Fig. 1。

```
输入 x (12 dim)
   │
   ├─► RF   ──► p_RF   ┐
   ├─► NRBoost ─► p_NR ┤
   ├─► MT-MLP ─► p_MT  ├─► [p_RF, p_NR, p_MT, p_SVM, p_ET] ─► LR_meta ─► Isotonic ─► p̂
   ├─► SVM-RBF ─► p_SVM┤
   └─► ExtraTrees ─► p_ET ┘
```

## 4.2 基学习器

| 基学习器 | 模型配置 | Sample Weight | 特征空间 |
|---|---|---|---|
| **RF** | RandomForest，`n_estimators=400, max_depth=10, min_samples_leaf=5, class_weight="balanced_subsample"` | 接 CL-v6 | 原始 12 维 |
| **NR-Boost** | XGBoost 基础上的 Noise-Robust Boost：在外循环 2 阶段内部 400 次迭代 boosting，每个 stage 丢弃残差最大的 10% 样本 | `pos_weight` 先验（无 CL） | 原始 12 维 |
| **MT-MLP** | 多任务 MLP：共享 2 个 64→32 全连接层，两个输出头分别做离职意向 5 级有序回归与离职行为二元分类；`λ_order = 0.3` 平衡两个损失；Sigmoid + Isotonic | 接 CL-v6 | 标准化 12 维 |
| **SVM-RBF** | RBF 核 SVM，`C=1.0, γ=scale, class_weight="balanced"` | 接 CL-v6 | 标准化 12 维 |
| **ExtraTrees** | ExtraTrees，`n_estimators=400, max_depth=12, min_samples_leaf=3, class_weight="balanced_subsample"` | 接 CL-v6 | 原始 12 维 |

其中 NR-Boost 的 `pos_weight` 先验是 `neg_count / pos_count`，这与逐样本 CL 权重互斥——两者都试图缓解类别不平衡，我们经消融实验发现同时使用两者会导致过度降权正类。

## 4.3 元学习器与堆叠协议

**堆叠协议**（Phase 6）：

1. 对训练集做 StratifiedKFold 5-fold 切分；
2. 对每一 fold $f$：
   - 在 train-of-fold 上训练 5 个基学习器
   - 对 validation-of-fold 生成 5 维概率向量 → 存入 `oof_preds`
3. 在全训练集上重训 5 个基学习器，对测试集生成 5 维概率向量 → `test_preds`；
4. 在 `oof_preds` 上训练 L2 Logistic Regression（`C=10`）作为元学习器；
5. 用元学习器对 `test_preds` 做预测 → 得到测试集 meta 概率；
6. 在 `oof_preds` 的元预测上拟合 Isotonic 回归，对测试 meta 概率做校准。

**元学习器系数**（Table 13，ablation A）：

| 基 | LR 系数（C=10） |
|---|---|
| RF | **0.376** |
| NR-Boost | 0.014 |
| MT-MLP | **0.316** |
| SVM | -0.048 |
| ET | **0.284** |

RF、MT-MLP、ET 承担主要贡献；SVM 略为负权重表明其在元学习器层面起到"去相关"作用；NR-Boost 的低权重反映出其预测与 RF/ET 高度相关，元学习器已在它们中完成主要信号提取。

## 4.4 Cleanlab v6 样本权重

**v6 设计**：对 Cleanlab 识别出的 299 条可疑训练样本赋 `weight = 0.3`，其余为 `weight = 1.0`。此二值方案在经验上优于连续权重（v5）与完全丢弃（v3）：前者在 CV OOF 上 AUC 提升 0.004，Bal-Acc 提升 0.006；后者导致训练集损失近 7% 样本、方差增大。

与基于 loss-reweighting 的 Focal Loss 等方法相比，CL-v6 的优势在于**无需改动基学习器的损失函数**——我们只需在每个 `fit` 调用传入 `sample_weight=w`。因此它对 RF、XGBoost、SVM、LR 等任意接受 `sample_weight` 的 scikit-learn 风格 API 都是透明的。

## 4.5 Isotonic 概率校准

元学习器的 Logistic 输出在验证集上呈现系统性的"高概率区欠自信、低概率区过自信"，这是类别不平衡数据集上 LR 元模型的常见现象。我们用 Isotonic 回归在 CV OOF 上做校准，测试 ECE 从 0.089 降至 **0.032**，MCE 从 0.221 降至 0.074，Brier 从 0.113 降至 0.110（Table 16）。

## 4.6 决策阈值

对校准后的概率，我们报告两个决策阈值：

- **Bal-Acc 最优阈值 τ = 0.135**：在 CV OOF 上以 Balanced Accuracy 为目标做网格搜索得到。此阈值偏向高 Sensitivity，适合"宁愿多报"的预警场景。
- **F1 最优阈值 τ = 0.185**：在 CV OOF 上以 F1 为目标得到，Sens 与 Spec 更均衡，适合决策成本对称的场景。

两阈值在测试集上的 Sens/Spec 分别为 (0.730, 0.698) 与 (0.679, 0.742)（详见 Table 21）。
