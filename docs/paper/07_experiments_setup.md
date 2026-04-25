# 实验设置

## 6.1 数据切分

- 固定随机种子 `seed=42`；Stratified 80/20 切分，按离职行为分层。
- 训练集 `train_idx.npy`（N = 4 375，正类 635）；测试集 `test_idx.npy`（N = 1 094，正类 159）。
- 所有实验（冠军、七基线、SHAP、Subgroup、HRCF、消融）共用此同一切分，以保证指标严格可比。
- 测试集在整个研究中**仅用于最终评估**，所有阈值、Cleanlab 权重、超参都在训练集的 CV OOF 上决定。

## 6.2 评估指标

- **AUC**：排序能力；对阈值无关。
- **PR-AUC**：类别不平衡下比 ROC-AUC 更能反映少数类识别能力。
- **Balanced Accuracy**：$\frac{1}{2}(\text{Sens} + \text{Spec})$，不平衡分类的首选总览指标。
- **Sens / Spec / F1 / MCC**：在 Bal-Acc-optimal 与 F1-optimal 两个阈值下分别报告。
- **Brier Score, ECE, MCE**：概率校准质量。ECE 以 10 个分位数桶计算。
- **Bootstrap 95% CI**：1 000 次有放回重采样；对 Sens/Spec，在每个重采样上重新计算 confusion matrix。
- **DeLong 配对检验**：对两个分类器在同一测试集上的 AUC 做相关样本比较，得到 z 值与双侧 p 值。

## 6.3 基线配置

为回答"堆叠集成是否真的比单模型好"，我们在**完全相同的训练/测试切分**上重训 7 个基线（Table 17）：

| 代号 | 模型 | 特征空间 | CL 权重 | 关键超参 |
|---|---|---|---|---|
| b1 | Logistic Regression | 标准化 12 维 | off | L2, `C = 1.0` |
| b2 | RandomForest | 原始 12 维 | on | `n_est = 500, max_depth = 10, class_weight="balanced"` |
| b3 | XGBoost | 原始 12 维 | on | `max_depth = 6, lr = 0.05, n_est = 500` |
| b4 | LightGBM | 原始 12 维 | on | `num_leaves = 31, lr = 0.05, n_est = 500` |
| b5 | CatBoost | 原始 12 维 | on | `depth = 6, lr = 0.05, iter = 500` |
| b6 | kNN | 标准化 12 维 | off（kNN 不支持） | `k = 50, 距离加权` |
| b7 | SVM-RBF | 标准化 12 维 | on | `C = 1.0, γ = scale, class_weight="balanced"` |

每个基线均执行：(i) 5-fold CV 得到 OOF 概率；(ii) 在 OOF 上拟合 Isotonic 校准器；(iii) 在全训练集上重训并对测试集生成概率；(iv) 用 CV 校准器校准测试概率；(v) 在 OOF 上找 F1/Bal-Acc 最优阈值；(vi) 在测试集上报告全部指标 + bootstrap CI + 对冠军的 DeLong 检验。

**超参策略**：固定常识默认值而非在测试集上网格搜索。此设计避免在测试集上调参导致的乐观偏差；其代价是每个基线的性能可能没有到绝对最优，但在同一协议下的相对比较仍然公平。

## 6.4 硬件与运行时

所有实验运行在 M2 Max MacBook Pro（12-core CPU，64 GB RAM）上，conda 环境 `yangbo`，Python 3.11。主要第三方依赖版本：

```
scikit-learn ≥ 1.4
xgboost ≥ 2.0   lightgbm ≥ 4.0   catboost ≥ 1.2
torch ≥ 2.2 (CPU)
cleanlab ≥ 2.6
shap ≥ 0.43
```

- **冠军 Stacking 全流程**：约 6 分钟（5-fold + 全量重训 + Isotonic）
- **七基线面板**：约 3 分钟
- **Tree-SHAP on RF**：约 30 秒
- **HRCF on 60 high-risk samples**：约 4 分钟（12 × restart × 60 samples × 2 algos）
- **Subgroup robustness**：< 10 秒（纯测试集切片）
