# HRCF 反事实干预算法

## 5.1 问题形式化

记高风险员工的原始特征向量为 $x \in \mathbb{R}^{12}$，目标分类器为 $f: \mathbb{R}^{12} \to [0, 1]$（预测离职概率），决策阈值为 $\tau_{\text{target}}$。我们希望找到反事实 $x'$，在满足硬约束集合 $\mathcal{C}$ 的前提下最小化成本 $c(x, x')$ 且使 $f(x') < \tau_{\text{target}}$：

$$
\min_{x'} \; c(x, x') \quad \text{s.t.} \quad f(x') < \tau_{\text{target}}, \quad x' \in \mathcal{C}
$$

硬约束 $\mathcal{C}$ 包含四类：

- **(a) Immutable**：`性别, 高校类型, 专业类型, 家庭所在地, 工作单位性质, 工作区域` 必须等于原值。
- **(b) Monotonic**：`工作压力` 只能降低；`工作满意度, 工作匹配度, 工作机会, 工作氛围, 收入水平` 只能升高。
- **(c) Bounded**：Likert 变量限制在原始取值区间；`收入水平` 限制在 $[x_{\text{收入}}, 2 x_{\text{收入}}]$（最多涨薪一倍）。
- **(d) Integer/Step**：`工作压力 ∈ {1, 2, 3}`，`工作氛围 ∈ {1, ..., 5}` 为整数；`工作满意度, 工作匹配度, 工作机会 ∈ {1, 1.25, ..., 4.75, 5}` 为 0.25 步长。

## 5.2 HRCF 算法

我们用**投影式 Adam 梯度下降**求解。单次求解的伪代码：

```python
# 输入: x0, f, cost_weights, τ_target, lr, max_iters
z = scale(x0) + noise   # 在 z-score 空间做优化
for t in 1..max_iters:
    x_cand   = unscale(z)
    prob     = f(x_cand)
    loss     = softplus(τ_target - prob) + α · cost(x_cand, x0)
    z        = Adam_step(z, ∇_z loss, lr)
    x_cand   = unscale(z)
    x_cand   = project_immutable(x_cand, x0)
    x_cand   = project_monotonic(x_cand, x0)
    x_cand   = project_bounded(x_cand, x0)
    x_cand   = snap_to_grid(x_cand)   # Likert 离散取值
    z        = scale(x_cand)
    if f(x_cand) < τ_target - margin: break
```

**关键设计**：

- **投影级联**：在每一步梯度下降之后，按 (a) → (b) → (c) → (d) 的顺序投影；这保证返回的 $x_{\text{cand}}$ 在整个优化轨迹上始终满足硬约束。
- **离散 snap**：`snap_to_grid` 将连续优化结果舍入至 Likert 允许的取值网格。因为舍入可能破坏单调性，snap 之后再做一次 $\text{clip}$ 回到 $[x_0, \infty)$（单调↑）或 $(-\infty, x_0]$（单调↓）。
- **多重启动 + 菜单选择**：用 $R = 12$ 次随机初始化（在 z-score 空间加 $\sigma = 0.3$ 高斯噪声）运行上述流程，得到多个候选；用贪心 L2 多样性选择保留 $k = 5$ 条最不相似的方案作为 Top-5 菜单。
- **成本函数**：$c(x, x') = \sum_i w_i \cdot |x'_i - x_{0,i}|$。收入权重 $w_{\text{收入}} = 1.0$（元），其他 Likert 项的权重按 HR 文献中"一个级别变化大约相当于 1.5k–3k 元月薪"的经验估计设定（见 `02a_hrcf_algo.py::DEFAULT_COSTS`）。

## 5.3 HRCF vs Soft-CF

我们通过设置 `hard_project=False` 得到 DiCE 风格的 **soft-CF** 基线——它在损失中加入 constraint-violation 惩罚项，但不做投影。soft-CF 更容易达到低概率目标（因为搜索空间是整个 $\mathbb{R}^{12}$，不受约束限制），但其返回的 $x'$ 几乎从不满足约束集合 $\mathcal{C}$。在实证上（Section 8），soft-CF 的 actionability 为 0%——每一条返回的方案都至少违反一项硬约束——而 HRCF 为 100%。

## 5.4 目标分类器选择

HRCF 需要对 $f(x')$ 求梯度。Phase 6 冠军（Stack + Isotonic）整体不是端到端可微的（Isotonic 是分段常数函数）。因此我们将 HRCF 的目标分类器设为冠军堆叠中权重最大的 MT-MLP 基学习器的校准版本（coef = 0.316，仅次于 RF 的 0.376；但 RF 非可微）。

附录 A 报告 MT-MLP 与整体堆叠的一致性验证：全测试集上 Pearson $r$ = 0.922、Spearman $\rho$ = 0.932；在高风险区（stack $p \geq 0.30$，$n$ = 160）top-30% 排名的 Cohen $\kappa$ = 0.791。这证明 MT-MLP 是 HRCF 的一个合理的可微代理。
