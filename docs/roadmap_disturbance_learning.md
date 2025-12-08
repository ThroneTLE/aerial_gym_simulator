# 抗扰动与奖励解耦改造任务清单

## 任务 1：奖励与归一化基线
- 奖励：使用 xadapt 风格（存活 +10、坠机 -10、角速度 0.2、线加速度 0.01、动作平滑 0.06），移除位置/补偿/模仿项，imitation_weight=0（模仿仅作为损失）。
- 步数同步 dt=0.002：`episode_len_steps`、release/window 步数、`horizon_length` 等已放大，训练命令保持当前配置。
- 归一化：base 用 rl-games 运行归一化；priv 用 `AERIAL_FIXED_OBS_NORM` 指向的 npz（41 维，无 NaN/0 std）。

## 任务 2：扰动经验池与采样
- 利用释放/扰动标志切分轨迹，建立平稳池 + 扰动池。
- 采样策略：扰动池过采样 5–20×，整体混合比例 7:3 或 8:2（平稳:扰动），保证每个 batch 含足够扰动样本。
- 归一化/优势统计：必要时对两池分别做归一化或合并后重置统计，避免少量扰动样本拉坏尺度。
- 实施要点（结合现有指标）：
  - 标记扰动段：用 `release_warning_flag` 或 `just_released`（已有 obs 字段），再配合 `debug/obs_base_norm`、`debug/policy_action_norm` 的尖峰确认标记正确。
  - 写入 buffer 时带上标记：`is_disturb = release_warning_flag | just_released`，其余样本进平稳池。
  - 采样：每次优化从扰动池过采样（如 10×），再与平稳池按 7:3 或 8:2 混合；可用 TB 的 `imitation/err` 峰值验证扰动样本覆盖是否足够（峰值应逐步压低）。
  - 统计防漂：小批量合并后重算优势/归一化统计，避免少量高扰动样本放大均值方差。

## 任务 3：条件损失加权
- 扰动窗口内提高模仿/抗扰动惩罚（角速度/加速度/振荡）权重，窗口外降低，平稳段维持基础表现。
- 保留 priv 中的质量/COM/惯量特征，鼓励在扰动窗口利用慢变量补偿惯量差。

## 任务 4：记忆与序列采样
- 确认 GRU/短历史开启，扰动样本按序列采样保持连续性，必要时调整 bptt/horizon。
- 可选增加短历史堆叠或更强的动作平滑/振荡惩罚，帮助高频响应。

## 任务 5：检查点保存与监控
- 添加定期保存策略（例如每 500 epoch 保存一次，文件名包含 reward/imit_err 或时间戳），避免仅保存 best。
- 监控：继续记录 obs/policy/teacher 范数、扰动/释放标志、池采样比例，便于诊断。

## 任务 6：验证与调参
- 小规模 smoke test（少量迭代）验证无 NaN/shape/device 错误。
- 观察 imitation_err、reward 曲线和 PPO 诊断（clip_frac/exp_var），按需调 learning rate、mini_epochs、权重/采样比例。***
