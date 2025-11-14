## Payload Compensation 任务调参指南

以下建议基于 `payload_compensation_task` 的默认设置，帮助你在看到不同现象时快速定位需要调整的参数。

### 1. 典型现象与应对

- **释放瞬间俯仰/横滚尖峰很大**  
  - 提高 `reward_parameters.attitude_penalty_coef`（基准惩罚）或 `release_attitude_boost`（仅在 `just_released`=True 时生效），促使策略在释放窗口内优先压姿态。  
  - 若仍抑不住，可适度增大 `controller_config.compensation_torque_limits` 上限，让策略拥有更强补偿能力。

- **未释放时出现慢速漂移或过度补偿**  
  - 增大 `reward_parameters.comp_torque_penalty_coef`，让策略在非必要时减少输出。  
  - 同时在任务动作中保留零均值噪声，避免策略只记忆单一偏差。

- **释放排序/时间单调，策略欠泛化**  
  - 扩大 `payload_parameters.release_start_range` 与 `release_interval_range`，增大随机性，使策略学会应对任意顺序与节奏。

- **持续位置误差较大**  
  - 增加 `reward_parameters.position_weight`，或在任务脚本中适度调低位置环 PD 增益，避免外环干扰姿态补偿。

### 2. 观测信号含义

- `payload_mass`：当前附着总质量（含所有未释放子机）。  
- `com_offset`：质心相对机体系原点的偏移，用于快速推断重力力矩方向。  
- `attached_mask`：每个挂点是否仍在位，未来可由载荷估计算法输出。  
- `last_release_norm` 与 `warning_flag`：分别标记最近释放的挂点索引（-1 表示未释放）与是否进入释放预警阶段。

### 3. 推荐调参流程

1. **固定补偿范围**：先将 `compensation_torque_limits` 设为稍大的常量（如 0.3 Nm），确认策略能对抗扰动。  
2. **调奖励结构**：根据落差在 `position_weight`、`attitude_penalty_coef`、`comp_torque_penalty_coef` 间平衡，直到释放瞬间姿态波动 < 期望阈值。  
3. **增强随机性**：打开 `release_start_range` / `release_interval_range`，观察训练曲线是否仍稳定；必要时减少范围逐步放开。  
4. **回收冗余**：训练收敛后再缩小 `compensation_torque_limits`，对策略输出做能量化约束，便于今后移植到真实系统。

> 提示：若需要模拟额外低频风扰，可以在 `PayloadManager` 中扩展一个缓慢变化的扰动力，并将其写入观测；后续只需让真实估计器输出同样的通道即可无缝替换。
