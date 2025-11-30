# PPO 参数与训练指标速览

> 目标：给不熟悉 PPO 的同学一个快速参考，解释常见超参、训练曲线含义，以及它们对稳定性的影响。结合本项目当前配置/现象撰写。

## 主要超参数（ppo_aerial_quad.yaml）

- `learning_rate`（当前 2e-4）
  - 控制每次梯度更新步长。过大：clip_frac/KL 飙升、训练不稳；过小：收敛慢。
- `kl_threshold`（当前 0.008）
  - 自适应调度器的 KL 上限。超过阈值会自动降学习率/回滚。值越小，更新越保守。
- `entropy_coef`（当前 5e-4）
  - 探索权重。增大可防止过早收敛到尖锐策略；过大则动作噪声过强、收敛慢。
- `e_clip`（当前 0.15）
  - PPO 裁剪范围。越小，单步更新幅度被限制得越紧，稳定性↑，提升速度↓。
- `mini_epochs`（当前 5）
  - 同一批数据的重复训练轮数。越多易过拟合当前 batch、KL 抬升；越少更新更保守。
- 其他相关（保持默认）
  - `minibatch_size`：每次优化的样本量。大 batch 降低梯度噪声但占显存。
  - `num_envs` × `horizon_length`：每次收集的数据量，决定 on-policy 的“数据宽度”。

## 常见训练指标解读

- `clip_frac/i`（各动作维度的裁剪比例）
  - 理想：0.1 左右；>0.2 表示更新过猛、大量样本被截断；<0.05 表示更新太保守。
- `info/kl`
  - 反映新旧策略差距。0.005~0.02 较安全；尖峰>0.03 需要降 lr/收紧剪裁。
- `info/last_lr`
  - 当前实际学习率（含自适应调整）。剧烈波动说明调度器频繁“踩刹车”。
- `info/e_clip`、`info/lr_mul`
  - 裁剪常数与 lr 乘子（通常是固定线/1.0）。
- `episode_lengths`/`rewards`
  - 平稳高位表示策略稳定；周期性跌落通常对应 KL/clip_frac 高企或动作撞限。
- `losses/a_loss`、`losses/c_loss`
  - 策略/价值损失。后期持续抬升常伴随 KL 高、clip_frac 高。
- `losses/bounds_loss`
  - 动作越界惩罚，高说明动作常打到限幅。
- `losses/entropy`
  - 探索程度，正常应缓慢下降；掉到极低/负值再乱升说明策略过尖后又被拉回。
- `actions/saturation_rate`、`actions/comp_saturation_rate`
  - 动作/补偿通道被夹到 ±1 的比例。高饱和率+高 bounds_loss 说明限幅和惩罚在“拉扯”。
- `diagnostics/exp_var`
  - 价值函数解释方差，接近 1 表示 value 拟合良好；大幅跌落需关注 c_loss。

## 调参思路（结合当前现象）

1) **KL/clip_frac 偏高**：
   - 降 `learning_rate`，或再收紧 `e_clip`；适度减小 `mini_epochs`；酌情再降 `kl_threshold`。
2) **熵过低/探索枯竭**：
   - 小幅上调 `entropy_coef`，或确保 log_std 有下限。
3) **动作频繁撞限**：
   - 适度放宽控制上限，或降低对应惩罚/ bounds_loss 源头；若不放宽，则更需降低 lr。
4) **奖励/episode length 周期性掉崖**：
   - 多半是步长过猛+限幅冲突。先稳住 lr/KL，再检查奖励项是否过度惩罚（尤其高段惩罚、方向性惩罚）。

## 快速健康检查

- clip_frac ~0.1；info/kl <0.02，无尖峰；entropy 平滑下降；bounds_loss 低；episode length/奖励曲线平台期波动小。满足这些，训练通常是健康的。

## 奖励参数（payload_compensation_task_config.py）

- `position_weight`（当前 2.0）
  - 距离/姿态主奖励的前置权重。越大，回到目标的引导越强；过大易盖过惩罚导致步子猛。
- `crash_penalty`（-120）
  - 终止时一次性扣分，防止策略频繁撞击。绝对值过大会放大方差。
- 姿态相关
  - `attitude_penalty_coef`（1.0）、`release_attitude_boost`（1.0）：滚俯仰误差惩罚及释放窗口放大系数（已取消放大）。
  - `tilt_excess_*`（阈值/系数/指数均为 0，关闭）：原本用于大倾角指数惩罚，现已关闭防止过度约束。
- 速度/角速度/平滑
- `velocity_penalty_coef`（0.1）：线速度惩罚，过高会抑制必要机动。
  - `angvel_penalty_coef`（0.0）：角速度惩罚，目前关闭，避免滞后变量带来的过激控制。
  - `action_smoothness_coef`（0.1）：动作差分惩罚，抑制高频震荡。可 0.01~0.1 调。
- 补偿能量惩罚
- `comp_torque_penalty_coef`（0.01）、`comp_thrust_penalty_coef`（0.02）：基础补偿幅值惩罚，过大导致“装死”。
  - 高段惩罚已关闭：`comp_penalty_high_threshold=1.0`，高段系数 0。若需限制打满，可适度恢复。
  - `comp_window_penalty_scale`（0.35）：在释放/预警窗口内放松补偿惩罚，便于机动。
- 其他关闭/弱化项
  - `height_*`、`hover_*`、`stability_*`、`position_error_penalty_coef`、`z_error_penalty_coef`、`yaw_penalty_coef` 等均为 0，表示不关心高度/稳定度奖励。
  - `vel_away_penalty_coef`、`accel_penalty_*` 为 0，已移除方向性塑形。

### TensorBoard 奖励分项含义（reward_components/*）
- `pos_reward`：位置项的指数奖励（越接近目标越大）。
- `dist_reward`：线性距离奖励，用于远距离引导。
- `pos_weighted`：位置主奖励 = position_weight × (pos_reward + dist_reward)。
- `up_reward`：姿态对齐奖励（机体 z 轴朝上），乘以 pos_reward。
- `ang_vel_reward`：角速度抑制奖励（随机体角速度减小而增大），乘以 pos_reward。
- `attitude`：滚/俯仰误差惩罚（含释放期放大）。
- `velocity`：线速度惩罚（当前系数较小）。
- `smooth`：动作平滑惩罚（相邻动作差）。
- `comp_torque` / `comp_thrust`：补偿力矩/推力的幅值惩罚（含高段）。
- 其他可能出现的分项：`tilt_warn`、`height_warn`、`release_tilt`、`stability`、`hover_bonus`、`delta_error_bonus` 等，对应配置中的阈值/系数，若为 0 则不会出现在 TB（倾角指数惩罚已关闭，不会出现）。

### 调参建议（奖励）
- 位置引导弱/回不来：提高 `position_weight`（如 1.0→2.0），但观察 clip_frac/KL 是否抬头。
- 释放姿态抖动大：可提高 `attitude_penalty_coef`；如需重启倾角惩罚，设置较高阈值（>5°）且小系数，避免过度限制。
- 动作过猛/高频：小幅提高 `action_smoothness_coef`，或增加补偿惩罚；反之动作不敢出力则反向调整。
- 补偿总是打满：放宽 `comp_penalty_high_threshold` 或降低高段系数/基础补偿惩罚；如果饱和率高且 bounds_loss 高，优先调整惩罚而非仅靠限幅。
- 探索不足/容易发散：结合 PPO 超参（entropy/lr/clip）一起看，奖励端避免过重惩罚导致“躺平”。

## 补偿通道限幅（lee_controller_with_comp_config.py）
- `compensation_thrust_limit = 0.3`：补偿推力最大绝对值（N）。
- `compensation_torque_limits = [0.5, 0.5, 0.1]`：补偿力矩上限（roll, pitch, yaw，单位 Nm）。
  - 若饱和率高且 bounds_loss 高，可放宽上限或降低补偿惩罚；若动作过猛，则反向调整。
