# Payload Compensation Teacher — 网络输入说明

本任务（`payload_compensation_task_teacher`）的策略输入拆成两路：
- `obs`：基础观测 29 维（干路直接使用）。
- `privileged_obs`：特权向量 52 维（单独送入策略侧 encoder，env 不再把 raw 拼到 obs）。

## 基础观测（29 维）
- 0:2 位置误差 `(target_position - robot_position)`，机体坐标系，m  
- 3:11 机体姿态旋转矩阵 3×3 展平（行优先），无单位  
- 12:14 机体系线速度 `robot_body_linvel`，m/s  
- 15:17 机体系角速度 `robot_body_angvel`，rad/s  
- 18      当前载荷质量，kg  
- 19:21   当前质心偏移 `com_offset`，m  
- 22:25   附着掩码（4 个载荷是否仍附着）  
- 26      上一次释放的索引归一化 `last_release_norm`（[-1,1]）  
- 27      上一次释放的质量，kg  
- 28      预警标志 `warning_flag`（0/1）

## 特权向量（52 维，Teacher 模式下启用）
- 0       当前载荷质量，kg  
- 1:3     当前质心偏移，m  
- 4:6     基础惯量对角项（名义机体）  
- 7:9     最近一次 payload 扭矩（机体系），Nm  
- 10      电机推力/力矩比 `thrust_to_torque_ratio`  
- 11:12   电机推力常数最小/最大值  
- 13      单电机最大推力 `max_thrust`  
- 14      最大推力变化率 `max_thrust_rate`  
- 15:18   电机时间常数（升/降，最小/最大各一）  
- 19      最小推力 `min_thrust`  
- 20:43   分配矩阵（4×6 展平，共 24 项）  
- 44:49   外扰最大力/力矩（6 项）  
- 50      施加扰动的概率 `prob_apply_disturbance`  
- 51      预留（当前为 0）

> 以上索引与维度对应 `aerial_gym/task/payload_compensation_task/payload_compensation_task.py::process_obs_for_task`。若修改特权维度，请同步调整 `privileged_observation_space_dim` 与策略侧编码器输入。***
