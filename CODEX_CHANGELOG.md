## 工作理念与当前状态
- 我们在 **aerial_gym_simulator** 中原型验证“母机挂载小机释放”场景，目标是通过 RL 学会补偿挂载造成的姿态扰动，并与传统 Lee 控制器对比。
- 目前维护两套任务：`payload_compensation_task`（Lee 控制 + RL 补偿）和 `payload_compensation_task_full_rl`（纯 RL 姿态控制）。二者都通过 `PayloadManager` 注入随机/固定的载荷质量、质心偏移与释放节奏。
- 奖励设计仍在迭代：我们引入了位置/姿态惩罚、动作平滑与速度惩罚，并可调 crash 阈值；但训练与回放的表现差异依旧需要关注，尤其是奖励尺度与终止逻辑要保持一致。
- 训练采用 RL-Games PPO，大批并行环境（默认 2048）+ 长 episode（1800 步）；`score_to_win` 设得很高避免训练过早退出，`save_best_after` 负责保存当前最优模型。
- 常见风险：play 环境与训练配置不一致（导致回放表现差）、reward 数值与真实稳定性脱节、以及终止条件修改后需重新训练。任何新策略或配置改动，都应同步更新 reward/crash 判据，并通过 TensorBoard（runs/<exp>/summaries）核对 episode/ reward 曲线。

## 2025-02-14
- `aerial_gym/rl_training/rl_games/__init__.py`
  - 对 RL-Games 的 `A2CBase` 进行 monkey patch，保证 checkpoint 保存/恢复 observation 与 value 的 `running_mean_std` 缓冲，从而继续训练或 `--play` 时不再经历奖励瞬间暴跌。
  - 同时修复 `set_stats_weights()` 在加载 value 归一化统计时使用错误 key（`'normalize_value'`）的问题。
- `aerial_gym/config/task_config/payload_compensation_task_config.py`
  - 将载荷释放固定为 `start=300`、`interval=150`（范围为 None 且 `randomize_release=False`），并把 `randomization_parameters` 设为 `None`，方便以确定性的 target/扰动验证表现。
  - 新增 `log_release_events` 参数（当前默认 True），在调试少量环境时可打印“某个无人机何时释放哪颗子机”的日志。
- `aerial_gym/task/payload_compensation_task/payload_compensation_task.py`
  - `randomization_parameters` 允许显式设为 `None`，通过 `getattr(... ) or {}` 兜底，避免在 play 模式关闭随机化时出现 `AttributeError: 'NoneType' object has no attribute 'get'`。
  - `PayloadManager` 支持 `log_release_events`：当配置开启时，会在每次释载记录环境 ID、payload 索引与偏移，便于四机调试时区分是哪台释放了哪颗挂点。
- `aerial_gym/control/controllers/position_control.py`
  - 复制原 Lee 位置控制器的核心逻辑，派生出 `LeePositionControllerWithCompensation`，明确动作切分（前 4 维为 `[x,y,z,yaw]`，后 3 维为姿态力矩补偿），在 `update()` 中对补偿量裁剪/缩放后直接叠加至 `wrench[:,3:6]`，用于抵消悬挂载荷带来的额外外力矩。
  - 在 `init_tensors()` 中检查并缓存 `compensation_torque_limits`，保障没有配置字段时立即报错，方便早期发现配置缺失。
- `aerial_gym/config/controller_config/lee_controller_with_comp_config.py`
  - 继承默认 Lee 配置，重写 `num_actions` 为 7，并新增 `compensation_dims`、`compensation_torque_limits`，便于今后通过配置调整补偿通道数与最大力矩。
- `aerial_gym/control/__init__.py`
  - 引入新控制类与配置文件，在 `controller_registry` 中注册 `lee_position_control_with_compensation`，这样任务/机器人只需切换 `controller_name` 即可试用补偿版本。
- `aerial_gym/task/payload_compensation_task/payload_compensation_task.py`
  - 动作空间扩展：RL 现在输出推力补偿 + 3 轴力矩补偿，并在观测中显式提供滚/俯仰误差；任务同时支持目标/初始状态随机化与观测噪声。
  - 奖励中加入推力/角速度惩罚、倾角/高度预警、稳定区自检以及载荷释放姿态约束。当前阶段为便于策略先学会稳态，默认随机化幅度及安全惩罚被调低、curriculum 暂停，后续可逐步恢复。
  - 为排查开局崩溃问题，在 reset/reset_idx 中加入调试日志，记录前几次重置的目标采样与载荷释放顺序。
- `aerial_gym/config/controller_config/lee_controller_with_comp_config.py`
  - 控制器支持推力补偿维度：`num_actions`/`compensation_dims` 增至 4，并新增 `compensation_thrust_limit`。
- `aerial_gym/control/controllers/position_control.py`
  - `LeePositionControllerWithCompensation` 可同时接收推力与力矩补偿，基于配置限幅后直接加到力/力矩输出。
  - 新增完整任务，实现 PayloadManager（含质量/惯量更新、释放调度、预警、理想观测特征与等效重力力矩注入），并在 Task 中复用传统 Lee 位置指令 + RL 补偿输入（3 维），将扩展的 payload 状态写入观测、奖励中增加姿态/补偿能量惩罚。
- `aerial_gym/config/task_config/payload_compensation_task_config.py`
  - 提供任务配置：指明补偿控制器、动作/观测维度、奖励系数以及 payload 的质量、偏移与释放节奏参数。
- `aerial_gym/task/__init__.py`
  - 注册 `payload_compensation_task`，确保 `task_registry` 可用于训练脚本/示例。
- `aerial_gym/rl_training/rl_games/runner.py`
  - 为 RL-Games 接入新的任务枚举，使 `--task payload_compensation_task` 通过 `env_configurations` 创建环境。
- `aerial_gym/task/payload_compensation_task/payload_compensation_task.py`
  - 随机化释放顺序与节奏：为每个环境生成独立的 payload 乱序队列，并支持 `release_start_range` / `release_interval_range` 配置，释放时重新采样下一次等待步数，使策略面临更加多样的扰动情况。
- `aerial_gym/config/task_config/payload_compensation_task_config.py`
  - 新增 `release_start_range`、`release_interval_range`，默认值覆盖 300~500、150~300 步，激活上述随机化逻辑。
- `docs/payload_compensation_tuning.md`
  - 编写调参与现象对照表，说明遇到姿态尖峰、过补偿、位置偏差等情况时应优先调节的参数，并解释观测信号含义与推荐流程。
- `aerial_gym/examples/new_my_position_control.py`
  - 新增示例脚本，直接通过 `task_registry` 构建 `payload_compensation_task`，在小规模环境中随机注入补偿动作并打印挂点释放事件，便于快速验证控制回路。
- `aerial_gym/task/payload_compensation_task/payload_compensation_task.py`
  - 移除 reward 相关函数的 TorchScript 装饰和字典索引限制，改为纯 Python 版本，避免因 `parameter_dict["..."]` 在 TorchScript 中不受支持而导致的初始化报错。
- `aerial_gym/task/payload_compensation_task/payload_compensation_task.py`
  - 任务内部保留 `SimBuilder` 实例并在 `close()` 调用其 `delete_env()`，避免直接访问 `EnvManager` 不存在的 `delete_env` 方法而导致关闭阶段抛出 `AttributeError`。
- `aerial_gym/task/payload_compensation_task/payload_compensation_task.py`
  - 新增 `_initialize_vehicle_state()`，在 `reset/reset_idx` 时将所有环境的机器人根状态重置为零姿态（单位四元数、零速度），以便复现实验基线并减少随机初态漂移；该函数通过 Isaac Gym root state tensor 写回，保证不会意外触发位置重置。
- `aerial_gym/examples/new_my_position_control.py`
  - 释放日志包含环境 ID，便于区分不同环境的挂点事件。
- `aerial_gym/examples/new_my_position_control.py`
  - 重构为策略推理与可视化脚本：加载 RL-Games YAML + checkpoint，构建与训练一致的 MLP 策略网络、生成补偿动作，记录 Z 轴/姿态/释放历史，并使用与原位置控制示例一致的双子图+滑块界面绘制结果。
- `aerial_gym/config/task_config/payload_compensation_task_config.py`
  - 将 `position_weight` 从 0.5 提高到 2.0，使奖励更加关注位置误差，避免策略只靠姿态/补偿项刷分而忽略悬停误差。
- `aerial_gym/config/task_config/payload_compensation_task_config.py`
  - 新增 `crash_distance_threshold`、`crash_tilt_threshold_deg`、`randomize_release` 与速度/动作平滑惩罚系数，可按阶段选择固定释放并收紧崩溃条件。
- `aerial_gym/task/payload_compensation_task/payload_compensation_task.py`
  - `PayloadManager` 支持随机/固定释放模式，奖励函数引入速度与动作平滑惩罚，并根据配置阈值触发 crash，用以提升训练稳定性。
- `aerial_gym/task/payload_compensation_task_full_rl/payload_compensation_task_full_rl.py`
  - 新增 `PayloadCompensationTaskFullRL`，继承 PositionSetpointTask 并嵌入 payload 释放逻辑，让策略直接输出姿态指令同时获取扩展观测与更严格的 crash 判据。
- `aerial_gym/config/task_config/payload_compensation_task_full_rl_config.py`
  - 定义上述任务的配置（观测维度、reward 参数、payload 释放设置），便于在 RL-Games 中直接训练/评估纯 RL 控制方案。
- `aerial_gym/rl_training/rl_games/runner.py`
  - 在 `env_configurations` 中注册 `payload_compensation_task_full_rl`，可以通过 `--task payload_compensation_task_full_rl` 直接创建新环境进行训练/推理。
- `aerial_gym/examples/new_my_position_control.py`
  - 在 `task.close()` 处添加兜底逻辑，遇到 `EnvManager` 不支持 `delete_env` 时仅打印警告并清空 CUDA 显存，保证示例脚本自行完成资源清理。
- `aerial_gym/examples/new_my_position_control.py`
  - 默认将 `--headless` 设为 False，打开 Isaac Gym viewer 以便直接查看飞行效果，仍可通过命令行显式指定 True 进入无头模式。
- `aerial_gym/config/task_config/payload_compensation_task_full_rl_config.py`
  - 增加 `randomization_parameters`：包含初始状态扰动、随机目标范围、观测噪声与质量/惯量/推力抖动配置（默认只开启前三项），为提升策略鲁棒性提供统一入口。
  - 当前为了排查问题，关闭所有随机项：`release_start_range`/`interval_range` 设为 `None`、初始/目标噪声设为 0、观测噪声清零，便于复现实验。
  - 强化安全相关奖励：加大 `crash_penalty`，并新增倾角/高度预警惩罚、线/角速度平滑系数以及载荷释放姿态限制参数。
- `aerial_gym/task/payload_compensation_task_full_rl/payload_compensation_task_full_rl.py`
  - 读取上述随机化配置并在 reset/reset_idx 中重采样目标点、添加初始位置/姿态噪声，同时在 `process_obs_for_task()` 为位置/速度观测叠加噪声；质量/惯量/推力抖动配置仅记录，后续可扩展。
  - 在 `step()` 中根据新参数追加安全惩罚：对临界倾角/高度、过大的线/角速度、以及释放瞬间超限姿态施加负奖励，使策略更警惕极端状态。
"""
python -m aerial_gym.rl_training.rl_games.runner --train   --file aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml   --task payload_compensation_task   --experiment_name payload_comp_rl_stage1_resume   --checkpoint aerial_gym/rl_training/rl_games/runs/payload_comp_rl_stage1_15-17-18-20/nn/payload_comp_rl_stage1.pth   --headless True --num_envs 4096

"""
python -m aerial_gym.rl_training.rl_games.runner --play   --file aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml   --task payload_compensation_task   --checkpoint runs/payload_comp_rl_stagetestresume_18-19-25-04/nn/payload_comp_rl_stagetestresume.pth --headless False --num_envs 64

"""
python -m aerial_gym.rl_training.rl_games.runner --train   --file aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml   --task payload_compensation_task   --experiment_name 3_2   --checkpoint runs/3_1_29-21-55-55/nn/3_1_ep_976_rew_100322.28.pth  --headless True --num_envs 8192

export AERIAL_TB_LOGDIR=./runs/diagnostics_payload
export AERIAL_TB_INTERVAL=100




export AERIAL_FIXED_OBS_NORM=/home/throne/workspaces/aerial_gym_ws/src/aerial_gym_simulator/fixed_stats.npz


export AERIAL_USE_PRIV_ENCODER=0  关闭
export AERIAL_USE_PRIV_ENCODER=1  打开


重新训练观测
/home/throne/miniconda3/envs/aerialgym/bin/python tools/resample_fixed_stats.py \
  --task payload_compensation_task_teacher \
  --num_envs 256 \
  --steps 2000 \
  --warmup 100 \
  --out /home/throne/workspaces/aerial_gym_ws/src/aerial_gym_simulator/fixed_stats.npz



export AERIAL_DEBUG_NAN_SOURCES=0
export AERIAL_GUARD_NAN_MODEL=0   # 关闭自动保护
export AERIAL_DEBUG_NAN_MODEL=1   # 只开日志

tensorboard --logdir ./runs --port 6006

python -m aerial_gym.rl_training.rl_games.runner --train   --file aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml   --task payload_compensation_task   --experiment_name 1_1     --headless False --num_envs 8192 --headless True


clip_frac/0：附加推力 ΔT（沿机体 z 轴的补偿 thrust）。
clip_frac/1：roll 方向补偿力矩 τx。
clip_frac/2：pitch 方向补偿力矩 τy。
clip_frac/3：yaw 方向补偿力矩 τz。
clip_frac/4、clip_frac/5：策略网络中连续动作分布的 log-std 参数（RL-Games 把它们也当作可训练参数来更新，所以同样记录裁剪比例）。


python -m aerial_gym.examples.teacher_expert_demo --steps 2000 --device cuda:0 --headless False


python -m aerial_gym.rl_training.rl_games.runner \
  --train \
  --file aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml \
  --task payload_compensation_task \
  --experiment_name teacher_residual_stage1 \
  --num_envs 1024 \
  --headless True

# 训练或示例前设置混合比例（0~1），越高越偏向专家
export AERIAL_DAGGER_FRAC=0.7

python -m aerial_gym.rl_training.rl_games.runner \
  --train \
  --file aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml \
  --task payload_compensation_task_teacher \
  --experiment_name teacher_residual_stage1 \
  --num_envs 8192 \
  --headless True \
  --checkpoint runs/teacher_residual_stage1_20-22-06-25/nn/teacher_residual_stage1.pth


python -m aerial_gym.rl_training.rl_games.runner \
  --play \
  --file aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml \
  --task payload_compensation_task_teacher \
  --experiment_name teacher_residual_stage1 \
  --num_envs 1024 \
  --headless False \
  --checkpoint runs/teacher_residual_stage1_06-22-18-25/nn/teacher_residual_stage1.pth 

python -m aerial_gym.rl_training.rl_games.runner \
  --train \
  --file aerial_gym/rl_training/rl_games/ppo_aerial_quad_aux.yaml \
  --task payload_compensation_task_teacher \
  --experiment_name teacher_aux_fixed_imitation \
  --num_envs 4096 \
  --headless True

python aerial_gym/rl_training/train_cnn_student.py --teacher_checkpoint runs/teacher_aux_fixed_imitation_17-14-16-10/nn/teacher_aux_fixed_imitation.pth --use_attention --history_len 200 --experiment_name cnn_stage2_waypoint

python aerial_gym/examples/validate_cnn_stage2.py \
    --teacher_checkpoint runs/teacher_aux_fixed_imitation_17-14-16-10/nn/teacher_aux_fixed_imitation.pth \
    --cnn_checkpoint runs/cnn_stage2_waypoint_17-15-33-13/nn/best_cnn_encoder.pth \
    --history_len 100 \
    --steps 1500 \
    --preflight_steps 1000 \
    --show_plot True

## 2025-02-21 Teacher 残差原型
- `aerial_gym/config/task_config/payload_compensation_task_config.py` 增加 `imitation_weight`（默认为 0）供模仿项权重使用。
- 新增 `aerial_gym/config/task_config/payload_compensation_task_teacher_config.py`：开启 `teacher_mode`，观测维度为 29+52（raw 特权拼接到 obs），特权 obs_dim=52，由策略侧可训练编码器处理，增加 `dagger_frac=0.7` 作为教师混合比例。
- `aerial_gym/task/payload_compensation_task/payload_compensation_task.py`：
  - 引入 `teacher_mode` 开关与补偿限幅缓存（从 `lee_controller_with_comp_config` 读取）。
  - 特权向量扩展为 52 维（质量/COM/基惯量对角/最近一次 payload torque/推力常数/时间常数/分配矩阵/扰动上限等），raw 输出到 `privileged_obs` 并拼接到 observations，编码由策略侧完成。
  - 计算教师残差目标：补偿 thrust=payload 质量×|g|，torque=payload 力矩（body），按限幅归一化到 [-1,1]。
  - 奖励中加入模仿项：`imitation_weight * ||action - teacher_residual||^2`（负向），仅在 teacher_mode 且权重>0 时生效。
- `aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml`：
  - 为 log_std 增加上下限配置：`fixed_sigma: False`，`min_logstd/max_logstd`，限制噪声放大。
  - 保留初始 logstd=-2.0 并添加注释，说明噪声 clamp 目的。
- 新增 `aerial_gym/examples/teacher_expert_demo.py`：使用教师残差直接作为动作跑若干步，打印位移漂移，用于快速 sanity check。
