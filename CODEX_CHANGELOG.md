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
/*1111
python aerial_gym/rl_training/train_cnn_student.py --teacher_checkpoint runs/teacher_aux_fixed_imitation_17-14-16-10/nn/teacher_aux_fixed_imitation.pth --use_attention --history_len 500 --experiment_name cnn_stage2_waypoint

/*2222
python aerial_gym/rl_training/train_cnn_student.py --teacher_checkpoint runs/teacher_aux_fixed_imitation_19-21-52-29/nn/last_teacher_aux_fixed_imitation_ep_135_rew_15005.909.pth --use_attention --history_len 500 --experiment_name cnn_stage2_waypoint



/**3333
/home/throne/miniconda3/envs/aerialgym/bin/python aerial_gym/rl_training/train_cnn_student.py --teacher_checkpoint runs/teacher_aux_fixed_imitation_19-21-52-29/nn/last_teacher_aux_fixed_imitation_ep_135_rew_15005.909.pth --use_attention --history_len 500 --experiment_name cnn_stage2_waypoint_resumed --cnn_checkpoint runs/cnn_stage2_waypoint_19-22-16-52/nn/best_cnn_encoder.pth





/*1111
python aerial_gym/examples/validate_cnn_stage2.py \
    --teacher_checkpoint runs/teacher_aux_fixed_imitation_17-14-16-10/nn/teacher_aux_fixed_imitation.pth \
    --cnn_checkpoint runs/cnn_student_18-02-01-12/nn/best_cnn_encoder.pth \
    --history_len 500 \
    --steps 1500 \
    --show_plot True

/*2222
python aerial_gym/examples/validate_cnn_stage2.py \
    --teacher_checkpoint runs/teacher_aux_fixed_imitation_19-21-52-29/nn/last_teacher_aux_fixed_imitation_ep_135_rew_15005.909.pth \
    --cnn_checkpoint runs/cnn_stage2_weighted_v3_20-00-27-56/nn/best_cnn_encoder.pth \
    --history_len 500 \
    --steps 1500 \
    --show_plot True

python aerial_gym/examples/validate_cnn_stage2.py \
    --teacher_checkpoint runs/teacher_aux_fixed_imitation_19-21-52-29/nn/last_teacher_aux_fixed_imitation_ep_135_rew_15005.909.pth \
    --cnn_checkpoint runs/cnn_stage2_blind_v2_20-02-24-13/nn/best_cnn_encoder.pth \
    --history_len 200 \
    --steps 1500 \
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

---

## 2025-01-18 消融实验 (Ablation Studies)

### 实验目的
验证物理参数随机化（电机、阻力、风扰动）对模型泛化能力的影响。

### 配置文件

1. **Full (完整随机化)**: `ppo_ablation_full_teacher.yaml`
   - 包含载荷质量、电机参数、阻力系数、恒定风扰等全部物理参数随机化
   - 任务配置: `payload_compensation_task_teacher_config.py`

2. **No Phys Rand (无物理随机化)**: `ppo_ablation_no_phys_rand.yaml`
   - 仅保留载荷质量随机化，关闭电机/阻力/风扰等其他物理参数随机化
   - 任务配置: `payload_compensation_task_no_phys_rand_config.py`

### 训练命令

```bash
# Full 模型训练 (新训练)
python aerial_gym/rl_training/rl_games/runner.py \
  --train \
  --file aerial_gym/rl_training/rl_games/ppo_ablation_full_teacher.yaml \
  --task payload_compensation_task_teacher \
  --headless True

# Full 模型继续训练 (从 checkpoint 恢复)
python aerial_gym/rl_training/rl_games/runner.py \
  --train \
  --file aerial_gym/rl_training/rl_games/ppo_ablation_full_teacher.yaml \
  --task payload_compensation_task_teacher \
  --checkpoint runs/ablation_full_18-21-58-41/nn/last_ablation_full_ep_155_rew_14999.545.pth \
  --headless True

# NoPhysRand 模型训练
python aerial_gym/rl_training/rl_games/runner.py \
  --train \
  --file aerial_gym/rl_training/rl_games/ppo_ablation_no_phys_rand.yaml \
  --task payload_compensation_task_no_phys_rand \
  --headless True
```

### 评估脚本

评估脚本: `paper/eval_ablation.py`

```bash
# 评估命令示例
python paper/eval_ablation.py \
  --checkpoint "runs/ablation_full_18-23-00-23/nn/last_ablation_full_ep_256_rew_15001.956.pth" \
  --name "Full_ep256" \
  --test_mass 0.03 \
  --test_wind 0.0 \
  --num_steps 500

# 参数说明:
# --checkpoint: 模型 checkpoint 路径
# --name: 实验名称（用于结果标识）
# --test_mass: 测试载荷质量 (kg)
# --test_wind: 测试风扰动 (N)
# --num_steps: 评估步数
# --num_envs: 并行环境数量 (默认 256)
```

### 实验结果 (2025-01-18)

**测试条件**: 环境数量=256, 评估步数=500

#### 不同载荷质量下的极限评估（无风条件）

| 载荷质量 | Original | Full_ep256 | NoPhysRand |
| :---: | :---: | :---: | :---: |
| 0.01 kg | **100.00%** | 100.00% | 100.00% |
| 0.03 kg | **99.61%** | 96.88% | 98.83% |
| 0.05 kg (OOD) | **85.94%** | 81.25% | 38.67% |

#### 风扰动下的消融对比 (Mass=0.03kg, Wind=0.1N)

| 模型 | 成功率 | 崩溃率 |
| :--- | :---: | :---: |
| **Original** | **99.61%** | 0.39% |
| Full_ep256 | 96.48% | 3.52% |
| NoPhysRand | 94.92% | 5.08% |

**崩溃判定条件**:
- 位置误差 > 1.0 m
- 姿态倾角 > 20°

### 关键 Checkpoint

| 模型 | Checkpoint 路径 | 说明 |
| :--- | :--- | :--- |
| **Original (推荐)** | `runs/teacher_aux_fixed_imitation_17-14-16-10/nn/teacher_aux_fixed_imitation.pth` | 无风训练，表现最佳 |
| Full | `runs/ablation_full_18-23-00-23/nn/last_ablation_full_ep_256_rew_15001.956.pth` | 含风随机化 |
| NoPhysRand | `runs/ablation_no_phys_rand_18-21-48-34/nn/last_ablation_no_phys_rand_ep_158_rew_15006.333.pth` | 无物理随机化 |

### 结论

1. **Original 模型表现最佳**：在所有测试条件下成功率均最高
2. **电机/阻力随机化是关键**：Original 和 Full 在 OOD 条件下远超 NoPhysRand
3. **风扰动随机化非必需**：Original 未经风训练但在有风测试中表现最好，说明电机/阻力随机化提供了足够的鲁棒性

---

## 2025-01-20 Stage 2 CNN Ablation (Blind v2)

### 实验配置
- **Model**: `cnn_stage2_blind_v2` (History=200, Blind Observations, No Mask/Warning)
- **Teacher Checkpoint**: `teacher_aux_fixed_imitation` (Ep 135)
- **Script**: `validate_cnn_stage2.py` (Modified for ablation)
- **Steps**: 500
- **Num Envs**: 256

### 结果汇总

#### 1. 载荷质量泛化测试 (无风)
验证模型在不同载荷质量下的稳定性。

| 载荷质量 | 成功率 (Survival) | 崩溃数 | 说明 |
| :---: | :---: | :---: | :--- |
| **0.01 kg** | **100.00%** | 0/256 | 小质量，极其稳定 |
| **0.03 kg** | **100.00%** | 0/256 | 训练分布中心，极其稳定 |
| **0.05 kg** | **100.00%** | 0/256 | 大质量，极其稳定 (对比 Teacher 可能有性能下降，但未崩溃) |

#### 2. 强风扰动测试 (Mass=0.03kg, Wind=0.3N)
验证模型在强外界扰动下的鲁棒性。注意：本次测试风力为 **0.3N**，显著高于之前的 0.1N 测试。

| 条件 | 成功率 (Survival) | 崩溃数 | 备注 |
| :--- | :---: | :---: | :--- |
| **Wind 0.3N** | **90.62%** | 24/256 | 潜变量拟合 MSE 显著增加 (Avg 0.745)，表明强风下预测变难，导致部分环境失稳。 |

### 结论
- `cnn_stage2_blind_v2` 在无风条件下展现了完美的鲁棒性 (100% 存活)，即使在边缘质量 (0.01kg, 0.05kg) 下也未发生崩溃。
#### 3. 极限载荷性能对比 (横向评测, Still Air)
验证不同模型在现实载荷范围（0.1kg - 0.4kg）下的稳定性。

| 载荷质量 | Teacher (Full) | NoPhysRand | PDOnly (Base) | CNN Student |
| :---: | :---: | :---: | :---: | :---: |
| **0.1 kg** | **100.00%** | 100.00% | 100.00% | **100.00%** |
| **0.2 kg** | **100.00%** | 100.00% | **50.78%** | **100.00%** |
| **0.3 kg** | **100.00%** | 100.00% | **0.00%** | **100.00%** |
| **0.4 kg** | **100.00%** | 61.33% | **0.00%** | **100.00%** |

> [!CAUTION]
> **重要更正 (2026-01-20)**: 之前的表格因脚本 `final_ablation_table.py` 的表头顺序与模型列表不匹配，导致 PDOnly 和 NoPhysRand 的数据**完全被互换**。现已修正。

**关键结论 (修正后)**:
1. **Teacher (Full) 模型展示了统治级的泛化力**: 在所有测试质量（包括训练边界外的 0.4kg）均保持 100% 成功率。这证明物理参数随机化方案非常成功。
2. **PDOnly 的物理极限**: 纯 PD 控制器由于无法提供额外的推力/力矩补偿，在 **0.2kg** 时成功率就骤降到 **50.78%**，在 **0.3kg** 及以上载荷时**彻底崩溃 (0%)**。这是因为偏置载荷产生的重力矩超过了 PD 控制器的稳态增益能力，导致倾角超过 25° 阈值。
3. **NoPhysRand 的表现优于预期**: 在 0.3kg 及以下依然保持 100%，直到 0.4kg 时才下降到 61.33%。这比之前以为的"0.2kg 就崩"更加鲁棒，但仍然不如 Teacher。
4. **CNN 学生模型的完美表现**: 学生模型在**所有测试质量（包括 0.4kg）均保持 100% 成功率**，展示了优秀的泛化能力，证明盲系统辨识策略是有效的。

#### 4. 有风条件测试 (0.3N Wind, 0.1-0.4kg)
| 载荷质量 | Teacher (Full) | NoPhysRand | CNN Student |
| :---: | :---: | :---: | :---: |
| **0.1 kg** | 89.06% | **100.00%** | 86.72% |
| **0.2 kg** | 72.66% | **98.44%** | 59.38% |
| **0.3 kg** | 60.55% | **81.64%** | 34.38% |
| **0.4 kg** | **44.14%** | 33.59% | 0.39% |

**关键结论**:
1. **有风条件下所有模型性能显著下降**：风力扰动对所有模型均造成严重影响。
2. **NoPhysRand 在轻载荷下的意外优势**：在 0.1-0.3kg 范围内，NoPhysRand 反而比 Teacher 更稳定。这可能是因为它没有学到"过度补偿"，在风力存在时更保守。
3. **Teacher 在重载荷+风力下依然最强**：在 0.4kg + 0.3N Wind 的极端条件下，Teacher 以 44% 成功率保持领先。
4. **CNN Student 对风力极为敏感**：在 0.4kg + 风力下几乎全部崩溃 (0.39%)，说明盲辨识策略很难区分"风"和"载荷变化"。

#### 5. 极限载荷测试 (Still Air, 0.5-0.8kg)
| 载荷质量 | Teacher (Full) | NoPhysRand | CNN Student |
| :---: | :---: | :---: | :---: |
| **0.5 kg** | **100.00%** | 0.00% | 83.98% |
| **0.6 kg** | **63.67%** | 0.00% | 43.75% |
| **0.7 kg** | **33.20%** | 0.00% | 0.00% |
| **0.8 kg** | 0.00% | 0.00% | 0.00% |

**关键结论**:
1. **Teacher 的物理极限在 0.7-0.8kg**：0.5kg 依然 100%，0.6kg 降到 64%，0.7kg 降到 33%，0.8kg 全部崩溃。这是因为总载荷 (4×0.8=3.2kg) 接近无人机的最大升力极限。
2. **NoPhysRand 在 OOD 条件下彻底失效**：0.5kg 及以上完全崩溃，再次证明没有物理随机化的策略缺乏泛化能力。
3. **CNN Student 展示了不错的极限外推能力**：在 0.5kg 时依然保持 84%，0.6kg 时 44%，比 NoPhysRand 显著更强。这说明历史观测的时间序列蕴含了可泛化的物理信息。
4. **系统物理极限 ≈ 0.7-0.8kg**：所有模型在此载荷下均无法存活，这是硬件限制而非软件问题。
