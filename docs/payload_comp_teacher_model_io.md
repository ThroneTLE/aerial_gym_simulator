# Payload Compensation Teacher — 策略网络输入/输出示意

> 代码路径：`aerial_gym/rl_training/rl_games/nn/privileged_actor_critic.py`  
> 配置：`aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml`

## 输入通道

- `obs`：基础 29 维（不再包含特权 raw）。
- `privileged_obs`：特权 raw 52 维（单独传给 encoder）。
- 网络内部：`base_dim=29`，`priv_dim=52`，encoder 输出 `priv_embed_dim=8`，与 base 拼成 37 维送干路。

```
env -> {obs:29, privileged_obs:52}
      │               │
      │        priv_encoder (52→128→128→8, ELU)
      │               ▼
      │         priv embed 8
      │               │
      └──concat──► fused 37 (=29+8) ──► actor/critic MLP
```

## 干路结构

- **Privileged encoder**：Linear(52→128) + ELU → Linear(128→128) + ELU → Linear(128→8)
- **共享 MLP**（`mlp.units`）：[256, 256, 128]，激活 ELU。输入维度 37（29 base + 8 embed）。
- **RNN**：YAML 里 GRU(64, 1 layer, layer_norm=True)，位于 MLP 之后（`before_mlp: False`）。若关闭 RNN，则直接用 MLP 输出。

## 输出头

- **动作均值 `mu`**：Linear → 4 维（thrust 补偿 + 3 轴力矩补偿），无激活；随后在环境侧再混合教师残差并裁剪到 [-1, 1]。
- **logstd**：可学习，受 `min_logstd/max_logstd` clamp；用于高斯策略。
- **价值 `value`**：独立 Linear 头，输入与策略共享干路输出（或 RNN 输出）。

## 数据流小结

1) 环境提供 obs（29 维基础）和 privileged_obs（52 维特权 raw）。  
2) 网络将 `privileged_obs` 52 维过 encoder → 8 维 embed；与 29 维 base 拼成 37 维。  
3) 37 维输入共享 MLP (+可选 GRU)，再分支到 `mu`、`logstd`、`value`。  
4) 输出动作经策略噪声采样，送入环境，在 Teacher 模式下按 `dagger_frac` 与教师残差混合，再缩放/限幅成物理补偿力/力矩。***



复刻 xadapt_ctrl-main 用残差补偿时，和当前 aerial_gym Teacher 方案的关键差异与潜在坑点：

控制接口

xadapt：网络直接输出 4 路电机转速归一化命令（0.5 为 hover），经 act_std/mean 反归一化后乘 maxMotorSpd，硬件拓扑还交换了 motor1/2。残差补偿思路（推力+力矩残差加在 Lee 控制器上）与它的直驱电机接口完全不同。若直接套残差网络，命令含义/尺度错位会导致崩溃。
我们：动作是 [-1,1] 的补偿 thrust + 3 力矩，乘限幅后加到 Lee 控制器输出；再经分配器到电机。想复刻 xadapt，需要把动作语义改成电机转速或把 xadapt 模型包一层映射到残差通道。
观测与归一化

xadapt：输入包含当前状态 8 维（ωxyz、prop_acc_z、cmd ωxyz、cmd thrust）+ 上一个动作 4 维 + 历史 100 步的状态/动作堆叠（共 8+4=12，历史 100 -> 1200 维）+ 适应模块 latent 8 维；总维度 > 1200。所有通道用 RMS/mean/var 做归一化（见 xadapt_controller/utils.py）。
我们：obs 81（基础 29 + 特权 raw 52），特权再编码成 8 维嵌入，最终给干路 37 维；按 running mean/std 归一化（rl-games），无 100 步历史。缺少历史和规范化方式不一致，会让策略感知/统计分布完全不同。
网络/训练架构

xadapt：两阶段（PPO+IL + DAGGER），SB3 实现，使用历史堆叠 + 独立适应模块 latent。策略是 ONNX 双模型（base + adap）。无 teacher 残差混合。
我们：rl-games PPO，在线 DAgger 风格混合教师 residual（dagger_frac），特权编码 52→8。没有动作/观测历史堆叠，靠 GRU 64 单层补记忆。
奖励/任务

xadapt：轨迹跟踪/姿态保持，奖励集中在角速度、线加速度惩罚、生存、振荡惩罚等（learning/hyperparam.yaml）。
我们：悬停 + 载荷释放补偿，奖励含位置/姿态/补偿惩罚 + 模仿项等。目标任务差异直接导致策略行为不可对齐。
仿真/物理

xadapt 仿真在自带轻量环境，dt=0.002，max_t=5s，num_envs=300，参数与真实机体匹配其论文设置；没有 payload/COM 扰动概念。
我们在 Isaac Gym，payload 质量/偏置、惯量差、补偿限幅都不一样；控制分配矩阵、推力/力矩限幅也不相同。
可能导致复刻失败的核心点

动作语义不匹配（电机转速 vs 力/力矩残差）——必须统一接口或包裹转换。
观测形状/历史堆叠缺失，且归一化方式不同——如果不用 100 步历史和 RMS，xadapt 训练思路不成立。
奖励/任务场景不同——需要对齐任务（轨迹跟踪 vs 载荷补偿）。
模型结构不同（双模型 + latent vs 单模型 + priv encoder）——复刻需保留 xadapt 的 base+adap 结构或改训练策略。
动力学与限幅不同——需要按 xadapt 的机体参数、allocation、推力/力矩上限、无 payload 偏置来配置。
如果想用残差补偿但靠近 xadapt 的做法，建议先决定接口：要么沿用电机转速输出，去掉残差概念；要么保持残差接口，但重训一套网络/奖励/归一化（不可直接用 xadapt 超参或模型）。


只看 Phase 1（你现在做的 teacher 对齐部分），它的关键模块是：

Intrinsics Encoder (μ)：输入环境参数 (e_t)（如动力学、外参），输出一个低维 (z_t)。代码里是 MLP：128→128→latent_size(8)，直接由环境参数喂给策略，不用历史。
Low-level Controller (π)：接收当前状态 (x_t)（线加速度、角速度/姿态等）和 (z_t)，输出电机命令。代码里 net_arch 是 [128,128,latent(8), dict(pi=[256,256], vf=[512,512])]，动作头再接 Tanh。
监督信号：模型基控制器提供参考 ( \hat z_t )，训练时最小化 |z_t - \hat z_t|，并监督动作（DAGGER-like）混合。
无历史堆叠：Phase 1 不用 100 步历史，只有当前观测 + 环境参数。
与你当前的 teacher 方案对照：

你有 52 维特权向量（动力/混控/载荷等），经 128→128→8 编码，功能上相当于他们的 Intrinsics Encoder 输出 8 维 latent；这一点是对齐的。
你使用的是残差补偿动作（thrust+torque），他们输出的是电机命令；接口不同。
你用 rl-games 的 PPO + 模仿（teacher residual 混合），他们 Phase 1 也是 PPO + 模型监督，但监督的是 (z_t) 和低层动作（基于模型控制器）。想贴近他们，需要：1) 让 8 维 encoder 明确表示“环境内参”而非单纯 raw 特权拼接；2) 有对应的监督项（目前你的监督是残差模仿，不是 z_t 对齐）；3) 注意归一化策略（他们用固定 RMS，当前你用 running mean/std）。
Phase 1 没有历史堆叠这一块，所以“缺少历史”不是 Phase 1 的阻碍；主要差异是动作接口和监督目标不同。


逐条展开说明：

让 8 维 encoder 表示“环境内参”  改了

含义：这 8 维应承载慢变量/内参（质量、COM、惯量、推力常数、扰动上限等），而不是把同样的原始特权向量再次拼到干路里，让网络去猜。
做法：确保 priv 向量只放环境/机体参数（固定或缓变的量），不要混入高频动态量；并且不要再把 raw 特权直接拼到干路末尾（只让 encoder→8 维参与）。这样 8 维就是“环境描述”，不会被重复信息淹没。

有对应的监督项（对齐 z，而不只是残差模仿）xxxxx错的，只有第二阶段有监督学习

含义：xadapt Phase 1 里有 “Intrinsics Encoder(μ) 输出 z，与模型基控制器给的 \hat z 对齐” 的监督。你现在只有模仿残差动作，没有教网络“z 应该代表什么”。
做法：给 encoder 输出加一个辅助损失：让它回归/分类已知的内参（质量、COM、推力常数等），或最简单地重构原始 priv 向量（autoencoder）。这样 8 维会被明确地压缩“环境内参”，而不是随意漂移。动作模仿可以保留，但再多加这个 z-level 监督。

注意归一化策略（固定 RMS vs running mean/std）也改了

含义：xadapt 用预统计的均值/方差做固定归一化；你用 rl-games 的 running stats，训练/推理分布可能不一致。
做法：若想贴近 xadapt，预先从数据/采样统计出 mean/var，训练时固定它，不更新；推理也用同一套参数。或者对特权向量用手动按物理量纲归一化（如质量除以 m0、COM 除以 0.5m 等），并关闭 running stats。这样训练/部署一致，避免分布漂移。
