#!/usr/bin/env python3
"""
诊断 imitation/err 瓶颈问题

检查三个假设：
1. 教师信号噪声 - 同样条件下教师输出是否一致
2. 网络容量 - 网络是否能表达映射
3. 观测信息 - 策略是否有足够信息区分不同情况
"""

import sys
import isaacgym
import torch
import numpy as np

from aerial_gym.registry.task_registry import task_registry

def check_teacher_noise():
    """检查教师信号的噪声水平"""
    print("=" * 80)
    print("检查 1: 教师信号噪声")
    print("=" * 80)
    
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    task = task_registry.make_task('payload_compensation_task_teacher', num_envs=8, headless=True)
    sys.argv = original_argv
    
    task.reset()
    
    # 统一设置所有环境的载荷质量
    for i in range(64):
        task.payload_manager.payload_mass_per_env[i] = 0.025  # 固定质量
    task.payload_manager._update_mass_properties(torch.arange(64, device=task.device))
    
    # 收集多步的教师输出
    dummy = torch.zeros((64, 3), device=task.device)
    teacher_outputs = []
    
    for step in range(100):
        task.step(dummy)
        teacher_outputs.append(task.teacher_residual.clone())
    
    teacher_stack = torch.stack(teacher_outputs)  # [100, 64, 3]
    
    # 分析变化
    print("\n同一质量(0.025kg)下，教师输出的变化:")
    for i, name in enumerate(['thrust', 'roll', 'pitch']):
        mean_val = teacher_stack[:, :, i].mean().item()
        std_across_envs = teacher_stack[:, :, i].std(dim=1).mean().item()  # 同一时刻不同环境的std
        std_across_time = teacher_stack[:, :, i].std(dim=0).mean().item()  # 同一环境不同时刻的std
        print(f"  {name:6s}: mean={mean_val:.4f}, std_env={std_across_envs:.4f}, std_time={std_across_time:.4f}")
    
    # 计算理论最小误差（如果策略输出 mean 值）
    mean_teacher = teacher_stack.mean(dim=(0, 1))  # [3]
    theoretical_min_err = ((teacher_stack - mean_teacher) ** 2).mean().item()
    print(f"\n理论最小 imitation err (策略输出均值): {theoretical_min_err:.4f}")
    print("如果这个值接近 0.15，说明教师噪声是主要原因")
    
    task.close()
    return theoretical_min_err

def check_observation_info():
    """检查策略观测是否包含足够信息"""
    print("\n" + "=" * 80)
    print("检查 2: 观测信息充分性")
    print("=" * 80)
    
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    task = task_registry.make_task('payload_compensation_task_teacher', num_envs=8, headless=True)
    sys.argv = original_argv
    
    task.reset()
    
    # 设置不同质量
    for i in range(64):
        task.payload_manager.payload_mass_per_env[i] = 0.01 + 0.03 * (i / 63)
    task.payload_manager._update_mass_properties(torch.arange(64, device=task.device))
    
    dummy = torch.zeros((64, 3), device=task.device)
    for _ in range(50):
        task.step(dummy)
    
    # 获取观测和教师
    obs = task.obs_dict.get("obs", None)
    priv_obs = task.task_obs.get("privileged_obs", None)
    teacher = task.teacher_residual.clone()
    
    print("\n观测结构:")
    print(f"  base obs 维度: {obs.shape if obs is not None else 'None'}")
    print(f"  privileged obs 维度: {priv_obs.shape if priv_obs is not None else 'None'}")
    print(f"  teacher residual 维度: {teacher.shape}")
    
    # 检查 base obs 是否包含质量相关信息
    if obs is not None:
        print(f"\nBase obs 统计:")
        for i in range(min(10, obs.shape[1])):
            col = obs[:, i]
            print(f"  dim {i}: mean={col.mean():.4f}, std={col.std():.4f}")
        
        # 检查 obs 与 teacher thrust 的相关性
        thrust = teacher[:, 0]
        correlations = []
        for i in range(obs.shape[1]):
            corr = torch.corrcoef(torch.stack([obs[:, i], thrust]))[0, 1].item()
            correlations.append((i, corr))
        
        print(f"\nBase obs 与 thrust 教师的相关性 (前5高):")
        sorted_corr = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)[:5]
        for idx, corr in sorted_corr:
            print(f"  dim {idx}: {corr:.4f}")
    
    # 检查 attached_mask 是否在 obs 中
    attached = task.payload_manager.attached_mask.float()
    print(f"\nAttached mask: {attached[0].tolist()}")
    
    task.close()

def check_network_capacity():
    """检查网络容量"""
    print("\n" + "=" * 80)
    print("检查 3: 网络容量")
    print("=" * 80)
    
    from aerial_gym.rl_training.rl_games.nn.privileged_actor_critic import PrivilegedA2CBuilder
    
    # 模拟网络配置
    params = {
        "privileged_dim": 7,
        "privileged_embed": 8,
        "privileged_hidden": 128,
        "aux_target_dim": 5,
        "mlp": {
            "units": [256, 256, 128],
            "activation": "elu",
        }
    }
    
    print("\n网络结构:")
    print(f"  Privileged encoder: {7} -> 128 -> 128 -> {8}")
    print(f"  MLP backbone: [256, 256, 128]")
    print(f"  Total input to MLP: base_obs(20) + priv_embed(8) = 28")
    
    # 估算参数量
    total_params = 0
    # priv_encoder: 7*128 + 128 + 128*128 + 128 + 128*8 + 8
    priv_encoder_params = 7*128 + 128 + 128*128 + 128 + 128*8 + 8
    total_params += priv_encoder_params
    # mlp: 28*256 + 256 + 256*256 + 256 + 256*128 + 128
    mlp_params = 28*256 + 256 + 256*256 + 256 + 256*128 + 128
    total_params += mlp_params
    
    print(f"  Privileged encoder 参数: {priv_encoder_params:,}")
    print(f"  MLP backbone 参数: {mlp_params:,}")
    print(f"  总参数量估算: ~{total_params:,}")
    
    print("\n评估: 网络容量应该足够表达 mass -> torque 的线性映射")
    print("如果问题不在容量，可能是：")
    print("  - 输入信息不足（策略看不到质量）")
    print("  - 编码器没有正确传递信息")

if __name__ == "__main__":
    min_err = check_teacher_noise()
    check_observation_info()
    check_network_capacity()
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print(f"理论最小 err (教师噪声下限): {min_err:.4f}")
    if min_err > 0.1:
        print("→ 教师信号噪声是主要瓶颈！需要改进教师计算或使用 force/torque 方案")
    else:
        print("→ 教师信号足够干净，问题可能在观测或网络")
