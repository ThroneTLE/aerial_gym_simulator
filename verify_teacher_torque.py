#!/usr/bin/env python3
"""
验证：释放载荷后，教师力矩是否随载荷质量线性变化。
核心问题：训练后策略似乎按"平均值"补偿，而不是随质量自适应。
"""

import sys
import isaacgym  # noqa: F401
import torch
import numpy as np

from aerial_gym.registry.task_registry import task_registry

def main():
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    
    try:
        task = task_registry.make_task(
            "payload_compensation_task_teacher",
            num_envs=8,
            headless=True,
        )
    finally:
        sys.argv = original_argv

    print("=" * 80)
    print("验证：释放载荷后，教师力矩是否随载荷质量变化")
    print("=" * 80)
    
    task.reset()
    
    # 手动设置极端不同的载荷质量，确保差异足够大
    masses = [0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03, 0.04]
    for i, m in enumerate(masses):
        task.payload_manager.payload_mass_per_env[i] = m
    
    # 更新质量属性
    task.payload_manager._update_mass_properties(torch.arange(8, device=task.device))
    
    print("\n手动设置的载荷质量 (kg):")
    for i in range(8):
        print(f"  Env {i}: {task.payload_manager.payload_mass_per_env[i].item():.4f}")
    
    dummy_action = torch.zeros((task.sim_env.num_envs, task.action_space_dim), device=task.device)
    
    # 强制释放所有环境的第一个载荷
    print("\n" + "=" * 80)
    print("强制释放所有环境的 Payload 0 (位于 X 轴)")
    print("=" * 80)
    
    for env_id in range(8):
        task.payload_manager.attached_mask[env_id, 0] = False
        task.payload_manager._update_mass_properties(torch.tensor([env_id], device=task.device))
    
    # 运行一步让物理和教师更新
    task.step(dummy_action)
    
    print(f"\n释放 Payload 0 后的教师残差:")
    print(f"{'Env':>4s} | {'Mass/ea':>8s} | {'Remaining':>10s} | {'Thrust':>8s} | {'Roll':>8s} | {'Pitch':>8s}")
    print("-" * 70)
    
    thrust_list = []
    pitch_list = []
    mass_list = []
    
    for i in range(8):
        mass_ea = task.payload_manager.payload_mass_per_env[i].item()
        remaining = task.payload_manager.current_payload_mass[i].item()
        thrust = task.teacher_residual[i, 0].item()
        roll = task.teacher_residual[i, 1].item()
        pitch = task.teacher_residual[i, 2].item()
        
        print(f"{i:4d} | {mass_ea:8.4f} | {remaining:10.4f} | {thrust:8.4f} | {roll:8.4f} | {pitch:8.4f}")
        
        thrust_list.append(thrust)
        pitch_list.append(pitch)
        mass_list.append(mass_ea)
    
    # 相关性分析
    thrust_t = torch.tensor(thrust_list)
    pitch_t = torch.tensor(pitch_list)
    mass_t = torch.tensor(mass_list)
    
    corr_thrust = torch.corrcoef(torch.stack([mass_t, thrust_t]))[0, 1].item()
    corr_pitch = torch.corrcoef(torch.stack([mass_t, pitch_t]))[0, 1].item()
    
    print(f"\n相关性分析:")
    print(f"  推力 vs 质量: {corr_thrust:.4f}")
    print(f"  Pitch vs 质量: {corr_pitch:.4f}")
    
    if abs(corr_pitch) > 0.9:
        print("  ✓ Pitch 力矩与质量高度相关")
    else:
        print(f"  ⚠️ Pitch 力矩与质量相关性不足 ({corr_pitch:.4f})")
    
    # 释放第二个载荷
    print("\n" + "=" * 80)
    print("强制释放所有环境的 Payload 1 (位于 Y 轴)")
    print("=" * 80)
    
    for env_id in range(8):
        task.payload_manager.attached_mask[env_id, 1] = False
        task.payload_manager._update_mass_properties(torch.tensor([env_id], device=task.device))
    
    task.step(dummy_action)
    
    print(f"\n释放 Payload 0+1 后的教师残差:")
    print(f"{'Env':>4s} | {'Mass/ea':>8s} | {'Remaining':>10s} | {'Thrust':>8s} | {'Roll':>8s} | {'Pitch':>8s}")
    print("-" * 70)
    
    roll_list = []
    for i in range(8):
        mass_ea = task.payload_manager.payload_mass_per_env[i].item()
        remaining = task.payload_manager.current_payload_mass[i].item()
        thrust = task.teacher_residual[i, 0].item()
        roll = task.teacher_residual[i, 1].item()
        pitch = task.teacher_residual[i, 2].item()
        
        print(f"{i:4d} | {mass_ea:8.4f} | {remaining:10.4f} | {thrust:8.4f} | {roll:8.4f} | {pitch:8.4f}")
        roll_list.append(roll)
    
    roll_t = torch.tensor(roll_list)
    corr_roll = torch.corrcoef(torch.stack([mass_t, roll_t]))[0, 1].item()
    print(f"\n  Roll vs 质量: {corr_roll:.4f}")
    
    # 释放所有载荷
    print("\n" + "=" * 80)
    print("释放所有载荷后 (应该不需要补偿)")
    print("=" * 80)
    
    for env_id in range(8):
        task.payload_manager.attached_mask[env_id, :] = False
        task.payload_manager._update_mass_properties(torch.tensor([env_id], device=task.device))
    
    task.step(dummy_action)
    
    print(f"\n所有载荷释放后:")
    for i in range(8):
        remaining = task.payload_manager.current_payload_mass[i].item()
        thrust = task.teacher_residual[i, 0].item()
        roll = task.teacher_residual[i, 1].item()
        pitch = task.teacher_residual[i, 2].item()
        print(f"  Env {i}: remaining={remaining:.4f}, thrust={thrust:.4f}, roll={roll:.4f}, pitch={pitch:.4f}")
    
    print("\n" + "=" * 80)
    print("关键检查：privileged_obs 是否包含正确的质量信息")
    print("=" * 80)
    
    # 重置并检查 privileged obs
    task.reset()
    for i, m in enumerate(masses):
        task.payload_manager.payload_mass_per_env[i] = m
    task.payload_manager._update_mass_properties(torch.arange(8, device=task.device))
    task.step(dummy_action)
    
    priv_obs = task.task_obs.get("privileged_obs", None)
    if priv_obs is not None:
        print(f"\nPrivileged obs 维度: {priv_obs.shape}")
        print("Privileged obs 内容 (前8个环境):")
        print(f"{'Env':>4s} | {'priv[0]':>8s} | {'priv[1]':>8s} | {'priv[2]':>8s} | {'priv[3]':>8s}")
        print("-" * 50)
        for i in range(8):
            vals = priv_obs[i, :4].tolist()
            print(f"{i:4d} | {vals[0]:8.4f} | {vals[1]:8.4f} | {vals[2]:8.4f} | {vals[3]:8.4f}")
        
        # 检查 priv[0] (应该是 payload_mass) 是否与实际质量匹配
        print(f"\n验证 priv[0] 是否等于 current_payload_mass:")
        for i in range(8):
            priv_mass = priv_obs[i, 0].item()
            actual_mass = task.payload_manager.current_payload_mass[i].item()
            match = "✓" if abs(priv_mass - actual_mass) < 1e-4 else "✗"
            print(f"  Env {i}: priv={priv_mass:.4f}, actual={actual_mass:.4f} {match}")
    else:
        print("  ⚠️ 未找到 privileged_obs")
    
    task.close()
    print("\n验证完成。")

if __name__ == "__main__":
    main()
