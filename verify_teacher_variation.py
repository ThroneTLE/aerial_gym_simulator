#!/usr/bin/env python3
"""
验证教师力矩输出是否随载荷质量变化。
检查不同环境是否有不同的教师残差。
"""

import sys

# IsaacGym must be imported before torch
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
            num_envs=16,  # 用少量环境便于观察
            headless=True,
        )
    finally:
        sys.argv = original_argv

    print("=" * 80)
    print("验证教师力矩输出是否随载荷质量变化")
    print("=" * 80)
    
    task.reset()
    
    # 显示每个环境的载荷质量
    payload_mass = task.payload_manager.payload_mass_per_env.clone()
    print(f"\n[初始化后] 每个环境的单个载荷质量 (应该各不相同):")
    for i in range(min(16, task.sim_env.num_envs)):
        print(f"  Env {i:2d}: payload_mass = {payload_mass[i].item():.4f} kg")
    
    print(f"\n  统计: min={payload_mass.min():.4f}, max={payload_mass.max():.4f}, "
          f"mean={payload_mass.mean():.4f}, std={payload_mass.std():.4f}")
    
    if payload_mass.std() < 1e-6:
        print("\n  ⚠️  警告: 所有环境的载荷质量相同！随机化可能未生效！")
    else:
        print("\n  ✓ 载荷质量在不同环境间有变化")
    
    # 运行一些步骤，让系统稳定
    dummy_action = torch.zeros((task.sim_env.num_envs, task.action_space_dim), device=task.device)
    for _ in range(50):
        task.step(dummy_action)
    
    print("\n" + "=" * 80)
    print("运行 50 步后，检查教师残差")
    print("=" * 80)
    
    # 触发教师残差计算
    task.step(dummy_action)
    
    teacher_residual = task.teacher_residual.clone()
    current_mass = task.payload_manager.current_payload_mass.clone()
    
    print(f"\n[Step 51] 每个环境的教师残差 (应该随质量变化):")
    print(f"{'Env':>4s} | {'Payload Mass':>12s} | {'Thrust':>8s} | {'Roll':>8s} | {'Pitch':>8s}")
    print("-" * 60)
    
    for i in range(min(16, task.sim_env.num_envs)):
        print(f"{i:4d} | {current_mass[i].item():12.4f} | "
              f"{teacher_residual[i, 0].item():8.4f} | "
              f"{teacher_residual[i, 1].item():8.4f} | "
              f"{teacher_residual[i, 2].item():8.4f}")
    
    # 检查教师残差的变化
    thrust_std = teacher_residual[:, 0].std().item()
    roll_std = teacher_residual[:, 1].std().item()
    pitch_std = teacher_residual[:, 2].std().item()
    
    print(f"\n教师残差统计:")
    print(f"  Thrust: mean={teacher_residual[:, 0].mean():.4f}, std={thrust_std:.4f}")
    print(f"  Roll:   mean={teacher_residual[:, 1].mean():.4f}, std={roll_std:.4f}")
    print(f"  Pitch:  mean={teacher_residual[:, 2].mean():.4f}, std={pitch_std:.4f}")
    
    if thrust_std < 1e-6 and roll_std < 1e-6 and pitch_std < 1e-6:
        print("\n  ⚠️  警告: 所有环境的教师残差相同！教师可能未正确使用每个环境的载荷参数！")
    else:
        print("\n  ✓ 教师残差在不同环境间有变化")
    
    # 检查推力与质量的相关性
    correlation = torch.corrcoef(torch.stack([current_mass, teacher_residual[:, 0]]))[0, 1].item()
    print(f"\n推力残差与载荷质量的相关系数: {correlation:.4f}")
    if correlation > 0.9:
        print("  ✓ 推力与质量高度正相关（符合预期）")
    elif correlation > 0.5:
        print("  ? 推力与质量有一定相关性")
    else:
        print("  ⚠️  推力与质量相关性较低，可能有问题")
    
    print("\n" + "=" * 80)
    print("测试释放载荷后教师残差的变化")
    print("=" * 80)
    
    # 继续运行直到有释放事件
    for step in range(200):
        task.step(dummy_action)
        if task.payload_manager.just_released_flag.any():
            released_envs = torch.nonzero(task.payload_manager.just_released_flag).squeeze(-1)
            print(f"\n[Step {51 + step}] 检测到释放事件！")
            for env_id in released_envs.tolist()[:5]:  # 只显示前5个
                print(f"  Env {env_id}: 释放了载荷 {task.payload_manager.last_release_index[env_id].item()}")
                print(f"    - Thrust residual: {task.teacher_residual[env_id, 0].item():.4f}")
                print(f"    - Roll residual:   {task.teacher_residual[env_id, 1].item():.4f}")
                print(f"    - Pitch residual:  {task.teacher_residual[env_id, 2].item():.4f}")
            break
    
    # 最终验证：创建极端差异的环境
    print("\n" + "=" * 80)
    print("手动设置极端载荷差异，验证教师响应")
    print("=" * 80)
    
    # 手动设置极端不同的载荷质量
    task.payload_manager.payload_mass_per_env[0] = 0.01  # 最小
    task.payload_manager.payload_mass_per_env[1] = 0.04  # 最大
    task.payload_manager.payload_mass_per_env[2] = 0.01
    task.payload_manager.payload_mass_per_env[3] = 0.04
    
    # 更新质量属性
    task.payload_manager._update_mass_properties(torch.arange(4, device=task.device))
    
    # 运行一步让教师更新
    task.step(dummy_action)
    
    print(f"\n手动设置后:")
    for i in range(4):
        mass = task.payload_manager.current_payload_mass[i].item()
        print(f"  Env {i}: total_payload_mass={mass:.4f}, "
              f"thrust={task.teacher_residual[i, 0].item():.4f}, "
              f"roll={task.teacher_residual[i, 1].item():.4f}, "
              f"pitch={task.teacher_residual[i, 2].item():.4f}")
    
    # 验证 env 0 和 env 1 的推力残差是否不同
    thrust_0 = task.teacher_residual[0, 0].item()
    thrust_1 = task.teacher_residual[1, 0].item()
    
    if abs(thrust_0 - thrust_1) > 0.01:
        print(f"\n  ✓ 推力残差对不同载荷质量有响应 (差异: {abs(thrust_0 - thrust_1):.4f})")
    else:
        print(f"\n  ⚠️ 推力残差对载荷质量变化不敏感！(差异: {abs(thrust_0 - thrust_1):.4f})")
        print("     这可能是问题所在！")
    
    task.close()
    print("\n验证完成。")

if __name__ == "__main__":
    main()
