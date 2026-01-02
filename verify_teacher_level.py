#!/usr/bin/env python3
"""
验证：在完美水平姿态下，力矩是否与质量线性相关。
"""

import sys
import isaacgym  # noqa: F401
import torch

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
    print("验证：在完美水平姿态下，力矩与质量的关系")
    print("=" * 80)
    
    task.reset()
    
    # 手动设置不同的载荷质量
    masses = [0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03, 0.04]
    for i, m in enumerate(masses):
        task.payload_manager.payload_mass_per_env[i] = m
    task.payload_manager._update_mass_properties(torch.arange(8, device=task.device))
    
    # 强制所有环境的姿态为完美水平 (identity quaternion: w=1, x=y=z=0)
    vec_root = task.sim_env.IGE_env.vec_root_tensor
    vec_root[:, 0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=task.device)
    # 同时重置速度
    vec_root[:, 0, 7:13] = 0.0
    
    # 应用状态
    from isaacgym import gymtorch
    task.sim_env.IGE_env.gym.set_actor_root_state_tensor(
        task.sim_env.IGE_env.sim,
        gymtorch.unwrap_tensor(task.sim_env.IGE_env.unfolded_vec_root_tensor)
    )
    
    # 释放 payload 0
    for env_id in range(8):
        task.payload_manager.attached_mask[env_id, 0] = False
        task.payload_manager._update_mass_properties(torch.tensor([env_id], device=task.device))
    
    # 立即获取姿态并计算力矩（在物理步进前）
    orientations = vec_root[:, 0, 3:7].clone()
    tau_payload = task.payload_manager.compute_body_torque(orientations)
    
    print("\n在强制水平姿态下:")
    print(f"{'Env':>4s} | {'Mass/ea':>8s} | {'tau_roll':>10s} | {'tau_pitch':>10s}")
    print("-" * 50)
    
    tau_pitch_list = []
    mass_list = []
    
    for i in range(8):
        mass_ea = task.payload_manager.payload_mass_per_env[i].item()
        tau_r = tau_payload[i, 0].item()
        tau_p = tau_payload[i, 1].item()
        print(f"{i:4d} | {mass_ea:8.4f} | {tau_r:10.6f} | {tau_p:10.6f}")
        tau_pitch_list.append(tau_p)
        mass_list.append(mass_ea)
    
    # 相关性
    import numpy as np
    corr = np.corrcoef(mass_list, tau_pitch_list)[0, 1]
    print(f"\n完美水平姿态下 Pitch 力矩与质量相关性: {corr:.4f}")
    
    if abs(corr) > 0.99:
        print("  ✓ 在完美水平姿态下，力矩与质量完美线性相关！")
        print("  结论：问题在于不同的姿态导致了教师信号的噪声")
    else:
        print("  ⚠️ 仍然不相关，需要进一步检查")
    
    task.close()

if __name__ == "__main__":
    main()
