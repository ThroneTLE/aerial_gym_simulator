#!/usr/bin/env python3
"""
详细分析 COM 偏移和力矩计算。
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
            num_envs=4,
            headless=True,
        )
    finally:
        sys.argv = original_argv

    print("=" * 80)
    print("详细分析 COM 偏移和力矩计算")
    print("=" * 80)
    
    task.reset()
    
    # 手动设置不同的载荷质量
    masses = [0.01, 0.02, 0.03, 0.04]
    for i, m in enumerate(masses):
        task.payload_manager.payload_mass_per_env[i] = m
    task.payload_manager._update_mass_properties(torch.arange(4, device=task.device))
    
    pm = task.payload_manager
    
    # 强制释放 payload 0 ([0.4, 0.0, -0.4])
    for env_id in range(4):
        pm.attached_mask[env_id, 0] = False
        pm._update_mass_properties(torch.tensor([env_id], device=task.device))
    
    print("\n配置:")
    print(f"  Base mass: {pm.base_mass[0].item():.4f} kg")
    print(f"  Offsets: {pm.offsets[0].tolist()}")
    print(f"  Attached mask (after release): {pm.attached_mask[0].tolist()}")
    
    print("\n" + "=" * 80)
    print("分析每个环境的计算过程")
    print("=" * 80)
    
    gravity_z = -9.81
    
    for env_id in range(4):
        mass_ea = pm.payload_mass_per_env[env_id].item()
        num_attached = pm.attached_mask[env_id].sum().item()
        payload_mass = num_attached * mass_ea
        total_mass = pm.base_mass[env_id].item() + payload_mass
        
        # 手动计算 weighted offset
        weighted_offset = torch.zeros(3, device=task.device)
        for idx in range(pm.num_payloads):
            if pm.attached_mask[env_id, idx]:
                weighted_offset += mass_ea * pm.offsets[env_id, idx]
        
        com_offset = weighted_offset / total_mass
        
        # 在水平姿态下，F_gravity_body = [0, 0, gravity_z * total_mass]
        F_gravity_body = torch.tensor([0.0, 0.0, gravity_z * total_mass], device=task.device)
        
        # τ = r_com × F_gravity
        tau = torch.cross(com_offset, F_gravity_body)
        
        print(f"\nEnv {env_id}:")
        print(f"  mass_per_payload: {mass_ea:.4f} kg")
        print(f"  num_attached: {num_attached:.0f}")
        print(f"  payload_mass_total: {payload_mass:.4f} kg")
        print(f"  base_mass: {pm.base_mass[env_id].item():.4f} kg")
        print(f"  total_mass: {total_mass:.4f} kg")
        print(f"  weighted_offset: [{weighted_offset[0]:.4f}, {weighted_offset[1]:.4f}, {weighted_offset[2]:.4f}]")
        print(f"  com_offset: [{com_offset[0]:.6f}, {com_offset[1]:.6f}, {com_offset[2]:.6f}]")
        print(f"  F_gravity_body: [{F_gravity_body[0]:.4f}, {F_gravity_body[1]:.4f}, {F_gravity_body[2]:.4f}]")
        print(f"  tau (pitch): {tau[1]:.6f} Nm")
        
        # 理论值：如果我们只考虑载荷贡献的力矩
        # tau_payload = r_payload × (m_payload * g)
        # 释放 payload0 后，剩余 3 个载荷在 [-0.4, 0], [0, 0.4], [0, -0.4]
        # 对于 pitch (绕 y 轴)，只有 x 分量的偏移有贡献
        # tau_y = offset_x * m * g * cos(0) = offset_x * (m * g)
        # 但这里 offset 是负的 (-0.4)，所以会产生负的 pitch 力矩
        
        # 更直接的理解：
        # 释放了 [0.4, 0, -0.4]，意味着少了一个在 +x 方向的质量
        # COM 会向 -x 偏移，产生向 +pitch 的力矩需求（或负的扰动力矩）
        
        print(f"  ")
        print(f"  ** 检查：稀释效应 **")
        print(f"  raw_weighted_x = {weighted_offset[0]:.6f}")
        print(f"  com_offset_x = weighted_x / total_mass = {weighted_offset[0]:.6f} / {total_mass:.4f} = {com_offset[0]:.6f}")
        
        # 关键：力矩 = com_offset_x * F_z = com_offset_x * (total_mass * g)
        #         = (weighted_x / total_mass) * (total_mass * g)
        #         = weighted_x * g
        # 所以力矩应该只依赖 weighted_offset，不应该被 total_mass 稀释！
        expected_tau = weighted_offset[0] * gravity_z
        print(f"  expected_tau_pitch = weighted_x * g = {weighted_offset[0]:.6f} * {gravity_z} = {expected_tau:.6f}")
        print(f"  actual_tau_pitch = {tau[1]:.6f}")
        print(f"  match: {'✓' if abs(expected_tau - tau[1]) < 1e-4 else '✗'}")
    
    task.close()

if __name__ == "__main__":
    main()
