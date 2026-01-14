#!/usr/bin/env python3
"""
调试脚本：验证推理时 privileged_obs 是否正确传递
"""

import sys
import isaacgym
import torch

from aerial_gym.registry.task_registry import task_registry

def main():
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    
    task = task_registry.make_task('payload_compensation_task_teacher', num_envs=4, headless=True)
    sys.argv = original_argv
    
    task.reset()
    
    print("=" * 80)
    print("当前配置的载荷质量范围:")
    print(f"  payload_mass: {task.task_config.payload_parameters['payload_mass']}")
    print(f"  payload_mass_range: {task.task_config.payload_parameters['payload_mass_range']}")
    print(f"  randomize_payload_mass: {task.task_config.payload_parameters['randomize_payload_mass']}")
    print("=" * 80)
    
    dummy = torch.zeros((4, 3), device=task.device)
    
    # 运行几步
    for step in range(50):
        task.step(dummy)
    
    print("\n当前状态:")
    print(f"  payload_mass_per_env: {task.payload_manager.payload_mass_per_env.tolist()}")
    print(f"  current_payload_mass: {task.payload_manager.current_payload_mass.tolist()}")
    print(f"  attached_mask: {task.payload_manager.attached_mask.tolist()}")
    
    print("\n教师残差:")
    print(f"  teacher_residual: {task.teacher_residual.tolist()}")
    
    print("\n privileged_obs:")
    priv = task.task_obs.get('privileged_obs', None)
    if priv is not None:
        print(f"  shape: {priv.shape}")
        print(f"  values env0: {priv[0].tolist()}")
        print(f"  values env1: {priv[1].tolist()}")
    else:
        print("  ⚠️ privileged_obs 为 None!")
    
    # 检查 infos 中的 teacher_actions
    print("\ninfos 中的 teacher_actions:")
    if hasattr(task, 'infos') and 'teacher_actions' in task.infos:
        ta = task.infos['teacher_actions']
        print(f"  shape: {ta.shape}")
        print(f"  values env0: {ta[0].tolist()}")
    else:
        print("  ⚠️ teacher_actions 不在 infos 中!")
    
    task.close()

if __name__ == "__main__":
    main()
