#!/usr/bin/env python3
"""
Verify Teacher Behavior:
Check if the Teacher (optimal controller) actually changes its output based on payload mass.
"""

import isaacgym
import torch
import numpy as np
import yaml
from aerial_gym.registry.task_registry import task_registry

def main():
    print("=" * 80)
    print("Verifying Teacher Action Sensitivity to Payload Mass")
    print("=" * 80)

    # 1. Load config
    config_path = "aerial_gym/rl_training/rl_games/ppo_aerial_quad_aux.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Setup environment with distinct masses
    # We will use a small num_envs but force different masses if possible,
    # or just observe random ones.
    # Ideally we'd force mass, but let's just observe 100 envs and check correlation.
    
    env_config = config['params']['config']['env_config']
    env_config['num_envs'] = 16
    env_config['headless'] = True
    
    task = task_registry.make_task("payload_compensation_task_teacher", **env_config)
    env = task
    
    print("\nEnvironment created.")
    
    # 3. Reset and step
    obs = env.reset()
    
    # Run a few steps to let things settle (though teacher is instantaneous usually)
    num_envs = task.sim_env.num_envs
    for _ in range(10):
        # random actions for the robot (policy) - doesn't matter for teacher calculation usually
        # but teacher depends on state.
        actions = torch.zeros(num_envs, 4, device=env.device)
        try:
            obs, rew, done, info = env.step(actions)
        except RuntimeError as e:
            print(f"Caught expected runtime error (dim mismatch): {e}")
            pass # Continue to analyze teacher_residual which should be computed by now
        
    # 4. Capture Data
    print("\nCapturing Teacher Actions...")
    
    # Teacher actions are in task.teacher_residual or info['teacher_actions']
    # Privileged obs (mass) is in task.task_obs['priv_obs'] (or similar, need to verify access)
    
    # Let's peek into task implementation details
    # task.payload_manager.payload_masses -> The actual masses
    # task.teacher_residual -> The calculated teacher output
    
    # Introspect to find mass
    # print(f"PayloadManager attributes: {dir(task.payload_manager)}")
    if hasattr(task.payload_manager, 'payload_mass_per_env'):
        masses = task.payload_manager.payload_mass_per_env.flatten().cpu().numpy()
        print(f"Retrieved masses from payload_manager.payload_mass_per_env. Values: {masses[:5]}")
    elif hasattr(task.payload_manager, '_payload_masses'):
         masses = task.payload_manager._payload_masses.flatten().cpu().numpy()
    else:
         print("Using placeholder masses.")
         masses = np.zeros(num_envs)

    teacher_actions = task.teacher_residual.cpu().numpy()
    
    # teacher_actions: [num_envs, 3] (thrust, roll_torque, pitch_torque) assuming simplified
    # Let's check shapes
    print(f"Masses shape: {masses.shape}")
    print(f"Teacher Actions shape: {teacher_actions.shape}")
    
    # 5. Analyze Correlation
    # We focus on Thrust (idx 0) vs Mass
    thrust_cmds = teacher_actions[:, 0]
    
    print("\nSample Data (First 10 envs):")
    print(f"{'Mass (PrivObs)':<15} {'Thrust Cmd (Norm)':<20} {'Ratio':<25}")
    print("-" * 60)
    
    for i in range(min(10, num_envs)):
        # Just print raw values
        ratio = thrust_cmds[i] / masses[i] if abs(masses[i]) > 1e-5 else 0
        print(f"{masses[i]:<15.4f} {thrust_cmds[i]:<20.4f} {ratio:<25.4f}")
        
    # Correlation metrics
    correlation = np.corrcoef(masses, thrust_cmds)[0, 1]
    
    print("-" * 60)
    print(f"Correlation (Mass vs Thrust Cmd): {correlation:.4f}")
    
    mass_std = np.std(masses)
    thrust_std = np.std(thrust_cmds)
    
    print(f"Mass Std: {mass_std:.4f}")
    print(f"Thrust Cmd Std: {thrust_std:.4f}")
    
    print("=" * 80)
    if thrust_std < 0.001:
        print("❌ TEACHER IS BROKEN! Outputs are potentially constant.")
        print("   This explains why the policy learns a constant.")
    elif np.isnan(correlation):
        print("⚠️ Correlation is NaN (Constant input or output).")
    elif correlation < 0.9:
        print("⚠️ Correlation is low. Teacher might be noisy or logic is complex.")
    else:
        print("✅ Teacher looks healthy. Outputs vary linearly with mass.")
        print("   The issue MUST be in the network learning (gradient flow).")
    print("=" * 80)

if __name__ == "__main__":
    main()
