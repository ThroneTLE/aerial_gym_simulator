"""
Ablation Experiment: Compare trained models in challenging test environments.

Usage:
    python paper/eval_ablation.py --checkpoint runs/ablation_full_18-21-58-41/nn/ablation_full.pth --name Full
    python paper/eval_ablation.py --checkpoint runs/ablation_no_phys_rand_18-21-48-34/nn/ablation_no_phys_rand.pth --name NoPhysRand
"""

import isaacgym
import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
import os
import json
import yaml
import copy
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aerial_gym.task.payload_compensation_task.payload_compensation_task import PayloadCompensationTask
from aerial_gym.config.task_config.payload_compensation_task_teacher_config import task_config

_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def build_teacher_policy(cfg, obs_dim, action_dim, num_envs, device):
    original_path = sys.path.copy()
    sys.path = [p for p in sys.path if "aerial_gym" not in p]
    
    try:
        from rl_games.algos_torch import model_builder
        
        _priv_ac_path = os.path.join(_base_path, "aerial_gym/rl_training/rl_games/nn/privileged_actor_critic.py")
        spec = importlib.util.spec_from_file_location("privileged_actor_critic", _priv_ac_path)
        priv_ac_module = importlib.util.module_from_spec(spec)
        sys.modules["aerial_gym.rl_training.rl_games.nn.privileged_actor_critic"] = priv_ac_module
        spec.loader.exec_module(priv_ac_module)
        
        params = cfg.get("params", {})
        builder = model_builder.ModelBuilder()
        model = builder.load(params)
        model_cfg = {
            "actions_num": action_dim,
            "input_shape": (obs_dim,),
            "num_seqs": num_envs,
            "value_size": 1,
            "normalize_value": bool(params.get("config", {}).get("normalize_value", False)),
            "normalize_input": bool(params.get("config", {}).get("normalize_input", False)),
        }
        network = model.build(model_cfg)
        network.to(device)
        network.eval()
        return network
    finally:
        sys.path = original_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to policy checkpoint")
    parser.add_argument("--pd_only", action="store_true", help="Run with zero compensation (Pure PD baseline)")
    parser.add_argument("--name", type=str, default="Model")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--config", type=str, default="aerial_gym/rl_training/rl_games/ppo_aerial_quad_aux.yaml")
    parser.add_argument("--headless", type=str, default="True", help="Run with or without GUI (True/False)")
    # Test environment parameters
    parser.add_argument("--test_mass", type=float, default=0.03, help="Fixed test payload mass")
    parser.add_argument("--test_wind", type=float, default=0.3, help="Fixed test wind disturbance")
    return parser.parse_args()

def main():
    args = parse_args()
    device = args.device
    headless = args.headless.lower() == "true"
    
    if not args.pd_only and args.checkpoint is None:
        print("Error: --checkpoint is required unless --pd_only is specified.")
        sys.exit(1)

    # 1. Create a modified config class dynamically
    class TestTaskConfig(task_config):
        pass
    
    # Set test environment parameters
    TestTaskConfig.num_envs = args.num_envs
    TestTaskConfig.headless = headless
    TestTaskConfig.device = device
    
    # Override payload parameters - Copy dict first to avoid mutation
    TestTaskConfig.payload_parameters = copy.deepcopy(task_config.payload_parameters)
    TestTaskConfig.payload_parameters["randomize_payload_mass"] = True
    TestTaskConfig.payload_parameters["payload_mass_range"] = [args.test_mass, args.test_mass]
    TestTaskConfig.payload_parameters["release_start"] = 100
    TestTaskConfig.payload_parameters["release_start_range"] = [100, 100]
    
    # Override randomization parameters - Enable wind for challenging test
    TestTaskConfig.randomization_parameters = copy.deepcopy(task_config.randomization_parameters)
    TestTaskConfig.randomization_parameters["randomize_external_disturbance"] = True
    TestTaskConfig.randomization_parameters["external_force_range"] = [args.test_wind, args.test_wind]
    
    print(f"=== Evaluating: {args.name} ===")
    if args.pd_only:
        print("Mode: Pure PD (Zero RL Compensation)")
    else:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Conditions: Mass={args.test_mass}kg, Wind={args.test_wind}N")
    print(f"Num Envs: {args.num_envs}, Steps: {args.num_steps}")
    
    task = PayloadCompensationTask(
        task_config=TestTaskConfig, seed=42, num_envs=args.num_envs, headless=headless, device=device
    )
    
    obs_dim = task.task_config.observation_space_dim
    priv_dim = task.task_config.privileged_observation_space_dim
    action_dim = task.task_config.action_space_dim
    
    # 2. Load Policy (if not PDOnly)
    policy = None
    if not args.pd_only:
        rl_config = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
        ckpt = torch.load(args.checkpoint, map_location=device)
        policy = build_teacher_policy(rl_config, obs_dim, action_dim, args.num_envs, device)
        policy.load_state_dict(ckpt["model"], strict=True)
        
        # Load running stats properly into the model if normalize_input is True
        if getattr(policy, "normalize_input", False) and "running_mean_std" in ckpt:
            print("Loading running_mean_std into policy...")
            policy.running_mean_std.load_state_dict(ckpt["running_mean_std"])

    # 3. Evaluation Loop
    task.reset()
    
    total_rewards = torch.zeros(args.num_envs, device=device)
    crash_count = 0
    success_count = 0
    episode_lengths = []
    
    max_tilt_envs = torch.zeros(args.num_envs, device=device)
    sum_angle_error_envs = torch.zeros(args.num_envs, device=device)
    steps_per_env = torch.zeros(args.num_envs, device=device)
    
    obs_dict = task.get_task_observations()
    
    # Track which envs have finished to avoid double counting
    env_finished = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    
    # Helper for quat to euler
    def get_euler_xyz(q):
        qx, qy, qz, qw = 0, 1, 2, 3
        # roll (x-axis rotation)
        sinr_cosp = 2 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
        cosr_cosp = 1 - 2 * (q[:, qx] * q[:, qx] + q[:, qy] * q[:, qy])
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        # pitch (y-axis rotation)
        sinp = 2 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
        pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * np.pi / 2, torch.asin(sinp))
        # yaw (z-axis rotation)
        siny_cosp = 2 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
        cosy_cosp = 1 - 2 * (q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz])
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return torch.stack([roll, pitch, yaw], dim=1)

    for step in range(args.num_steps):
        # Get policy inputs
        task_obs = task.get_task_observations()
        obs = torch.as_tensor(task_obs["observations"], device=device, dtype=torch.float32)
        priv_obs = task_obs.get("privileged_obs", None)
        if priv_obs is not None:
             priv_obs = torch.as_tensor(priv_obs, device=device, dtype=torch.float32)

        with torch.no_grad():
            if args.pd_only:
                # Zero compensation: Lee controller only
                actions = torch.zeros((args.num_envs, action_dim), device=device)
            else:
                # RL Policy compensation
                input_dict = {"obs": obs, "privileged_obs": priv_obs, "is_train": False}
                result = policy(input_dict)
                actions = torch.clamp(result["mus"], -1.0, 1.0)
        
        task.step(actions)
        
        # Refresh observations
        task_obs = task.get_task_observations() # For next step/rewards/terms
        
        # Access RAW state for metrics from task.obs_dict
        raw_obs = task.obs_dict 
        
        # Get rewards from task_obs (processed)
        rewards = task_obs.get("rewards", torch.zeros(args.num_envs, device=device))
        if isinstance(rewards, np.ndarray):
            rewards = torch.as_tensor(rewards, device=device)
        total_rewards += rewards
        
        # --- Metrics Calculation ---
        # Try to get euler from raw_obs
        if 'robot_euler_angles' in raw_obs:
            euler = raw_obs['robot_euler_angles']
        elif 'robot_orientation' in raw_obs:
            quat = raw_obs['robot_orientation']
            euler = get_euler_xyz(quat)
        else:
            euler = torch.zeros((args.num_envs, 3), device=device)
            
        angle_error = torch.norm(euler[:, 0:2], dim=1) * (180.0 / np.pi) # Degrees
        
        # Only update stats for active envs
        active_mask = ~env_finished
        if active_mask.any():
            sum_angle_error_envs[active_mask] += angle_error[active_mask]
            steps_per_env[active_mask] += 1.0
            max_tilt_envs[active_mask] = torch.max(max_tilt_envs[active_mask], angle_error[active_mask])
        # ---------------------------
        
        # Check termination (from task_obs)
        truncs = task_obs.get("truncations")
        terms = task_obs.get("terminations")
        
        if truncs is not None and terms is not None:
            # Identify newly finished envs
            finished_now = ((truncs > 0) | (terms > 0)) & (~env_finished)
            finished_ids = torch.where(finished_now)[0]
            
            if len(finished_ids) > 0:
                env_finished[finished_ids] = True
                for env_id in finished_ids:
                    if truncs[env_id] > 0:
                        success_count += 1
                    else:
                        crash_count += 1
                    episode_lengths.append(step + 1)

    # Remaining active envs are considered successful
    # But wait, if they didn't finish, are they successful? Yes, surviving 500 steps is success.
    # We already count them? No, we count finished_ids. 
    # If using 'terms' usually implies crash. 'truncs' implies timeout (success).
    # If neither happened, they are still running -> survival -> success.
    remaining_ids = torch.where(~env_finished)[0]
    success_count += len(remaining_ids)

    # Results
    avg_reward = total_rewards.mean().item()
    success_rate = success_count / args.num_envs
    crash_rate = crash_count / args.num_envs
    
    print(f"\n=== Results: {args.name} ===")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Crash Rate: {crash_rate:.2%}")
    print(f"Crash Count: {crash_count} / {args.num_envs}")
    
    # Calculate detailed stats
    avg_angle_error = (sum_angle_error_envs / torch.clamp(steps_per_env, min=1.0)).mean().item()
    max_tilt_overall = max_tilt_envs.max().item()
    print(f"Max Tilt: {max_tilt_overall:.2f} deg")
    print(f"Avg Angle Error: {avg_angle_error:.2f} deg")
    
    # Save result JSON
    result = {
        "name": args.name,
        "checkpoint": args.checkpoint,
        "test_mass": args.test_mass,
        "test_wind": args.test_wind,
        "avg_reward": avg_reward,
        "success_rate": success_rate,
        "crash_rate": crash_rate,
        "crash_count": crash_count,
        "total_envs": args.num_envs,
        "max_tilt": max_tilt_overall,
        "avg_angle_error": avg_angle_error
    }
    print(f"RESULT_JSON:{json.dumps(result)}")
    
    task.close()

if __name__ == "__main__":
    main()
