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
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--name", type=str, default="Model")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--config", type=str, default="aerial_gym/rl_training/rl_games/ppo_aerial_quad_aux.yaml")
    # Test environment parameters
    parser.add_argument("--test_mass", type=float, default=0.03, help="Fixed test payload mass")
    parser.add_argument("--test_wind", type=float, default=0.3, help="Fixed test wind disturbance")
    return parser.parse_args()

def main():
    args = parse_args()
    device = args.device
    
    # 1. Create a modified config class dynamically
    class TestTaskConfig(task_config):
        pass
    
    # Set test environment parameters
    TestTaskConfig.num_envs = args.num_envs
    TestTaskConfig.headless = True
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
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Conditions: Mass={args.test_mass}kg, Wind={args.test_wind}N")
    print(f"Num Envs: {args.num_envs}, Steps: {args.num_steps}")
    
    task = PayloadCompensationTask(
        task_config=TestTaskConfig, seed=42, num_envs=args.num_envs, headless=True, device=device
    )
    
    obs_dim = task.task_config.observation_space_dim
    priv_dim = task.task_config.privileged_observation_space_dim
    action_dim = task.task_config.action_space_dim
    
    # 2. Load Policy
    rl_config = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    ckpt = torch.load(args.checkpoint, map_location=device)
    policy = build_teacher_policy(rl_config, obs_dim, action_dim, args.num_envs, device)
    policy.load_state_dict(ckpt["model"], strict=True)
    
    # Load running stats if available
    normalize_obs = lambda x: x
    rms_state = ckpt.get("running_mean_std", None)
    if rms_state:
        # Check both possible key formats
        if "running_mean" in rms_state:
            rms_mean = rms_state["running_mean"].to(device).float()
            rms_var = rms_state["running_var"].to(device).float()
        elif "running_mean_std.running_mean" in rms_state:
            rms_mean = rms_state["running_mean_std.running_mean"].to(device).float()
            rms_var = rms_state["running_mean_std.running_var"].to(device).float()
        else:
            rms_mean = None
            rms_var = None
        
        if rms_mean is not None:
            def normalize_obs(obs):
                return torch.clamp((obs - rms_mean) / torch.sqrt(rms_var + 1e-8), -5.0, 5.0)

    # 3. Evaluation Loop
    task.reset()
    
    total_rewards = torch.zeros(args.num_envs, device=device)
    crash_count = 0
    success_count = 0
    episode_lengths = []
    
    obs_dict = task.get_task_observations()
    
    for step in range(args.num_steps):
        obs = torch.as_tensor(obs_dict["observations"], device=device, dtype=torch.float32)
        priv_obs = obs_dict.get("privileged_obs", None)
        if priv_obs is not None:
             priv_obs = torch.as_tensor(priv_obs, device=device, dtype=torch.float32)

        with torch.no_grad():
            norm_obs = normalize_obs(obs)
            input_dict = {"obs": norm_obs, "privileged_obs": priv_obs, "is_train": False}
            result = policy(input_dict)
            actions = torch.clamp(result["mus"], -1.0, 1.0)
        
        task.step(actions)
        obs_dict = task.get_task_observations()
        
        # Get rewards
        rewards = obs_dict.get("rewards", torch.zeros(args.num_envs, device=device))
        if isinstance(rewards, np.ndarray):
            rewards = torch.as_tensor(rewards, device=device)
        total_rewards += rewards
        
        # Check termination
        truncs = obs_dict.get("truncations")
        terms = obs_dict.get("terminations")
        
        if truncs is not None and terms is not None:
            finished = (truncs > 0) | (terms > 0)
            finished_ids = torch.where(finished)[0]
            
            for env_id in finished_ids:
                if truncs[env_id] > 0:
                    success_count += 1
                else:
                    crash_count += 1
                episode_lengths.append(step + 1)

    # Remaining active envs are considered successful
    remaining = args.num_envs - crash_count - success_count
    success_count += remaining

    # Results
    avg_reward = total_rewards.mean().item()
    success_rate = success_count / args.num_envs
    crash_rate = crash_count / args.num_envs
    
    print(f"\n=== Results: {args.name} ===")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Crash Rate: {crash_rate:.2%}")
    print(f"Crash Count: {crash_count} / {args.num_envs}")
    
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
        "total_envs": args.num_envs
    }
    print(f"RESULT_JSON:{json.dumps(result)}")
    
    task.close()

if __name__ == "__main__":
    main()
