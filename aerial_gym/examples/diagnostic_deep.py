#!/usr/bin/env python3
"""
深度诊断脚本：分析 aux_decoder 在零载荷时的预测
"""

# IMPORTANT: Import aerial_gym BEFORE torch
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.rl_training.rl_games.nn import privileged_actor_critic  # noqa

import argparse
import numpy as np
import torch
import yaml
from rl_games.algos_torch import model_builder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, 
                        default="aerial_gym/rl_training/rl_games/ppo_aerial_quad_aux.yaml")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    return parser.parse_args()


def load_model(cfg, checkpoint_path, obs_dim, action_dim, num_envs, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
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
    network.load_state_dict(ckpt["model"], strict=True)
    if getattr(network, "normalize_input", False) and "running_mean_std" in ckpt:
        network.running_mean_std.load_state_dict(ckpt["running_mean_std"])
    return network


def run_diagnostic(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    import sys
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        task = task_registry.make_task("payload_compensation_task_teacher", 
                                       num_envs=args.num_envs, headless=True)
    finally:
        sys.argv = original_argv
    
    device = torch.device(task.device)
    obs_dim = task.task_config.observation_space_dim
    action_dim = task.task_config.action_space_dim
    
    model = load_model(cfg, args.checkpoint, obs_dim, action_dim, args.num_envs, device)
    
    task.reset()
    
    # 收集零载荷状态的详细信息
    zero_payload_data = {
        "attached_mask": [],
        "priv_mass": [],      # 特权观测中的 mass
        "aux_pred_all": [],   # aux_decoder 预测的全部输出
        "policy_thrust": [],
        "teacher_thrust": [],
        "base_obs": [],       # 完整的 base 观测
    }
    
    print(f"Running diagnostic: {args.steps} steps, {args.num_envs} envs")
    
    with torch.no_grad():
        for step in range(args.steps):
            obs = torch.as_tensor(task.task_obs["observations"], device=device, dtype=torch.float32)
            priv = task.task_obs.get("privileged_obs", None)
            if priv is not None:
                priv = torch.as_tensor(priv, device=device, dtype=torch.float32)
            
            input_dict = {
                "is_train": False,
                "prev_actions": None,
                "obs": obs,
                "privileged_obs": priv,
                "rnn_states": None,
                "seq_length": 1,
            }
            
            result = model(input_dict)
            action = result["mus"]
            action = torch.clamp(action, -1.0, 1.0)
            
            # 检查 aux_pred
            aux_pred = getattr(model.a2c_network, "_aux_pred", None)
            
            # 检测零载荷状态
            attached_mask = task.payload_manager.attached_mask
            payload_count = attached_mask.sum(dim=1)
            zero_mask = (payload_count == 0)
            
            if zero_mask.sum() > 0:
                zero_payload_data["attached_mask"].extend(
                    attached_mask[zero_mask].cpu().numpy().tolist()
                )
                if priv is not None:
                    zero_payload_data["priv_mass"].extend(
                        priv[zero_mask, 0].cpu().numpy().tolist()  # 第0维是 mass
                    )
                if aux_pred is not None:
                    zero_payload_data["aux_pred_all"].extend(
                        aux_pred[zero_mask].cpu().numpy().tolist()
                    )
                zero_payload_data["policy_thrust"].extend(
                    action[zero_mask, 0].cpu().numpy().tolist()
                )
                zero_payload_data["teacher_thrust"].extend(
                    task.teacher_residual[zero_mask, 0].cpu().numpy().tolist()
                )
                # 采样一些 base_obs
                if len(zero_payload_data["base_obs"]) < 100:
                    zero_payload_data["base_obs"].extend(
                        obs[zero_mask][:5].cpu().numpy().tolist()
                    )
            
            task.step(action)
            
            if step % 500 == 0:
                print(f"Step {step}/{args.steps}, zero_payload samples: {len(zero_payload_data['policy_thrust'])}")
    
    task.close()
    
    # 分析结果
    print("\n" + "="*60)
    print("零载荷状态详细分析")
    print("="*60)
    
    if len(zero_payload_data["priv_mass"]) > 0:
        priv_mass = np.array(zero_payload_data["priv_mass"])
        print(f"\n特权观测中的 mass (归一化后):")
        print(f"  均值: {priv_mass.mean():.6f}")
        print(f"  标准差: {priv_mass.std():.6f}")
        print(f"  范围: [{priv_mass.min():.6f}, {priv_mass.max():.6f}]")
    
    if len(zero_payload_data["aux_pred_all"]) > 0:
        aux_pred = np.array(zero_payload_data["aux_pred_all"])
        print(f"\naux_decoder 预测 (5维: [mass, com_x, com_y, com_z, released_mass]):")
        for i in range(aux_pred.shape[1]):
            print(f"  dim[{i}]: 均值={aux_pred[:, i].mean():.4f}, 标准差={aux_pred[:, i].std():.4f}")
    
    policy_thrust = np.array(zero_payload_data["policy_thrust"])
    teacher_thrust = np.array(zero_payload_data["teacher_thrust"])
    print(f"\n输出对比:")
    print(f"  策略推力: 均值={policy_thrust.mean():.4f}, 标准差={policy_thrust.std():.4f}")
    print(f"  教师推力: 均值={teacher_thrust.mean():.4f}, 标准差={teacher_thrust.std():.4f}")
    print(f"  误差: {(policy_thrust - teacher_thrust).mean():.4f}")
    
    if len(zero_payload_data["base_obs"]) > 0:
        base_obs = np.array(zero_payload_data["base_obs"])
        print(f"\nBase 观测采样 (前3个):")
        for i, obs in enumerate(base_obs[:3]):
            print(f"  [{i}] rot_mat[0:9]: {obs[0:9]}")
            print(f"       angvel[9:12]: {obs[9:12]}")
            print(f"       attached[12:16]: {obs[12:16]}")
            print(f"       warning[16]: {obs[16]}")
            print(f"       prev_actions[17:20]: {obs[17:20]}")


if __name__ == "__main__":
    args = parse_args()
    run_diagnostic(args)
