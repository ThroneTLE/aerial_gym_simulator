#!/usr/bin/env python3
"""
诊断脚本：分析策略在不同载荷状态下的行为
目标：找出为什么策略在零载荷状态输出 -0.2 而不是 0

实验设计：
1. 收集不同 payload_count (0,1,2,3,4) 下的策略输出和教师输出
2. 分析 aux_decoder 预测的 mass 值
3. 观察策略输入的各个维度
4. 统计分析误差分布
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

# IMPORTANT: Import aerial_gym BEFORE torch to avoid isaacgym import error
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.rl_training.rl_games.nn import privileged_actor_critic  # noqa

import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml

from rl_games.algos_torch import model_builder

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser(description="诊断策略在不同载荷状态下的行为")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint 路径")
    parser.add_argument("--config", type=str, 
                        default="aerial_gym/rl_training/rl_games/ppo_aerial_quad_aux.yaml")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--save_dir", type=str, default="diagnostic_results")
    return parser.parse_args()


def load_model(cfg, checkpoint_path, obs_dim, action_dim, num_envs, device):
    """加载训练好的模型"""
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
    """运行诊断实验"""
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    env_name = cfg.get("params", {}).get("config", {}).get("env_name", "payload_compensation_task_teacher")
    
    # 创建环境
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        task = task_registry.make_task(env_name, num_envs=args.num_envs, headless=True)
    finally:
        sys.argv = original_argv
    
    device = torch.device(task.device)
    obs_dim = task.task_config.observation_space_dim
    action_dim = task.task_config.action_space_dim
    
    # 加载模型
    model = load_model(cfg, args.checkpoint, obs_dim, action_dim, args.num_envs, device)
    
    task.reset()
    
    # 数据收集容器 - 按 payload_count 分组
    data_by_payload_count = defaultdict(lambda: {
        "policy_thrust": [],
        "teacher_thrust": [],
        "aux_pred_mass": [],
        "true_mass": [],
        "attached_mask": [],
        "step": [],
    })
    
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
            action = result["mus"]  # 使用确定性输出
            action = torch.clamp(action, -1.0, 1.0)
            
            # 收集数据
            attached_mask = task.payload_manager.attached_mask  # [N, 4]
            payload_count = attached_mask.sum(dim=1).long()  # [N]
            true_mass = task.payload_manager.current_payload_mass  # [N]
            teacher_residual = task.teacher_residual  # [N, 3]
            
            # 获取 aux_pred (如果模型有这个属性)
            aux_pred = getattr(model.a2c_network, "_aux_pred", None)
            aux_pred_mass = aux_pred[:, 0] if aux_pred is not None else torch.zeros(args.num_envs, device=device)
            
            # 按 payload_count 分组收集
            for pc in range(5):  # 0, 1, 2, 3, 4
                mask = (payload_count == pc)
                if mask.sum() > 0:
                    data_by_payload_count[pc]["policy_thrust"].extend(action[mask, 0].cpu().numpy().tolist())
                    data_by_payload_count[pc]["teacher_thrust"].extend(teacher_residual[mask, 0].cpu().numpy().tolist())
                    data_by_payload_count[pc]["aux_pred_mass"].extend(aux_pred_mass[mask].cpu().numpy().tolist())
                    data_by_payload_count[pc]["true_mass"].extend(true_mass[mask].cpu().numpy().tolist())
                    data_by_payload_count[pc]["step"].extend([step] * mask.sum().item())
            
            # Step environment
            task.step(action)
            
            if step % 500 == 0:
                print(f"Step {step}/{args.steps}")
    
    task.close()
    
    # 分析和可视化
    analyze_results(data_by_payload_count, args.save_dir)


def analyze_results(data_by_payload_count, save_dir):
    """分析和可视化结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("诊断结果分析")
    print("="*60)
    
    # 统计表格
    print("\n按 payload_count 分组的统计:")
    print("-"*80)
    print(f"{'Count':>6} | {'Samples':>8} | {'Policy μ':>10} | {'Policy σ':>10} | {'Teacher μ':>10} | {'Error μ':>10}")
    print("-"*80)
    
    results = {}
    for pc in range(5):
        data = data_by_payload_count[pc]
        if len(data["policy_thrust"]) == 0:
            continue
        
        policy_arr = np.array(data["policy_thrust"])
        teacher_arr = np.array(data["teacher_thrust"])
        error_arr = policy_arr - teacher_arr
        
        results[pc] = {
            "n_samples": len(policy_arr),
            "policy_mean": policy_arr.mean(),
            "policy_std": policy_arr.std(),
            "teacher_mean": teacher_arr.mean(),
            "teacher_std": teacher_arr.std(),
            "error_mean": error_arr.mean(),
            "error_std": error_arr.std(),
        }
        
        print(f"{pc:>6} | {results[pc]['n_samples']:>8} | {results[pc]['policy_mean']:>10.4f} | "
              f"{results[pc]['policy_std']:>10.4f} | {results[pc]['teacher_mean']:>10.4f} | "
              f"{results[pc]['error_mean']:>10.4f}")
    
    print("-"*80)
    
    # 关键发现
    print("\n关键发现:")
    if 0 in results:
        r0 = results[0]
        print(f"  [零载荷状态 (payload_count=0)]")
        print(f"    - 策略输出均值: {r0['policy_mean']:.4f}")
        print(f"    - 教师输出均值: {r0['teacher_mean']:.4f}")
        print(f"    - 误差: {r0['error_mean']:.4f}")
        if abs(r0['error_mean']) > 0.1:
            print(f"    ⚠️  误差显著！策略在零载荷时输出偏离教师 {r0['error_mean']:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 按 payload_count 的输出分布
    ax = axes[0, 0]
    positions = []
    labels = []
    for pc in range(5):
        if pc in results:
            positions.append(pc)
            labels.append(f"{pc}")
    if positions:
        policy_data = [data_by_payload_count[pc]["policy_thrust"] for pc in positions]
        teacher_data = [data_by_payload_count[pc]["teacher_thrust"] for pc in positions]
        
        bp1 = ax.boxplot(policy_data, positions=np.array(positions) - 0.15, widths=0.25, 
                         patch_artist=True, boxprops=dict(facecolor='C0', alpha=0.6))
        bp2 = ax.boxplot(teacher_data, positions=np.array(positions) + 0.15, widths=0.25,
                         patch_artist=True, boxprops=dict(facecolor='C1', alpha=0.6))
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Payload Count")
        ax.set_ylabel("Thrust Compensation")
        ax.set_title("策略 vs 教师输出分布 (按载荷数量)")
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Policy", "Teacher"])
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 2. 误差分布
    ax = axes[0, 1]
    for pc in range(5):
        if pc in results and len(data_by_payload_count[pc]["policy_thrust"]) > 100:
            error = np.array(data_by_payload_count[pc]["policy_thrust"]) - np.array(data_by_payload_count[pc]["teacher_thrust"])
            ax.hist(error, bins=50, alpha=0.5, label=f"count={pc}")
    ax.set_xlabel("Error (Policy - Teacher)")
    ax.set_ylabel("Frequency")
    ax.set_title("误差分布 (按载荷数量)")
    ax.legend()
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # 3. aux_pred_mass vs true_mass (零载荷)
    ax = axes[1, 0]
    if 0 in results and len(data_by_payload_count[0]["aux_pred_mass"]) > 0:
        aux_pred = np.array(data_by_payload_count[0]["aux_pred_mass"])
        true_mass = np.array(data_by_payload_count[0]["true_mass"])
        ax.scatter(true_mass, aux_pred, alpha=0.3, s=5)
        ax.plot([0, true_mass.max()], [0, true_mass.max()], 'r--', label='Perfect')
        ax.set_xlabel("True Mass")
        ax.set_ylabel("Predicted Mass (aux_pred)")
        ax.set_title("零载荷状态: aux_decoder 预测 vs 真实")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No zero-payload data", ha='center', va='center', transform=ax.transAxes)
    
    # 4. 策略输出 vs 载荷数量 (均值+标准差)
    ax = axes[1, 1]
    pcs = sorted(results.keys())
    policy_means = [results[pc]["policy_mean"] for pc in pcs]
    policy_stds = [results[pc]["policy_std"] for pc in pcs]
    teacher_means = [results[pc]["teacher_mean"] for pc in pcs]
    teacher_stds = [results[pc]["teacher_std"] for pc in pcs]
    
    x = np.arange(len(pcs))
    ax.errorbar(x - 0.1, policy_means, yerr=policy_stds, fmt='o-', capsize=5, label="Policy", color='C0')
    ax.errorbar(x + 0.1, teacher_means, yerr=teacher_stds, fmt='s-', capsize=5, label="Teacher", color='C1')
    ax.set_xticks(x)
    ax.set_xticklabels(pcs)
    ax.set_xlabel("Payload Count")
    ax.set_ylabel("Thrust Compensation")
    ax.set_title("均值 ± 标准差")
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "diagnostic_results.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n图像已保存到: {save_path}")
    
    # 保存数据
    import json
    data_path = os.path.join(save_dir, "diagnostic_data.json")
    with open(data_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"数据已保存到: {data_path}")
    
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    run_diagnostic(args)
