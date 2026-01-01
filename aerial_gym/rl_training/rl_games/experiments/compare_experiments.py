#!/usr/bin/env python3
"""
比较不同 MLP 配置训练结果的验证脚本。
用法：
    python compare_experiments.py                        # 自动查找所有实验
    python compare_experiments.py --runs_dir runs/       # 指定 runs 目录
    python compare_experiments.py --steps 1000           # 评估步数
"""
import argparse
import os
import sys
from pathlib import Path
from glob import glob
import re

# 必须先导入 isaacgym
import isaacgym  # noqa: F401

import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt

# 添加项目路径
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from aerial_gym.task.payload_compensation_task.payload_compensation_task import PayloadCompensationTask
from aerial_gym.config.task_config import payload_compensation_task_teacher_config as teacher_cfg
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.model_builder import ModelBuilder

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def find_experiment_checkpoints(runs_dir: Path, pattern: str = "exp_*"):
    """查找所有实验的 checkpoint"""
    experiments = {}
    for exp_dir in runs_dir.glob(f"{pattern}*"):
        if not exp_dir.is_dir():
            continue
        nn_dir = exp_dir / "nn"
        if not nn_dir.exists():
            continue
        
        # 查找 checkpoint
        ckpts = list(nn_dir.glob("*.pth"))
        if not ckpts:
            continue
        
        # 找最新的
        latest_ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)
        
        # 解析实验名称
        match = re.match(r"exp_(\w+)_\d+", exp_dir.name)
        if match:
            config_name = match.group(1)
        else:
            config_name = exp_dir.name
        
        experiments[config_name] = {
            "dir": exp_dir,
            "ckpt": latest_ckpt,
        }
    
    return experiments


def load_policy(ckpt_path: Path, obs_dim: int = 29, priv_dim: int = 41, action_dim: int = 4,
                mlp_units: list = None, device: str = "cuda:0"):
    """加载策略网络"""
    from aerial_gym.rl_training.rl_games.nn.privileged_actor_critic import PrivilegedA2CBuilder
    
    # 默认 MLP 结构
    if mlp_units is None:
        mlp_units = [256, 256, 128]
    
    network_cfg = {
        "name": "privileged_actor_critic",
        "separate": False,
        "privileged_dim": priv_dim,
        "privileged_embed": 8,
        "privileged_hidden": 128,
        "aux_target_dim": 5,
        "aux_loss_weight": 0.5,
        "space": {
            "continuous": {
                "mu_activation": "None",
                "sigma_activation": "None",
                "mu_init": {"name": "default"},
                "sigma_init": {"name": "const_initializer", "val": -1.0},
                "fixed_sigma": False,
                "min_logstd": -3.0,
                "max_logstd": 1.0,
            }
        },
        "mlp": {
            "units": mlp_units,
            "activation": "elu",
            "initializer": {"name": "default", "scale": 2},
            "d2rl": False,
        },
    }
    
    builder = PrivilegedA2CBuilder()
    model = builder.build("privileged_a2c", **{
        "actions_num": action_dim,
        "input_shape": (obs_dim,),
        "num_seqs": 1,
        "value_size": 1,
        "normalize_value": True,
        "normalize_input": True,
        "network": network_cfg,
    })
    model.to(device)
    
    # 加载权重
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()
    
    return model


def evaluate_policy(model, env, num_steps: int = 1000, device: str = "cuda:0"):
    """评估策略性能"""
    obs, *_ = env.reset()
    
    total_reward = 0.0
    crash_count = 0
    total_episodes = 0
    pos_errors = []
    
    for step in range(num_steps):
        with torch.no_grad():
            # 构造输入
            base_obs = obs[:, :29]  # base observation
            priv_obs = obs[:, 29:]  # privileged observation
            input_dict = {
                "obs": base_obs,
                "privileged_obs": priv_obs,
                "is_train": False,
            }
            action = model(input_dict)["mus"]
        
        obs, rewards, terms, truncs, infos = env.step(action)
        total_reward += rewards.sum().item()
        
        # 记录位置误差
        pos = env.obs_dict["robot_position"]
        target = env.target_position
        pos_err = torch.norm(pos - target, dim=1).mean().item()
        pos_errors.append(pos_err)
        
        # 统计 crash
        crashed = terms.squeeze() > 0.5
        crash_count += crashed.sum().item()
        total_episodes += crashed.sum().item() + (truncs.squeeze() > 0.5).sum().item()
    
    results = {
        "avg_reward": total_reward / num_steps / env.num_envs,
        "crash_rate": crash_count / max(1, total_episodes),
        "avg_pos_error": np.mean(pos_errors),
        "total_crashes": crash_count,
        "total_episodes": total_episodes,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="比较 MLP 实验结果")
    parser.add_argument("--runs_dir", type=str, default="runs", help="实验结果目录")
    parser.add_argument("--steps", type=int, default=1000, help="评估步数")
    parser.add_argument("--num_envs", type=int, default=512, help="评估环境数")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    args = parser.parse_args()
    
    runs_dir = REPO_ROOT / args.runs_dir
    
    print("\n" + "=" * 60)
    print("MLP 架构对比实验 - 结果验证")
    print("=" * 60)
    
    # 查找实验
    experiments = find_experiment_checkpoints(runs_dir)
    
    if not experiments:
        print("[ERROR] 未找到实验结果")
        print(f"        请确保在 {runs_dir} 下有 exp_* 目录")
        return
    
    print(f"\n找到 {len(experiments)} 个实验:")
    for name, exp in experiments.items():
        print(f"  - {name}: {exp['ckpt'].name}")
    
    # 创建测试环境
    print("\n[INFO] 创建测试环境...")
    cfg = teacher_cfg.task_config
    cfg.headless = True
    cfg.device = args.device
    cfg.num_envs = args.num_envs
    env = PayloadCompensationTask(cfg)
    
    # 评估每个实验
    results = {}
    for name, exp in experiments.items():
        print(f"\n[评估] {name}...")
        try:
            # 根据配置名推断 MLP 结构
            mlp_map = {
                "small": [128, 128],
                "medium": [256, 256, 128],
                "large": [512, 256, 128],
                "xlarge": [512, 512, 256],
            }
            mlp_units = mlp_map.get(name, [256, 256, 128])
            
            model = load_policy(exp["ckpt"], mlp_units=mlp_units, device=args.device)
            res = evaluate_policy(model, env, num_steps=args.steps, device=args.device)
            results[name] = res
            print(f"        Reward: {res['avg_reward']:.2f}, Crash: {res['crash_rate']*100:.1f}%, PosErr: {res['avg_pos_error']:.4f}m")
        except Exception as e:
            print(f"        [ERROR] {e}")
            results[name] = None
    
    # 关闭环境
    try:
        env.close()
    except:
        pass
    
    # 打印结果表格
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    print(f"{'配置':<10} {'Reward':>10} {'Crash%':>10} {'PosErr(m)':>12}")
    print("-" * 45)
    
    for name, res in sorted(results.items()):
        if res:
            print(f"{name:<10} {res['avg_reward']:>10.2f} {res['crash_rate']*100:>9.1f}% {res['avg_pos_error']:>12.4f}")
        else:
            print(f"{name:<10} {'ERROR':>10} {'-':>10} {'-':>12}")
    
    # 找最佳
    valid_results = {k: v for k, v in results.items() if v}
    if valid_results:
        best = max(valid_results.items(), key=lambda x: x[1]["avg_reward"])
        print(f"\n🏆 最佳配置: {best[0]} (Reward: {best[1]['avg_reward']:.2f})")


if __name__ == "__main__":
    main()
