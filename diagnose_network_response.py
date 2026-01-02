#!/usr/bin/env python3
"""
简化诊断：直接用 new_my_position_control2.py 的加载方式测试网络响应
"""

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import isaacgym
import torch
import numpy as np
import yaml

from rl_games.algos_torch import model_builder
from aerial_gym.rl_training.rl_games.nn import privileged_actor_critic

def main():
    ckpt_path = "runs/teacher_scaled_10x_02-22-44-03/nn/teacher_scaled_10x.pth"
    config_path = "aerial_gym/rl_training/rl_games/ppo_aerial_quad_aux.yaml"
    
    print("=" * 80)
    print("诊断：网络对 privileged_obs 的响应")
    print("=" * 80)
    
    # 加载配置和 checkpoint
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # 构建模型
    params = cfg.get("params", {})
    builder = model_builder.ModelBuilder()
    model = builder.load(params)
    
    obs_dim = 20
    action_dim = 3
    num_envs = 4
    device = torch.device("cuda:0")
    
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
    network.load_state_dict(ckpt["model"])
    
    print(f"已加载模型: {ckpt_path}")
    
    # 创建固定的 base observation
    base_obs = torch.zeros(num_envs, obs_dim, device=device)
    base_obs[:, 0] = 1.0  # R[0,0]
    base_obs[:, 4] = 1.0  # R[1,1]
    base_obs[:, 8] = 1.0  # R[2,2]
    
    # 测试不同的 privileged_obs (7维: mass, com_x, com_y, com_z, Ixx, Iyy, Izz)
    # IMPORTANT: 训练时的 payload_mass_range 是 [0.01, 0.04] kg!
    # 我们应该在这个范围内测试，而不是超出分布的 0~0.16
    test_cases = [
        ("质量0.01kg", torch.tensor([[0.01, 0, 0, 0, 0.4, 0.4, 0.8]])),  # min
        ("质量0.025kg", torch.tensor([[0.025, 0, 0, 0, 0.4, 0.4, 0.8]])), # mid
        ("质量0.04kg", torch.tensor([[0.04, 0, 0, 0, 0.4, 0.4, 0.8]])),   # max
        ("偏心X+", torch.tensor([[0.025, 0.05, 0, 0, 0.4, 0.4, 0.8]])),
        ("偏心X-", torch.tensor([[0.025, -0.05, 0, 0, 0.4, 0.4, 0.8]])),
    ]
    
    print("\n测试不同 privileged_obs 下的网络输出：")
    print(f"{'条件':<15} {'thrust':>10} {'roll':>10} {'pitch':>10}")
    print("-" * 50)
    
    results = []
    with torch.no_grad():
        for name, priv_obs in test_cases:
            priv_obs = priv_obs.expand(num_envs, -1).to(device)
            input_dict = {
                'is_train': False,
                'obs': base_obs,
                'privileged_obs': priv_obs,
            }
            output = network(input_dict)
            mu = output['mus'][0].cpu().numpy()
            results.append((name, mu))
            print(f"{name:<15} {mu[0]:>10.4f} {mu[1]:>10.4f} {mu[2]:>10.4f}")
    
    # 分析结果
    print("\n" + "=" * 80)
    output_values = [r[1] for r in results]
    output_array = np.array(output_values)
    std_per_dim = output_array.std(axis=0)
    
    print(f"输出标准差 (跨不同privileged_obs):")
    print(f"  thrust: {std_per_dim[0]:.6f}")
    print(f"  roll:   {std_per_dim[1]:.6f}")  
    print(f"  pitch:  {std_per_dim[2]:.6f}")
    
    # 特别检查质量差异
    zero_mass = results[0][1]
    large_mass = results[2][1]
    print(f"\n零质量 vs 大质量(0.16) 的差异:")
    print(f"  thrust 差: {abs(large_mass[0] - zero_mass[0]):.6f}")
    print(f"  roll 差:   {abs(large_mass[1] - zero_mass[1]):.6f}")
    print(f"  pitch 差:  {abs(large_mass[2] - zero_mass[2]):.6f}")
    
    if std_per_dim.max() < 0.01:
        print("\n❌ 网络输出对 privileged_obs 几乎没有响应！")
        print("   这说明 priv_encoder 的梯度没有流向 actor 网络")
    else:
        print("\n✅ 网络输出随 privileged_obs 变化！")
    print("=" * 80)

if __name__ == "__main__":
    main()
