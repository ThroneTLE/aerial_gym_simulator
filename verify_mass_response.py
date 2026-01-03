#!/usr/bin/env python3
"""验证策略是否响应质量变化 - 简化版本"""

import isaacgym  # noqa
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

CKPT_PATH = "runs/teacher_aux_fixed_imitation_03-22-01-44/nn/teacher_aux_fixed_imitation.pth"

def main():
    print("=" * 60)
    print("验证策略是否响应质量变化")
    print("=" * 60)
    
    device = 'cuda'
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model_state = {k: v.float() for k, v in ckpt['model'].items()}
    
    # 从模型结构可知: actor_mlp 输入 25 维 = 20 (base) + 5 (latent)
    # latent 5 维来自特权编码器处理 7 维特权观测
    
    # 获取特权编码器权重
    priv_enc_0_w = model_state['a2c_network.priv_encoder.0.weight']  # [128, 7]
    priv_enc_0_b = model_state['a2c_network.priv_encoder.0.bias']
    priv_enc_2_w = model_state['a2c_network.priv_encoder.2.weight']  # [128, 128]
    priv_enc_2_b = model_state['a2c_network.priv_encoder.2.bias']
    priv_enc_4_w = model_state['a2c_network.priv_encoder.4.weight']  # [8, 128]
    priv_enc_4_b = model_state['a2c_network.priv_encoder.4.bias']
    
    # 获取 actor MLP 权重
    actor_0_w = model_state['a2c_network.actor_mlp.0.weight']  # [256, 25]
    actor_0_b = model_state['a2c_network.actor_mlp.0.bias']
    actor_2_w = model_state['a2c_network.actor_mlp.2.weight']  # [256, 256]
    actor_2_b = model_state['a2c_network.actor_mlp.2.bias']
    actor_4_w = model_state['a2c_network.actor_mlp.4.weight']  # [128, 256]
    actor_4_b = model_state['a2c_network.actor_mlp.4.bias']
    mu_w = model_state['a2c_network.mu.weight']  # [3, 128]
    mu_b = model_state['a2c_network.mu.bias']
    
    # 获取归一化参数
    obs_mean = model_state['running_mean_std.running_mean']
    obs_var = model_state['running_mean_std.running_var']
    priv_mean = model_state['a2c_network.priv_norm.running_mean']
    priv_var = model_state['a2c_network.priv_norm.running_var']
    
    def forward(base_obs, priv_obs):
        """手动前向传播"""
        # 归一化 base obs
        base_norm = (base_obs - obs_mean) / (obs_var.sqrt() + 1e-5)
        
        # 归一化 priv obs
        priv_norm = (priv_obs - priv_mean) / (priv_var.sqrt() + 1e-5)
        
        # 特权编码器
        h = torch.relu(priv_norm @ priv_enc_0_w.T + priv_enc_0_b)
        h = torch.relu(h @ priv_enc_2_w.T + priv_enc_2_b)
        latent = h @ priv_enc_4_w.T + priv_enc_4_b  # [1, 8]
        
        # 只取前 5 维（aux decoder input）或全 8 维拼接
        # 从 actor_mlp.0.weight 形状 [256, 25] 可知用的是 5 维
        # 检查实际拼接方式...
        
        # 拼接: base (20) + latent[:5] (5) = 25
        combined = torch.cat([base_norm, latent[:, :5]], dim=1)
        
        # Actor MLP
        h = F.elu(combined @ actor_0_w.T + actor_0_b)
        h = F.elu(h @ actor_2_w.T + actor_2_b)
        h = F.elu(h @ actor_4_w.T + actor_4_b)
        mu = h @ mu_w.T + mu_b
        
        return mu
    
    # 创建测试观测
    base_obs = torch.zeros((1, 20), device=device)
    base_obs[0, 0] = 1.0
    base_obs[0, 4] = 1.0
    base_obs[0, 8] = 1.0
    base_obs[0, 12:16] = 1.0
    
    # 测试不同质量
    test_masses = [0.0, 0.04, 0.08, 0.12, 0.16]
    
    print("\n" + "-" * 70)
    print(f"{'质量 (kg)':<12} | {'Thrust':<12} | {'Roll':<12} | {'Pitch':<12}")
    print("-" * 70)
    
    results = []
    for mass in test_masses:
        priv_obs = torch.zeros((1, 7), device=device)
        priv_obs[0, 0] = mass * 10.0
        priv_obs[0, 1:4] = torch.tensor([0.1, 0.0, -0.1], device=device) * 2.5
        priv_obs[0, 4:7] = torch.tensor([0.0008, 0.0008, 0.0016], device=device) * 1000.0
        
        with torch.no_grad():
            mu = forward(base_obs, priv_obs)[0].cpu().numpy()
        
        print(f"{mass:<12.3f} | {mu[0]:<12.4f} | {mu[1]:<12.4f} | {mu[2]:<12.4f}")
        results.append((mass, mu.copy()))
    
    print("-" * 70)
    
    delta = results[-1][1] - results[0][1]
    print(f"\n质量 0→0.16 kg 时变化: Thrust={delta[0]:+.4f}, Roll={delta[1]:+.4f}, Pitch={delta[2]:+.4f}")
    
    if any(abs(d) > 0.02 for d in delta):
        print("✅ 策略对质量变化有响应")
    else:
        print("❌ 策略对质量变化无响应")

if __name__ == "__main__":
    main()
