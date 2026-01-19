#!/usr/bin/env python3
"""Generate combined training curves (Reward + BC Loss)."""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    epochs = [d[1] for d in data]
    values = [d[2] for d in data]
    return epochs, values

def plot_combined():
    base_dir = "paper"
    
    # Load data
    epochs_r, rewards = load_json(f"{base_dir}/teacher_aux_fixed_imitation_17-14-16-10_summaries.json")
    epochs_t, bc_thrust = load_json(f"{base_dir}/teacher_aux_fixed_imitation_17-14-16-10_summaries (1).json")
    epochs_q, bc_torque = load_json(f"{base_dir}/teacher_aux_fixed_imitation_17-14-16-10_summaries (2).json")
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Top: Reward curve
    ax1.plot(epochs_r, rewards, 'b-', linewidth=2, label='Episode Reward')
    ax1.axhline(y=15000, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Max Reward')
    ax1.set_ylabel('Episode Reward', fontsize=12)
    ax1.set_title('Training Convergence: PPO + BC Loss', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 16000)
    
    # Bottom: BC Loss curves
    ax2.semilogy(epochs_t, bc_thrust, 'orange', linewidth=2, label='BC Loss (Thrust)')
    ax2.semilogy(epochs_q, bc_torque, 'green', linewidth=2, label='BC Loss (Torque)')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('BC Loss (log scale)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Add annotation
    ax2.annotate('Policy converges to teacher\n(RL exploration -> BC convergence)', 
                 xy=(340, 0.0003), fontsize=10, 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    os.makedirs("paper/figures", exist_ok=True)
    plt.savefig("paper/figures/fig_training_combined.png", dpi=150, bbox_inches='tight')
    print("Saved: paper/figures/fig_training_combined.png")
    plt.close()

if __name__ == "__main__":
    plot_combined()
