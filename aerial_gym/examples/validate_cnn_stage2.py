"""
Stage 2 CNN Student Validation - Enhanced Visualization.

This script validates the CNN student encoder against the teacher's privileged
encoder and produces comprehensive comparison plots showing:
1. Flight state: Z-height, Euler angles, XY trajectory
2. 8-dim latent comparison: Teacher vs CNN per dimension
3. Error metrics: MSE and Cosine similarity over time
4. Payload release event markers

Usage:
    python aerial_gym/examples/validate_cnn_stage2.py \
        --teacher_checkpoint runs/<teacher_run>/nn/best_*.pth \
        --cnn_checkpoint runs/cnn_student_*/nn/best_cnn_encoder.pth \
        --steps 1500 \
        --show_plot True
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import importlib.util

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import yaml

from aerial_gym.registry.task_registry import task_registry
from aerial_gym.utils.math import get_euler_xyz_tensor

# Ensure privileged network is registered before building the model.
from aerial_gym.rl_training.rl_games.nn import privileged_actor_critic  # noqa: F401
from rl_games.algos_torch import model_builder
import torch
import torch.nn as nn

DEFAULT_ENV_NAME = "payload_compensation_task_teacher"
DEFAULT_CONFIG = "aerial_gym/rl_training/rl_games/ppo_aerial_quad_aux.yaml"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1"):
        return True
    if value in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Unable to parse bool: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 CNN Student Validation with Enhanced Visualization."
    )
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel envs for statistical validation.")
    parser.add_argument("--steps", type=int, default=1500, help="Number of simulation steps.")
    parser.add_argument(
        "--headless",
        type=_str2bool,
        nargs="?",
        const=False,
        default=False,
        help="Headless mode (True/False).",
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        required=True,
        help="Path to teacher RL-Games .pth checkpoint.",
    )
    parser.add_argument(
        "--cnn_checkpoint",
        type=str,
        required=True,
        help="Path to trained CNN student encoder checkpoint.",
    )
    parser.add_argument(
        "--history_len",
        type=int,
        default=50,
        help="Observation history length for CNN.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Training YAML config path.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default=None,
        help="Override env_name.",
    )
    parser.add_argument(
        "--deterministic",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Use mean action instead of sampling.",
    )
    parser.add_argument(
        "--show_plot",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Show plot window after run.",
    )
    parser.add_argument(
        "--save_plot",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Save plots to file.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Save path for plots.",
    )
    parser.add_argument(
        "--preflight_steps",
        type=int,
        default=0,
        help="Number of random waypoint preflight steps before release phase (0 to disable).",
    )
    parser.add_argument(
        "--preflight_waypoint_range",
        type=float,
        default=0.5,
        help="Range for random waypoint targets during preflight (meters).",
    )
    parser.add_argument(
        "--preflight_waypoint_interval",
        type=int,
        default=100,
        help="Steps between waypoint changes during preflight.",
    )
    return parser.parse_args()


def load_training_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path: str) -> Tuple[Dict[str, Any], int]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_state = ckpt.get("model", {})
    action_tensor = model_state.get("a2c_network.mu.weight")
    if action_tensor is None:
        raise RuntimeError("Checkpoint missing a2c_network.mu.weight, cannot infer action_dim.")
    action_dim = int(action_tensor.shape[0])
    return ckpt, action_dim


def build_model(
    cfg: Dict[str, Any],
    obs_dim: int,
    action_dim: int,
    num_envs: int,
    device: torch.device,
):
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


def load_teacher_encoder(checkpoint_path: str, device: str, priv_dim: int = 7):
    """Load teacher's privileged encoder with auto-detection and RunningObsNorm support."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint.get("model", checkpoint)
    
    priv_encoder_state = {}
    priv_norm_running_mean = None
    priv_norm_running_var = None
    priv_norm_mean = None
    priv_norm_std = None
    
    for key, value in model_state.items():
        if "a2c_network.priv_encoder" in key:
            local_key = key.replace("a2c_network.priv_encoder.", "")
            priv_encoder_state[local_key] = value
        elif "a2c_network.priv_norm.running_mean" in key:
            priv_norm_running_mean = value
        elif "a2c_network.priv_norm.running_var" in key:
            priv_norm_running_var = value
        elif "a2c_network.priv_norm.mean" in key:
            priv_norm_mean = value
        elif "a2c_network.priv_norm.std" in key:
            priv_norm_std = value
    
    if not priv_encoder_state:
        raise RuntimeError("No priv_encoder weights found in checkpoint")
    
    # Auto-detect priv_dim
    first_layer_key = "0.weight"
    if first_layer_key in priv_encoder_state:
        detected = priv_encoder_state[first_layer_key].shape[1]
        if detected != priv_dim:
            print(f"  [Auto-detect] priv_dim: {detected}")
            priv_dim = detected
    
    hidden, embed_dim = 128, 8
    use_running = priv_norm_running_mean is not None
    
    class TeacherEncoder(nn.Module):
        def __init__(self, priv_dim, hidden, embed_dim, running_mean, running_var, fixed_mean, fixed_std, use_run):
            super().__init__()
            self.eps, self.clip_range = 1e-6, 10.0
            if use_run and running_mean is not None:
                self.register_buffer("running_mean", running_mean.float())
                self.register_buffer("running_var", running_var.float() if running_var is not None else torch.ones(priv_dim))
            elif fixed_mean is not None:
                self.register_buffer("running_mean", fixed_mean.float())
                self.register_buffer("running_var", (fixed_std.float() if fixed_std is not None else torch.ones(priv_dim)) ** 2)
            else:
                self.register_buffer("running_mean", torch.zeros(priv_dim))
                self.register_buffer("running_var", torch.ones(priv_dim))
            self.encoder = nn.Sequential(
                nn.Linear(priv_dim, hidden), nn.ELU(),
                nn.Linear(hidden, hidden), nn.ELU(),
                nn.Linear(hidden, embed_dim),
            )
        
        def forward(self, x):
            std = torch.sqrt(self.running_var + self.eps)
            normalized = torch.clamp((x - self.running_mean) / std, -self.clip_range, self.clip_range)
            return self.encoder(normalized)
    
    encoder = TeacherEncoder(priv_dim, hidden, embed_dim, priv_norm_running_mean, priv_norm_running_var,
                             priv_norm_mean, priv_norm_std, use_running).to(device)
    encoder.encoder.load_state_dict(priv_encoder_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"Loaded teacher encoder: {priv_dim} -> {embed_dim}")
    return encoder


def load_cnn_encoder(checkpoint_path: str, obs_dim: int, history_len: int, latent_dim: int, device: str):
    """Load trained CNN student encoder (auto-detects standard vs attention version)."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cnn_encoder_path = os.path.join(base_path, "rl_training/rl_games/nn/cnn_student_encoder.py")
    
    spec = importlib.util.spec_from_file_location("cnn_student_encoder", cnn_encoder_path)
    cnn_module = importlib.util.module_from_spec(spec)
    sys.modules["cnn_student_encoder"] = cnn_module
    spec.loader.exec_module(cnn_module)
    
    CNNStudentEncoder = cnn_module.CNNStudentEncoder
    CNNWithTemporalAttention = cnn_module.CNNWithTemporalAttention
    ObsHistoryBuffer = cnn_module.ObsHistoryBuffer
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Auto-detect encoder type from checkpoint keys
    has_attention = any("temporal_attention" in k for k in state_dict.keys())
    
    if has_attention:
        print("Detected CNN with Temporal Attention")
        encoder = CNNWithTemporalAttention(obs_dim=obs_dim, history_len=history_len, latent_dim=latent_dim).to(device)
    else:
        print("Detected standard CNN encoder")
        encoder = CNNStudentEncoder(obs_dim=obs_dim, history_len=history_len, latent_dim=latent_dim).to(device)
    
    encoder.load_state_dict(state_dict)
    encoder.eval()
    
    buffer = ObsHistoryBuffer(num_envs=1, obs_dim=obs_dim, history_len=history_len, device=device)
    return encoder, buffer, ObsHistoryBuffer


def _to_device(states, device):
    if states is None:
        return None
    if isinstance(states, (list, tuple)):
        return tuple(s.to(device) for s in states)
    return states.to(device)


def _reset_rnn_states(states, env_ids):
    if states is None or env_ids is None or (hasattr(env_ids, 'numel') and env_ids.numel() == 0):
        return states
    if not torch.is_tensor(env_ids):
        env_ids = torch.as_tensor(env_ids, device=states[0].device if isinstance(states, (list, tuple)) else states.device)
    
    def _zero_state(state):
        if state is None:
            return state
        if state.dim() >= 2:
            state[:, env_ids, :] = 0.0
        else:
            state.zero_()
        return state
    
    if isinstance(states, (list, tuple)):
        return tuple(_zero_state(s) for s in states)
    return _zero_state(states)


def plot_comprehensive_results(
    z_history: List[float],
    euler_history: List[np.ndarray],
    release_history: List[Tuple[int, int]],
    policy_actions: List[np.ndarray],
    teacher_actions: List[np.ndarray],
    teacher_latents: np.ndarray,
    cnn_latents: np.ndarray,
    pos_history: List[np.ndarray],
    save_path: Optional[str] = None,
    num_envs: int = 1,
):
    """Generate comprehensive comparison plots for CNN vs Teacher latents."""
    # Reshape latents if num_envs > 1: (steps*num_envs, latent_dim) -> (steps, num_envs, latent_dim)
    if num_envs > 1:
        steps_count = teacher_latents.shape[0] // num_envs
        teacher_all = teacher_latents.reshape(steps_count, num_envs, -1)
        cnn_all = cnn_latents.reshape(steps_count, num_envs, -1)
        
        # Use Env 0 for time series plotting
        teacher_plot = teacher_all[:, 0, :]
        cnn_plot = cnn_all[:, 0, :]
        
        # Compute stats across ALL environments (flattened)
        # teacher_latents/cnn_latents are already flattened, so we can use them directly for global stats
    else:
        teacher_plot = teacher_latents
        cnn_plot = cnn_latents
        
    steps = np.arange(len(z_history))
    
    # === Figure 1: Flight State ===
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Z-height
    axes1[0].plot(steps, z_history, 'b-', linewidth=1.5, label="高度 Z")
    for rs, pi in release_history:
        axes1[0].axvline(rs, color='r', linestyle='--', alpha=0.6)
        axes1[0].text(rs, max(z_history), f"释放{pi}", color='r', fontsize=8, rotation=90, va='top')
    axes1[0].set_ylabel("Z (m)")
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)
    axes1[0].set_title("飞行高度变化")
    
    # Euler angles - normalize to ±180° for better visualization
    eulers = np.rad2deg(np.array(euler_history))
    # Wrap to [-180, 180]
    eulers = ((eulers + 180) % 360) - 180
    axes1[1].plot(steps, eulers[:, 0], label="Roll", alpha=0.8)
    axes1[1].plot(steps, eulers[:, 1], label="Pitch", alpha=0.8)
    axes1[1].plot(steps, eulers[:, 2], label="Yaw", alpha=0.8)
    axes1[1].axhline(0, color='k', linestyle='-', linewidth=1.0, alpha=0.6, label="基准线 0°")
    axes1[1].set_ylim(-30, 30)  # Reasonable range for small angle variations
    for rs, _ in release_history:
        axes1[1].axvline(rs, color='r', linestyle='--', alpha=0.4)
    axes1[1].set_ylabel("角度 (°)")
    axes1[1].legend()
    axes1[1].grid(True, alpha=0.3)
    axes1[1].set_title("姿态角变化")
    
    # XY position
    pos_arr = np.vstack(pos_history)
    xy_dist = np.linalg.norm(pos_arr[:, :2], axis=1)
    axes1[2].plot(steps, xy_dist, 'g-', linewidth=1.5, label="|XY| 平移距离")
    for rs, _ in release_history:
        axes1[2].axvline(rs, color='r', linestyle='--', alpha=0.4)
    axes1[2].set_xlabel("步数")
    axes1[2].set_ylabel("距离 (m)")
    axes1[2].legend()
    axes1[2].grid(True, alpha=0.3)
    axes1[2].set_title("XY平面平移距离")
    
    fig1.suptitle("飞行状态监测", fontsize=14, fontweight='bold')
    fig1.tight_layout()
    
    # === Figure 2: Latent Comparison (8 dims) ===
    latent_dim = teacher_plot.shape[1]
    fig2, axes2 = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
    axes2 = axes2.ravel()
    
    for i in range(latent_dim):
        ax = axes2[i]
        ax.plot(steps, teacher_plot[:, i], 'b-', linewidth=1.2, label="Teacher", alpha=0.9)
        ax.plot(steps, cnn_plot[:, i], 'r--', linewidth=1.2, label="CNN", alpha=0.9)
        for rs, _ in release_history:
            ax.axvline(rs, color='g', linestyle=':', alpha=0.5)
        ax.set_ylabel(f"z[{i}]")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right')
    
    axes2[-2].set_xlabel("步数")
    axes2[-1].set_xlabel("步数")
    fig2.suptitle("潜变量对比: Teacher (蓝) vs CNN (红)", fontsize=14, fontweight='bold')
    fig2.tight_layout()
    
    # === Figure 3: Error Metrics ===
    fig3, axes3 = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    
    # Per-step MSE
    # Per-step MSE (using Env 0 for plot visualization)
    # Note: Global usage MSE is printed, but for plot we show Env 0 time series
    mse_plot = np.mean((teacher_plot - cnn_plot) ** 2, axis=1)
    
    # Global MSE for title (using all envs)
    global_mse = np.mean((teacher_latents - cnn_latents) ** 2)
    
    axes3[0].plot(steps, mse_plot, 'C2-', linewidth=1.5)
    axes3[0].fill_between(steps, 0, mse_plot, alpha=0.3, color='C2')
    for rs, _ in release_history:
        axes3[0].axvline(rs, color='r', linestyle='--', alpha=0.4)
    axes3[0].set_ylabel("MSE (Env 0)")
    axes3[0].set_title(f"逐步 MSE (Env 0) | 全局平均 MSE: {global_mse:.6f}")
    axes3[0].grid(True, alpha=0.3)
    
    # Cosine similarity (Env 0 for plot)
    t_norm_plot = np.linalg.norm(teacher_plot, axis=1, keepdims=True)
    c_norm_plot = np.linalg.norm(cnn_plot, axis=1, keepdims=True)
    cos_sim_plot = np.sum(teacher_plot * cnn_plot, axis=1) / (t_norm_plot.squeeze() * c_norm_plot.squeeze() + 1e-8)
    
    # Global Cosine Sim
    t_norm_all = np.linalg.norm(teacher_latents, axis=1, keepdims=True)
    c_norm_all = np.linalg.norm(cnn_latents, axis=1, keepdims=True)
    cos_sim_all = np.sum(teacher_latents * cnn_latents, axis=1) / (t_norm_all.squeeze() * c_norm_all.squeeze() + 1e-8)
    global_cos_sim = np.mean(cos_sim_all)
    
    axes3[1].plot(steps, cos_sim_plot, 'C3-', linewidth=1.5)
    axes3[1].axhline(1.0, color='k', linestyle='--', alpha=0.3)
    axes3[1].axhline(0.9, color='g', linestyle=':', alpha=0.5, label="阈值 0.9")
    for rs, _ in release_history:
        axes3[1].axvline(rs, color='r', linestyle='--', alpha=0.4)
    axes3[1].set_ylabel("Cosine Sim (Env 0)")
    axes3[1].set_ylim(-0.1, 1.1)
    axes3[1].set_title(f"余弦相似度 (Env 0) | 全局平均: {global_cos_sim:.4f}")
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)
    
    # Per-dimension contribution to error
    dim_mse = np.mean((teacher_latents - cnn_latents) ** 2, axis=0)
    bar_colors = plt.cm.viridis(np.linspace(0.2, 0.8, latent_dim))
    axes3[2].bar(range(latent_dim), dim_mse, color=bar_colors)
    axes3[2].set_xlabel("潜变量维度")
    axes3[2].set_ylabel("平均 MSE")
    axes3[2].set_title("各维度平均误差")
    axes3[2].set_xticks(range(latent_dim))
    axes3[2].set_xticklabels([f"z[{i}]" for i in range(latent_dim)], rotation=0)
    axes3[2].grid(True, alpha=0.3, axis='y')
    
    fig3.suptitle("潜变量拟合误差分析", fontsize=14, fontweight='bold')
    fig3.tight_layout(rect=[0, 0.02, 1, 0.98])
    
    # === Figure 4: Action Comparison ===
    if policy_actions and teacher_actions:
        act_arr = np.vstack(policy_actions)
        teacher_arr = np.vstack(teacher_actions) if teacher_actions else None
        action_dim = act_arr.shape[1]
        
        fig4, axes4 = plt.subplots(action_dim, 1, figsize=(14, 2.5 * action_dim), sharex=True)
        if action_dim == 1:
            axes4 = [axes4]
        
        labels = ["Thrust"] + [f"Torque {['X', 'Y', 'Z'][i-1]}" for i in range(1, action_dim)]
        for i, ax in enumerate(axes4):
            ax.plot(steps, act_arr[:, i], 'b-', linewidth=1, label="Policy", alpha=0.8)
            if teacher_arr is not None:
                ax.plot(steps, teacher_arr[:, i], 'r--', linewidth=1, label="Teacher", alpha=0.8)
            for rs, _ in release_history:
                ax.axvline(rs, color='g', linestyle=':', alpha=0.5)
            ax.set_ylabel(labels[i])
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
        
        axes4[-1].set_xlabel("步数")
        fig4.suptitle("残差补偿动作对比", fontsize=14, fontweight='bold')
        fig4.tight_layout()
    else:
        fig4 = None
    
    # Print summary statistics
    # Calculate per-dimension MSE on global data
    dim_mse = np.mean((teacher_latents - cnn_latents) ** 2, axis=0)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("潜变量拟合统计")
    print("=" * 60)
    print(f"平均 MSE: {global_mse:.6f}")
    print(f"最大 MSE (Global): {np.max(np.mean((teacher_latents - cnn_latents) ** 2, axis=1)):.6f}")
    print(f"平均 Cosine Similarity: {global_cos_sim:.4f}")
    print(f"最小 Cosine Similarity (Global): {np.min(np.sum(teacher_latents * cnn_latents, axis=1) / (np.linalg.norm(teacher_latents, axis=1) * np.linalg.norm(cnn_latents, axis=1) + 1e-8)):.4f}")
    print(f"各维度 MSE: {[f'{v:.4f}' for v in dim_mse]}")
    print("=" * 60)
    
    # Save plots
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        base = os.path.splitext(save_path)[0]
        fig1.savefig(f"{base}_flight_state.png", dpi=150, bbox_inches="tight")
        fig2.savefig(f"{base}_latent_compare.png", dpi=150, bbox_inches="tight")
        fig3.savefig(f"{base}_error_metrics.png", dpi=150, bbox_inches="tight")
        if fig4:
            fig4.savefig(f"{base}_action_compare.png", dpi=150, bbox_inches="tight")
        print(f"图像已保存到 {base}_*.png")
    
    return fig1, fig2, fig3, fig4


def main() -> None:
    args = parse_args()
    
    if not os.path.isfile(args.teacher_checkpoint):
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_checkpoint}")
    if not os.path.isfile(args.cnn_checkpoint):
        raise FileNotFoundError(f"CNN checkpoint not found: {args.cnn_checkpoint}")

    cfg = load_training_config(args.config) or {}
    env_name = args.env_name or cfg.get("params", {}).get("config", {}).get("env_name", DEFAULT_ENV_NAME)

    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        task = task_registry.make_task(env_name, num_envs=args.num_envs, headless=args.headless)
    finally:
        sys.argv = original_argv

    obs_dim = task.task_config.observation_space_dim
    priv_dim = task.task_config.privileged_observation_space_dim
    latent_dim = 8
    device = torch.device(task.device)
    
    # Relax crash thresholds for validation with random waypoints
    # This prevents crashes during trajectory tracking which has larger angles
    if args.preflight_steps > 0:
        task.crash_distance_threshold = 20.0  # meters (relaxed from 1.0)
        task.crash_tilt_threshold_rad = np.deg2rad(60.0)  # 60° (relaxed from 20°)
        print(f"Relaxed crash thresholds: distance={task.crash_distance_threshold}m, tilt=60°")

    # Load models
    checkpoint, checkpoint_action_dim = load_checkpoint(args.teacher_checkpoint)
    action_dim = task.task_config.action_space_dim
    if action_dim != checkpoint_action_dim:
        task.close()
        raise RuntimeError(f"Action dim mismatch: task={action_dim}, ckpt={checkpoint_action_dim}")

    model = build_model(cfg, obs_dim, action_dim, task.sim_env.num_envs, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if getattr(model, "normalize_input", False) and "running_mean_std" in checkpoint:
        model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    teacher_encoder = load_teacher_encoder(args.teacher_checkpoint, str(device), priv_dim)
    cnn_encoder, history_buffer, ObsHistoryBuffer = load_cnn_encoder(
        args.cnn_checkpoint, obs_dim, args.history_len, latent_dim, str(device)
    )
    history_buffer = ObsHistoryBuffer(num_envs=args.num_envs, obs_dim=obs_dim, 
                                      history_len=args.history_len, device=str(device))

    print("=" * 60)
    print("Stage 2 CNN Student Validation")
    print("=" * 60)
    print(f"Task: {env_name}, Envs: {args.num_envs}, Steps: {args.steps}")
    print(f"Dims: obs={obs_dim}, priv={priv_dim}, action={action_dim}, latent={latent_dim}")
    print(f"History length: {args.history_len}")
    print("=" * 60)

    task.reset()
    rnn_states = _to_device(model.get_default_rnn_state(), device)
    
    # Recording lists
    z_history, euler_history, pos_history = [], [], []
    release_history = []
    policy_actions, teacher_actions = [], []
    teacher_latents_list, cnn_latents_list = [], []

    # Warm up history buffer
    print(f"Warming up history buffer ({args.history_len} steps)...")
    with torch.no_grad():
        for _ in range(args.history_len):
            obs = torch.as_tensor(task.task_obs["observations"], device=device, dtype=torch.float32)
            history_buffer.push(obs)
            actions = torch.zeros((args.num_envs, action_dim), device=device)
            task.step(actions)

    print(f"Running validation for {args.steps} steps...")
    preflight_step = 0
    with torch.no_grad():
        for step in range(args.steps):
            obs = torch.as_tensor(task.task_obs["observations"], device=device, dtype=torch.float32)
            priv = task.task_obs.get("privileged_obs", None)
            if priv is not None:
                priv = torch.as_tensor(priv, device=device, dtype=torch.float32)
            
            # Random waypoint navigation during preflight
            if args.preflight_steps > 0 and preflight_step < args.preflight_steps:
                if preflight_step % args.preflight_waypoint_interval == 0:
                    random_target = (torch.rand(1, 3, device=device) * 2 - 1) * args.preflight_waypoint_range
                    random_target[:, 2] = random_target[:, 2].abs()
                    task.target_position[:] = random_target
                preflight_step += 1
            elif args.preflight_steps > 0 and preflight_step == args.preflight_steps:
                # Reset to origin after preflight
                task.target_position[:] = 0.0
                preflight_step += 1
            
            # Get latents
            teacher_z = teacher_encoder(priv) if priv is not None else torch.zeros((args.num_envs, latent_dim), device=device)
            history_buffer.push(obs)
            cnn_z = cnn_encoder(history_buffer.get())
            
            teacher_latents_list.append(teacher_z.cpu().numpy())  # (num_envs, latent_dim)
            cnn_latents_list.append(cnn_z.cpu().numpy())  # (num_envs, latent_dim)
            
            # Policy forward
            input_dict = {"is_train": False, "prev_actions": None, "obs": obs,
                          "privileged_obs": priv, "rnn_states": rnn_states, "seq_length": 1}
            result = model(input_dict)
            action = torch.clamp(result["mus"] if args.deterministic else result["actions"], -1.0, 1.0)
            
            # Zero compensation during preflight
            if args.preflight_steps > 0 and step < args.preflight_steps:
                action = torch.zeros_like(action)
            
            task_obs, rewards, terms, truncs, infos = task.step(action)
            rnn_states = _to_device(result.get("rnn_states"), device)
            done_envs = torch.nonzero(terms | truncs, as_tuple=False).squeeze(-1)
            rnn_states = _reset_rnn_states(rnn_states, done_envs)
            if done_envs.numel() > 0:
                history_buffer.reset(done_envs)
                preflight_step = 0  # Reset preflight counter on env reset
            
            # Record for env 0
            env_id = 0
            pos = task.obs_dict["robot_position"][env_id].detach().cpu().numpy()
            quat = task.obs_dict["robot_orientation"][env_id:env_id+1]
            euler = get_euler_xyz_tensor(quat)[0].detach().cpu().numpy()
            
            pos_history.append(pos.copy())
            z_history.append(pos[2])
            euler_history.append(euler)
            policy_actions.append(action[env_id].detach().cpu().numpy())
            
            if hasattr(task, "teacher_residual"):
                teacher_actions.append(task.teacher_residual[env_id].detach().cpu().numpy())
            
            if task.payload_manager.just_released_flag[env_id]:
                payload_idx = int(task.payload_manager.last_release_index[env_id].item())
                release_history.append((step, payload_idx))

    # Stack all envs: (steps, num_envs, latent_dim) -> (steps*num_envs, latent_dim)
    teacher_latents = np.concatenate(teacher_latents_list, axis=0)
    cnn_latents = np.concatenate(cnn_latents_list, axis=0)

    save_path = args.save_path or (f"validate_cnn_stage2_{args.steps}steps.png" if args.save_plot else None)
    
    plot_comprehensive_results(
        z_history, euler_history, release_history,
        policy_actions, teacher_actions,
        teacher_latents, cnn_latents,
        pos_history, save_path,
        num_envs=args.num_envs
    )

    if args.show_plot:
        plt.show()
    else:
        plt.close("all")

    try:
        task.close()
    except AttributeError as exc:
        print(f"Warning: task.close() failed ({exc})")


if __name__ == "__main__":
    main()
