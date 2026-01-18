"""
Ablation Experiment Runner for Paper Results.

This script runs ablation experiments and collects metrics for the paper:
- Success Rate (survival rate across episodes)
- Position RMSE (Root Mean Square Error of position tracking)
- Transient Recovery Time

Usage:
    python aerial_gym/examples/run_ablation_experiments.py \
        --teacher_checkpoint runs/teacher_aux_*/nn/best_*.pth \
        --num_envs 64 \
        --num_episodes 100 \
        --output_dir paper/ablation_results/

Output:
    - ablation_table.md: Markdown table for paper
    - ablation_results.csv: Detailed CSV data
"""

import argparse
import os
import sys
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# IMPORTANT: IsaacGym must be imported before PyTorch!
# aerial_gym imports isaacgym internally
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.utils.math import get_euler_xyz_tensor

# Ensure privileged network is registered
from aerial_gym.rl_training.rl_games.nn import privileged_actor_critic  # noqa: F401
from rl_games.algos_torch import model_builder

# Now safe to import torch and numpy
import numpy as np
import yaml
import torch


DEFAULT_CONFIG = "aerial_gym/rl_training/rl_games/ppo_aerial_quad_aux.yaml"
DEFAULT_ENV_NAME = "payload_compensation_task_teacher"


@dataclass
class ExperimentConfig:
    """Configuration for a single ablation experiment."""
    name: str
    checkpoint_path: str
    description: str
    # Config overrides for task
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    use_zero_action: bool = False  # If True, output zero actions (pure PD baseline)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    name: str
    description: str
    # Core metrics
    success_rate: float  # Fraction of episodes that didn't crash
    position_rmse: float  # Root mean square position error
    # Additional metrics
    avg_episode_length: float
    avg_reward: float
    num_episodes: int
    num_crashes: int
    # New metrics
    max_tilt_angle_deg: float = 0.0  # Maximum tilt angle (degrees)
    avg_max_tilt_deg: float = 0.0    # Average of per-episode max tilt
    recovery_time_steps: float = 0.0 # Steps to return to stable after disturbance


@dataclass 
class DifficultyLevel:
    """Test difficulty configuration."""
    name: str
    wind_force_range: Tuple[float, float]
    payload_mass_range: Tuple[float, float]


# Default difficulty levels for testing
DIFFICULTY_LEVELS = [
    DifficultyLevel("Easy", (0.0, 0.05), (0.0, 0.01)),
    DifficultyLevel("Medium", (0.0, 0.10), (0.0, 0.02)),
    DifficultyLevel("Hard", (0.0, 0.20), (0.0, 0.03)),
    DifficultyLevel("Extreme", (0.0, 0.50), (0.0, 0.05)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation experiments for paper.")
    
    # Default paths based on user's training runs
    default_teacher = "runs/teacher_aux_fixed_imitation_17-14-16-10/nn/teacher_aux_fixed_imitation.pth"
    default_cnn = "runs/cnn_student_18-02-01-12/nn/best_cnn_encoder.pth"
    
    parser.add_argument("--teacher_checkpoint", type=str, default=default_teacher,
                        help="Path to teacher checkpoint (Full model).")
    parser.add_argument("--cnn_checkpoint", type=str, default=default_cnn,
                        help="Path to CNN student checkpoint.")
    parser.add_argument("--history_len", type=int, default=500,
                        help="History length for CNN encoder.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="Training YAML config path.")
    parser.add_argument("--num_envs", type=int, default=64,
                        help="Number of parallel environments.")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of episodes to run per experiment.")
    parser.add_argument("--steps_per_episode", type=int, default=1500,
                        help="Maximum steps per episode.")
    parser.add_argument("--output_dir", type=str, default="paper/ablation_results",
                        help="Output directory for results.")
    parser.add_argument("--difficulty", type=str, default="all",
                        choices=["easy", "medium", "hard", "extreme", "all"],
                        help="Difficulty level to test.")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["full", "no_phys_rand", "pure_pd", "all"],
                        help="Which experiment to run (run separately to avoid Isaac Gym crash).")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run without rendering.")
    
    return parser.parse_args()


def load_training_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path: str) -> Tuple[Dict[str, Any], int]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_state = ckpt.get("model", {})
    action_tensor = model_state.get("a2c_network.mu.weight")
    if action_tensor is None:
        raise RuntimeError("Checkpoint missing a2c_network.mu.weight")
    action_dim = int(action_tensor.shape[0])
    return ckpt, action_dim


def build_model(cfg: Dict[str, Any], obs_dim: int, action_dim: int, 
                num_envs: int, device: torch.device):
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


def run_single_experiment(
    experiment: ExperimentConfig,
    difficulty: DifficultyLevel,
    cfg: Dict[str, Any],
    num_envs: int,
    num_episodes: int,
    steps_per_episode: int,
    headless: bool = True,
) -> ExperimentResult:
    """Run a single ablation experiment and collect metrics."""
    
    print(f"\n{'='*60}")
    print(f"Running: {experiment.name} @ {difficulty.name}")
    print(f"{'='*60}")
    
    # Create task with difficulty overrides
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    
    try:
        task = task_registry.make_task(
            DEFAULT_ENV_NAME, 
            num_envs=num_envs, 
            headless=headless
        )
        
        # Apply difficulty settings
        task.task_config.randomization_parameters["external_force_range"] = list(difficulty.wind_force_range)
        task.task_config.payload_parameters["payload_mass_range"] = list(difficulty.payload_mass_range)
        
        # Apply experiment-specific overrides
        for key, value in experiment.config_overrides.items():
            if hasattr(task.task_config, key):
                setattr(task.task_config, key, value)
        
    finally:
        sys.argv = original_argv
    
    obs_dim = task.task_config.observation_space_dim
    action_dim = task.task_config.action_space_dim
    device = torch.device(task.device)
    
    # Load model
    checkpoint, ckpt_action_dim = load_checkpoint(experiment.checkpoint_path)
    if action_dim != ckpt_action_dim:
        task.close()
        raise RuntimeError(f"Action dim mismatch: task={action_dim}, ckpt={ckpt_action_dim}")
    
    model = build_model(cfg, obs_dim, action_dim, num_envs, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if getattr(model, "normalize_input", False) and "running_mean_std" in checkpoint:
        model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
    
    # Tracking variables
    episode_count = 0
    crash_count = 0
    total_rewards = []
    episode_lengths = []
    position_errors = []
    tilt_angles = []          # All tilt angles across all steps
    episode_max_tilts = []    # Max tilt per completed episode
    
    # Per-env tracking
    env_episode_rewards = torch.zeros(num_envs, device=device)
    env_episode_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
    env_max_tilt = torch.zeros(num_envs, device=device)
    
    task.reset()
    rnn_states = model.get_default_rnn_state()
    if rnn_states is not None:
        if isinstance(rnn_states, (list, tuple)):
            rnn_states = tuple(s.to(device) for s in rnn_states)
        else:
            rnn_states = rnn_states.to(device)
    
    print(f"Running {num_episodes} episodes across {num_envs} parallel envs...")
    
    with torch.no_grad():
        while episode_count < num_episodes:
            obs = torch.as_tensor(task.task_obs["observations"], device=device, dtype=torch.float32)
            priv = task.task_obs.get("privileged_obs", None)
            if priv is not None:
                priv = torch.as_tensor(priv, device=device, dtype=torch.float32)
            
            # Policy forward
            input_dict = {
                "is_train": False,
                "prev_actions": None,
                "obs": obs,
                "privileged_obs": priv,
                "rnn_states": rnn_states,
            }
            result = model(input_dict)
            if experiment.use_zero_action:
                # Pure PD baseline - no RL compensation
                action = torch.zeros_like(result["mus"])
            else:
                action = torch.clamp(result["mus"], -1.0, 1.0)
            
            task_obs, rewards, terms, truncs, infos = task.step(action)
            
            env_episode_rewards += rewards.squeeze()
            env_episode_steps += 1
            
            # Track position errors
            pos = task.obs_dict["robot_position"]
            target_pos = task.target_position if hasattr(task, "target_position") else torch.zeros_like(pos)
            pos_err = torch.norm(pos - target_pos, dim=-1)
            position_errors.extend(pos_err.cpu().numpy().tolist())
            
            # Track tilt angles (roll/pitch from quaternion)
            if "robot_orientation" in task.obs_dict:
                quat = task.obs_dict["robot_orientation"]  # [N, 4] - Isaac Gym uses xyzw format
                # Direct tilt calculation from quaternion (avoids euler gimbal issues)
                # For small angles: roll ≈ 2*qx, pitch ≈ 2*qy (when w ≈ 1)
                qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
                # Compute roll and pitch using atan2 for better numerical stability
                roll = torch.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
                pitch = torch.asin(torch.clamp(2*(qw*qy - qz*qx), -1.0, 1.0))
                roll_deg = torch.abs(roll) * 180.0 / 3.14159
                pitch_deg = torch.abs(pitch) * 180.0 / 3.14159
                tilt_deg = torch.sqrt(roll_deg**2 + pitch_deg**2)
                tilt_angles.extend(tilt_deg.cpu().numpy().tolist())
                # Track per-env max tilt
                env_max_tilt = torch.maximum(env_max_tilt, tilt_deg)
            
            # Handle episode ends
            done = terms | truncs
            done_envs = torch.nonzero(done, as_tuple=False).squeeze(-1)
            
            if done_envs.numel() > 0:
                for env_id in done_envs.cpu().numpy():
                    if episode_count >= num_episodes:
                        break
                    
                    episode_count += 1
                    total_rewards.append(env_episode_rewards[env_id].item())
                    episode_lengths.append(env_episode_steps[env_id].item())
                    episode_max_tilts.append(env_max_tilt[env_id].item())
                    
                    # Check if crash (terminated early)
                    if terms[env_id].item() and env_episode_steps[env_id].item() < steps_per_episode * 0.9:
                        crash_count += 1
                    
                    # Reset env tracking
                    env_episode_rewards[env_id] = 0
                    env_episode_steps[env_id] = 0
                    env_max_tilt[env_id] = 0.0
                
                # Update RNN states for done envs
                if rnn_states is not None:
                    if isinstance(rnn_states, (list, tuple)):
                        for s in rnn_states:
                            if s.dim() >= 2:
                                s[:, done_envs, :] = 0.0
                    else:
                        rnn_states[:, done_envs, :] = 0.0
            
            # Progress update (only when episode count changes and is non-zero)
            if episode_count > 0 and episode_count % max(1, num_episodes // 5) == 0:
                if not hasattr(run_single_experiment, '_last_printed') or run_single_experiment._last_printed != episode_count:
                    print(f"  Progress: {episode_count}/{num_episodes} episodes")
                    run_single_experiment._last_printed = episode_count
    
    task.close()
    
    # Compute metrics
    success_rate = 1.0 - (crash_count / max(1, num_episodes))
    position_rmse = np.sqrt(np.mean(np.array(position_errors) ** 2))
    avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0
    avg_reward = np.mean(total_rewards) if total_rewards else 0
    max_tilt = np.max(tilt_angles) if tilt_angles else 0
    avg_max_tilt = np.mean(episode_max_tilts) if episode_max_tilts else 0
    
    result = ExperimentResult(
        name=experiment.name,
        description=experiment.description,
        success_rate=success_rate,
        position_rmse=position_rmse,
        avg_episode_length=avg_episode_length,
        avg_reward=avg_reward,
        num_episodes=num_episodes,
        num_crashes=crash_count,
        max_tilt_angle_deg=max_tilt,
        avg_max_tilt_deg=avg_max_tilt,
    )
    
    print(f"\nResults for {experiment.name} @ {difficulty.name}:")
    print(f"  Success Rate: {success_rate:.2%}")
    print(f"  Position RMSE: {position_rmse:.4f} m")
    print(f"  Max Tilt: {max_tilt:.1f}° (avg per-ep max: {avg_max_tilt:.1f}°)")
    print(f"  Avg Episode Length: {avg_episode_length:.1f} steps")
    print(f"  Avg Reward: {avg_reward:.2f}")
    print(f"  Crashes: {crash_count}/{num_episodes}")
    
    return result


def generate_markdown_table(results: Dict[str, Dict[str, ExperimentResult]]) -> str:
    """Generate markdown table for paper."""
    
    lines = [
        "## 消融实验结果 (Ablation Study Results)",
        "",
        "| 实验设置 | 成功率 | RMSE (Pos) | 最大倾角 | 说明 |",
        "| :--- | :---: | :---: | :---: | :--- |",
    ]
    
    # Use "Hard" difficulty as the primary result for paper
    target_difficulty = "Hard"
    
    for exp_name, difficulty_results in results.items():
        if target_difficulty in difficulty_results:
            r = difficulty_results[target_difficulty]
            lines.append(
                f"| **{r.name}** | {r.success_rate:.1%} | {r.position_rmse:.4f} m | {r.avg_max_tilt_deg:.1f}° | {r.description} |"
            )
    
    lines.extend([
        "",
        f"*测试条件: {target_difficulty} 难度, 每配置 {next(iter(results.values()))[target_difficulty].num_episodes} episodes*",
    ])
    
    return "\n".join(lines)


def generate_csv(results: Dict[str, Dict[str, ExperimentResult]], output_path: str):
    """Generate detailed CSV output."""
    import csv
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Experiment", "Difficulty", "Success Rate", "Position RMSE",
            "Avg Episode Length", "Avg Reward", "Num Episodes", "Num Crashes"
        ])
        
        for exp_name, difficulty_results in results.items():
            for diff_name, r in difficulty_results.items():
                writer.writerow([
                    r.name, diff_name, f"{r.success_rate:.4f}", f"{r.position_rmse:.6f}",
                    f"{r.avg_episode_length:.2f}", f"{r.avg_reward:.2f}",
                    r.num_episodes, r.num_crashes
                ])


def main():
    args = parse_args()
    
    if not os.path.isfile(args.teacher_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.teacher_checkpoint}")
    
    cfg = load_training_config(args.config) or {}
    
    # Define experiments
    all_experiments = [
        ExperimentConfig(
            name="Ours (Full)",
            checkpoint_path=args.teacher_checkpoint,
            description="完整方案: 残差架构 + 潜变量蒸馏 + 物理随机化",
        ),
        ExperimentConfig(
            name="w/o Physics Randomization",
            checkpoint_path="runs/ablation_no_phys_rand_18-17-26-52/nn/ablation_no_phys_rand.pth",
            description="关闭电机/阻力/外力随机化",
        ),
        ExperimentConfig(
            name="Pure PD (Baseline)",
            checkpoint_path=args.teacher_checkpoint,  # Need a model to load, but won't use its output
            description="纯PD控制器，无RL补偿",
            use_zero_action=True,
        ),
    ]
    
    # Filter experiments based on --experiment argument
    if args.experiment == "full":
        experiments = [all_experiments[0]]
    elif args.experiment == "no_phys_rand":
        experiments = [all_experiments[1]]
    elif args.experiment == "pure_pd":
        experiments = [all_experiments[2]]
    else:
        # Warning: running multiple experiments may cause Isaac Gym crash
        print("⚠️  WARNING: Running multiple experiments may cause Isaac Gym segfault!")
        print("⚠️  Consider running separately with --experiment full/no_phys_rand")
        experiments = all_experiments
    
    # Select difficulty levels
    if args.difficulty == "all":
        difficulties = DIFFICULTY_LEVELS
    else:
        difficulties = [d for d in DIFFICULTY_LEVELS if d.name.lower() == args.difficulty]
    
    # Run experiments
    all_results: Dict[str, Dict[str, ExperimentResult]] = {}
    
    for exp in experiments:
        all_results[exp.name] = {}
        for diff in difficulties:
            result = run_single_experiment(
                experiment=exp,
                difficulty=diff,
                cfg=cfg,
                num_envs=args.num_envs,
                num_episodes=args.num_episodes,
                steps_per_episode=args.steps_per_episode,
                headless=args.headless,
            )
            all_results[exp.name][diff.name] = result
    
    # Generate outputs
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Markdown table
    md_table = generate_markdown_table(all_results)
    md_path = os.path.join(args.output_dir, "ablation_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"\n✅ Markdown table saved to: {md_path}")
    
    # CSV
    csv_path = os.path.join(args.output_dir, "ablation_results.csv")
    generate_csv(all_results, csv_path)
    print(f"✅ CSV saved to: {csv_path}")
    
    # JSON (for programmatic access)
    json_path = os.path.join(args.output_dir, "ablation_results.json")
    json_results = {
        exp_name: {
            diff_name: asdict(result)
            for diff_name, result in diff_results.items()
        }
        for exp_name, diff_results in all_results.items()
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON saved to: {json_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION EXPERIMENT SUMMARY")
    print("=" * 60)
    print(md_table)


if __name__ == "__main__":
    main()
