"""
Stage 2: CNN Student Supervised Learning Training Script.

This script trains a CNN encoder to predict the 8-dim latent variable z_t
from historical observations, supervised by the frozen teacher's privileged encoder.

Usage:
    python aerial_gym/rl_training/train_cnn_student.py \
        --teacher_checkpoint runs/teacher_residual_stage1_27-17-06-04/nn/teacher_residual_stage1.pth \
        --num_envs 1024 \
        --history_len 50 \
        --epochs 500 \
        --lr 1e-4 \
        --experiment_name cnn_student_stage2
"""

# Isaac Gym MUST be imported before torch
import isaacgym  # noqa: F401

import os
import sys
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Use importlib to load modules directly, avoiding rl_games/__init__.py
import importlib.util

def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Get the base path
_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load cnn_student_encoder directly
_cnn_encoder_path = os.path.join(_base_path, "rl_training/rl_games/nn/cnn_student_encoder.py")
cnn_encoder_module = load_module_from_file("cnn_student_encoder", _cnn_encoder_path)
CNNStudentEncoder = cnn_encoder_module.CNNStudentEncoder
CNNWithTemporalAttention = cnn_encoder_module.CNNWithTemporalAttention
ObsHistoryBuffer = cnn_encoder_module.ObsHistoryBuffer
create_cnn_encoder = cnn_encoder_module.create_cnn_encoder

# Load PayloadCompensationTask
from aerial_gym.task.payload_compensation_task.payload_compensation_task import (
    PayloadCompensationTask,
)

# Import yaml for config loading
import yaml


def load_training_config(config_path: str):
    """Load training YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_teacher_policy(cfg, obs_dim, action_dim, num_envs, device):
    """Build the full Teacher policy network."""
    import importlib.util
    
    # Temporarily modify sys.path to prioritize conda rl_games over local
    original_path = sys.path.copy()
    # Remove paths containing aerial_gym/rl_training/rl_games
    sys.path = [p for p in sys.path if "aerial_gym" not in p]
    
    try:
        # Import rl_games from conda installation
        from rl_games.algos_torch import model_builder
        
        # Register privileged actor critic by loading directly
        _base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _priv_ac_path = os.path.join(_base_path, "rl_training/rl_games/nn/privileged_actor_critic.py")
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
        # Restore original sys.path
        sys.path = original_path


def load_teacher_encoder(checkpoint_path: str, device: str, priv_dim: int = 7):
    """
    Load the trained teacher model and extract the privileged encoder.
    
    Args:
        checkpoint_path: Path to teacher checkpoint
        device: Device to load on
        priv_dim: Privileged observation dimension (default: 7 to match current config)
    
    Returns:
        priv_encoder: Frozen privileged encoder (priv_dim -> 8) WITH normalization
        embed_dim: Detected embedding dimension
    """
    print(f"Loading teacher checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # rl-games saves model under 'model' key
    model_state = checkpoint.get("model", checkpoint)
    
    # Extract privileged encoder weights and normalization params
    priv_encoder_state = {}
    priv_norm_running_mean = None
    priv_norm_running_var = None
    priv_norm_count = None
    # Also check for FixedObsNorm style (mean/std)
    priv_norm_mean = None
    priv_norm_std = None
    
    for key, value in model_state.items():
        if "a2c_network.priv_encoder" in key:
            # Remove prefix: a2c_network.priv_encoder.0.weight -> 0.weight
            local_key = key.replace("a2c_network.priv_encoder.", "")
            priv_encoder_state[local_key] = value
        # RunningObsNorm uses running_mean/running_var
        elif "a2c_network.priv_norm.running_mean" in key:
            priv_norm_running_mean = value
        elif "a2c_network.priv_norm.running_var" in key:
            priv_norm_running_var = value
        elif "a2c_network.priv_norm.count" in key:
            priv_norm_count = value
        # FixedObsNorm uses mean/std
        elif "a2c_network.priv_norm.mean" in key:
            priv_norm_mean = value
        elif "a2c_network.priv_norm.std" in key:
            priv_norm_std = value
    
    if not priv_encoder_state:
        raise RuntimeError(f"No priv_encoder weights found in checkpoint")
    
    # Auto-detect priv_dim from first encoder layer weights
    first_layer_key = "0.weight"
    if first_layer_key in priv_encoder_state:
        detected_priv_dim = priv_encoder_state[first_layer_key].shape[1]
        if detected_priv_dim != priv_dim:
            print(f"  [Auto-detect] priv_dim from checkpoint: {detected_priv_dim} (overriding {priv_dim})")
            priv_dim = detected_priv_dim
    
    hidden = 128
    # embed_dim = 8  # No longer hardcoded
    
    # Auto-detect embed_dim from last layer weights if possible, else 8
    embed_dim = 8
    last_layer_key = "4.weight" # Linear(priv) -> Linear(hidden) -> Linear(hidden) -> Linear(embed)
    if last_layer_key in priv_encoder_state:
        embed_dim = priv_encoder_state[last_layer_key].shape[0]
        print(f"  [Auto-detect] embed_dim from checkpoint: {embed_dim}")
    else:
        print(f"  [Warning] Could not detect embed_dim, using default: {embed_dim}")
    
    # Determine normalization type from checkpoint
    use_running_norm = priv_norm_running_mean is not None
    
    # Build encoder WITH normalization
    class TeacherEncoder(nn.Module):
        def __init__(self, priv_dim, hidden, embed_dim, running_mean, running_var, fixed_mean, fixed_std, use_running):
            super().__init__()
            self.use_running_norm = use_running
            self.eps = 1e-6
            self.clip_range = 10.0
            
            if use_running and running_mean is not None:
                # RunningObsNorm style
                self.register_buffer("running_mean", running_mean.float())
                self.register_buffer("running_var", running_var.float() if running_var is not None else torch.ones(priv_dim))
            elif fixed_mean is not None:
                # FixedObsNorm style
                self.register_buffer("running_mean", fixed_mean.float())
                std = fixed_std.float() if fixed_std is not None else torch.ones(priv_dim)
                self.register_buffer("running_var", std ** 2)  # Convert std to var
            else:
                # Default (no normalization stored)
                self.register_buffer("running_mean", torch.zeros(priv_dim))
                self.register_buffer("running_var", torch.ones(priv_dim))
            
            # Encoder layers
            self.encoder = nn.Sequential(
                nn.Linear(priv_dim, hidden),
                nn.ELU(),
                nn.Linear(hidden, hidden),
                nn.ELU(),
                nn.Linear(hidden, embed_dim),
            )
        
        def forward(self, x):
            # Apply normalization (same as RunningObsNorm.forward)
            std = torch.sqrt(self.running_var + self.eps)
            normalized = (x - self.running_mean) / std
            normalized = torch.clamp(normalized, -self.clip_range, self.clip_range)
            return self.encoder(normalized)
    
    priv_encoder = TeacherEncoder(
        priv_dim, hidden, embed_dim, 
        priv_norm_running_mean, priv_norm_running_var,
        priv_norm_mean, priv_norm_std,
        use_running_norm
    ).to(device)
    
    # Load encoder weights
    priv_encoder.encoder.load_state_dict(priv_encoder_state)
    
    # Freeze encoder
    for param in priv_encoder.parameters():
        param.requires_grad = False
    priv_encoder.eval()
    
    print(f"Loaded teacher encoder: {priv_dim} -> {embed_dim}")
    norm_type = "RunningObsNorm" if use_running_norm else "FixedObsNorm"
    print(f"  Normalization type: {norm_type}")
    print(f"  running_mean[:5]: {priv_encoder.running_mean[:min(5, priv_dim)].tolist()}")
    print(f"  running_var[:5]: {priv_encoder.running_var[:min(5, priv_dim)].tolist()}")
    
    return priv_encoder, embed_dim


def create_task(num_envs: int, device: str):
    """Create the payload compensation task for data collection.
    
    Overrides release_start to ensure immediate payload release at training start.
    """
    from aerial_gym.config.task_config.payload_compensation_task_teacher_config import (
        task_config,
    )
    
    # Override some settings for data collection without polluting global config
    import copy
    cfg_copy = copy.deepcopy(task_config)
    cfg_copy.num_envs = num_envs
    cfg_copy.headless = True
    cfg_copy.device = device
    # Relax crash thresholds for training stability
    cfg_copy.crash_distance_threshold = 10.0
    cfg_copy.crash_tilt_threshold_deg = 60.0
    
    # Force release to start after warm-up so all 4 releases are captured in training
    # With history_len=200 (typical), release_start=250 ensures first release is visible
    cfg_copy.payload_parameters["release_start"] = 250
    cfg_copy.payload_parameters["release_start_range"] = [200, 300]
    
    task = PayloadCompensationTask(
        task_config=cfg_copy,
        seed=42,
        num_envs=num_envs,
        headless=True,
        device=device,
    )
    return task


def collect_training_sample(
    task_obs: dict,
    history_buffer: ObsHistoryBuffer,
    teacher_encoder: nn.Module,
    obs_override: torch.Tensor = None,
):
    """
    Collect one training sample: (obs_history, teacher_z).
    """
    # Get current observation
    if obs_override is not None:
        base_obs = obs_override
    else:
        base_obs = task_obs["observations"]
        
    priv_obs = task_obs.get("privileged_obs", None)
    
    if priv_obs is None:
        raise RuntimeError("Task must provide privileged observations for teacher supervision")
    
    # Push current obs to history
    history_buffer.push(base_obs)
    
    # Get history for CNN input
    obs_history = history_buffer.get()
    
    # Get teacher's latent encoding (frozen)
    with torch.no_grad():
        teacher_z = teacher_encoder(priv_obs)
    
    return obs_history, teacher_z


def train_step(
    cnn_encoder: nn.Module,
    optimizer: optim.Optimizer,
    obs_history: torch.Tensor,
    teacher_z: torch.Tensor,
    criterion: nn.Module,
    weighting: torch.Tensor = None,
):
    """
    One training step.
    
    Returns:
        loss: scalar loss value
    """
    optimizer.zero_grad()
    
    # Forward pass
    student_z = cnn_encoder(obs_history)
    
    # Compute Weighted MSE loss
    # teacher_z and student_z: (num_envs, latent_dim)
    # weighting: (num_envs,)
    if weighting is not None:
        raw_mse = torch.mean((student_z - teacher_z) ** 2, dim=1)  # (num_envs,)
        loss = torch.mean(raw_mse * weighting)
    else:
        loss = criterion(student_z, teacher_z)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(cnn_encoder.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()


def compute_metrics(student_z: torch.Tensor, teacher_z: torch.Tensor):
    """Compute additional metrics for monitoring."""
    with torch.no_grad():
        mse = torch.mean((student_z - teacher_z) ** 2).item()
        mae = torch.mean(torch.abs(student_z - teacher_z)).item()
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(student_z, teacher_z, dim=1)
        cos_sim_mean = cos_sim.mean().item()
        
        # Per-dimension MSE
        dim_mse = torch.mean((student_z - teacher_z) ** 2, dim=0).cpu().numpy()
        
    return {
        "mse": mse,
        "mae": mae,
        "cosine_similarity": cos_sim_mean,
        "dim_mse": dim_mse,
    }


def main():
    parser = argparse.ArgumentParser(description="Train CNN Student Encoder")
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        required=True,
        help="Path to teacher checkpoint",
    )
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel envs")
    parser.add_argument("--history_len", type=int, default=100, help="Observation history length (Shortened for faster reaction)")
    parser.add_argument("--latent_dim", type=int, default=8, help="Latent dimension")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=1500, help="Steps per epoch (match or exceed release timeline)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--experiment_name", type=str, default="cnn_student", help="Experiment name")
    parser.add_argument("--save_interval", type=int, default=50, help="Checkpoint save interval")
    parser.add_argument(
        "--config",
        type=str,
        default="aerial_gym/rl_training/rl_games/ppo_aerial_quad_aux.yaml",
        help="Training YAML config path for building Teacher policy",
    )

    parser.add_argument(
        "--use_attention",
        action="store_true",
        help="Use CNN with Temporal Attention instead of standard CNN",
    )
    
    parser.add_argument(
        "--cnn_checkpoint",
        type=str,
        default=None,
        help="Path to existing CNN student checkpoint to resume/inherit from",
    )
    
    args = parser.parse_args()
    
    device = args.device
    
    # Create task first to get dimensions from config
    print("Creating simulation environment to detect dimensions...")
    task = create_task(args.num_envs, device)
    obs_dim = task.task_config.observation_space_dim
    priv_dim = task.task_config.privileged_observation_space_dim
    action_dim = task.task_config.action_space_dim
    print(f"Detected dimensions: obs_dim={obs_dim}, priv_dim={priv_dim}, action_dim={action_dim}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%d-%H-%M-%S")
    run_dir = f"runs/{args.experiment_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/nn", exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=f"{run_dir}/summaries")
    
    print("=" * 60)
    print("CNN Student Encoder Training - Stage 2")
    print("=" * 60)
    print(f"Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"Num envs: {args.num_envs}")
    print(f"Obs dim: {obs_dim}, Priv dim: {priv_dim}, Action dim: {action_dim}")
    print(f"History length: {args.history_len}")
    print(f"Requested Latent dim: {args.latent_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Output dir: {run_dir}")
    print("=" * 60)
    
    # Load teacher encoder (auto-detects priv_dim and embed_dim from checkpoint)
    teacher_encoder, detected_embed_dim = load_teacher_encoder(args.teacher_checkpoint, device, priv_dim)

    # Consistency check
    if detected_embed_dim != args.latent_dim:
        print(f"[WARNING] Config latent_dim ({args.latent_dim}) != Teacher embed_dim ({detected_embed_dim}).")
        print(f"  -> Overriding latent_dim to {detected_embed_dim} to match Teacher.")
        args.latent_dim = detected_embed_dim
    
    # Define Student Observation Dimension (Manual Assembly)
    # Rot(9) + LinVel(3) + AngVel(3) + Action(3) = 18
    # Excluding Mask(4) and Warning(1)
    student_obs_dim = 18
    
    # Create CNN student encoder (with or without attention)
    if args.use_attention:
        cnn_encoder = CNNWithTemporalAttention(
            obs_dim=student_obs_dim,
            history_len=args.history_len,
            latent_dim=args.latent_dim,
        ).to(device)
        print(f"CNN Encoder (with Temporal Attention): {cnn_encoder}")
    else:
        cnn_encoder = create_cnn_encoder(
            obs_dim=student_obs_dim,
            history_len=args.history_len,
            latent_dim=args.latent_dim,
            device=device,
        )
        print(f"CNN Encoder: {cnn_encoder}")
    
    # Optional: Load CNN student checkpoint
    if args.cnn_checkpoint is not None:
        print(f"Loading CNN student checkpoint: {args.cnn_checkpoint}")
        student_ckpt = torch.load(args.cnn_checkpoint, map_location=device)
        
        # Determine if it's a full checkpoint or just model state dict
        if "model_state_dict" in student_ckpt:
            cnn_encoder.load_state_dict(student_ckpt["model_state_dict"])
            print(f"  [Loaded] Model weights (Epoch: {student_ckpt.get('epoch', 'N/A')})")
        else:
            cnn_encoder.load_state_dict(student_ckpt)
            print(f"  [Loaded] Model weights (standalone state dict)")
    
    # Load Teacher policy for generating realistic flight data
    print("Loading Teacher policy for data generation...")
    cfg = load_training_config(args.config)
    
    # Load checkpoint to get model weights
    teacher_ckpt = torch.load(args.teacher_checkpoint, map_location=device)
    teacher_policy = build_teacher_policy(cfg, obs_dim, action_dim, args.num_envs, device)
    teacher_policy.load_state_dict(teacher_ckpt["model"], strict=True)
    if getattr(teacher_policy, "normalize_input", False) and "running_mean_std" in teacher_ckpt:
        teacher_policy.running_mean_std.load_state_dict(teacher_ckpt["running_mean_std"])
    print("Teacher policy loaded successfully.")
    
    # Create history buffer
    history_buffer = ObsHistoryBuffer(
        num_envs=args.num_envs,
        obs_dim=student_obs_dim,
        history_len=args.history_len,
        device=device,
    )
    
    # Optimizer and loss
    optimizer = optim.Adam(cnn_encoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.MSELoss()
    
    # Initialize environment
    print("Resetting environment...")
    task.reset()
    
    target_zero = torch.zeros((args.num_envs, 3), device=device)

    # Warm up history buffer
    print(f"Warming up history buffer ({args.history_len} steps)...")
    for _ in range(args.history_len):
        # Use Teacher actions so early release doesn't destabilize warm-up
        task.update_target_position(target_zero)
        task_obs = task.get_task_observations()
        obs = torch.as_tensor(
            task_obs["observations"], device=device, dtype=torch.float32
        )
        priv = task_obs.get("privileged_obs", None)
        if priv is not None:
            priv = torch.as_tensor(priv, device=device, dtype=torch.float32)
        with torch.no_grad():
            input_dict = {
                "is_train": False,
                "prev_actions": None,
                "obs": obs,
                "privileged_obs": priv,
                "rnn_states": None,
                "seq_length": 1,
            }
            result = teacher_policy(input_dict)
            actions = torch.clamp(result["mus"], -1.0, 1.0)
        task.step(actions)
        task_obs = task.get_task_observations()
        terminations = task_obs.get("terminations")
        truncations = task_obs.get("truncations")
        if terminations is not None and truncations is not None:
            reset_ids = (terminations > 0) | (truncations > 0)
            reset_ids = torch.where(reset_ids)[0]
            if reset_ids.numel() > 0:
                history_buffer.reset(reset_ids)
        teacher_obs_warm = torch.as_tensor(task_obs["observations"], device=device, dtype=torch.float32)
        linvel_obs_warm = task.obs_dict["robot_linvel"].to(device)
        student_obs_warm = torch.cat([
            teacher_obs_warm[:, 0:9],   # Rot
            linvel_obs_warm,           # LinVel
            teacher_obs_warm[:, 9:12],  # AngVel
            teacher_obs_warm[:, 17:20]  # Action
        ], dim=1)
        history_buffer.push(student_obs_warm)
    
    print("Starting training...")
    global_step = 0
    best_loss = float("inf")
    

    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_cos_sim = 0.0
        
        cnn_encoder.train()
        
        # Track previous mass to detect release
        prev_mass = None
        transition_counter = torch.zeros(args.num_envs, device=device)
        
        for step in range(args.steps_per_epoch):
            # Always Release Phase: use Teacher policy for compensation
            task.update_target_position(target_zero)  # Target is origin during release
            task_obs = task.get_task_observations()
            
            # --- Detect Release for Weighting ---
            current_mass = task.payload_manager.current_payload_mass
            if prev_mass is not None:
                # If mass decreases, reset counter to high weighting
                release_event = (current_mass < prev_mass - 0.01)
                transition_counter[release_event] = 50 # Weight for 50 steps
            prev_mass = current_mass.clone()
            
            # Compute weighting: 20.0 for transitions, 1.0 for steady state
            weighting = torch.ones(args.num_envs, device=device)
            weighting[transition_counter > 0] = 20.0
            transition_counter = torch.max(torch.zeros_like(transition_counter), transition_counter - 1)

            # --- Construct Teacher Obs (20 dims) for Policy ---
            teacher_obs = torch.as_tensor(
                task_obs["observations"], device=device, dtype=torch.float32
            )
            
            # --- Construct Student Obs (18 dims) for Encoder ---
            # Rot(9) | LinVel(3) | AngVel(3) | Action(3)
            # Access LinVel from task internals (not in teacher_obs)
            linvel_obs = task.obs_dict["robot_linvel"].to(device) # (N, 3)
            
            # Extract components from teacher_obs (original index based on 20-dim config)
            # Ref: [0-8] Rot | [9-11] AngVel | [12-15] Mask | [16] Warn | [17-19] Act
            rot_obs = teacher_obs[:, 0:9]
            angvel_obs = teacher_obs[:, 9:12]
            action_obs = teacher_obs[:, 17:20]
            
            # Concatenate for student
            student_obs = torch.cat([rot_obs, linvel_obs, angvel_obs, action_obs], dim=1) # (N, 18)
            priv = task_obs.get("privileged_obs", None)
            if priv is not None:
                priv = torch.as_tensor(priv, device=device, dtype=torch.float32)
            with torch.no_grad():
                input_dict = {
                    "is_train": False,
                    "prev_actions": None,
                    "obs": teacher_obs,
                    "privileged_obs": priv,
                    "rnn_states": None,
                    "seq_length": 1,
                }
                result = teacher_policy(input_dict)
                actions = torch.clamp(result["mus"], -1.0, 1.0)
            
            task.step(actions)

            # --- FIX: Handle Resets BEFORE Sampling ---
            # If an env reset happened this step, its obs is now the specific initial state,
            # and its history should be cleared. Sampling from it mixed with old history is wrong.
            # We must identify reset envs and clear their history buffer FIRST.
            task_obs = task.get_task_observations()
            terminations = task_obs.get("terminations")
            truncations = task_obs.get("truncations")
            if terminations is not None and truncations is not None:
                reset_ids = (terminations > 0) | (truncations > 0)
                reset_ids = torch.where(reset_ids)[0]
                if reset_ids.numel() > 0:
                    history_buffer.reset(reset_ids)

            # Collect training sample (now safe: history is clean for new episodes, valid for ongoing ones)
            obs_history, teacher_z = collect_training_sample(
                task_obs, history_buffer, teacher_encoder, obs_override=student_obs
            )
            
            # Train step with weighting
            loss = train_step(cnn_encoder, optimizer, obs_history, teacher_z, criterion, weighting=weighting)
            epoch_loss += loss
            
            # Compute metrics
            with torch.no_grad():
                student_z = cnn_encoder(obs_history)
                metrics = compute_metrics(student_z, teacher_z)
            epoch_cos_sim += metrics["cosine_similarity"]
            
            
            global_step += 1
            
            # Log to TensorBoard
            if global_step % 50 == 0:
                writer.add_scalar("loss/mse", loss, global_step)
                writer.add_scalar("metrics/cosine_similarity", metrics["cosine_similarity"], global_step)
                writer.add_scalar("metrics/mae", metrics["mae"], global_step)
        
        # Epoch summary
        avg_loss = epoch_loss / args.steps_per_epoch
        avg_cos_sim = epoch_cos_sim / args.steps_per_epoch
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Loss: {avg_loss:.6f} | "
            f"CosSim: {avg_cos_sim:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        
        writer.add_scalar("epoch/avg_loss", avg_loss, epoch)
        writer.add_scalar("epoch/avg_cosine_similarity", avg_cos_sim, epoch)
        writer.add_scalar("epoch/learning_rate", current_lr, epoch)
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or avg_loss < best_loss:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": cnn_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "cosine_similarity": avg_cos_sim,
            }
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, f"{run_dir}/nn/best_cnn_encoder.pth")
                print(f"  -> Saved best model (loss: {best_loss:.6f})")
            
            torch.save(checkpoint, f"{run_dir}/nn/cnn_encoder_ep{epoch + 1}.pth")
    
    # Final save
    torch.save(
        {"model_state_dict": cnn_encoder.state_dict()},
        f"{run_dir}/nn/final_cnn_encoder.pth",
    )
    
    print("=" * 60)
    print("Training complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {run_dir}/nn/")
    print("=" * 60)
    
    writer.close()
    task.close()


if __name__ == "__main__":
    main()
