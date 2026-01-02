"""
NaN Prevention Utilities for RL Training Stability

This module provides numerical stability utilities to prevent NaN/Inf values
during RL training. Based on best practices from:
- OpenAI Spinning Up
- CleanRL
- Stable-Baselines3
- NVIDIA Isaac Gym

Author: Aerial Gym Team
Date: 2026-01-02
"""

import torch
import numpy as np
from typing import Dict, Optional, Union, Tuple
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("nan_prevention")


def safe_normalize(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    epsilon: float = 1e-8,
    clip_range: float = 10.0,
) -> torch.Tensor:
    """
    Safely normalize observations to prevent NaN in normalization.
    
    Improvements over naive (x - mean) / std:
    1. Adds epsilon to std to avoid division by zero
    2. Clips normalized values to prevent extreme outliers
    
    Args:
        x: Input tensor to normalize
        mean: Running mean for normalization
        std: Running std for normalization
        epsilon: Small constant added to std (default: 1e-8)
        clip_range: Clip normalized values to [-clip_range, clip_range] (default: 10.0)
    
    Returns:
        Normalized and clipped tensor
    
    Example:
        >>> obs_normalized = safe_normalize(obs, running_mean, running_std)
    """
    std_safe = torch.clamp(std, min=epsilon)
    normalized = (x - mean) / std_safe
    return torch.clamp(normalized, -clip_range, clip_range)


def clip_rewards(
    rewards: torch.Tensor,
    min_r: float = -100.0,
    max_r: float = 100.0,
) -> torch.Tensor:
    """
    Clip rewards to prevent extreme values that can destabilize training.
    
    Extreme rewards can cause:
    - Value function divergence
    - Advantage estimation issues
    - Gradient explosion
    
    Args:
        rewards: Reward tensor (any shape)
        min_r: Minimum allowed reward (default: -100.0)
        max_r: Maximum allowed reward (default: 100.0)
    
    Returns:
        Clipped rewards
    
    Example:
        >>> rewards_safe = clip_rewards(rewards, min_r=-50.0, max_r=50.0)
    """
    return torch.clamp(rewards, min_r, max_r)


def clip_observations(
    obs_dict: Dict[str, torch.Tensor],
    limits: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Clip each observation component to safe ranges to prevent extreme values.
    
    Default limits (if not specified):
    - position: ±50 m
    - velocity: ±50 m/s
    - angular_velocity: ±20 rad/s
    - quaternion: normalized (no clipping)
    
    Args:
        obs_dict: Dictionary of observation tensors
        limits: Optional custom limits dict, e.g., {"robot_linvel": (-30, 30)}
    
    Returns:
        Dictionary with clipped observations
    
    Example:
        >>> obs_dict_safe = clip_observations(obs_dict, limits={"robot_linvel": (-20, 20)})
    """
    if limits is None:
        # Default safe limits for common observation types
        limits = {
            "robot_position": (-50.0, 50.0),
            "robot_linvel": (-50.0, 50.0),
            "robot_body_linvel": (-50.0, 50.0),
            "robot_angvel": (-20.0, 20.0),
            "robot_body_angvel": (-20.0, 20.0),
        }
    
    clipped_dict = {}
    for key, tensor in obs_dict.items():
        if key in limits:
            min_val, max_val = limits[key]
            clipped_dict[key] = torch.clamp(tensor, min_val, max_val)
        else:
            clipped_dict[key] = tensor
    
    return clipped_dict


def detect_and_log_nan(
    tensor: torch.Tensor,
    name: str,
    step: int,
    logger_instance: Optional[CustomLogger] = None,
    raise_on_nan: bool = False,
) -> bool:
    """
    Detect NaN/Inf in tensor and log detailed diagnostic information.
    
    Args:
        tensor: Tensor to check
        name: Name of the tensor (for logging)
        step: Current training step
        logger_instance: Logger instance (uses module logger if None)
        raise_on_nan: Whether to raise exception on NaN (default: False)
    
    Returns:
        True if NaN/Inf detected, False otherwise
    
    Example:
        >>> has_nan = detect_and_log_nan(gradients, "policy_gradients", step=1000)
        >>> if has_nan:
        >>>     # Take recovery action
    """
    log = logger_instance if logger_instance else logger
    
    if not torch.isfinite(tensor).all():
        # Detailed diagnostics
        nan_mask = torch.isnan(tensor)
        inf_mask = torch.isinf(tensor)
        nan_count = nan_mask.sum().item()
        inf_count = inf_mask.sum().item()
        
        # Get locations of first few NaN/Inf values
        nan_indices = torch.nonzero(nan_mask)[:5].cpu().numpy()
        inf_indices = torch.nonzero(inf_mask)[:5].cpu().numpy()
        
        # Statistics on finite values
        finite_mask = torch.isfinite(tensor)
        if finite_mask.any():
            finite_values = tensor[finite_mask]
            stats = {
                "min": finite_values.min().item(),
                "max": finite_values.max().item(),
                "mean": finite_values.mean().item(),
                "std": finite_values.std().item(),
            }
        else:
            stats = {"all_nonfinite": True}
        
        # Log detailed information
        log.error(
            f"[NaN Detection] Step {step}, Tensor '{name}':\n"
            f"  NaN count: {nan_count} ({100*nan_count/tensor.numel():.2f}%)\n"
            f"  Inf count: {inf_count} ({100*inf_count/tensor.numel():.2f}%)\n"
            f"  NaN locations (first 5): {nan_indices.tolist()}\n"
            f"  Inf locations (first 5): {inf_indices.tolist()}\n"
            f"  Finite value stats: {stats}"
        )
        
        if raise_on_nan:
            raise ValueError(f"NaN/Inf detected in {name} at step {step}")
        
        return True
    
    return False


def safe_exp(x: torch.Tensor, max_input: float = 20.0) -> torch.Tensor:
    """
    Safe exponential function that clips input to prevent overflow.
    
    torch.exp(x) overflows when x > ~88, producing Inf.
    This function clips input to a safe range.
    
    Args:
        x: Input tensor
        max_input: Maximum allowed input value (default: 20.0)
    
    Returns:
        exp(clipped_x)
    
    Example:
        >>> reward = safe_exp(-dist**2)
    """
    x_clipped = torch.clamp(x, max=-max_input, min=-max_input)
    return torch.exp(x_clipped)


def safe_sqrt(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Safe square root that prevents NaN from negative inputs.
    
    Args:
        x: Input tensor
        epsilon: Small positive constant (default: 1e-8)
    
    Returns:
        sqrt(max(x, epsilon))
    
    Example:
        >>> norm = safe_sqrt((vec**2).sum(dim=-1))
    """
    return torch.sqrt(torch.clamp(x, min=epsilon))


def safe_normalize_quaternion(quat: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Safely normalize quaternions to unit length.
    
    Handles edge cases:
    - Zero or near-zero quaternions
    - Already normalized quaternions
    
    Args:
        quat: Quaternion tensor [..., 4] (w, x, y, z) or (x, y, z, w)
        epsilon: Small constant for numerical stability
    
    Returns:
        Normalized quaternion
    
    Example:
        >>> quat_normalized = safe_normalize_quaternion(quat)
    """
    norm = torch.sqrt(torch.sum(quat**2, dim=-1, keepdim=True))
    norm_safe = torch.clamp(norm, min=epsilon)
    return quat / norm_safe


def check_inertia_matrix_condition(
    inertia: Union[np.ndarray, torch.Tensor],
    max_condition: float = 1000.0,
    regularization: float = 1e-6,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Check inertia matrix condition number and regularize if ill-conditioned.
    
    Ill-conditioned inertia matrices can cause:
    - Numerical instability in dynamics
    - Gradient issues
    - Simulation artifacts
    
    Args:
        inertia: 3x3 inertia matrix (numpy or torch)
        max_condition: Maximum allowed condition number (default: 1000)
        regularization: Amount to add to diagonal if ill-conditioned (default: 1e-6)
    
    Returns:
        Regularized inertia matrix (same type as input)
    
    Example:
        >>> inertia_stable = check_inertia_matrix_condition(inertia_matrix)
    """
    is_numpy = isinstance(inertia, np.ndarray)
    
    if is_numpy:
        inertia_np = inertia
    else:
        inertia_np = inertia.cpu().numpy()
    
    # Compute condition number via eigenvalues
    try:
        eigenvalues = np.linalg.eigvalsh(inertia_np)
        if eigenvalues.min() <= 0:
            logger.warning(
                f"Inertia matrix has non-positive eigenvalue: {eigenvalues.min():.2e}, "
                f"adding regularization"
            )
            inertia_np += np.eye(3) * regularization
            eigenvalues = np.linalg.eigvalsh(inertia_np)
        
        condition_number = eigenvalues.max() / (eigenvalues.min() + 1e-12)
        
        if condition_number > max_condition:
            logger.warning(
                f"Ill-conditioned inertia matrix detected: "
                f"condition number = {condition_number:.1f} > {max_condition}, "
                f"adding regularization"
            )
            inertia_np += np.eye(3) * regularization
    
    except np.linalg.LinAlgError:
        logger.error("Failed to compute eigenvalues for inertia matrix, adding regularization")
        inertia_np += np.eye(3) * regularization
    
    if is_numpy:
        return inertia_np
    else:
        return torch.from_numpy(inertia_np).to(inertia.device).to(inertia.dtype)


def check_physical_limits(
    obs_dict: Dict[str, torch.Tensor],
    velocity_limit: float = 100.0,
    angvel_limit: float = 50.0,
    acceleration_limit: Optional[float] = None,
    prev_velocity: Optional[torch.Tensor] = None,
    dt: float = 0.01,
) -> torch.Tensor:
    """
    Check if observations violate physical plausibility limits.
    
    Returns a boolean mask indicating which environments have violated limits.
    These environments should typically be reset.
    
    Args:
        obs_dict: Observation dictionary
        velocity_limit: Maximum plausible velocity magnitude (m/s)
        angvel_limit: Maximum plausible angular velocity magnitude (rad/s)
        acceleration_limit: Maximum plausible acceleration (m/s²), None to skip
        prev_velocity: Previous velocity for acceleration check
        dt: Time step for acceleration computation
    
    Returns:
        Boolean tensor [num_envs] indicating limit violations
    
    Example:
        >>> violation_mask = check_physical_limits(obs_dict, velocity_limit=50.0)
        >>> if violation_mask.any():
        >>>     env_ids = torch.nonzero(violation_mask).squeeze(-1)
        >>>     # Reset these environments
    """
    num_envs = None
    violation_mask = None
    
    # Check linear velocity
    if "robot_linvel" in obs_dict:
        linvel = obs_dict["robot_linvel"]
        num_envs = linvel.shape[0]
        vel_norm = torch.norm(linvel, dim=1)
        vel_violation = vel_norm > velocity_limit
        violation_mask = vel_violation
        
        if vel_violation.any():
            logger.warning(
                f"Velocity limit exceeded in {vel_violation.sum().item()} envs, "
                f"max vel: {vel_norm.max().item():.2f} m/s"
            )
    
    # Check angular velocity
    if "robot_body_angvel" in obs_dict:
        angvel = obs_dict["robot_body_angvel"]
        if num_envs is None:
            num_envs = angvel.shape[0]
        angvel_norm = torch.norm(angvel, dim=1)
        angvel_violation = angvel_norm > angvel_limit
        
        if violation_mask is None:
            violation_mask = angvel_violation
        else:
            violation_mask = violation_mask | angvel_violation
        
        if angvel_violation.any():
            logger.warning(
                f"Angular velocity limit exceeded in {angvel_violation.sum().item()} envs, "
                f"max angvel: {angvel_norm.max().item():.2f} rad/s"
            )
    
    # Check acceleration (if previous velocity provided)
    if acceleration_limit is not None and prev_velocity is not None and "robot_linvel" in obs_dict:
        current_velocity = obs_dict["robot_linvel"]
        acceleration = (current_velocity - prev_velocity) / dt
        accel_norm = torch.norm(acceleration, dim=1)
        accel_violation = accel_norm > acceleration_limit
        
        violation_mask = violation_mask | accel_violation
        
        if accel_violation.any():
            logger.warning(
                f"Acceleration limit exceeded in {accel_violation.sum().item()} envs, "
                f"max accel: {accel_norm.max().item():.2f} m/s²"
            )
    
    if violation_mask is None:
        # No checks performed, return all False
        if num_envs is None:
            raise ValueError("obs_dict must contain at least 'robot_linvel' or 'robot_body_angvel'")
        violation_mask = torch.zeros(num_envs, dtype=torch.bool, device=list(obs_dict.values())[0].device)
    
    return violation_mask


def sanitize_observation_dict(
    obs_dict: Dict[str, torch.Tensor],
    clip_limits: Optional[Dict[str, Tuple[float, float]]] = None,
    normalize_quaternions: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Comprehensive observation sanitization pipeline.
    
    Applies:
    1. Clipping to safe ranges
    2. Quaternion normalization
    3. NaN/Inf replacement with zeros (last resort)
    
    Args:
        obs_dict: Raw observation dictionary
        clip_limits: Custom clipping limits
        normalize_quaternions: Whether to normalize quaternion observations
    
    Returns:
        Sanitized observation dictionary
    
    Example:
        >>> obs_dict_safe = sanitize_observation_dict(obs_dict)
    """
    # Step 1: Clip to safe ranges
    sanitized = clip_observations(obs_dict, limits=clip_limits)
    
    # Step 2: Normalize quaternions
    if normalize_quaternions:
        quat_keys = ["robot_orientation", "robot_vehicle_orientation"]
        for key in quat_keys:
            if key in sanitized:
                sanitized[key] = safe_normalize_quaternion(sanitized[key])
    
    # Step 3: Replace any remaining NaN/Inf with zeros (last resort)
    for key, value in sanitized.items():
        # Skip non-tensor values (e.g., int, float, None)
        if not torch.is_tensor(value):
            continue
        if not torch.isfinite(value).all():
            logger.error(
                f"NaN/Inf found in '{key}' after clipping, replacing with zeros"
            )
            sanitized[key] = torch.where(
                torch.isfinite(value),
                value,
                torch.zeros_like(value)
            )
    
    return sanitized
