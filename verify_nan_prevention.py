#!/usr/bin/env python3
"""
Quick verification script for NaN prevention implementation.

This script performs basic sanity checks on the NaN prevention utilities
and configuration to ensure everything is properly integrated.

Usage:
    python verify_nan_prevention.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from aerial_gym.utils.nan_prevention_utils import (
    safe_normalize,
    clip_rewards,
    clip_observations,
    detect_and_log_nan,
    safe_exp,
    safe_sqrt,
    safe_normalize_quaternion,
    check_inertia_matrix_condition,
    check_physical_limits,
    sanitize_observation_dict,
)

def test_safe_normalize():
    """Test safe normalization with near-zero std."""
    print("\n[Test 1/10] safe_normalize...")
    x = torch.tensor([1.0, 2.0, 3.0])
    mean = torch.tensor([1.0, 1.0, 1.0])
    std = torch.tensor([1e-10, 1.0, 1.0])  # Near-zero std for first element
    
    result = safe_normalize(x, mean, std, epsilon=1e-8, clip_range=10.0)
    assert torch.isfinite(result).all(), "safe_normalize produced NaN/Inf!"
    assert (result.abs() <= 10.0).all(), "safe_normalize clip failed!"
    print("  ✓ Passed")

def test_clip_rewards():
    """Test reward clipping."""
    print("[Test 2/10] clip_rewards...")
    rewards = torch.tensor([-200.0, 50.0, 150.0, float('nan')])
    
    clipped = clip_rewards(rewards[:3], min_r=-100.0, max_r=100.0)
    assert clipped[0] == -100.0, "Reward min clip failed!"
    assert clipped[2] == 100.0, "Reward max clip failed!"
    print("  ✓ Passed")

def test_clip_observations():
    """Test observation clipping."""
    print("[Test 3/10] clip_observations...")
    obs_dict = {
        "robot_linvel": torch.tensor([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]]),
        "robot_body_angvel": torch.tensor([[30.0, 0.0, 0.0], [-30.0, 0.0, 0.0]]),
    }
    
    clipped = clip_observations(obs_dict)
    assert (clipped["robot_linvel"].abs() <= 50.0).all(), "Velocity clip failed!"
    assert (clipped["robot_body_angvel"].abs() <= 20.0).all(), "Angvel clip failed!"
    print("  ✓ Passed")

def test_detect_nan():
    """Test NaN detection."""
    print("[Test 4/10] detect_and_log_nan...")
    tensor_with_nan = torch.tensor([1.0, float('nan'), 3.0])
    
    has_nan = detect_and_log_nan(tensor_with_nan, "test_tensor", step=0, raise_on_nan=False)
    assert has_nan, "NaN detection failed!"
    
    tensor_finite = torch.tensor([1.0, 2.0, 3.0])
    has_nan = detect_and_log_nan(tensor_finite, "test_tensor", step=0, raise_on_nan=False)
    assert not has_nan, "False positive NaN detection!"
    print("  ✓ Passed")

def test_safe_exp():
    """Test safe exponential function."""
    print("[Test 5/10] safe_exp...")
    x_extreme = torch.tensor ([100.0, -100.0, 1.0])
    
    result = safe_exp(x_extreme, max_input=20.0)
    assert torch.isfinite(result).all(), "safe_exp produced Inf!"
    assert result[0] < 1e10, "safe_exp overflow not prevented!"
    print("  ✓ Passed")

def test_safe_sqrt():
    """Test safe square root."""
    print("[Test 6/10] safe_sqrt...")
    x = torch.tensor([4.0, -1.0, 0.0])  # Negative input
    
    result = safe_sqrt(x, epsilon=1e-8)
    assert torch.isfinite(result).all(), "safe_sqrt produced NaN!"
    assert result[1] > 0, "safe_sqrt negative handling failed!"
    print("  ✓ Passed")

def test_safe_normalize_quaternion():
    """Test quaternion normalization."""
    print("[Test 7/10] safe_normalize_quaternion...")
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1e-10, 1e-10, 1e-10, 1e-10]])  # Near-zero quat
    
    result = safe_normalize_quaternion(quat, epsilon=1e-8)
    assert torch.isfinite(result).all(), "Quaternion normalization produced NaN!"
    norms = torch.sqrt((result ** 2).sum(dim=1))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Quaternion not unit length!"
    print("  ✓ Passed")

def test_check_inertia_matrix():
    """Test inertia matrix condition check."""
    print("[Test 8/10] check_inertia_matrix_condition...")
    # Well-conditioned matrix
    inertia_good = np.eye(3) * 0.1
    result = check_inertia_matrix_condition(inertia_good, max_condition=1000.0)
    assert np.allclose(result, inertia_good, atol=1e-6), "Good matrix was modified!"
    
    # Ill-conditioned matrix
    inertia_bad = np.diag([1.0, 0.001, 0.001])  # High condition number
    result = check_inertia_matrix_condition(inertia_bad, max_condition=100.0, regularization=1e-5)
    eigenvalues = np.linalg.eigvalsh(result)
    condition = eigenvalues.max() / eigenvalues.min()
    assert condition < 1000.0, "Ill-conditioned matrix not regularized!"
    print("  ✓ Passed")

def test_check_physical_limits():
    """Test physical limits checking."""
    print("[Test 9/10] check_physical_limits...")
    obs_dict = {
        "robot_linvel": torch.tensor([[50.0, 0.0, 0.0], [150.0, 0.0, 0.0]]),  # Second exceeds limit
        "robot_body_angvel": torch.tensor([[10.0, 0.0, 0.0], [60.0, 0.0, 0.0]]),  # Second exceeds limit
    }
    
    violations = check_physical_limits(obs_dict, velocity_limit=100.0, angvel_limit=50.0)
    assert violations[0] == False, "False positive violation!"
    assert violations[1] == True, "Missed physical limit violation!"
    print("  ✓ Passed")

def test_sanitize_observation_dict():
    """Test comprehensive observation sanitization."""
    print("[Test 10/10] sanitize_observation_dict...")
    obs_dict = {
        "robot_position": torch.tensor([[100.0, 0.0, 0.0]]),  # Extreme position
        "robot_linvel": torch.tensor([[200.0, 0.0, 0.0]]),  # Extreme velocity
        "robot_orientation": torch.tensor([[2.0, 0.0, 0.0, 0.0]]),  # Non-unit quaternion
    }
    
    sanitized = sanitize_observation_dict(obs_dict, normalize_quaternions=True)
    
    # Check clipping
    assert (sanitized["robot_position"].abs() <= 50.0).all(), "Position clip failed!"
    assert (sanitized["robot_linvel"].abs() <= 50.0).all(), "Velocity clip failed!"
    
    # Check quaternion normalization
    quat_norm = torch.sqrt((sanitized["robot_orientation"] ** 2).sum())
    assert torch.isclose(quat_norm, torch.tensor(1.0), atol=1e-5), "Quaternion not normalized!"
    
    # Check no NaN/Inf
    for key, tensor in sanitized.items():
        assert torch.isfinite(tensor).all(), f"Sanitized dict contains NaN/Inf in {key}!"
    
    print("  ✓ Passed")

def test_config_integration():
    """Test that config parameters are properly loaded."""
    print("\n[Test 11/11] Configuration integration...")
    
    # Import task config
    try:
        from aerial_gym.config.task_config.payload_compensation_task_teacher_config import task_config
        print("  ✓ Config import successful")
    except Exception as e:
        print(f"  ✗ Config import failed: {e}")
        return False
    
    # Check YAML config
    import yaml
    yaml_path = os.path.join(
        os.path.dirname(__file__), 
        '../aerial_gym/rl_training/rl_games/ppo_aerial_quad_aux.yaml'
    )
    
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check key parameters
        assert config['params']['config']['grad_norm'] == 0.5, "grad_norm not updated!"
        assert config['params']['config']['e_clip'] == 0.15, "e_clip not updated!"
        assert config['params']['config']['mini_epochs'] == 3, "mini_epochs not updated!"
        
        # Check NaN prevention parameters
        assert 'reward_clip_min' in config['params']['config'], "reward_clip_min missing!"
        assert 'skip_nan_gradients' in config['params']['config'], "skip_nan_gradients missing!"
        
        print("  ✓ YAML config validation passed")
        print(f"    - grad_norm: {config['params']['config']['grad_norm']}")
        print(f"    - e_clip: {config['params']['config']['e_clip']}")
        print(f"    - mini_epochs: {config['params']['config']['mini_epochs']}")
        print(f"    - skip_nan_gradients: {config['params']['config']['skip_nan_gradients']}")
        
    except Exception as e:
        print(f"  ✗ YAML validation failed: {e}")
        return False
    
    return True

def main():
    """Run all verification tests."""
    print("="*60)
    print("NaN Prevention Implementation Verification")
    print("="*60)
    
    tests = [
        test_safe_normalize,
        test_clip_rewards,
        test_clip_observations,
        test_detect_nan,
        test_safe_exp,
        test_safe_sqrt,
        test_safe_normalize_quaternion,
        test_check_inertia_matrix,
        test_check_physical_limits,
        test_sanitize_observation_dict,
        test_config_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test()
            if result is False:
                failed += 1
            else:
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if failed == 0:
        print("\n✅ All tests passed! NaN prevention is properly integrated.")
        print("\nNext steps:")
        print("1. Run a short training test (5000 epochs)")
        print("2. Monitor TensorBoard for NaN skip counts")
        print("3. Check training logs for physical limit violations")
        print("\nSee walkthrough.md for detailed usage guide.")
        return 0
    else:
        print(f"\n❌ {failed} tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
