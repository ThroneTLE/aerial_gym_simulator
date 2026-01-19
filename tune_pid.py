#!/usr/bin/env python3
"""
PID Tuning Script for 1.5kg Quadrotor with 450mm Wheelbase

This script tests the Lee position controller with the new mass/inertia parameters
and helps tune the PID gains through step response tests.

Features:
- Runtime PID gain adjustment via command line
- Proper payload loading sequence for disturbance tests
- Settling time with sustained-in-band check

Usage:
    # Hover test with default gains
    python tune_pid.py --test_type hover

    # Step response with custom gains
    python tune_pid.py --test_type step --k_pos 6.0 --k_rot 24.0

    # Disturbance test with payload
    python tune_pid.py --test_type disturbance --payload_mass 0.4
"""

import argparse
from aerial_gym.registry.task_registry import task_registry
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="PID Tuning for Updated Quadrotor")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration in seconds")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to file")
    parser.add_argument("--test_type", type=str, default="hover", 
                        choices=["hover", "step", "disturbance", "all"],
                        help="Type of test to run")
    
    # Runtime PID gain adjustment
    parser.add_argument("--k_pos", type=float, default=None, help="Override K_pos gain (single value for all axes)")
    parser.add_argument("--k_vel", type=float, default=None, help="Override K_vel gain")
    parser.add_argument("--k_rot", type=float, default=None, help="Override K_rot gain")
    parser.add_argument("--k_angvel", type=float, default=None, help="Override K_angvel gain")
    
    # Payload for disturbance test
    parser.add_argument("--payload_mass", type=float, default=0.4, help="Payload mass per attachment (kg)")
    
    # Settling criteria
    parser.add_argument("--settle_tolerance", type=float, default=0.05, help="Settling tolerance (m)")
    parser.add_argument("--settle_duration", type=float, default=0.5, help="Required time to stay in tolerance (s)")
    
    return parser.parse_args()


def create_task(num_envs: int, headless: bool = True, payload_mass: float = 0.0):
    """Create the payload compensation task for testing."""
    from aerial_gym.config.task_config.payload_compensation_task_teacher_config import task_config
    
    
    # Enable viewer for user demonstration
    task_config.headless = False
    
    # Configure payload - this will be respected on reset
    task_config.payload_parameters["randomize_payload_mass"] = False
    task_config.payload_parameters["payload_mass"] = payload_mass  # Will be used by reset
    task_config.payload_parameters["release_start"] = 999999  # No release
    
    # Ensure zero initial noise so reset() puts us at 0,0,1 safely
    if "randomization_parameters" in task_config.__dict__:
        task_config.randomization_parameters["initial_position_noise"] = [0.0, 0.0, 0.0]
        task_config.randomization_parameters["initial_orientation_noise_deg"] = [0.0, 0.0, 0.0]
        # Disable other randomizations to prevent instability at spawn
        task_config.randomization_parameters["randomize_motor_thrust_constant"] = False
        task_config.randomization_parameters["randomize_motor_time_constant"] = False
        task_config.randomization_parameters["randomize_drag_coefficients"] = False
        task_config.randomization_parameters["randomize_external_disturbance"] = False
        
    # Relax crash thresholds for tuning
    task_config.crash_distance_threshold = 10.0  # Allow large drops
    task_config.crash_tilt_threshold_deg = 80.0  # Allow extreme tilts
    
    task = task_registry.make_task(
        "payload_compensation_task_teacher",
        num_envs=num_envs,
        headless=headless,
    )
    return task


def apply_pid_gains(task, args):
    """Apply runtime PID gain overrides to the controller."""
    controller = task.sim_env.robot_manager.robot.controller
    
    gains_modified = False
    
    if args.k_pos is not None:
        controller.K_pos_tensor_current[:] = args.k_pos
        print(f"  K_pos overridden to: {args.k_pos}")
        gains_modified = True
    
    if args.k_vel is not None:
        controller.K_linvel_tensor_current[:] = args.k_vel
        print(f"  K_vel overridden to: {args.k_vel}")
        gains_modified = True
    
    if args.k_rot is not None:
        controller.K_rot_tensor_current[:] = args.k_rot
        print(f"  K_rot overridden to: {args.k_rot}")
        gains_modified = True
    
    if args.k_angvel is not None:
        controller.K_angvel_tensor_current[:] = args.k_angvel
        print(f"  K_angvel overridden to: {args.k_angvel}")
        gains_modified = True
    
    return gains_modified


def get_sim_dt(task) -> float:
    """Get the actual simulation dt from the task."""
    try:
        return task.sim_env.IGE_env.sim_config.dt
    except:
        return 0.01  # Default fallback


def run_hover_test(task, duration: float, args):
    """Test hovering stability at origin."""
    print("\n" + "="*60)
    print("HOVER TEST: Checking if drone can maintain stable hover")
    print("="*60)
    
    dt = get_sim_dt(task)
    num_steps = int(duration / dt)
    print(f"Sim dt: {dt}s, Steps: {num_steps}")
    
    # Data collection
    positions = []
    velocities = []
    angular_velocities = []
    times = []
    
    # Reset and set target to origin
    task.reset()
    
    # Re-apply PID gains after reset (in case randomize_params wiped them)
    apply_pid_gains(task, args)
    
    task.target_position[:] = torch.tensor([0.0, 0.0, 1.0], device=task.device)
    
    # Zero compensation actions
    actions = torch.zeros((task.num_envs, task.action_space_dim), device=task.device)
    
    for step in range(num_steps):
        obs_dict = task.step(actions)
        
        # Collect data
        pos = task.sim_env.IGE_env.global_tensor_dict["robot_position"].clone().cpu().numpy()
        vel = task.sim_env.IGE_env.global_tensor_dict["robot_velocity"].clone().cpu().numpy()
        angvel = task.sim_env.IGE_env.global_tensor_dict["robot_body_angvel"].clone().cpu().numpy()
        
        positions.append(pos)
        velocities.append(vel)
        angular_velocities.append(angvel)
        times.append(step * dt)
        
        # Check for crashes
        if task.terminations.any():
            print(f"⚠️ Crash detected at step {step} ({step*dt:.2f}s)")
            break
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    angular_velocities = np.array(angular_velocities)
    times = np.array(times)
    
    # Analyze results
    print("\n--- Hover Test Results ---")
    final_pos = positions[-1, 0]  # First env
    target_pos = np.array([0.0, 0.0, 1.0])
    pos_error = np.linalg.norm(final_pos - target_pos)
    
    print(f"Target Position: {target_pos}")
    print(f"Final Position:  {final_pos}")
    print(f"Position Error:  {pos_error:.4f} m")
    
    # Check stability
    pos_std = positions[:, 0].std(axis=0)
    vel_std = velocities[:, 0].std(axis=0)
    angvel_std = angular_velocities[:, 0].std(axis=0)
    
    print(f"\nPosition Std Dev: {pos_std}")
    print(f"Velocity Std Dev: {vel_std}")
    print(f"AngVel Std Dev:   {angvel_std}")
    
    # Verdict
    stable = pos_error < 0.5 and np.all(pos_std < 0.2)
    print(f"\n{'✅ STABLE' if stable else '❌ UNSTABLE'}")
    
    return {
        "positions": positions,
        "velocities": velocities,
        "angular_velocities": angular_velocities,
        "times": times,
        "stable": stable,
        "pos_error": pos_error,
    }


def run_step_response_test(task, duration: float, args):
    """Test step response to position change."""
    print("\n" + "="*60)
    print("STEP RESPONSE TEST: Checking position tracking")
    print("="*60)
    
    dt = get_sim_dt(task)
    num_steps = int(duration / dt)
    step_time = duration / 3  # Step at 1/3 of duration
    
    print(f"Sim dt: {dt}s, Steps: {num_steps}")
    print(f"Step will occur at t={step_time:.2f}s")
    
    # Data collection
    positions = []
    targets = []
    times = []
    
    # Reset
    task.reset()
    
    # Re-apply PID gains
    apply_pid_gains(task, args)
    
    initial_target = torch.tensor([0.0, 0.0, 1.0], device=task.device)
    step_target = torch.tensor([1.0, 0.0, 1.0], device=task.device)  # Step in X
    task.target_position[:] = initial_target
    
    actions = torch.zeros((task.num_envs, task.action_space_dim), device=task.device)
    
    for step in range(num_steps):
        t = step * dt
        
        # Apply step at step_time
        if t >= step_time:
            task.target_position[:] = step_target
        
        obs_dict = task.step(actions)
        
        pos = task.sim_env.IGE_env.global_tensor_dict["robot_position"].clone().cpu().numpy()
        positions.append(pos)
        targets.append(task.target_position[0].cpu().numpy())
        times.append(t)
        
        if task.terminations.any():
            print(f"⚠️ Crash detected at step {step}")
            break
    
    positions = np.array(positions)
    targets = np.array(targets)
    times = np.array(times)
    
    # Analyze step response with sustained-in-band check
    print("\n--- Step Response Analysis ---")
    
    step_idx = int(step_time / dt)
    final_target = step_target.cpu().numpy()
    import math
    settle_steps_required = max(1, math.ceil(args.settle_duration / dt))  # Ensure at least 1 step, use ceil
    
    settling_time = None
    consecutive_in_band = 0
    
    for i in range(step_idx, len(positions)):
        error = np.linalg.norm(positions[i, 0] - final_target)
        if error < args.settle_tolerance:
            consecutive_in_band += 1
            if consecutive_in_band >= settle_steps_required:
                settling_time = times[i - settle_steps_required + 1] - step_time
                break
        else:
            consecutive_in_band = 0
    
    if settling_time is not None:
        print(f"✅ Settling Time: {settling_time:.3f}s (stayed in {args.settle_tolerance}m band for {args.settle_duration}s)")
    else:
        print(f"❌ Did not settle (need to stay in {args.settle_tolerance}m for {args.settle_duration}s)")
    
    # Check overshoot
    x_positions = positions[step_idx:, 0, 0]  # X component after step
    overshoot = max(0, x_positions.max() - final_target[0])
    undershoot = max(0, final_target[0] - x_positions.min())
    print(f"Overshoot: {overshoot:.4f} m ({overshoot*100:.1f}%)")
    print(f"Undershoot: {undershoot:.4f} m")
    
    return {
        "positions": positions,
        "targets": targets,
        "times": times,
        "settling_time": settling_time,
        "overshoot": overshoot,
    }


def run_disturbance_test(task, duration: float, args):
    """Test disturbance rejection with payload attached."""
    print("\n" + "="*60)
    print("DISTURBANCE TEST: Testing with payload attached")
    print(f"Payload mass per attachment: {args.payload_mass} kg")
    print(f"Total payload: {args.payload_mass * 4} kg")
    print("="*60)
    
    dt = get_sim_dt(task)
    num_steps = int(duration / dt)
    
    # IMPORTANT: Reset FIRST, then modify payload mass
    # reset() calls _sample_payload_mass which uses the config value
    # So we already set payload_mass in create_task, but let's also
    # directly set it after reset to be sure
    # 2. Reset Envs (this puts them at 0,0,1 because noise is disabled in create_task)
    # We call reset() again to ensure everything is clean
    task.reset()
    
    # Force 5m height
    task.sim_env.IGE_env.global_tensor_dict["robot_position"][:, 2] = 5.0
    task.sim_env.IGE_env.global_tensor_dict["robot_linvel"][:] = 0.0
    task.sim_env.IGE_env.write_to_sim()
    task.sim_env.IGE_env.refresh_tensors()
    
    # Re-apply PID again just in case reset cleared it
    apply_pid_gains(task, args)
    
    # 3. Apply Payload
    # Override payload mass after reset
    task.payload_manager.payload_mass_per_env[:] = args.payload_mass
    
    # Enable only ONE payload per env
    # attached_mask shape: (num_envs, num_payloads)
    task.payload_manager.attached_mask[:] = False
    task.payload_manager.attached_mask[:, 0] = True
    
    task.payload_manager._update_mass_properties(
        torch.arange(task.sim_env.num_envs, device=task.device)
    )
    
    # Verify payload
    actual_payload = task.payload_manager.current_payload_mass[0].item()
    print(f"Verified current payload mass: {actual_payload:.4f} kg")
    
    # Wait one step to let physics settle
    # task.step(torch.zeros((task.sim_env.num_envs, 4), device=task.device)) -> No, step 0 check will catch it
    
    positions = []
    times = []
    
    actions = torch.zeros((task.sim_env.num_envs, task.action_space_dim), device=task.device)
    
    for step in range(num_steps):
        # Keep target at 5m
        task.target_position[:] = torch.tensor([0.0, 0.0, 5.0], device=task.device)
        
        obs_dict = task.step(actions)
        
        pos = task.sim_env.IGE_env.global_tensor_dict["robot_position"].clone().cpu().numpy()
        positions.append(pos)
        times.append(step * dt)
        
        if task.terminations.any():
            print(f"⚠️ Crash detected at step {step}")
            break
    
    positions = np.array(positions)
    times = np.array(times)
    
    # Analyze
    print("\n--- Disturbance Test Results ---")
    final_pos = positions[-1, 0]
    target = np.array([0.0, 0.0, 5.0])
    error = np.linalg.norm(final_pos - target)
    
    print(f"With {args.payload_mass}kg × 1 = {args.payload_mass*1}kg payload attached:")
    print(f"Final Position: {final_pos}")
    print(f"Final Position Error: {error:.4f} m")
    print(f"Final Z Position: {final_pos[2]:.4f} m (target: 5.0)")
    
    # Z drop is expected with payload if no thrust compensation
    z_drop = 5.0 - final_pos[2]
    print(f"Z Drop due to payload: {z_drop:.4f} m")
    
    # Check if hover is maintained
    stable = len(positions) == num_steps and final_pos[2] > 0.3
    print(f"\n{'✅ STABLE with payload' if stable else '❌ CRASHED or UNSTABLE'}")
    
    if stable and z_drop > 0.1:
        print(f"⚠️ Note: Z drop of {z_drop:.2f}m is expected without thrust compensation")
    
    return {
        "positions": positions,
        "times": times,
        "stable": stable,
        "error": error,
        "z_drop": z_drop,
    }


def plot_results(results: dict, test_type: str, save: bool = False):
    """Plot test results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"PID Tuning Test: {test_type.upper()}", fontsize=14)
    
    times = results["times"]
    positions = results["positions"]
    
    # Position X, Y, Z
    ax = axes[0, 0]
    ax.plot(times, positions[:, 0, 0], label="X")
    ax.plot(times, positions[:, 0, 1], label="Y")
    ax.plot(times, positions[:, 0, 2], label="Z")
    if "targets" in results:
        targets = results["targets"]
        ax.plot(times, targets[:, 0], "--", alpha=0.5, label="Target X")
        ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.3, label="Target Z")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.set_title("Position vs Time")
    ax.legend()
    ax.grid(True)
    
    # Velocities
    ax = axes[0, 1]
    if "velocities" in results:
        velocities = results["velocities"]
        ax.plot(times, velocities[:, 0, 0], label="Vx")
        ax.plot(times, velocities[:, 0, 1], label="Vy")
        ax.plot(times, velocities[:, 0, 2], label="Vz")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity vs Time")
    ax.legend()
    ax.grid(True)
    
    # Position error
    ax = axes[1, 0]
    target = np.array([0.0, 0.0, 1.0])
    if "targets" in results:
        errors = np.linalg.norm(positions[:, 0] - results["targets"], axis=1)
    else:
        errors = np.linalg.norm(positions[:, 0] - target, axis=1)
    ax.plot(times, errors)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Position Error vs Time")
    ax.grid(True)
    
    # Angular velocities
    ax = axes[1, 1]
    if "angular_velocities" in results:
        angvel = results["angular_velocities"]
        ax.plot(times, angvel[:, 0, 0], label="ωx")
        ax.plot(times, angvel[:, 0, 1], label="ωy")
        ax.plot(times, angvel[:, 0, 2], label="ωz")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.set_title("Angular Velocity vs Time")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pid_tuning_{test_type}_{timestamp}.png"
        plt.savefig(filename, dpi=150)
        print(f"Plot saved to: {filename}")
    else:
        plt.show()


def print_current_gains():
    """Print current PID gains from config."""
    from aerial_gym.config.controller_config.lee_controller_config import control
    print("\nCurrent PID Gains (from config):")
    print(f"  K_pos: {control.K_pos_tensor_min} ~ {control.K_pos_tensor_max}")
    print(f"  K_vel: {control.K_vel_tensor_min} ~ {control.K_vel_tensor_max}")
    print(f"  K_rot: {control.K_rot_tensor_min} ~ {control.K_rot_tensor_max}")
    print(f"  K_angvel: {control.K_angvel_tensor_min} ~ {control.K_angvel_tensor_max}")


def main():
    args = parse_args()
    
    print("="*60)
    print("PID TUNING SCRIPT")
    print("="*60)
    print(f"Quadrotor: 1.5kg, 450mm wheelbase")
    print(f"Test Type: {args.test_type}")
    print(f"Duration: {args.duration}s")
    print(f"Num Envs: {args.num_envs}")
    print("="*60)
    
    # Print current gains
    print_current_gains()
    
    # Create task with appropriate payload
    payload_for_create = args.payload_mass if args.test_type == "disturbance" else 0.0
    task = create_task(args.num_envs, headless=not args.visualize, payload_mass=payload_for_create)
    
    # Apply runtime PID overrides
    print("\nApplying PID overrides:")
    if apply_pid_gains(task, args):
        print("  (Gains modified)")
    else:
        print("  (Using default gains)")
    
    # Run tests
    all_results = {}
    
    if args.test_type == "hover" or args.test_type == "all":
        results = run_hover_test(task, args.duration, args)
        all_results["hover"] = results
        if args.save_plots or args.visualize:
            plot_results(results, "hover", save=args.save_plots)
    
    if args.test_type == "step" or args.test_type == "all":
        results = run_step_response_test(task, args.duration, args)
        all_results["step"] = results
        if args.save_plots or args.visualize:
            plot_results(results, "step", save=args.save_plots)
    
    if args.test_type == "disturbance" or args.test_type == "all":
        results = run_disturbance_test(task, args.duration, args)
        all_results["disturbance"] = results
        if args.save_plots or args.visualize:
            plot_results(results, "disturbance", save=args.save_plots)
    
    # Cleanup
    task.close()
    
    print("\n" + "="*60)
    print("PID TUNING COMPLETE")
    print("="*60)
    
    # Provide tuning suggestions based on results
    print("\n📋 TUNING SUGGESTIONS:")
    
    if "hover" in all_results:
        r = all_results["hover"]
        if not r.get("stable", True):
            print("  [Hover] ❌ Unstable - try reducing K_rot and K_angvel")
        elif r.get("pos_error", 0) > 0.1:
            print("  [Hover] ⚠️ Large steady-state error - increase K_pos")
    
    if "step" in all_results:
        r = all_results["step"]
        if r.get("settling_time") is None:
            print("  [Step] ❌ Did not settle - check for oscillation, reduce K_rot")
        elif r.get("settling_time", 0) > 2.0:
            print("  [Step] ⚠️ Slow settling - increase K_pos and K_vel")
        if r.get("overshoot", 0) > 0.2:
            print("  [Step] ⚠️ Large overshoot - reduce K_pos, increase K_vel")
    
    if "disturbance" in all_results:
        r = all_results["disturbance"]
        if not r.get("stable", True):
            print("  [Disturbance] ❌ Crashed with payload - gains too aggressive")
        elif r.get("z_drop", 0) > 0.3:
            print("  [Disturbance] ⚠️ Large Z drop - expected without compensation, but check K_pos[2]")
    
    print("\n💡 To try different gains:")
    print("   python tune_pid.py --test_type hover --k_pos 5.0 --k_rot 20.0")


if __name__ == "__main__":
    main()
