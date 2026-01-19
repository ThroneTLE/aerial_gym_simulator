#!/usr/bin/env python3
"""
Parallel PID Auto-Tuning Script using CMA-ES / Evolution Strategy
For Aerial Gym Simulator

Features:
- Massively Parallel: Takes full advantage of Isaac Gym by running DIFFERENT PID parameters 
  in each environment simultaneously.
- Evolution Strategy:
  1. Initialize population of parameters around a mean.
  2. Evaluate all candidates in parallel (one simulation run).
  3. Select top performers (Elitism).
  4. Generate next generation via mutation and crossover.
- Speed: Can tune PID in minutes instead of hours.

Usage:
    python auto_tune_pid.py --num_envs 2048 --generations 20
"""

import argparse
import numpy as np
import time
import copy
from aerial_gym.registry.task_registry import task_registry
import torch

# Define tuning parameters config
# Bounds: [min, max] for [K_pos, K_vel, K_rot, K_angvel]
# Expanded ranges based on recent manual tuning experience
PARAM_BOUNDS = torch.tensor([
    [1.0, 40.0],   # K_pos (Mass 2x -> 2x gain)
    [0.1, 20.0],   # K_vel (Mass 2x -> 2x gain)
    [1.0, 400.0],   # K_rot (Inertia 11x -> 11x gain)
    [0.1, 50.0]     # K_angvel (Inertia 11x -> 11x gain)
])

PARAM_NAMES = ["K_pos", "K_vel", "K_rot", "K_angvel"]
NUM_PARAMS = 4

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel PID Auto-Tuning")
    parser.add_argument("--num_envs", type=int, default=2048, help="Population size (must be large, e.g. 2048)")
    parser.add_argument("--generations", type=int, default=30, help="Number of evolutionary generations")
    parser.add_argument("--duration", type=float, default=2.5, help="Eval duration per generation (seconds)")
    parser.add_argument("--top_k", type=int, default=64, help="Number of elite survivors to breed next gen")
    parser.add_argument("--payloads", type=int, default=4, help="Number of payloads to attach (0-4)")
    return parser.parse_args()

def create_task(num_envs: int, headless: bool = True, num_payloads: int = 4):
    from aerial_gym.config.task_config.payload_compensation_task_teacher_config import task_config
    
    # 1. Enable Payload for Tuning
    task_config.payload_parameters["payload_mass"] = 0.4  # 0.4kg per payload
    
    # Dynamic Payload Count Logic
    # We DO NOT slice offsets because Observation Space expects 4 payloads.
    # Instead, we will handle the active payloads by forcing attached_mask in main()
    # But we DO need to handle the pure mass calculation if any.
    
    print(f"DEBUG: Configuring Task for {num_payloads} payloads (Configured to spawn 4, but will mask to {num_payloads})")
    
    task_config.payload_parameters["release_start"] = 999999 # Never release
    task_config.payload_parameters["randomize_release"] = False
    task_config.payload_parameters["release_start_range"] = [999999, 999999]
    task_config.payload_parameters["randomize_payload_mass"] = False
    task_config.teacher_mode = True # Enable Physical Comp (Gravity/Inertia)
    task_config.use_omniscient_gains = False # Disable Shadow Controller during tuning
    
    # 2. Disable Initial State Randomization
    if "randomization_parameters" in task_config.__dict__:
        rp = task_config.randomization_parameters
        rp["initial_position_noise"] = [0.0, 0.0, 0.0]
        rp["initial_orientation_noise_deg"] = [0.0, 0.0, 0.0]
        
        # 3. Disable Physical Randomizations (Keep Mass/Inertia Fixed for tuning)
        rp["mass_jitter"] = {"enabled": False}
        rp["inertia_jitter"] = {"enabled": False}
        rp["thrust_scale_jitter"] = {"enabled": False}
        
        rp["randomize_motor_thrust_constant"] = False
        rp["randomize_motor_time_constant"] = False
        rp["randomize_drag_coefficients"] = False
        rp["randomize_external_disturbance"] = False
        
        rp["external_force_range"] = [0.0, 0.0]
        rp["external_torque_range"] = [0.0, 0.0]
    
    # Create task
    task = task_registry.make_task(
        "payload_compensation_task_teacher",
        num_envs=num_envs,
        headless=headless,
    )
    return task

def set_parallel_gains(task, population_params):
    """
    Apply a DIFFERENT set of gains to EACH environment.
    population_params: tensor of shape (num_envs, 4) -> [K_pos, K_vel, K_rot, K_angvel]
    """
    controller = task.sim_env.robot_manager.robot.controller
    
    # The controller usually expects single values or (1,3) tensors for constants.
    # We need to hack/ensure it uses the per-env tensors if it supports it, 
    # OR we modify the underlying tensor data directly.
    
    # In LeeController, K_*_tensor_current are typically broadcasted or set from config.
    # We need to overwrite them with our per-env population data.
    # Assuming LeeController uses: self.K_pos_tensor_current of shape (num_envs, 3)
    
    # Map population (num_envs, 4) to controller tensors (num_envs, 3)
    # K_pos -> (K_pos, K_pos, K_pos)
    # K_vel -> (K_vel, K_vel, K_vel)
    # ...
    
    # [num_envs, 1] -> [num_envs, 3]
    k_pos_expanded = population_params[:, 0].unsqueeze(1).expand(-1, 3)
    k_vel_expanded = population_params[:, 1].unsqueeze(1).expand(-1, 3)
    k_rot_expanded = population_params[:, 2].unsqueeze(1).expand(-1, 3)
    k_angvel_expanded = population_params[:, 3].unsqueeze(1).expand(-1, 3)
    
    # Modify the tensors in place
    controller.K_pos_tensor_current[:] = k_pos_expanded
    controller.K_linvel_tensor_current[:] = k_vel_expanded
    controller.K_rot_tensor_current[:] = k_rot_expanded
    controller.K_angvel_tensor_current[:] = k_angvel_expanded

def get_sim_dt(task) -> float:
    try:
        return task.sim_env.IGE_env.global_tensor_dict["dt"]
    except:
        return 0.01

def evaluate_generation(task, population_params, duration=5.0):
    """
    Evaluate the entire population in ONE simulation run.
    Returns: costs (num_envs,)
    """
    num_envs = task.sim_env.num_envs
    device = task.device
    
    # 1. Apply Gains
    set_parallel_gains(task, population_params)
    
    # 2. Reset Env
    task.reset()
    # Re-apply gains after reset (just in case reset overwrites them)
    set_parallel_gains(task, population_params)
    
    # 3. Force Deterministic Initial State (0,0,1)
    task.sim_env.IGE_env.global_tensor_dict["robot_position"][:] = torch.tensor([0.0, 0.0, 1.0], device=device)
    task.sim_env.IGE_env.global_tensor_dict["robot_linvel"][:] = 0.0
    task.sim_env.IGE_env.global_tensor_dict["robot_angvel"][:] = 0.0
    task.sim_env.IGE_env.global_tensor_dict["robot_orientation"][:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    task.sim_env.IGE_env.global_tensor_dict["robot_body_angvel"][:] = 0.0
    
    task.sim_env.IGE_env.write_to_sim()
    task.sim_env.IGE_env.refresh_tensors()
    
    # 4. Target Trajectory: Step X from 0.0 to 1.0
    start_pos = torch.tensor([0.0, 0.0, 1.0], device=device)
    target_pos = torch.tensor([1.0, 0.0, 1.0], device=device)
    
    task.target_position[:] = start_pos
    
    dt = get_sim_dt(task)
    num_steps = int(duration / dt)
    step_time = 1.0 # Step happens at 1.0s
    
    # Cost Accumulators
    total_ise = torch.zeros(num_envs, device=device)
    max_overshoot = torch.zeros(num_envs, device=device)
    crashed_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
    
    actions = torch.zeros((num_envs, task.action_space_dim), device=device)
    
    # Simulation Loop
    for i in range(num_steps):
        if i % 100 == 0:
            print(f"  Step {i}/{num_steps}", flush=True)
        t = i * dt
        
        # Step Signal
        if t >= step_time:
            task.target_position[:] = target_pos
            current_target = target_pos
        else:
            task.target_position[:] = start_pos
            current_target = start_pos
            
        task.step(actions)
        
        # -- Cost Calculation Term by Term --
        
        # 1. Crash Check
        if task.terminations.any():
            crashed_envs |= task.terminations
            
        # 2. Tracking Error (ISE) & Overshoot
        if t >= step_time:
            pos = task.sim_env.IGE_env.global_tensor_dict["robot_position"]
            error_vec = pos - current_target
            squared_error = torch.sum(error_vec**2, dim=1) # (num_envs,)
            
            # Add to total ISE
            total_ise += squared_error * dt
            
            # Overshoot (X axis only)
            overshoot_val = pos[:, 0] - current_target[0]
            overshoot_val = torch.clamp(overshoot_val, min=0.0)
            max_overshoot = torch.max(max_overshoot, overshoot_val)

    # Final Cost Calculation
    # Cost = ISE + 5.0 * Overshoot + Penalty(Crash)
    costs = total_ise + 5.0 * max_overshoot
    
    # Apply huge penalty to crashed envs
    costs[crashed_envs] += 10000.0
    
    # Sanity check for NaNs
    costs[torch.isnan(costs)] = 20000.0
    
    return costs

def generate_initial_population(num_envs, device):
    """
    Uniform random sampling within bounds.
    """
    bounds_min = PARAM_BOUNDS[:, 0].to(device)
    bounds_max = PARAM_BOUNDS[:, 1].to(device)
    
    # Random values in [0, 1]
    rand_norm = torch.rand((num_envs, NUM_PARAMS), device=device)
    
    # Scale to bounds
    population = bounds_min + rand_norm * (bounds_max - bounds_min)
    return population

def mutate(parents, num_offspring, mutation_rate=0.1, mutation_scale=0.2):
    """
    Generate offspring by adding Gaussian noise to parents.
    """
    device = parents.device
    num_parents = parents.shape[0]
    
    # Repeat parents to fill offspring count
    # e.g. if we need 2000 offspring from 64 parents
    repeats = num_offspring // num_parents + 1
    offspring = parents.repeat(repeats, 1)[:num_offspring]
    
    # Apply mutation
    # Mask for mutation probability
    mask = (torch.rand_like(offspring) < mutation_rate)
    
    # Gaussian noise relative to current value (e.g. +/- 20%)
    noise = torch.randn_like(offspring) * mutation_scale * offspring
    
    offspring = offspring + mask * noise
    
    # Clip to Bounds
    bounds_min = PARAM_BOUNDS[:, 0].to(device)
    bounds_max = PARAM_BOUNDS[:, 1].to(device)
    
    offspring = torch.max(torch.min(offspring, bounds_max), bounds_min)
    
    return offspring

def main():
    args = parse_args()
    
    print("="*60)
    print(f"PARALLEL PID AUTO-TUNING (Evolution Strategy)")
    print(f"Population Size: {args.num_envs}")
    print(f"Generations:     {args.generations}")
    print("="*60)
    
    # 1. Init Task
    # 1. Create Environment
    task = create_task(num_envs=args.num_envs, headless=True, num_payloads=args.payloads)
    env = task.sim_env
    device = task.device
    
    print(f"DEBUG: task type: {type(task)}")
    print(f"DEBUG: task attrs: {dir(task)}")

    # --- FORCE PAYLOAD MASK ---
    # Since we kept 4 slots, we must disable the extra ones.
    # Reset first to ensure default state
    task.reset()
    
    pm = task.payload_manager
    desired_count = args.payloads
    
    # Set all to False first
    pm.attached_mask[:] = False
    
    if desired_count > 0:
        # Enable first N columns
        pm.attached_mask[:, :desired_count] = True
    
    # FORCE UPDATE MASS PROPERTIES
    pm._update_mass_properties(torch.arange(task.sim_env.num_envs, device=device))
    print(f"DEBUG: Enforced attached_mask for {desired_count} payloads. Total Mass (Env 0): {pm.current_payload_mass[0].item()}")
    
    # 2. Initial Population
    print("Generating initial population...")
    population = generate_initial_population(args.num_envs, device)
    
    global_best_cost = float('inf')
    global_best_params = None
    
    start_time = time.time()
    print("Starting optimization loop...", flush=True)
    
    for gen in range(args.generations):
        gen_start = time.time()
        print(f"DEBUG: Starting Gen {gen+1} evaluation...", flush=True)
        
        # --- Evaluate ---
        costs = evaluate_generation(task, population, duration=args.duration)
        print("DEBUG: Evaluation complete.", flush=True)
        
        # --- Select Elites ---
        # Sort indices by cost (ascending)
        sorted_indices = torch.argsort(costs)
        top_k_indices = sorted_indices[:args.top_k]
        
        elites = population[top_k_indices]
        elite_costs = costs[top_k_indices]
        
        best_gen_cost = elite_costs[0].item()
        best_gen_params = elites[0].cpu().numpy()
        
        # Update Global Best
        if best_gen_cost < global_best_cost:
            global_best_cost = best_gen_cost
            global_best_params = best_gen_params
            new_record = "** NEW BEST **"
        else:
            new_record = ""
            
        print(f"Gen {gen+1}/{args.generations} | Best Cost: {best_gen_cost:.4f} | Avg Cost: {costs.mean():.2f} | Time: {time.time()-gen_start:.1f}s {new_record}")
        print(f"   Best Params: K_p={best_gen_params[0]:.2f}, K_v={best_gen_params[1]:.2f}, K_r={best_gen_params[2]:.2f}, K_w={best_gen_params[3]:.2f}")
        
        # --- Evolving Next Generation ---
        if gen < args.generations - 1:
            # Elitism: Keep top K unchanged
            next_gen_elites = elites.clone()
            
            # Offspring: Mutate the elites to fill the rest
            num_to_fill = args.num_envs - args.top_k
            
            # Adaptive Mutation
            # If cost is high, explore more (high scale). If cost low, refine (low scale).
            scale = 0.2 if best_gen_cost > 10.0 else 0.05
            
            offspring = mutate(elites, num_to_fill, mutation_rate=0.3, mutation_scale=scale)
            
            # Combine
            population = torch.cat([next_gen_elites, offspring], dim=0)

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Total Time: {total_time:.2f}s")
    print(f"Global Min Cost: {global_best_cost:.4f}")
    print("\nGlobal Best Parameters:")
    print(f"  K_pos    : {global_best_params[0]:.4f}")
    print(f"  K_vel    : {global_best_params[1]:.4f}")
    print(f"  K_rot    : {global_best_params[2]:.4f}")
    print(f"  K_angvel : {global_best_params[3]:.4f}")
    
    print("\nRecommended for lee_controller_config.py:")
    print(f"  K_pos_tensor_min = [{global_best_params[0]:.1f}, {global_best_params[0]:.1f}, {global_best_params[0]*2:.1f}]")
    print(f"  K_vel_tensor_min = [{global_best_params[1]:.1f}, {global_best_params[1]:.1f}, ...]")
    print(f"  K_rot_tensor_min = [{global_best_params[2]:.1f}, {global_best_params[2]:.1f}, ...]")
    print(f"  K_angvel_tensor_min = [{global_best_params[3]:.1f}, {global_best_params[3]:.1f}, ...]")
    
    task.close()

if __name__ == "__main__":
    main()
