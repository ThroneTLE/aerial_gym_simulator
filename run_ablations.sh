#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/throne/下载/isaacgym/python:/home/throne/workspaces/aerial_gym_ws/src/aerial_gym_simulator

echo "Starting Ablation 1: No Physics Randomization (300 epochs)"
/home/throne/miniconda3/envs/aerialgym/bin/python aerial_gym/rl_training/rl_games/runner.py --file aerial_gym/rl_training/rl_games/ppo_ablation_no_phys_rand.yaml --task payload_compensation_task_no_phys_rand --train --headless True

echo "Starting Ablation 2: Full Physics Randomization (500 epochs)"
/home/throne/miniconda3/envs/aerialgym/bin/python aerial_gym/rl_training/rl_games/runner.py --file aerial_gym/rl_training/rl_games/ppo_ablation_full_teacher.yaml --task payload_compensation_task_teacher --train --headless True

echo "All ablations complete."
