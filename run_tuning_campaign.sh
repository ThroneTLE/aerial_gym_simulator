#!/bin/bash
echo "Starting PID Tuning Campaign..."

# Payloads 0
echo "=== Tuning 0 Payloads ==="
python auto_tune_pid.py --num_envs 1024 --generations 15 --duration 1.5 --payloads 0 > tune_0.log 2>&1
grep "Best Params" tune_0.log | tail -n 1

# Payloads 1
echo "=== Tuning 1 Payload ==="
python auto_tune_pid.py --num_envs 1024 --generations 15 --duration 1.5 --payloads 1 > tune_1.log 2>&1
grep "Best Params" tune_1.log | tail -n 1

# Payloads 2
echo "=== Tuning 2 Payloads ==="
python auto_tune_pid.py --num_envs 1024 --generations 15 --duration 1.5 --payloads 2 > tune_2.log 2>&1
grep "Best Params" tune_2.log | tail -n 1

# Payloads 3
echo "=== Tuning 3 Payloads ==="
python auto_tune_pid.py --num_envs 1024 --generations 15 --duration 1.5 --payloads 3 > tune_3.log 2>&1
grep "Best Params" tune_3.log | tail -n 1

# Payloads 4
echo "=== Tuning 4 Payloads ==="
python auto_tune_pid.py --num_envs 1024 --generations 15 --duration 1.5 --payloads 4 > tune_4.log 2>&1
grep "Best Params" tune_4.log | tail -n 1

echo "Campaign Complete."
