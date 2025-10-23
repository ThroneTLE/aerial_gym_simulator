python -m aerial_gym.examples.adjust_lee_rl_gamer_control \
    --device cuda:0 \
    --num-envs 4 \
    --total-timesteps 256 \
    --eval-horizon 600 \
    --max-epochs 128 \
    --best-path best_lee_gains_rlg.json