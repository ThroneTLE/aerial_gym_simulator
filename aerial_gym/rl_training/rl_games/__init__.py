"""RL-Games patches used by aerial_gym.

RL-Games (v1.7.2 as shipped with Isaac Gym) drops the observation/value
normalization statistics when saving PPO checkpoints, which means resumed
runs or `--play` sessions start with zero-mean/identity-variance stats and
perform poorly for the first few rollouts. We patch the base algorithm so
that checkpoints always contain those buffers and they are restored
properly when loading.
"""

from rl_games.common import a2c_common
from aerial_gym.rl_training.rl_games.nn import privileged_actor_critic  # noqa: F401


def _patch_a2c_base():
    """Ensure PPO saves and loads running mean/std statistics."""

    base_cls = a2c_common.A2CBase
    if getattr(base_cls, "_aerialgym_stats_patch", False):
        return

    def _get_weights_with_stats(self):
        state = self.get_stats_weights(model_stats=True)
        state["model"] = self.model.state_dict()
        return state

    def _set_stats_weights_with_stats(self, weights):
        if self.normalize_rms_advantage and "advantage_mean_std" in weights:
            self.advantage_mean_std.load_state_dict(weights["advantage_mean_std"])
        if self.normalize_input and "running_mean_std" in weights:
            self.model.running_mean_std.load_state_dict(weights["running_mean_std"])
        if self.normalize_value and "reward_mean_std" in weights:
            self.model.value_mean_std.load_state_dict(weights["reward_mean_std"])
        if self.mixed_precision and "scaler" in weights:
            self.scaler.load_state_dict(weights["scaler"])

    base_cls.get_weights = _get_weights_with_stats
    base_cls.set_stats_weights = _set_stats_weights_with_stats
    base_cls._aerialgym_stats_patch = True


_patch_a2c_base()
