from aerial_gym.config.controller_config.lee_controller_config import (
    control as BaseLeeControllerConfig,
)


class control(BaseLeeControllerConfig):
    """
    Lee position controller configuration extended with torque-compensation inputs.

    The first four actions are identical to the classic Lee position controller
    ([x, y, z, yaw] set-points). The remaining actions are interpreted as
    additive body-frame torques that are scaled by `compensation_torque_limits`.
    """

    # 4 Lee inputs + 3 torque compensation commands.
    num_actions = BaseLeeControllerConfig.num_actions + 3

    # Number of torque compensation dimensions (roll, pitch, yaw).
    compensation_dims = 3

    # Maximum absolute torque (Nm) applied per compensation input axis.
    compensation_torque_limits = [0.2, 0.2, 0.2]
