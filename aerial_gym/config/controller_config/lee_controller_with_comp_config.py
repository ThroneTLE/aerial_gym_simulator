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

    # 4 Lee inputs + thrust compensation + 3 torque commands.
    num_actions = BaseLeeControllerConfig.num_actions + 4

    # Number of compensation dimensions (1 thrust + 3 torques).
    compensation_dims = 4

    # Maximum absolute thrust (N) contributed by the compensation channel.
    compensation_thrust_limit = 20.0  # Updated for 1.5kg + 1.6kg payload

    # Maximum absolute torque (Nm) applied per compensation input axis (roll, pitch, yaw).
    compensation_torque_limits = [12.0, 12.0, 1.0]  # Updated for 450mm wheelbase
