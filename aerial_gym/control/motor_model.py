import torch # 导入 PyTorch
from aerial_gym.utils.math import torch_rand_float_tensor, tensor_clamp # 导入随机张量和张量限幅工具


class MotorModel:
    def __init__(self, num_envs, motors_per_robot, dt, config, device="cuda:0"):
        self.num_envs = num_envs # 环境数量
        self.dt = dt # 仿真步长
        self.cfg = config # 电机模型配置
        self.device = device # 设备（如 cuda:0）
        self.num_motors_per_robot = motors_per_robot # 每个机器人上的电机数量
        
        # --- 积分方案设置 ---
        try:
            self.integration_scheme = config.integration_scheme
            if self.integration_scheme not in ["euler", "rk4"]:
                # 如果未指定或无效，则默认设置为 rk4
                self.integration_scheme = "rk4"
        except:
            self.integration_scheme = "rk4" # 默认使用 RK4

        # --- 配置参数转换为张量并扩展到所有环境和电机 ---
        self.max_thrust = torch.tensor(self.cfg.max_thrust, device=self.device, dtype=torch.float32).expand(
            self.num_envs, self.num_motors_per_robot
        )
        self.min_thrust = torch.tensor(self.cfg.min_thrust, device=self.device, dtype=torch.float32).expand(
            self.num_envs, self.num_motors_per_robot
        )
        # 推力增加时的时间常数（最小/最大）
        self.motor_time_constant_increasing_min = torch.tensor(
            self.cfg.motor_time_constant_increasing_min, device=self.device
        ).expand(self.num_envs, self.num_motors_per_robot)
        self.motor_time_constant_increasing_max = torch.tensor(
            self.cfg.motor_time_constant_increasing_max, device=self.device
        ).expand(self.num_envs, self.num_motors_per_robot)
        # 推力减小时的时间常数（最小/最大）
        self.motor_time_constant_decreasing_min = torch.tensor(
            self.cfg.motor_time_constant_decreasing_min, device=self.device
        ).expand(self.num_envs, self.num_motors_per_robot)
        self.motor_time_constant_decreasing_max = torch.tensor(
            self.cfg.motor_time_constant_decreasing_max, device=self.device
        ).expand(self.num_envs, self.num_motors_per_robot)
        # 最大推力变化率
        self.max_rate = torch.tensor(self.cfg.max_thrust_rate, device=self.device).expand(
            self.num_envs, self.num_motors_per_robot
        )
        self.init_tensors() # 初始化内部状态张量

    def init_tensors(self, global_tensor_dict=None):
        """初始化电机模型的内部张量状态。"""
        # 当前电机推力 (在 min/max 范围内随机初始化)
        self.current_motor_thrust = torch_rand_float_tensor(
            torch.tensor(self.min_thrust, device=self.device, dtype=torch.float32).expand(
                self.num_envs, self.num_motors_per_robot
            ),
            torch.tensor(self.max_thrust, device=self.device, dtype=torch.float32).expand(
                self.num_envs, self.num_motors_per_robot
            ),
        )
        # 增加和减少时的时间常数（在 min/max 范围内随机化）
        self.motor_time_constants_increasing = torch_rand_float_tensor(
            self.motor_time_constant_increasing_min, self.motor_time_constant_increasing_max
        )
        self.motor_time_constants_decreasing = torch_rand_float_tensor(
            self.motor_time_constant_decreasing_min, self.motor_time_constant_decreasing_max
        )
        # 电机变化率（初始化为 0）
        self.motor_rate = torch.zeros(
            (self.num_envs, self.num_motors_per_robot), device=self.device
        )
        
        # --- RPM (每秒转数) 模型的特殊处理 ---
        if self.cfg.use_rps:
            # 推力常数 (k_f) 的最小/最大张量
            self.motor_thrust_constant_min = (
                torch.ones(
                    self.num_envs,
                    self.num_motors_per_robot,
                    device=self.device,
                    requires_grad=False,
                )
                * self.cfg.motor_thrust_constant_min
            )
            self.motor_thrust_constant_max = (
                torch.ones(
                    self.num_envs,
                    self.num_motors_per_robot,
                    device=self.device,
                    requires_grad=False,
                )
                * self.cfg.motor_thrust_constant_max
            )
            # 随机化推力常数
            self.motor_thrust_constant = torch_rand_float_tensor(
                self.motor_thrust_constant_min, self.motor_thrust_constant_max
            )
            
        # --- 混合因子函数选择 ---
        # 如果使用离散近似，则使用离散混合因子，否则使用连续混合因子
        if self.cfg.use_discrete_approximation:
            self.mixing_factor_function = discrete_mixing_factor
        else:
            self.mixing_factor_function = continuous_mixing_factor

    def update_motor_thrusts(self, ref_thrust):
        """
        根据参考推力 (ref_thrust) 更新电机当前推力 (current_motor_thrust)。
        """
        # 1. 对参考推力进行限幅
        ref_thrust = torch.clamp(ref_thrust, self.min_thrust, self.max_thrust)
        thrust_error = ref_thrust - self.current_motor_thrust
        
        # 2. 根据推力变化方向选择时间常数（非对称响应）
        # 如果符号相反（即推力增加或减少），则使用 time_constants_decreasing
        motor_time_constants = torch.where(
            torch.sign(self.current_motor_thrust) * torch.sign(thrust_error) < 0,
            self.motor_time_constants_decreasing,
            self.motor_time_constants_increasing,
        )
        
        # 3. 计算混合因子
        mixing_factor = self.mixing_factor_function(self.dt, motor_time_constants)
        
        # 4. 根据模型类型和积分方案更新推力
        if self.cfg.use_rps:
            # --- RPS/RPM 模型 (推力 ~ RPM^2) ---
            if self.integration_scheme == "euler":
                self.current_motor_thrust[:] = compute_thrust_with_rpm_time_constant(
                    ref_thrust,
                    self.current_motor_thrust,
                    mixing_factor,
                    self.motor_thrust_constant,
                    self.max_rate,
                    self.dt,
                )
            elif self.integration_scheme == "rk4":
                self.current_motor_thrust[:] = compute_thrust_with_rpm_time_constant_rk4(
                    ref_thrust,
                    self.current_motor_thrust,
                    mixing_factor,
                    self.motor_thrust_constant,
                    self.max_rate,
                    self.dt,
                )
            else:
                raise ValueError("integration scheme unknown")
        else:
            # --- 推力/力模型 (推力直接作为一阶滞后系统的输出) ---
            if self.integration_scheme == "euler":
                self.current_motor_thrust[:] = compute_thrust_with_force_time_constant(
                    ref_thrust,
                    self.current_motor_thrust,
                    mixing_factor,
                    self.max_rate,
                    self.dt,
                )
            elif self.integration_scheme == "rk4":
                self.current_motor_thrust[:] = compute_thrust_with_force_time_constant_rk4(
                    ref_thrust,
                    self.current_motor_thrust,
                    mixing_factor,
                    self.max_rate,
                    self.dt,
                )
            else:
                raise ValueError("integration scheme unknown")
                
        return self.current_motor_thrust

    def reset_idx(self, env_ids):
        """
        重置指定环境索引的电机状态和随机参数。
        """
        # 重新随机化增加/减少时间常数
        self.motor_time_constants_increasing[env_ids] = torch_rand_float_tensor(
            self.motor_time_constant_increasing_min, self.motor_time_constant_increasing_max
        )[env_ids]

        self.motor_time_constants_decreasing[env_ids] = torch_rand_float_tensor(
            self.motor_time_constant_decreasing_min, self.motor_time_constant_decreasing_max
        )[env_ids]
        
        # 重新随机化当前推力
        self.current_motor_thrust[env_ids] = torch_rand_float_tensor(
            self.min_thrust, self.max_thrust
        )[env_ids]
        
        # 如果是 RPS 模型，重新随机化推力常数
        if self.cfg.use_rps:
            self.motor_thrust_constant[env_ids] = torch_rand_float_tensor(
                self.motor_thrust_constant_min, self.motor_thrust_constant_max
            )[env_ids]

    def reset(self):
        """重置所有环境的电机状态。"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))


# --- JIT 脚本函数 (Torch JIT Scripted Functions) ---

@torch.jit.script
def motor_model_rate(error, mixing_factor, max_rate):
    """
    计算电机模型（RPM 或推力）的变化率，并进行限幅。
    """
    # 变化率 = 混合因子 * 误差，并限制在 [-max_rate, max_rate] 范围内
    return tensor_clamp(mixing_factor * (error), -max_rate, max_rate)


@torch.jit.script
def rk4_integration(error, mixing_factor, max_rate, dt):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    """
    四阶龙格-库塔 (RK4) 积分方法。
    """
    k1 = motor_model_rate(error, mixing_factor, max_rate)
    k2 = motor_model_rate(error + 0.5 * dt * k1, mixing_factor, max_rate)
    k3 = motor_model_rate(error + 0.5 * dt * k2, mixing_factor, max_rate)
    k4 = motor_model_rate(error + dt * k3, mixing_factor, max_rate)
    return (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

@torch.jit.script
def discrete_mixing_factor(dt, time_constant):
    # type: (float, Tensor) -> Tensor
    """离散混合因子（用于推力变化率的离散近似）。"""
    return 1.0 / (dt + time_constant)


@torch.jit.script
def continuous_mixing_factor(dt, time_constant):
    # type: (float, Tensor) -> Tensor
    """连续混合因子（常数）。"""
    return 1.0 / time_constant


@torch.jit.script
def compute_thrust_with_rpm_time_constant(
    ref_thrust, current_thrust, mixing_factor, thrust_constant, max_rate, dt
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    """
    使用 RPM 时间常数和欧拉积分计算下一推力。
    推力 T = k_f * RPM^2。
    """
    # 1. 推力 -> RPM
    current_rpm = torch.sqrt(current_thrust / thrust_constant)
    desired_rpm = torch.sqrt(ref_thrust / thrust_constant)
    rpm_error = desired_rpm - current_rpm
    # 2. 欧拉积分更新 RPM
    current_rpm += motor_model_rate(rpm_error, mixing_factor, max_rate) * dt
    # 3. RPM -> 推力
    return thrust_constant * current_rpm**2


@torch.jit.script
def compute_thrust_with_force_time_constant(
    ref_thrust, current_thrust, mixing_factor, max_rate, dt
):
    # type: (Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    """
    使用推力时间常数和欧拉积分计算下一推力。
    """
    thrust_error = ref_thrust - current_thrust
    # 欧拉积分更新推力
    current_thrust[:] += motor_model_rate(thrust_error, mixing_factor, max_rate) * dt
    return current_thrust

@torch.jit.script
def compute_thrust_with_rpm_time_constant_rk4(
    ref_thrust, current_thrust, mixing_factor, thrust_constant, max_rate, dt
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    """
    使用 RPM 时间常数和 RK4 积分计算下一推力。
    """
    current_rpm = torch.sqrt(current_thrust / thrust_constant)
    desired_rpm = torch.sqrt(ref_thrust / thrust_constant)
    rpm_error = desired_rpm - current_rpm
    # RK4 积分更新 RPM
    current_rpm += rk4_integration(rpm_error, mixing_factor, max_rate, dt)
    return thrust_constant * current_rpm**2


@torch.jit.script
def compute_thrust_with_force_time_constant_rk4(
    ref_thrust, current_thrust, mixing_factor, max_rate, dt
):
    # type: (Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    """
    使用推力时间常数和 RK4 积分计算下一推力。
    """
    thrust_error = ref_thrust - current_thrust
    # RK4 积分更新推力
    current_thrust[:] += rk4_integration(thrust_error, mixing_factor, max_rate, dt)
    return current_thrust