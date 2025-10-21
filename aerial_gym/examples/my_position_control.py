from aerial_gym.utils.logging import CustomLogger
import torch
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.helpers import get_args
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
import matplotlib
from isaacgym import gymapi
from aerial_gym.utils.math import quat_rotate_inverse, get_euler_xyz_tensor  # CODEx
from dataclasses import dataclass
from typing import Dict, List, Tuple

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

logger = CustomLogger(__name__)


@dataclass
class Payload:  # CODEx
    """定义子无人机（负载）的参数；修改 mass/offset 用于快速调参。"""  # CODEx
    name: str
    mass: float
    offset: torch.Tensor  # 3x tensor in base frame meters


# === 子机布置参数 ===
# 直接修改下面列表即可调整每个子机的质量和相对于母机（base_link）质心的偏移。
# offset 为 [x, y, z]，单位米；mass 单位 kg。
PAYLOAD_LAYOUT: List[Payload] = [  # CODEx
    Payload("front_left", 0.095, torch.tensor([0.16, 0.16, -0.05])),
    Payload("front_right", 0.095, torch.tensor([0.16, -0.16, -0.05])),
    Payload("rear_left", 0.095, torch.tensor([-0.16, 0.16, -0.05])),
    Payload("rear_right", 0.095, torch.tensor([-0.16, -0.16, -0.05])),
]


class PayloadManager:  # CODEx
    def __init__(self, env_manager, payloads: List[Payload], device: torch.device):
        self.env_manager = env_manager
        self.payloads = {payload.name: payload for payload in payloads}
        self.attached = {payload.name: True for payload in payloads}
        self.device = device

        self.gym = env_manager.IGE_env.gym
        self.env_ptr = env_manager.IGE_env.env_handles[0]
        self.robot_handle = env_manager.robot_manager.robot_handles[0]

        # Deep copy rigid body properties so we can modify in-place
        self.body_props = self.gym.get_actor_rigid_body_properties(self.env_ptr, self.robot_handle)
        base_prop = self.body_props[0]

        # 基础母机质量（来自 URDF）。如需修改母机本体质量，可在资源文件或此处覆写 base_prop.mass。  # CODEx
        self.base_mass = base_prop.mass
        self.base_com = torch.tensor([base_prop.com.x, base_prop.com.y, base_prop.com.z], device=device)
        self.base_inertia = torch.tensor(
            [
                [base_prop.inertia.x.x, base_prop.inertia.x.y, base_prop.inertia.x.z],
                [base_prop.inertia.y.x, base_prop.inertia.y.y, base_prop.inertia.y.z],
                [base_prop.inertia.z.x, base_prop.inertia.z.y, base_prop.inertia.z.z],
            ],
            device=device,
        )

        self.mass_history: List[float] = []
        self.com_history: List[torch.Tensor] = []

    def compute_total_properties(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute total mass and inertia with currently attached payloads."""
        total_mass = torch.tensor(self.base_mass, device=self.device)
        numerator_com = self.base_mass * self.base_com.clone()

        for name, payload in self.payloads.items():
            if not self.attached[name]:
                continue
            total_mass += payload.mass
            numerator_com += payload.mass * payload.offset.to(self.device)

        if total_mass.item() <= 0.0:
            combined_com = torch.zeros(3, device=self.device)
        else:
            combined_com = numerator_com / total_mass

        # Parallel axis theorem adjustments
        identity = torch.eye(3, device=self.device)
        d_base = self.base_com - combined_com
        inertia_total = self.base_inertia + self.base_mass * (
            torch.dot(d_base, d_base) * identity - torch.outer(d_base, d_base)
        )

        for name, payload in self.payloads.items():
            if not self.attached[name]:
                continue
            r = payload.offset.to(self.device) - combined_com
            inertia_total += payload.mass * (torch.dot(r, r) * identity - torch.outer(r, r))

        return total_mass, inertia_total, combined_com

    def apply_to_sim(self):
        total_mass, inertia_total, combined_com = self.compute_total_properties()

        base_prop = self.body_props[0]
        base_prop.mass = float(total_mass.item())
        base_prop.com = gymapi.Vec3(*combined_com.cpu().tolist())
        inertia_mat = gymapi.Mat33()
        inertia_cpu = inertia_total.detach().cpu().numpy()
        inertia_mat.x = gymapi.Vec3(*inertia_cpu[0])
        inertia_mat.y = gymapi.Vec3(*inertia_cpu[1])
        inertia_mat.z = gymapi.Vec3(*inertia_cpu[2])
        base_prop.inertia = inertia_mat

        self.gym.set_actor_rigid_body_properties(
            self.env_ptr, self.robot_handle, self.body_props, recomputeInertia=False
        )

        # Synchronize tensors used by controllers and logging
        env = self.env_manager
        env.robot_manager.robot_mass = float(total_mass.item())
        env.robot_manager.robot_masses.fill_(float(total_mass.item()))
        env.IGE_env.global_tensor_dict["robot_mass"].fill_(float(total_mass.item()))

        inertia_tensor = torch.tensor(inertia_cpu, device=self.device, dtype=torch.float32)
        env.robot_manager.robot_inertia = inertia_tensor
        env.robot_manager.robot_inertias[:] = inertia_tensor
        env.IGE_env.global_tensor_dict["robot_inertia"][:] = inertia_tensor

        controller_mass = torch.full((env.num_envs, 1), float(total_mass.item()), device=self.device)
        env.robot_manager.robot.controller.mass = controller_mass

        self.mass_history.append(float(total_mass.item()))
        self.com_history.append(combined_com.detach().cpu())

    def release_payload(self, name: str):
        if name not in self.payloads:
            logger.error(f"Payload '{name}' 未定义，忽略释放命令。")
            return
        if not self.attached[name]:
            logger.warning(f"Payload '{name}' 已经释放，忽略重复命令。")
            return
        self.attached[name] = False
        self.apply_to_sim()
        logger.info(f"释放子无人机 {name}，剩余质量 {self.mass_history[-1]:.3f} kg。")

    def attach_payload(self, name: str):
        if name not in self.payloads:
            logger.error(f"Payload '{name}' 未定义，无法重新挂载。")
            return
        self.attached[name] = True
        self.apply_to_sim()

    def current_mass(self) -> float:
        return self.mass_history[-1] if self.mass_history else self.base_mass

    def current_com(self) -> torch.Tensor:
        return self.com_history[-1] if self.com_history else self.base_com.cpu()


def run_controller(controller_name, args, results):
    """运行控制器并在指定时刻施加外力与质量扰动。"""
    logger.warning(f"测试控制器: {controller_name}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")

    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",
        robot_name="base_quadrotor",
        controller_name=controller_name,
        args=None,
        device=device,
        num_envs=1,
        headless=args.headless,
        use_warp=True,
    )

    env_manager.reset()  # CODEx

    initial_state = torch.zeros(13, device=device)
    initial_state[6] = 1.0
    root_state = env_manager.IGE_env.global_tensor_dict["vec_root_tensor"]  # CODEx
    root_state[:, 0, :13] = initial_state  # CODEx
    env_manager.IGE_env.write_to_sim()  # CODEx

    target_position = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device)
    actions = target_position.repeat(env_manager.num_envs, 1)

    payload_manager = PayloadManager(env_manager, PAYLOAD_LAYOUT, device=device)  # CODEx
    payload_manager.apply_to_sim()  # CODEx

    env_ptr = env_manager.IGE_env.env_handles[0]
    robot_handle = env_manager.robot_manager.robot_handles[0]
    gym = env_manager.IGE_env.gym

    num_bodies = gym.get_actor_rigid_body_count(env_ptr, robot_handle)
    body_index = 0
    try:
        rigid_body_names = gym.get_actor_rigid_body_names(env_ptr, robot_handle)
    except AttributeError:
        rigid_body_names = [f"body_{j}" for j in range(num_bodies)]

    logger.warning(f"刚体数量: {num_bodies}, 列表: {rigid_body_names}")
    logger.warning(f"施加力目标刚体索引: {body_index}, 名称: {rigid_body_names[body_index]}")

    NUM_STEPS = 3000
    # 外力窗口设置：键为 (start_step, end_step)，值为世界坐标系力向量 (N)。  # CODEx
    # 可按需增删用于模拟风/扰动等情景。
    FORCE_WINDOWS: Dict[Tuple[int, int], torch.Tensor] = {  # CODEx
        (400, 410): torch.tensor([0.0, 0.0, 0.0], device=device),
        (800, 820): torch.tensor([0.0, 0.0, 0.0], device=device),
    }
    # 子机释放时间表：列表中的 (time_step, payload_name) 可自由编辑，以控制各子机释放时间及顺序。  # CODEx
    release_plan = [  # CODEx
        (500, "rear_right"),
        (1000, "front_left"),
        (1500, "front_right"),
        (2000, "rear_left"),
    ]
    release_lookup = {step: name for step, name in release_plan}  # CODEx

    external_force_state: Dict[Tuple[int, int], bool] = {window: False for window in FORCE_WINDOWS}

    def external_force_callback(manager):  # CODEx
        robot_force = manager.IGE_env.global_tensor_dict["robot_force_tensor"]
        orientations = manager.IGE_env.global_tensor_dict["robot_orientation"]
        for window, force_vec in FORCE_WINDOWS.items():
            active = window[0] <= current_step <= window[1]
            external_force_state[window] = active
            if not active:
                continue
            world_force = force_vec.unsqueeze(0).expand(orientations.shape[0], -1)
            body_force = quat_rotate_inverse(orientations, world_force)
            robot_force[:, body_index, :] += body_force

    errors, positions, velocities = [], [], []
    attitudes, mass_log, com_log = [], [], []
    motor_force_history, motor_torque_history = [], []

    for current_step in range(NUM_STEPS):
        if current_step in release_lookup:
            payload_manager.release_payload(release_lookup[current_step])  # CODEx

        env_manager.reset_tensors()
        env_manager.step(actions=actions, external_force_callback=external_force_callback)  # CODEx
        if not args.headless:
            env_manager.render()

        obs = env_manager.get_obs()
        error = torch.norm(target_position[:, 0:3] - obs["robot_position"], dim=1)
        errors.append(error.cpu().numpy())
        positions.append(obs["robot_position"].cpu().numpy())
        velocities.append(obs["robot_vehicle_linvel"].cpu().numpy())
        attitudes.append(get_euler_xyz_tensor(obs["robot_orientation"]).cpu().numpy())  # CODEx
        mass_log.append(payload_manager.current_mass())  # CODEx
        com_log.append(payload_manager.current_com().numpy())  # CODEx

        robot_force = env_manager.IGE_env.global_tensor_dict["robot_force_tensor"][0].cpu().numpy()  # CODEx
        robot_torque = env_manager.IGE_env.global_tensor_dict["robot_torque_tensor"][0].cpu().numpy()  # CODEx
        motor_force_history.append(robot_force)
        motor_torque_history.append(robot_torque)

        if current_step % 100 == 0:
            logger.info(
                f"Step {current_step}, 位置: {obs['robot_position']}, 误差: {error}, 当前质量: {mass_log[-1]:.3f}"
            )

    results[controller_name] = {
        "errors": np.array(errors),
        "positions": np.array(positions),
        "velocities": np.array(velocities),
        "attitudes": np.array(attitudes),
        "mass": np.array(mass_log),
        "com": np.array(com_log),
        "motor_forces": np.array(motor_force_history),
        "motor_torques": np.array(motor_torque_history),
        "release_plan": release_plan,
    }

    del env_manager
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = get_args()
    controller_name = "lee_position_control"
    results = {}
    run_controller(controller_name, args, results)

    if controller_name in results:
        sim_results = results[controller_name]
        timestep = np.arange(len(sim_results["errors"])) * 0.01

        # 位置误差
        plt.figure(figsize=(10, 4))
        plt.plot(timestep, sim_results["errors"], label=f"{controller_name} 误差")
        plt.xlabel("时间 (秒)")
        plt.ylabel("位置误差 (米)")
        plt.title("位置误差曲线")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # 位置 XYZ
        pos = sim_results["positions"][:, 0, :]
        fig_pos, axes_pos = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for idx, axis_label in enumerate(["X", "Y", "Z"]):
            axes_pos[idx].plot(timestep, pos[:, idx], label=f"{axis_label} 位置")
            axes_pos[idx].set_ylabel(f"{axis_label} (m)")
            axes_pos[idx].grid(True)
        axes_pos[-1].set_xlabel("时间 (秒)")
        fig_pos.suptitle("无人机位置轨迹")
        fig_pos.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 姿态 (roll/pitch/yaw)
        attitude = sim_results["attitudes"][:, 0, :] * 180.0 / np.pi
        fig_att, axes_att = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for idx, axis_label in enumerate(["Roll", "Pitch", "Yaw"]):
            axes_att[idx].plot(timestep, attitude[:, idx], label=axis_label)
            axes_att[idx].set_ylabel(f"{axis_label} (deg)")
            axes_att[idx].grid(True)
        axes_att[-1].set_xlabel("时间 (秒)")
        fig_att.suptitle("无人机姿态（欧拉角）")
        fig_att.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 质量与质心
        fig_mass, ax_mass = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax_mass[0].step(timestep, sim_results["mass"], where="post")
        ax_mass[0].set_ylabel("质量 (kg)")
        ax_mass[0].grid(True)
        com = sim_results["com"]
        for idx, axis_label in enumerate(["X", "Y", "Z"]):
            ax_mass[1].plot(timestep, com[:, idx], label=f"{axis_label}")
        ax_mass[1].set_ylabel("质心 (m)")
        ax_mass[1].set_xlabel("时间 (秒)")
        ax_mass[1].grid(True)
        ax_mass[1].legend()
        fig_mass.suptitle("整体质量与质心演化")
        fig_mass.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 电机力与扭矩
        motor_forces = sim_results["motor_forces"]
        motor_torques = sim_results["motor_torques"]
        num_links = motor_forces.shape[1]
        motor_indices = range(1, num_links) if num_links > 1 else range(num_links)

        fig_force, ax_force = plt.subplots(figsize=(10, 5))
        for idx in motor_indices:
            ax_force.plot(
                timestep,
                motor_forces[:, idx, 2],
                label=f"Link {idx} Fz",
            )
        ax_force.set_xlabel("时间 (秒)")
        ax_force.set_ylabel("推力 (N)")
        ax_force.set_title("各电机推力 (Z 轴分量)")
        ax_force.grid(True)
        ax_force.legend()
        fig_force.tight_layout()

        fig_torque, ax_torque = plt.subplots(figsize=(10, 5))
        for idx in motor_indices:
            ax_torque.plot(
                timestep,
                motor_torques[:, idx, 2],
                label=f"Link {idx} Tz",
            )
        ax_torque.set_xlabel("时间 (秒)")
        ax_torque.set_ylabel("扭矩 (Nm)")
        ax_torque.set_title("各电机扭矩 (Z 轴分量)")
        ax_torque.grid(True)
        ax_torque.legend()
        fig_torque.tight_layout()

        # 3D 轨迹
        fig_traj = plt.figure(figsize=(10, 8))
        ax = fig_traj.add_subplot(111, projection="3d")
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=f"{controller_name} 轨迹")
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color="g", label="起始点")
        ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color="b", label="结束点")
        ax.set_xlabel("X (米)")
        ax.set_ylabel("Y (米)")
        ax.set_zlabel("Z (米)")
        ax.legend()
        fig_traj.tight_layout()

        plt.show()
