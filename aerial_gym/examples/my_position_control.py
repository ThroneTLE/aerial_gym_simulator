"""Minimal example showing payload release by editing mass properties only."""
from __future__ import annotations

import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import random

_deprecated_aliases = {
    "int": int,
    "float": float,
}
for _alias, _target in _deprecated_aliases.items():
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)  # type: ignore[attr-defined]

from isaacgym import gymapi, gymtorch
import torch

from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.helpers import get_args
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import get_euler_xyz_tensor, quat_rotate_inverse

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

logger = CustomLogger(__name__)

PAYLOAD_OFFSETS = [
    np.array([0.4, 0.4, -0.4], dtype=np.float32),
    np.array([0.4, -0.4, -0.4], dtype=np.float32),
    np.array([-0.4, 0.4, -0.4], dtype=np.float32),
    np.array([-0.4, -0.4, -0.4], dtype=np.float32),
]
PAYLOAD_MASS = 0.02  # kg
RELEASE_START_STEP = 400
RELEASE_INTERVAL = 200
RELEASE_START_RANGE = (max(0, RELEASE_START_STEP - 100), RELEASE_START_STEP + 100)
RELEASE_INTERVAL_RANGE = (max(1, RELEASE_INTERVAL - 50), RELEASE_INTERVAL + 50)
RANDOMIZE_RELEASE = True


def _add_axis_sliders(fig, axes):
    fig.subplots_adjust(bottom=0.25)
    slider_color = "#24a8a8"
    ax_xscale = fig.add_axes([0.15, 0.1, 0.7, 0.03], facecolor=slider_color)
    ax_yscale = fig.add_axes([0.15, 0.05, 0.7, 0.03], facecolor=slider_color)
    slider_x = Slider(ax_xscale, "X轴缩放", 0.2, 5.0, valinit=1.0)
    slider_y = Slider(ax_yscale, "Y轴缩放", 0.2, 5.0, valinit=1.0)

    base_limits = {axis: {"x": axis.get_xlim(), "y": axis.get_ylim()} for axis in axes}

    def _apply_scale(_):
        x_scale = slider_x.val
        y_scale = slider_y.val

        for axis in axes:
            base_xlim = base_limits[axis]["x"]
            base_ylim = base_limits[axis]["y"]

            x_mid = 0.5 * (base_xlim[0] + base_xlim[1])
            x_span = (base_xlim[1] - base_xlim[0]) / x_scale
            axis.set_xlim(x_mid - 0.5 * x_span, x_mid + 0.5 * x_span)

            y_mid = 0.5 * (base_ylim[0] + base_ylim[1])
            y_span = (base_ylim[1] - base_ylim[0]) / y_scale
            axis.set_ylim(y_mid - 0.5 * y_span, y_mid + 0.5 * y_span)

        fig.canvas.draw_idle()

    slider_x.on_changed(_apply_scale)
    slider_y.on_changed(_apply_scale)
    return slider_x, slider_y


def plot_results(
    z_history,
    euler_history,
    release_history,
    policy_actions,
    teacher_actions=None,
    xy_history=None,
    xy_error_history=None,
    target_xy_history=None,
):
    if not z_history:
        print("无可绘制数据。")
        return

    steps = np.arange(len(z_history))
    eulers = np.unwrap(np.array(euler_history), axis=0)
    eulers_deg = np.rad2deg(eulers)
    eulers_deg -= np.round(eulers_deg[0] / 360.0) * 360.0

    fig, (ax_z, ax_euler) = plt.subplots(2, 1, figsize=(10, 8))

    ax_z.plot(steps, z_history, label="Z 轴高度")
    ax_z.set_xlabel("步数")
    ax_z.set_ylabel("Z 轴高度 (米)")
    ax_z.set_title("无人机 Z 轴位置曲线")
    ax_z.grid(True)
    ax_z.legend()

    for release_step, payload_idx in release_history:
        ax_z.axvline(release_step, color="r", linestyle="--", alpha=0.6)
        ax_z.text(
            release_step,
            ax_z.get_ylim()[1],
            f"释放 {payload_idx}",
            color="r",
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="center",
            rotation=90,
        )

    ax_euler.plot(steps, eulers_deg[:, 0], label="Roll (°)")
    ax_euler.plot(steps, eulers_deg[:, 1], label="Pitch (°)")
    ax_euler.plot(steps, eulers_deg[:, 2], label="Yaw (°)")
    ax_euler.set_xlabel("步数")
    ax_euler.set_ylabel("角度 (°)")
    ax_euler.set_title("无人机姿态角 (XYZ)")
    ax_euler.grid(True)
    ax_euler.legend()

    _add_axis_sliders(fig, [ax_z, ax_euler])

    fig_xy = None
    if xy_history:
        xy_arr = np.vstack(xy_history)
        steps_xy = np.arange(xy_arr.shape[0])
        target_xy_arr = np.vstack(target_xy_history) if target_xy_history else None
        xy_err_arr = np.vstack(xy_error_history) if xy_error_history else None

        fig_xy, (ax_xy, ax_xy_err) = plt.subplots(2, 1, figsize=(10, 8))
        ax_xy.plot(xy_arr[:, 0], xy_arr[:, 1], label="轨迹", color="C0")
        ax_xy.scatter(
            xy_arr[0, 0],
            xy_arr[0, 1],
            label="起点",
            facecolors="none",
            edgecolors="#2ca02c",
            s=120,
            linewidth=1.5,
            zorder=6,
        )
        ax_xy.scatter(
            xy_arr[-1, 0],
            xy_arr[-1, 1],
            label="终点",
            color="#d62728",
            s=110,
            marker="^",
            edgecolor="k",
            linewidth=0.5,
            zorder=5,
        )
        if target_xy_arr is not None:
            ax_xy.plot(target_xy_arr[:, 0], target_xy_arr[:, 1], label="目标", color="C1", linestyle="--")
            ax_xy.scatter(
                target_xy_arr[-1, 0],
                target_xy_arr[-1, 1],
                color="C1",
                marker="x",
                s=60,
            )
        for release_step, _ in release_history:
            if 0 <= release_step < xy_arr.shape[0]:
                ax_xy.scatter(xy_arr[release_step, 0], xy_arr[release_step, 1], color="r", s=30)
        ax_xy.set_xlabel("X (米)")
        ax_xy.set_ylabel("Y (米)")
        ax_xy.set_title("XY 平面轨迹")
        ax_xy.grid(True)
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.legend()

        if xy_err_arr is not None:
            ax_xy_err.plot(steps_xy, xy_err_arr[:, 0], label="X 误差", color="C0")
            ax_xy_err.plot(steps_xy, xy_err_arr[:, 1], label="Y 误差", color="C1")
            xy_err_norm = np.linalg.norm(xy_err_arr, axis=1)
            ax_xy_err.plot(steps_xy, xy_err_norm, label="平面误差", color="C2", linestyle="--")
            for release_step, _ in release_history:
                ax_xy_err.axvline(release_step, color="r", linestyle=":", alpha=0.5)
            ax_xy_err.set_xlabel("步数")
            ax_xy_err.set_ylabel("误差 (米)")
            ax_xy_err.set_title("XY 平面位置误差")
            ax_xy_err.grid(True)
            ax_xy_err.legend()
        else:
            ax_xy_err.axis("off")

        _add_axis_sliders(fig_xy, [ax_xy, ax_xy_err] if xy_err_arr is not None else [ax_xy])

    if policy_actions:
        act_arr = np.vstack(policy_actions)
        action_dim = act_arr.shape[1]
        teacher_arr = (
            np.vstack(teacher_actions)
            if teacher_actions is not None and len(teacher_actions) == len(policy_actions)
            else None
        )
        fig_act, axes = plt.subplots(
            action_dim, 1, figsize=(10, max(4, 2 * action_dim)), sharex=True
        )
        if action_dim == 1:
            axes = [axes]
        labels = ["thrust"] + [f"torque_{i}" for i in range(1, action_dim)]
        for i, ax in enumerate(axes):
            ax.plot(steps, act_arr[:, i], label="policy", color="C0")
            if teacher_arr is not None:
                ax.plot(steps, teacher_arr[:, i], label="teacher", color="C1", linestyle="--")
            for release_step, _ in release_history:
                ax.axvline(release_step, color="r", linestyle=":", alpha=0.5)
            ax.set_ylabel(labels[i])
            ax.grid(True)
            ax.legend()
        axes[-1].set_xlabel("步数")
        fig_act.suptitle("残差补偿对比（策略 vs 教师）")

    save_path = os.environ.get("AERIAL_SAVE_PLOT")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if fig_xy is not None:
            fig_xy.savefig(
                os.path.splitext(save_path)[0] + "_xy.png", dpi=150, bbox_inches="tight"
            )
        if policy_actions:
            fig_act.savefig(
                os.path.splitext(save_path)[0] + "_actions.png", dpi=150, bbox_inches="tight"
            )
        print(f"已保存绘图到 {save_path}")
    else:
        plt.show()

# 将 Mat33 拆解成 numpy 数组，方便后续做线性代数运算（例如缩放惯量）。
def _mat33_to_np(mat: gymapi.Mat33) -> np.ndarray:
    """将 Isaac Gym 的 Mat33 按列主序转换为 numpy 数组。"""
    return np.array(
        [
            [mat.x.x, mat.y.x, mat.z.x],
            [mat.x.y, mat.y.y, mat.z.y],
            [mat.x.z, mat.y.z, mat.z.z],
        ],
        dtype=np.float32,
    )

# 依据自定义的 numpy 矩阵重建 Mat33，以便写回 Isaac Gym。
def _np_to_mat33(arr: np.ndarray) -> gymapi.Mat33:
    """把 numpy 数组重新封装为 Mat33。"""
    mat = gymapi.Mat33()
    mat.x = gymapi.Vec3(arr[0, 0], arr[1, 0], arr[2, 0])
    mat.y = gymapi.Vec3(arr[0, 1], arr[1, 1], arr[2, 1])
    mat.z = gymapi.Vec3(arr[0, 2], arr[1, 2], arr[2, 2])
    return mat

# 将最新质量值同步进控制器内部的张量，避免控制器仍用旧质量计算推力。
def _sync_controller_mass(env_manager, new_mass: float) -> None:
    """在示例内部同步控制器保存的质量张量，保持物理属性一致。"""
    controller = env_manager.robot_manager.robot.controller
    if hasattr(controller, "mass"):
        mass_tensor = torch.full_like(controller.mass, new_mass)
        controller.mass = mass_tensor
        logger.info("Controller mass tensor updated to %.3f kg", new_mass)
    else:
        logger.warning("控制器未暴露质量张量，跳过同步。")

# 在所有环境中写入统一的初始根状态（零位移+单位四元数），确保每次仿真可复现。
def initialize_vehicle_state(env_manager, device: str) -> None:
    """将根状态清零并设为单位四元数，保证仿真以一致初态开始。"""
    root_state = gymtorch.wrap_tensor(
        env_manager.IGE_env.gym.acquire_actor_root_state_tensor(env_manager.IGE_env.sim)
    )
    single_state = torch.zeros(13, device=device)
    single_state[6] = 1.0  # unit quaternion
    root_state[:] = single_state.repeat(env_manager.num_envs, 1)
    env_manager.IGE_env.gym.refresh_actor_root_state_tensor(env_manager.IGE_env.sim)

# 依据平行轴定理计算点质量在母机坐标系下的惯量增量，用于构建复合惯量矩。
def point_mass_inertia(mass: float, offset: np.ndarray) -> np.ndarray:
    """点质量相对于母机原点的惯量贡献。"""
    r_sq = float(np.dot(offset, offset))
    return mass * (r_sq * np.eye(3) - np.outer(offset, offset))


class PayloadManager:
    """管理四角子机的释放、惯量更新以及等效的重力力矩补偿。"""

    # 初始化载荷管理器：采样子机的质量/位置，缓存初始惯量矩并准备释放调度。
    def __init__(
        self,
        env_manager,
        payload_mass: float,
        offsets,
        release_start: int,
        release_interval: int,
        release_start_range=None,
        release_interval_range=None,
        randomize_release: bool = False,
    ):
        self.env_manager = env_manager
        self.device = env_manager.device
        self.gym = env_manager.IGE_env.gym
        self.env_handle = env_manager.IGE_env.env_handles[0]
        self.robot_handle = env_manager.robot_manager.robot_handles[0]
        self.payloads = [
            {
                "mass": payload_mass,
                "offset": np.array(offset, dtype=np.float32),
                "attached": True,
                "name": f"payload_{idx}",
            }
            for idx, offset in enumerate(offsets)
        ]
        self.release_idx = 0
        self.release_start = int(release_start)
        self.release_interval = int(release_interval)
        self.release_start_range = release_start_range
        self.release_interval_range = release_interval_range
        self.randomize_release = bool(randomize_release)
        self.next_release_step = self._sample_release_start()
        self.release_history = []

        props = self.gym.get_actor_rigid_body_properties(self.env_handle, self.robot_handle)
        self.props = props
        self.base_prop = props[0]
        self.initial_mass = float(self.base_prop.mass)
        self.initial_inertia = _mat33_to_np(self.base_prop.inertia)

        self.total_payload_mass = sum(p["mass"] for p in self.payloads)
        payload_inertia_total = sum(point_mass_inertia(p["mass"], p["offset"]) for p in self.payloads)

        # 空载母机的属性直接来自 URDF，需要在写回模拟前先保留下来
        self.empty_mass = self.initial_mass
        self.empty_inertia = self.initial_inertia

        # 将初始状态更新为“母机 + 全部子机”的质量与惯量
        self.current_mass = self.initial_mass + self.total_payload_mass
        full_inertia = self.initial_inertia + payload_inertia_total
        self.base_prop.mass = self.current_mass
        self.base_prop.inertia = _np_to_mat33(full_inertia)
        self.gym.set_actor_rigid_body_properties(
            self.env_handle, self.robot_handle, self.props, recomputeInertia=False
        )

        self.gravity_vec = (
            env_manager.IGE_env.global_tensor_dict["gravity"][0].detach().to(self.device)
        )
        self.global_torque_tensor = env_manager.IGE_env.global_tensor_dict["global_torque_tensor"]
        self.base_body_index = 0  # base_link 假设是第一个刚体
        self.current_mass = self.initial_mass

    def _sample_release_start(self) -> int:
        if self.randomize_release and self.release_start_range is not None:
            low, high = self.release_start_range
            low, high = int(low), int(high)
            if low > high:
                low, high = high, low
            return random.randint(low, high)
        return self.release_start

    def _sample_release_interval(self) -> int:
        if self.randomize_release and self.release_interval_range is not None:
            low, high = self.release_interval_range
            low, high = int(low), int(high)
            if low > high:
                low, high = high, low
            return max(1, random.randint(low, high))
        return max(1, self.release_interval)

    # 返回当前仍附着在母机上的子机列表。
    def _attached_payloads(self):
        return [p for p in self.payloads if p["attached"]]

    # 在满足触发步数时释放一个子机，并记录释放历史、更新质量属性。
    def maybe_release(self, step: int):
        if self.release_idx >= len(self.payloads):
            return
        if step < self.next_release_step:
            return
        attached_payloads = self._attached_payloads()
        if not attached_payloads:
            return
        payload = random.choice(attached_payloads)
        payload["attached"] = False
        self.release_idx += 1
        self.release_history.append((step, payload.get("name", f"{self.release_idx}")))
        self.next_release_step = step + self._sample_release_interval()
        self.update_mass_properties()
        logger.info("Payload %d released -> new mass %.3f kg", self.release_idx, self.current_mass)

    # 根据当前仍附着的子机重新计算质量与惯量，并写入物理属性。
    def update_mass_properties(self):
        
        attached = self._attached_payloads()
        payload_mass_sum = sum(p["mass"] for p in attached)
        self.current_mass = self.empty_mass + payload_mass_sum
        inertia_np = self.empty_inertia.copy()#self.initial_inertia
        for payload in attached:
            inertia_np += point_mass_inertia(payload["mass"], payload["offset"])

        self.base_prop.mass = self.current_mass
        self.base_prop.inertia = _np_to_mat33(inertia_np)
        self.gym.set_actor_rigid_body_properties(
            self.env_handle, self.robot_handle, self.props, recomputeInertia=False
        )
        _sync_controller_mass(self.env_manager, self.current_mass)# 同步控制器质量张量
    # 计算当前剩余子机导致的世界系扭矩（针对每个 env 返回一个 3D 向量）。
    def compute_world_torque(self) -> torch.Tensor:
        attached = self._attached_payloads()
        if not attached:
            return torch.zeros((self.env_manager.num_envs, 3), device=self.device)
        total_mass = self.current_mass
        if total_mass <= 0:
            return torch.zeros((self.env_manager.num_envs, 3), device=self.device)
        weighted_offset = torch.zeros(3, device=self.device)
        for payload in attached:
            offset = torch.as_tensor(
                payload["offset"], device=self.device, dtype=torch.float32
            )
            weighted_offset += payload["mass"] * offset
        com_offset = weighted_offset / total_mass
        #print(f"[PayloadTorque] COM offset: {com_offset.tolist()}")
        torque = torch.cross(com_offset, self.gravity_vec * total_mass)
        return torque.unsqueeze(0).expand(self.env_manager.num_envs, -1)

    # 将世界系扭矩转换到机体系，供控制器的 wrench 注入。
    def compute_body_torque(self, orientations: torch.Tensor) -> torch.Tensor:
        world_torque = self.compute_world_torque()
        if world_torque.shape[0] != orientations.shape[0]:
            world_torque = world_torque[: orientations.shape[0]]
        return quat_rotate_inverse(orientations, world_torque)


if __name__ == "__main__":
    args = get_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    sim_builder = SimBuilder()
    env_manager = sim_builder.build_env(
        sim_name="base_sim",
        env_name="empty_env",
        robot_name="base_quadrotor",
        controller_name="lee_position_control",
        args=None,
        device=device,
        num_envs=args.num_envs,
        headless=args.headless,
        use_warp=args.use_warp,
    )

    initialize_vehicle_state(env_manager, device)
    env_manager.reset()
    payload_manager = PayloadManager(
        env_manager,
        payload_mass=PAYLOAD_MASS,
        offsets=PAYLOAD_OFFSETS,
        release_start=RELEASE_START_STEP,
        release_interval=RELEASE_INTERVAL,
        release_start_range=RELEASE_START_RANGE,
        release_interval_range=RELEASE_INTERVAL_RANGE,
        randomize_release=RANDOMIZE_RELEASE,
    )

    orig_pre_physics_step = env_manager.robot_manager.pre_physics_step

    def patched_pre_physics_step(actions, _orig=orig_pre_physics_step):
        _orig(actions)
        orientations = env_manager.IGE_env.global_tensor_dict["robot_orientation"]
        body_torque = payload_manager.compute_body_torque(orientations)
        env_manager.robot_manager.robot.robot_torque_tensors[:, 0, :] += body_torque

    env_manager.robot_manager.pre_physics_step = patched_pre_physics_step

    actions = torch.zeros((env_manager.num_envs, 4), device=device)
    target_position = torch.zeros(3, device=device)
    target_yaw = 0.0
    actions[:, 0:3] = target_position
    actions[:, 3] = target_yaw
    target_xy_np = target_position[:2].detach().cpu().numpy()

    z_history = []
    euler_history = []
    xy_history = []
    xy_error_history = []
    target_xy_history = []

    obs = env_manager.get_obs()

    for step in range(2000):
        payload_manager.maybe_release(step)
        
        env_manager.step(actions=actions)

        obs = env_manager.get_obs()
        position = obs["robot_position"][0].detach().cpu().numpy()
        z_history.append(position[2])
        xy_history.append(position[:2].copy())
        target_xy_history.append(target_xy_np.copy())
        xy_error_history.append((target_xy_np - position[:2]).copy())
        quat = obs["robot_orientation"][0:1]
        euler = get_euler_xyz_tensor(quat)
        euler_history.append(euler[0].detach().cpu().numpy())

        if step % 200 == 0:
            logger.info("Step %d | position: [%.2f, %.2f, %.2f]", step, *position.tolist())

    sim_builder.delete_env()
    plot_results(
        z_history,
        euler_history,
        payload_manager.release_history,
        [],
        teacher_actions=None,
        xy_history=xy_history,
        xy_error_history=xy_error_history,
        target_xy_history=target_xy_history,
    )
