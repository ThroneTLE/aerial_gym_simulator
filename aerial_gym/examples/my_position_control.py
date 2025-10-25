from aerial_gym.utils.logging import CustomLogger
import torch
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.helpers import get_args
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from matplotlib import animation  # CODEx: 用于三维动画重放
import matplotlib
from isaacgym import gymapi
import json
from pathlib import Path
from aerial_gym.utils.math import quat_rotate_inverse, get_euler_xyz_tensor, ssa, quat_rotate  # CODEx
from dataclasses import dataclass
from typing import Dict, List, Tuple

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

logger = CustomLogger(__name__)

DEFAULT_THRUST_MARGIN = 55  # CODEx: 推力裕度，可调，用于适应更大总质量。
DEBUG_DIAG_STEPS = {500, 1000, 1500, 1600, 1700, 1800, 1900}
DIAG_JSON_PATH = "payload_diag_log.jsonl"
LAST_DIAG_STATE = {"linvel": None, "angvel": None}
N=True
def adjust_motor_thrust_limits(  # CODEx
    env_manager,
    margin: float = DEFAULT_THRUST_MARGIN,
    log: bool = True,
):
    """根据当前总质量调高电机推力上限，避免增重后推力不足导致坠落。
    margin（建议 1.5~2.0）越大，允许的最大推力越高。"""
    robot = env_manager.robot_manager.robot
    allocator = robot.control_allocator
    motor_model = allocator.motor_model
    
    num_motors = allocator.cfg.num_motors

    total_mass = env_manager.robot_manager.robot_mass
    gravity_vec = env_manager.IGE_env.global_tensor_dict["gravity"][0]
    gravity_mag = float(torch.linalg.norm(gravity_vec))
    if gravity_mag == 0.0:
        return

    hover_thrust_per_motor = (total_mass * gravity_mag) / num_motors
    max_thrust = hover_thrust_per_motor * margin

    motor_model.max_thrust[:] = max_thrust
    motor_model.min_thrust[:] = 0.0
    if log:
        logger.info(
            f"[Thrust Tuning] mass={total_mass:.3f}kg, hover≈{hover_thrust_per_motor:.2f}N/电机, "
            f"max_thrust=>{max_thrust:.2f}N (margin={margin:.2f})"
        )


@dataclass
class Payload:  # CODEx
    """定义子无人机（负载）的参数；修改 mass/offset 用于快速调参。"""  # CODEx
    name: str
    mass: float
    offset: torch.Tensor  # 3x tensor in base frame meters
    radius: float = 0.05  # CODEx: 可单独调节每个子机可视化球体的半径。


# === 子机布置参数 ===
# 直接修改下面列表即可调整每个子机的质量和相对于母机（base_link）质心的偏移。
# offset 为 [x, y, z]，单位米；mass 单位 kg。
PAYLOAD_LAYOUT: List[Payload] = [  # CODEx
    Payload("front_left", 1, torch.tensor([0.16, 0.16, -0.05]), radius=0.045),
    Payload("front_right", 1, torch.tensor([0.16, -0.16, -0.05]), radius=0.045),
    Payload("rear_left", 1, torch.tensor([-0.16, 0.16, -0.05]), radius=0.045),
    Payload("rear_right", 1, torch.tensor([-0.16, -0.16, -0.05]), radius=0.045),
]

def torch_to_vec3(t: torch.Tensor) -> gymapi.Vec3:
    return gymapi.Vec3(float(t[0].item()), float(t[1].item()), float(t[2].item()))


def tensor33_to_mat33(m: np.ndarray) -> gymapi.Mat33:
    mat = gymapi.Mat33()
    mat.x = gymapi.Vec3(*m[0])
    mat.y = gymapi.Vec3(*m[1])
    mat.z = gymapi.Vec3(*m[2])
    return mat


def log_physics_snapshot(step: int, env_manager, payload_manager: "PayloadManager"):
    """记录全面的诊断数据到日志和 jsonl。"""
    tensors = env_manager.IGE_env.global_tensor_dict
    robot_force = tensors["robot_force_tensor"][0, 0].detach().cpu().numpy().tolist()
    robot_torque = tensors["robot_torque_tensor"][0, 0].detach().cpu().numpy().tolist()
    robot_state = tensors["robot_state_tensor"][0]
    robot_position = robot_state[0:3].detach().cpu().numpy()
    robot_linvel_world = robot_state[7:10].detach().cpu().numpy()
    robot_angvel_world = robot_state[10:13].detach().cpu().numpy()
    robot_linvel_body = tensors.get("robot_vehicle_linvel")
    if robot_linvel_body is not None:
        robot_linvel_body = robot_linvel_body[0].detach().cpu().numpy()
    robot_angvel_body = tensors.get("robot_body_angvel")
    if robot_angvel_body is not None:
        robot_angvel_body = robot_angvel_body[0].detach().cpu().numpy()

    controller = env_manager.robot_manager.robot.controller
    controller_mass = controller.mass[0].item() if controller.mass is not None else float("nan")
    robot_mass = env_manager.robot_manager.robot_mass

    allocator = env_manager.robot_manager.robot.control_allocator
    motor_model = allocator.motor_model

    def summarize_tensor(tensor):
        if isinstance(tensor, torch.Tensor):
            tensor_cpu = tensor.detach().cpu()
            return {
                "min": tensor_cpu.min().item(),
                "max": tensor_cpu.max().item(),
                "mean": tensor_cpu.mean().item(),
            }
        return {"min": tensor, "max": tensor, "mean": tensor}

    sim_dt = 0.01
    sim_cfg_obj = getattr(env_manager, "sim_config", None)
    if sim_cfg_obj is not None and hasattr(sim_cfg_obj, "sim") and hasattr(sim_cfg_obj.sim, "dt"):
        sim_dt = float(sim_cfg_obj.sim.dt)
    elif hasattr(env_manager, "cfg") and hasattr(env_manager.cfg, "sim") and hasattr(env_manager.cfg.sim, "dt"):
        sim_dt = float(env_manager.cfg.sim.dt)
    last_linvel = LAST_DIAG_STATE.get("linvel")
    last_angvel = LAST_DIAG_STATE.get("angvel")
    lin_acc = None
    ang_acc = None
    if last_linvel is not None and sim_dt > 0:
        lin_acc = ((robot_linvel_world - last_linvel) / sim_dt).tolist()
    if last_angvel is not None and sim_dt > 0:
        ang_acc = ((robot_angvel_world - last_angvel) / sim_dt).tolist()

    diag_data = {
        "step": int(step),
        "robot_mass": float(robot_mass),
        "controller_mass": float(controller_mass),
        "motor_min_thrust": summarize_tensor(motor_model.min_thrust),
        "motor_max_thrust": summarize_tensor(motor_model.max_thrust),
        "motor_current_thrust": summarize_tensor(
            getattr(motor_model, "current_motor_thrust", None)
        ),
        "robot_force_body": robot_force,
        "robot_torque_body": robot_torque,
        "robot_position": robot_position.tolist(),
        "robot_linvel_world": robot_linvel_world.tolist(),
        "robot_angvel_world": robot_angvel_world.tolist(),
        "robot_linvel_body": robot_linvel_body.tolist() if robot_linvel_body is not None else None,
        "robot_angvel_body": robot_angvel_body.tolist() if robot_angvel_body is not None else None,
        "robot_lin_acc_world": lin_acc,
        "robot_ang_acc_world": ang_acc,
        "current_com": payload_manager.current_com().tolist(),
        "pos_integral": controller.pos_integral.detach().cpu().numpy().tolist()
        if hasattr(controller, "pos_integral")
        else None,
        "robot_orientation": tensors["robot_orientation"][0].detach().cpu().numpy().tolist(),
    }

    controller_wrench = getattr(controller, "wrench_command", None)
    if isinstance(controller_wrench, torch.Tensor):
        diag_data["controller_wrench"] = controller_wrench[0].detach().cpu().numpy().tolist()
    else:
        diag_data["controller_wrench"] = controller_wrench

    logger.error(
        f"[Diag][Step {step}] mass={robot_mass:.3f}kg, controller_mass={controller_mass:.3f}, "
        f"min_thrust={diag_data['motor_min_thrust']}, max_thrust={diag_data['motor_max_thrust']}, "
        f"motor_thrust={diag_data['motor_current_thrust']}, controller_wrench={diag_data['controller_wrench']}"
    )

    log_path = Path(DIAG_JSON_PATH)
    with log_path.open("a", encoding="utf-8") as f:
        json.dump(diag_data, f)
        f.write("\n")

    LAST_DIAG_STATE["linvel"] = robot_linvel_world
    LAST_DIAG_STATE["angvel"] = robot_angvel_world

class PayloadManager:

    def __init__(self, env_manager, payloads: List[Payload], device: torch.device):
        self.env_manager = env_manager
        self.device = device
        self.payloads = {payload.name: payload for payload in payloads}
        self.attached = {payload.name: True for payload in payloads}

        self.gym = env_manager.IGE_env.gym
        self.sim = env_manager.IGE_env.sim
        self.env_ptr = env_manager.IGE_env.env_handles[0]
        self.robot_handle = env_manager.robot_manager.robot_handles[0]

        self.body_props = self.gym.get_actor_rigid_body_properties(self.env_ptr, self.robot_handle)
        base_prop = self.body_props[0]
        self.base_mass = base_prop.mass
        self.base_com = torch.tensor(
            [base_prop.com.x, base_prop.com.y, base_prop.com.z],
            device=self.device,
            dtype=torch.float32,
        )
        self.base_inertia = torch.tensor(
            [
                [base_prop.inertia.x.x, base_prop.inertia.x.y, base_prop.inertia.x.z],
                [base_prop.inertia.y.x, base_prop.inertia.y.y, base_prop.inertia.y.z],
                [base_prop.inertia.z.x, base_prop.inertia.z.y, base_prop.inertia.z.z],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        self.pending_impulses: List[Dict[str, torch.Tensor]] = []
        self.mass_history: List[float] = []
        self.com_history: List[torch.Tensor] = []
        self.base_state_view = env_manager.IGE_env.vec_root_tensor

        self.apply_to_sim()

    def _compute_total_properties(self):
        total_mass = torch.tensor(self.base_mass, device=self.device)
        numerator_com = self.base_mass * self.base_com.clone()

        for name, payload in self.payloads.items():
            if not self.attached[name]:
                continue
            total_mass += payload.mass
            numerator_com += payload.mass * payload.offset.to(self.device)

        combined_com = torch.zeros(3, device=self.device)
        if total_mass.item() > 1e-6:
            combined_com[:] = numerator_com / total_mass

        identity = torch.eye(3, device=self.device)
        inertia_total = self.base_inertia.clone()
        d_base = self.base_com - combined_com
        inertia_total += self.base_mass * (
            torch.dot(d_base, d_base) * identity - torch.outer(d_base, d_base)
        )

        for name, payload in self.payloads.items():
            if not self.attached[name]:
                continue
            r = payload.offset.to(self.device) - combined_com
            inertia_total += payload.mass * (torch.dot(r, r) * identity - torch.outer(r, r))

        return total_mass, inertia_total, combined_com

    # === anchor: PayloadManager.apply_to_sim (BEGIN) ===
    def apply_to_sim(self):
        total_mass, inertia_total, combined_com = self._compute_total_properties()

        # 每次都计算虚拟惯量 (about new COM)
        inertia_np = inertia_total.detach().cpu().numpy()

        # --- 仅首轮把惯量写回物理引擎，其后不再写入 ---
        # 如果你前面没有 global N，请在文件顶部定义 N=True，并在此声明 global
        global N
        if N is True:
            N = False
            print("[PayloadManager] 应用到仿真(仅一次写物理惯量): 总质量={:.3f}kg, COM={}".format(
                float(total_mass.item()), combined_com.detach().cpu().numpy().tolist()))
            base_prop = self.body_props[0]
            base_prop.mass = float(total_mass.item())
            base_prop.com = torch_to_vec3(combined_com)
            # 只在第一次真正写入物理惯量
            base_prop.inertia = tensor33_to_mat33(inertia_np)

        # 仍然把 body_props 写回，以保证 mass/COM 生效；惯量在第一次之后不再改变
        self.gym.set_actor_rigid_body_properties(
            self.env_ptr, self.robot_handle, self.body_props, recomputeInertia=False
        )

        env = self.env_manager
        # 把“虚拟惯量”同步到张量，供控制器使用（重要！）
        inertia_tensor = torch.tensor(inertia_np, device=self.device, dtype=torch.float32)
        env.robot_manager.robot_inertia = inertia_tensor
        env.robot_manager.robot_inertias[:] = inertia_tensor
        env.IGE_env.global_tensor_dict["robot_inertia"][:] = inertia_tensor

        # 质量相关同步（保持你原逻辑）
        env.robot_manager.robot_mass = float(total_mass.item())
        env.robot_manager.robot_masses.fill_(float(total_mass.item()))
        env.IGE_env.global_tensor_dict["robot_mass"].fill_(float(total_mass.item()))
        controller_mass = torch.full((env.num_envs, 1), float(total_mass.item()), device=self.device)
        env.robot_manager.robot.controller.mass = controller_mass

        adjust_motor_thrust_limits(env_manager=self.env_manager, margin=DEFAULT_THRUST_MARGIN)

        self.mass_history.append(float(total_mass.item()))
        self.com_history.append(combined_com.detach().cpu())



    def release_payload(self, name: str):
        if name not in self.payloads:
            logger.error(f"Payload '{name}' 未定义，忽略释放命令。")
            return
        if not self.attached[name]:
            logger.warning(f"Payload '{name}' 已经释放，忽略重复命令。")
            return

        payload = self.payloads[name]
        prev_mass = self.current_mass()

        # ---- 读取释放前的惯量与角速度（机体系） ----
        I_old = self.env_manager.robot_manager.robot_inertia.clone()  # 3x3
        # 机体系角速度张量：env_manager.IGE_env.global_tensor_dict["robot_body_angvel"]
        angvel_body_view = self.env_manager.IGE_env.global_tensor_dict.get("robot_body_angvel", None)
        if angvel_body_view is not None:
            omega_old_body = angvel_body_view[0].clone()
        else:
            # 兜底：从世界系角速旋回到机体系（若必须）
            omega_old_body = self.env_manager.IGE_env.global_tensor_dict["robot_angvel"][0].clone()

        # ---- 标记释放并应用新的 mass/COM/惯量 ----
        offset = payload.offset.to(self.device)
        self.attached[name] = False
        self.apply_to_sim()  # 会写入新的 mass/COM/inertia，并同步到 controller 等

        # ---- 线速度缩放（你原有逻辑，守恒线动量近似）----
        new_mass = self.current_mass()
        root_state = self.env_manager.IGE_env.global_tensor_dict["vec_root_tensor"]
        base_states = root_state[:, 0] if root_state.ndim == 3 else root_state
        if new_mass > 1e-5:
            scale = prev_mass / new_mass
            base_states[:, 7:10] *= scale

        # ---- 角动量守恒：I_old * ω_old = I_new * ω_new => ω_new = I_new^{-1} * (I_old * ω_old) ----
        I_new = self.env_manager.robot_manager.robot_inertia  # 3x3 (apply_to_sim 后的新值)
        try:
            omega_new_body = torch.linalg.solve(I_new, I_old @ omega_old_body)
            if angvel_body_view is not None:
                angvel_body_view[0] = omega_new_body
            else:
                # 若没有机体系视图，则写回世界系对应张量（实际工程中基本都有机体系视图）
                self.env_manager.IGE_env.global_tensor_dict["robot_angvel"][0] = omega_new_body
        except RuntimeError as e:
            logger.error(f"[release_payload] 角动量守恒映射失败: {e}")

        self.env_manager.IGE_env.write_to_sim()


        #gravity_vec = self.env_manager.IGE_env.global_tensor_dict["gravity"][0].to(self.device)
        #torque_impulse = -payload.mass * torch.cross(offset, gravity_vec)
        #if torch.linalg.norm(torque_impulse).item() > 1e-6:
        #    self.pending_impulses.append(
        #        {"torque": torque_impulse, "steps": torch.tensor(0, device=self.device)}
        #    )

    def attach_payload(self, name: str):
        if name not in self.payloads:
            logger.error(f"Payload '{name}' 未定义，无法重新挂载。")
            return
        self.attached[name] = True
        self.apply_to_sim()
        self.env_manager.IGE_env.write_to_sim()

    def current_mass(self) -> float:
        return self.mass_history[-1] if self.mass_history else self.base_mass

    def current_com(self) -> torch.Tensor:
        return self.com_history[-1] if self.com_history else self.base_com.cpu()

    def consume_torque_impulse(self) -> torch.Tensor:
        if not self.pending_impulses:
            return torch.zeros(3, device=self.device)
        total = torch.zeros(3, device=self.device)
        remaining: List[Dict[str, torch.Tensor]] = []
        for item in self.pending_impulses:
            total += item["torque"]
            steps_left = item["steps"] - 1
            if steps_left.item() > 0:
                item["steps"] = steps_left
                remaining.append(item)
        self.pending_impulses = remaining
        return total

    def update_payload_visuals(self):
        viewer_ctrl = self.env_manager.IGE_env.viewer
        if viewer_ctrl is None or viewer_ctrl.viewer is None:
            return

        viewer = viewer_ctrl.viewer

        if hasattr(self.gym, "clear_lines"):
            self.gym.clear_lines(viewer)

        base_state = self.base_state_view[0, 0]
        base_pos = base_state[0:3]
        base_quat = base_state[3:7]

        for name, payload in self.payloads.items():
            if not self.attached[name]:
                continue
            offset_world = quat_rotate(
                base_quat.unsqueeze(0), payload.offset.unsqueeze(0).to(self.device)
            ).squeeze(0)
            payload_pos = base_pos + offset_world

            axes = [
                torch.tensor([payload.radius, 0.0, 0.0], device=self.device),
                torch.tensor([0.0, payload.radius, 0.0], device=self.device),
                torch.tensor([0.0, 0.0, payload.radius], device=self.device),
            ]

            verts = []
            for axis in axes:
                p1 = payload_pos - axis
                p2 = payload_pos + axis
                verts.append(p1.cpu().numpy())
                verts.append(p2.cpu().numpy())

            if verts:
                verts_np = np.asarray(verts, dtype=np.float32)
                colors_np = np.asarray([[0.2, 0.6, 1.0]] * len(verts), dtype=np.float32)
                self.gym.add_lines(viewer, self.env_ptr, len(verts) // 2, verts_np, colors_np)




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
    controller = env_manager.robot_manager.robot.controller
    # [解决方案]：注释掉下面这几行，以匹配 RL 训练时的配置

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

    def external_force_callback(manager):
        robot_force = manager.IGE_env.global_tensor_dict["robot_force_tensor"]
        orientations = manager.IGE_env.global_tensor_dict["robot_orientation"]
        # 外力（风/扰动）窗口：保留你的实现
        for window, force_vec in FORCE_WINDOWS.items():
            active = window[0] <= current_step <= window[1]
            external_force_state[window] = active
            if not active:
                continue
            world_force = force_vec.unsqueeze(0).expand(orientations.shape[0], -1)
            body_force = quat_rotate_inverse(orientations, world_force)
            robot_force[:, body_index, :] += body_force

        # 不再从 pending_impulses 注入扭矩脉冲；角动量已由释放瞬间的映射来保证
        # 若未来确需注入外扭矩，请确保只调用一次 consume_torque_impulse 且来源明确
        # robot_torque = manager.IGE_env.global_tensor_dict["robot_torque_tensor"]
        # torque_world = payload_manager.consume_torque_impulse()
        # if torch.linalg.norm(torque_world).item() > 0.0:
        #     torque_body = quat_rotate_inverse(orientations, torque_world.unsqueeze(0).expand(orientations.shape[0], -1))
        #     robot_torque[:, body_index, :] += torque_body


    errors, positions, velocities = [], [], []
    attitudes, mass_log, com_log = [], [], []
    motor_force_history, motor_torque_history = [], []

    for current_step in range(NUM_STEPS):
        if current_step in release_lookup:
            payload_manager.release_payload(release_lookup[current_step])  # CODEx

        payload_manager.update_payload_visuals()  # CODEx
        env_manager.reset_tensors()
        env_manager.step(actions=actions, external_force_callback=external_force_callback)  # CODEx
        if not args.headless:
            env_manager.render()

        obs = env_manager.get_obs()
        error = torch.norm(target_position[:, 0:3] - obs["robot_position"], dim=1)
        errors.append(error.cpu().numpy())
        positions.append(obs["robot_position"].cpu().numpy())
        velocities.append(obs["robot_vehicle_linvel"].cpu().numpy())
        attitudes.append(ssa(get_euler_xyz_tensor(obs["robot_orientation"])).cpu().numpy())  # CODEx
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
        
        #log_physics_snapshot(current_step, env_manager, payload_manager)

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

        # --- 三维动画重放（含简易飞机模型） ---  # CODEx
        pos_anim = sim_results["positions"][:, 0, :]
        att_anim = sim_results["attitudes"][:, 0, :]

        def rpy_to_rot(rpy):
            roll, pitch, yaw = rpy
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            rot = np.array(
                [
                    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                    [-sp, cp * sr, cp * cr],
                ]
            )
            return rot

        fig_anim = plt.figure(figsize=(8, 6))
        ax_anim = fig_anim.add_subplot(111, projection="3d")
        ax_anim.set_title("无人机姿态与轨迹动画")
        ax_anim.set_xlabel("X (米)")
        ax_anim.set_ylabel("Y (米)")
        ax_anim.set_zlabel("Z (米)")

        padding = 0.2
        ax_anim.set_xlim(pos_anim[:, 0].min() - padding, pos_anim[:, 0].max() + padding)
        ax_anim.set_ylim(pos_anim[:, 1].min() - padding, pos_anim[:, 1].max() + padding)
        ax_anim.set_zlim(pos_anim[:, 2].min() - padding, pos_anim[:, 2].max() + padding)

        path_line, = ax_anim.plot([], [], [], "k--", alpha=0.4)
        arm_colors = ["r", "g", "b"]
        arm_lines = [ax_anim.plot([], [], [], color=col, lw=2)[0] for col in arm_colors]

        arm_length = 0.25  # CODEx: 飞机臂长度，可根据模型调整
        arm_vectors = np.array(
            [
                [[-arm_length, 0.0, 0.0], [arm_length, 0.0, 0.0]],
                [[0.0, -arm_length, 0.0], [0.0, arm_length, 0.0]],
                [[0.0, 0.0, -0.08], [0.0, 0.0, 0.08]],
            ]
        )

        def init_anim():
            path_line.set_data([], [])
            path_line.set_3d_properties([])
            for line in arm_lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return [path_line, *arm_lines]

        def update_anim(frame):
            pos = pos_anim[frame]
            rot = rpy_to_rot(att_anim[frame])

            path_line.set_data(pos_anim[: frame + 1, 0], pos_anim[: frame + 1, 1])
            path_line.set_3d_properties(pos_anim[: frame + 1, 2])

            for idx, line in enumerate(arm_lines):
                endpoints = arm_vectors[idx] @ rot.T + pos
                line.set_data(endpoints[:, 0], endpoints[:, 1])
                line.set_3d_properties(endpoints[:, 2])
            return [path_line, *arm_lines]

        anim = animation.FuncAnimation(
            fig_anim,
            update_anim,
            init_func=init_anim,
            frames=len(pos_anim),
            interval=40,
            blit=True,
            repeat=True,
        )

        plt.show()
