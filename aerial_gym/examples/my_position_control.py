from aerial_gym.utils.logging import CustomLogger
import torch
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.helpers import get_args
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from isaacgym import gymapi, gymtorch

# 设置支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

logger = CustomLogger(__name__)

def add_noise(tensor, noise_std=0.01):
    """添加高斯噪声"""
    return tensor + torch.randn_like(tensor, device=tensor.device) * noise_std

def update_mass_properties(env_manager, mass_reduction=0.5):
    """更新质量、惯性张量和质心，模拟子机释放"""
    gym = env_manager.IGE_env.gym
    sim = env_manager.IGE_env.sim
    robot_handle = env_manager.robot_manager.robot_handles[0]
    
    # 获取当前机器人属性
    env_ptr = env_manager.IGE_env.env_handles[0]
    props = gym.get_actor_rigid_body_properties(env_ptr, robot_handle)
    
    # 初始质量（母机 0.25kg + 子机 0.5kg）
    initial_mass = 0.75
    props[0].mass = initial_mass
    new_mass = max(initial_mass - mass_reduction, 0.25)
    
    # URDF 初始惯性张量（base_link）
    initial_inertia = np.array([
        [0.0004225, 0.0, 0.0],
        [0.0, 0.0004225, 0.0],
        [0.0, 0.0, 0.000845]
    ])
    
    # 子机（0.5kg, z=-0.1m）增加 Ixx, Iyy
    sub_inertia = 0.5 * 0.1**2  # 0.005 kg·m²
    initial_inertia[0, 0] += sub_inertia
    initial_inertia[1, 1] += sub_inertia
    
    # 按质量比例缩放（0.25/0.75）
    mass_ratio = new_mass / initial_mass
    new_inertia = initial_inertia * mass_ratio
    
    # 质心：初始 z = (0.25*0 + 0.5*(-0.1))/0.75 = -0.0667，释放后 z=0
    props[0].com = gymapi.Vec3(0.0, 0.0, -0.0667)
    new_com = gymapi.Vec3(0.0, 0.0, 0.0)
    
    # 更新 Mat33（使用 Vec3 构造）
    # 更新惯性矩阵（你的 gym_38.so 版本专用）
    inertia_mat = gymapi.Mat33()
    inertia_mat.x = gymapi.Vec3(new_inertia[0, 0], new_inertia[1, 0], new_inertia[2, 0])
    inertia_mat.y = gymapi.Vec3(new_inertia[0, 1], new_inertia[1, 1], new_inertia[2, 1])
    inertia_mat.z = gymapi.Vec3(new_inertia[0, 2], new_inertia[1, 2], new_inertia[2, 2])
    props[0].inertia = inertia_mat

    props[0].com = new_com
    props[0].mass = new_mass
    
    # 应用物理属性
    gym.set_actor_rigid_body_properties(env_ptr, robot_handle, props)
    
    logger.info(f"子机释放, 质量从 {initial_mass:.4f} 减少到 {props[0].mass:.4f}")
    logger.info(f"惯性张量更新: \n{new_inertia}")
    logger.info(f"质心更新: ({props[0].com.x:.4f}, {props[0].com.y:.4f}, {props[0].com.z:.4f})")
    
    return new_mass, new_inertia

def run_controller(controller_name, args, results):
    """运行 lee_position_control 控制器，模拟子机释放"""
    
    # -----------------------------------------------------------------
    # 错误修复：以下所有代码行都已正确缩进
    # -----------------------------------------------------------------

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

    # --- 【建议 1: 状态初始化修改】 ---
    # 1. 首先重置环境
    env_manager.reset()
    
    # 2. 定义期望的初始状态
    initial_state = torch.zeros(13, device=device)
    initial_state[0:3] = torch.tensor([0.0, 0.0, 0.0], device=device)
    initial_state[6] = 1.0 # 四元数 w
    
    # 3. 获取 env_manager 的根状态张量视图
    root_state = env_manager.IGE_env.global_tensor_dict["vec_root_tensor"]
    
    # 4. 将初始状态写入张量视图
    #    形状为 [num_envs, num_actors_per_env, 13]
    root_state[:, 0, :13] = initial_state
    
    # 5. 调用 write_to_sim 将状态应用到仿真
    env_manager.IGE_env.write_to_sim()
    # --- 【建议 1 结束】 ---
    
    target_position = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device)
    actions = target_position.repeat(env_manager.num_envs, 1)
    
    errors = []
    positions = []
    velocities = [] # <--- 用于存储真实速度
    release_triggered = False
    release_step = 500
    apply_external_force = False
    
    env_ptr = env_manager.IGE_env.env_handles[0]
    robot_handle = env_manager.robot_manager.robot_handles[0]
    gym = env_manager.IGE_env.gym
    sim = env_manager.IGE_env.sim
    
    # 确保 num_bodies 和 body_index 已定义
    num_bodies = gym.get_actor_rigid_body_count(env_ptr, robot_handle)
    body_index = 0 # 施加到根链接 (base_link)
    
    # 获取刚体名称
    try:
        rigid_body_names = gym.get_actor_rigid_body_names(env_ptr, robot_handle)
    except AttributeError:
        rigid_body_names = [f"body_{j}" for j in range(num_bodies)]
        
    logger.warning(f"【循环前确认】刚体数量: {num_bodies}, 刚体列表: {rigid_body_names}")
    logger.warning(f"【循环前确认】施加力目标刚体索引: {body_index}, 名称: {rigid_body_names[body_index]}")


    for i in range(1500):
        obs = env_manager.get_obs()
        
        # 触发时
        if i == 500 and not release_triggered:
            release_step = i
            new_mass, new_inertia = update_mass_properties(env_manager, mass_reduction=0.5)
            controller_mass_tensor = torch.full((env_manager.num_envs,), new_mass, device=device, dtype=torch.float32)
            env_manager.robot_manager.robot.controller.mass = controller_mass_tensor.unsqueeze(1)
            release_triggered = True
            logger.info("子机释放并同步控制器质量")


        # --- 【建议 2: 外力注入修改】 ---
        # 施加外力：持续0.2秒（20个step）
        # (修正： 501 -> 520)
        if 500 <= i < 501:
            # 1. （可选）关闭控制器，观察纯外力效果
            #    保留此行，如建议中所述
            #env_manager.robot_manager.robot.controller.wrench_command[:] = 0.0

            # 2. 获取机器人力张量视图
            #    形状为 [num_envs, num_bodies_per_env, 3]
            robot_force = env_manager.IGE_env.global_tensor_dict["robot_force_tensor"]
            
            # 3. 使用 += 累加外力
            #    施加到第0个环境(env=0)，第0个刚体(body_index=0)
            force_to_apply = torch.tensor([0., 0., 1.], device=device)
            robot_force[:, body_index, :] += force_to_apply
            apply_external_force=True
            if i == 500: # 仅在第一次施加时打印日志
                logger.info(f"Step {i}: 开始施加扰动力 {force_to_apply[2].item():.1f} N 到 {rigid_body_names[body_index]}")
        
        if i == 520:
             logger.info(f"Step {i}: 停止施加扰动力")
        # --- 【建议 2 结束】 ---

        # 然后执行控制与仿真步
        actions = target_position.repeat(env_manager.num_envs, 1)
        
        
        env_manager.reset_tensors()
        env_manager.robot_manager.pre_physics_step(actions)     # 机器人先跑控制器，写入 robot_force_tensor

        if apply_external_force:
            robot_force = env_manager.IGE_env.global_tensor_dict["robot_force_tensor"]
            robot_force[:, body_index, :] += force_to_apply      # 此时再叠加就不会被覆盖

        env_manager.asset_manager.pre_physics_step(None)
        env_manager.obstacle_manager.pre_physics_step(None)
        env_manager.IGE_env.pre_physics_step(actions)
        env_manager.IGE_env.physics_step()
        env_manager.IGE_env.post_physics_step()
        env_manager.robot_manager.post_physics_step()
        env_manager.asset_manager.post_physics_step()
        # 需要的话再调用 compute_observations/render 等
        env_manager.IGE_env.step_graphics()
        env_manager.IGE_env.render_viewer()

        
        
        
        #env_manager.step(actions=actions)
        env_manager.render()

        
        error = torch.norm(target_position[:, 0:3] - obs["robot_position"], dim=1)
        errors.append(error.cpu().numpy())
        positions.append(obs["robot_position"].cpu().numpy())
        velocities.append(obs["robot_vehicle_linvel"].cpu().numpy()) # <--- 存储真实速度
        
        if i % 100 == 0:
            logger.info(f"Step {i}, 当前位置: {obs['robot_position']}, 误差: {error}")
    
    results[controller_name] = {
        "errors": np.array(errors),
        "positions": np.array(positions),
        "velocities": np.array(velocities), # <--- 将真实速度添加到结果
        "release_time": (release_step * 0.01) if release_triggered else None
    }

    
    del env_manager
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
if __name__ == "__main__":
    args = get_args()
    controller_name = "lee_position_control"
    results = {}
    release_step = None

    run_controller(controller_name, args, results)
    
    plt.figure(figsize=(10, 6))
    if controller_name in results:
        plt.plot(np.arange(len(results[controller_name]["errors"])) * 0.01, 
                 results[controller_name]["errors"], 
                 label=f"{controller_name} 误差")
        if results[controller_name]["release_time"] is not None:
            plt.axvline(x=results[controller_name]["release_time"], color='r', linestyle='--', label="子机释放")
    plt.xlabel("时间 (秒)")
    plt.ylabel("位置误差 (米)")
    plt.title("控制器性能")
    plt.legend()
    plt.grid(True)
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    if controller_name in results:
        pos = results[controller_name]["positions"]
        ax.plot(pos[:, 0, 0], pos[:, 0, 1], pos[:, 0, 2], label=f"{controller_name} 轨迹")
        ax.scatter(pos[0, 0, 0], pos[0, 0, 1], pos[0, 0, 2], color='g', label="起始点")
        release_idx = int(results[controller_name]["release_time"] / 0.01)
        ax.scatter(pos[release_idx, 0, 0], pos[release_idx, 0, 1], pos[release_idx, 0, 2], color='r', label="释放点")
        ax.scatter(pos[-1, 0, 0], pos[-1, 0, 1], pos[-1, 0, 2], color='b', label="结束点")
    ax.set_xlabel("X (米)")
    ax.set_ylabel("Y (米)")
    ax.set_zlabel("Z (米)")
    ax.set_title("无人机3D轨迹")
    ax.legend()
    plt.show()
    
    # --- 【额外修正: 绘制真实 Z 轴速度】 ---
    plt.figure(figsize=(10, 6))
    if controller_name in results:
        # 使用 obs 中记录的真实速度，而不是从位置中差分
        vel = results[controller_name]["velocities"]
        # 索引 [:, 0, 2] 对应 [all_steps, env_0, z_axis]
        z_vel = vel[:, 0, 2] 
        plt.plot(np.arange(len(z_vel)) * 0.01, z_vel, label=f"{controller_name} Z轴速度 (真实)")
        if results[controller_name]["release_time"] is not None:
            plt.axvline(x=results[controller_name]["release_time"], color='r', linestyle='--', label="子机释放/扰动开始")
    plt.xlabel("时间 (秒)")
    plt.ylabel("Z轴速度 (米/秒)")
    plt.title("无人机Z轴速度")
    plt.legend()
    plt.grid(True)
    plt.show()
    # --- 【修正结束】 ---
    
    if controller_name in results:
        mean_error = np.mean(results[controller_name]["errors"])
        logger.info(f"平均位置误差: {mean_error:.4f} 米")