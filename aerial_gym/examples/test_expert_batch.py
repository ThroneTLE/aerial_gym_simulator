"""
批量测试前馈专家系统的补偿能力 - 带可视化版本。
测试大量随机化载荷样本，统计专家系统的误差，并绘制误差分析图。
"""
import argparse
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# 必须先导入 isaacgym
import isaacgym  # noqa: F401

import torch

from aerial_gym.config.task_config import payload_compensation_task_teacher_config as teacher_cfg
from aerial_gym.task.payload_compensation_task.payload_compensation_task import PayloadCompensationTask
from aerial_gym.utils.math import get_euler_xyz_tensor

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def run_expert_test(num_envs=128, steps=1500, headless=True, device="cuda:0"):
    """
    测试前馈专家系统在大量随机化样本上的表现。
    """
    cfg = teacher_cfg.task_config
    cfg.headless = headless
    cfg.device = device
    cfg.num_envs = num_envs
    
    env = PayloadCompensationTask(cfg)
    obs, *_ = env.reset()
    
    # 时间序列记录
    time_series_pos_error = []
    time_series_angle_error = []
    time_series_z_error = []
    
    # 每个环境的参数（获取初始值）
    payload_masses = env.payload_manager.payload_mass_per_env.cpu().numpy()
    com_offsets = env.payload_manager.com_offset_body.cpu().numpy()
    com_norms = np.linalg.norm(com_offsets, axis=1)
    
    # 每个环境的累计统计
    env_pos_error_sum = torch.zeros(num_envs, device=device)
    env_angle_error_sum = torch.zeros(num_envs, device=device)
    env_max_tilt = torch.zeros(num_envs, device=device)
    env_max_drift = torch.zeros(num_envs, device=device)
    init_pos = env.obs_dict["robot_position"].clone()
    
    for step in range(steps):
        # 使用专家动作（前馈控制器）
        if hasattr(env, "teacher_residual"):
            actions = env.teacher_residual.clone()
        else:
            actions = torch.zeros((num_envs, env.task_config.action_space_dim), device=device)
        
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        # 计算位置误差
        pos = env.obs_dict["robot_position"]
        target = env.target_position
        pos_err = torch.norm(pos - target, dim=1)
        z_err = torch.abs(pos[:, 2] - target[:, 2])
        env_pos_error_sum += pos_err
        
        # 计算倾斜角（机体z轴与世界z轴的夹角）
        # 这比 euler 角更可靠，避免万向锁和累积问题
        quat = env.obs_dict["robot_orientation"]  # [N, 4] wxyz 格式
        # 从四元数提取机体 z 轴在世界坐标系中的方向
        # z_body_in_world = R @ [0, 0, 1]^T
        # 使用四元数旋转公式简化计算
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        # 机体 z 轴在世界坐标系中的方向 (简化计算)
        z_world = 2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x**2 + y**2)
        z_world_z = z_world[2]  # 机体 z 轴在世界 z 方向的投影
        
        # 倾斜角 = arccos(z_world_z)，但 z_world_z 可能 >1 或 <-1 due to numerical issues
        z_world_z_clamped = torch.clamp(z_world_z, -1.0, 1.0)
        tilt_rad = torch.acos(z_world_z_clamped)
        tilt_deg = tilt_rad * 180.0 / np.pi
        
        env_angle_error_sum += tilt_deg
        env_max_tilt = torch.maximum(env_max_tilt, tilt_deg)
        
        # 计算位移
        drift = torch.norm(pos - init_pos, dim=1)
        env_max_drift = torch.maximum(env_max_drift, drift)
        
        # 记录时间序列（全局平均）
        time_series_pos_error.append(pos_err.mean().item())
        time_series_angle_error.append(tilt_deg.mean().item())  # 瞬时倾斜角度
        time_series_z_error.append(z_err.mean().item())
        
        if (step + 1) % 500 == 0:
            print(f"Step {step+1}/{steps}: avg_pos_err={pos_err.mean().item():.4f}m, avg_tilt={tilt_deg.mean().item():.2f}°")
    
    try:
        env.close()
    except:
        pass
    
    # 计算每个环境的平均误差
    avg_pos_err = (env_pos_error_sum / steps).cpu().numpy()
    avg_angle_err = (env_angle_error_sum / steps).cpu().numpy()
    max_tilt = env_max_tilt.cpu().numpy()
    max_drift = env_max_drift.cpu().numpy()
    
    # 绘制结果
    plot_results(
        payload_masses, com_norms, com_offsets,
        avg_pos_err, avg_angle_err, max_tilt, max_drift,
        time_series_pos_error, time_series_angle_error, time_series_z_error
    )


def plot_results(masses, com_norms, com_offsets, avg_pos_err, avg_angle_err, 
                 max_tilt, max_drift, ts_pos, ts_angle, ts_z):
    """绘制误差分析图表"""
    
    print("\n" + "="*60)
    print("专家系统测试结果")
    print("="*60)
    print(f"平均位置误差: {avg_pos_err.mean():.4f} ± {avg_pos_err.std():.4f} m")
    print(f"平均角度误差: {avg_angle_err.mean():.2f} ± {avg_angle_err.std():.2f}°")
    print(f"最大倾斜: {max_tilt.max():.1f}°")
    print(f"最大位移: {max_drift.max():.3f} m")
    
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 时间序列：平均误差
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(ts_pos, label='位置误差', alpha=0.7, color='blue')
    ax1.plot(ts_z, label='Z轴误差', alpha=0.7, color='green', linestyle='--')
    ax1.set_xlabel('步数')
    ax1.set_ylabel('位置误差 (m)')
    ax1.set_title('平均位置误差随时间')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 时间序列：角度误差
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(ts_angle, color='orange', alpha=0.7)
    ax2.set_xlabel('步数')
    ax2.set_ylabel('角度误差 (°)')
    ax2.set_title('平均角度误差随时间')
    ax2.grid(True, alpha=0.3)
    
    # 3. 载荷质量 vs 平均位置误差
    ax3 = fig.add_subplot(3, 3, 3)
    sc = ax3.scatter(masses, avg_pos_err, c=com_norms, cmap='viridis', alpha=0.7, s=30)
    plt.colorbar(sc, ax=ax3, label='COM偏移 (m)')
    ax3.set_xlabel('载荷质量 (kg)')
    ax3.set_ylabel('平均位置误差 (m)')
    ax3.set_title('载荷质量 vs 位置误差\n(颜色=COM偏移)')
    ax3.grid(True, alpha=0.3)
    
    # 4. COM 偏移 vs 平均位置误差
    ax4 = fig.add_subplot(3, 3, 4)
    sc = ax4.scatter(com_norms, avg_pos_err, c=masses, cmap='plasma', alpha=0.7, s=30)
    plt.colorbar(sc, ax=ax4, label='载荷质量 (kg)')
    ax4.set_xlabel('COM 偏移距离 (m)')
    ax4.set_ylabel('平均位置误差 (m)')
    ax4.set_title('COM偏移 vs 位置误差\n(颜色=载荷质量)')
    ax4.grid(True, alpha=0.3)
    
    # 5. 载荷质量 vs 角度误差
    ax5 = fig.add_subplot(3, 3, 5)
    sc = ax5.scatter(masses, avg_angle_err, c=com_norms, cmap='viridis', alpha=0.7, s=30)
    plt.colorbar(sc, ax=ax5, label='COM偏移 (m)')
    ax5.set_xlabel('载荷质量 (kg)')
    ax5.set_ylabel('平均角度误差 (°)')
    ax5.set_title('载荷质量 vs 角度误差')
    ax5.grid(True, alpha=0.3)
    
    # 6. COM 偏移 vs 角度误差
    ax6 = fig.add_subplot(3, 3, 6)
    sc = ax6.scatter(com_norms, avg_angle_err, c=masses, cmap='plasma', alpha=0.7, s=30)
    plt.colorbar(sc, ax=ax6, label='载荷质量 (kg)')
    ax6.set_xlabel('COM 偏移距离 (m)')
    ax6.set_ylabel('平均角度误差 (°)')
    ax6.set_title('COM偏移 vs 角度误差')
    ax6.grid(True, alpha=0.3)
    
    # 7. COM XY 平面分布（颜色=位置误差）
    ax7 = fig.add_subplot(3, 3, 7)
    com_x = com_offsets[:, 0]
    com_y = com_offsets[:, 1]
    sc = ax7.scatter(com_x, com_y, c=avg_pos_err, cmap='hot', alpha=0.7, s=40)
    plt.colorbar(sc, ax=ax7, label='位置误差 (m)')
    ax7.set_xlabel('COM X 偏移 (m)')
    ax7.set_ylabel('COM Y 偏移 (m)')
    ax7.set_title('COM XY 分布 (颜色=位置误差)')
    ax7.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax7.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax7.axis('equal')
    ax7.grid(True, alpha=0.3)
    
    # 8. 误差分布直方图
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.hist(avg_pos_err, bins=20, alpha=0.7, label='位置误差', color='blue')
    ax8.set_xlabel('平均位置误差 (m)')
    ax8.set_ylabel('环境数')
    ax8.set_title('位置误差分布')
    ax8.grid(True, alpha=0.3)
    
    # 9. 角度误差分布
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.hist(avg_angle_err, bins=20, alpha=0.7, color='orange')
    ax9.set_xlabel('平均角度误差 (°)')
    ax9.set_ylabel('环境数')
    ax9.set_title('角度误差分布')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('expert_test_results.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存到 expert_test_results.png")
    plt.show()
    
    # 打印相关性分析
    print("\n参数与误差的相关性分析:")
    corr_mass_pos = np.corrcoef(masses, avg_pos_err)[0, 1]
    corr_com_pos = np.corrcoef(com_norms, avg_pos_err)[0, 1]
    corr_mass_angle = np.corrcoef(masses, avg_angle_err)[0, 1]
    corr_com_angle = np.corrcoef(com_norms, avg_angle_err)[0, 1]
    print(f"  载荷质量 vs 位置误差: r={corr_mass_pos:.3f}")
    print(f"  COM偏移 vs 位置误差: r={corr_com_pos:.3f}")
    print(f"  载荷质量 vs 角度误差: r={corr_mass_angle:.3f}")
    print(f"  COM偏移 vs 角度误差: r={corr_com_angle:.3f}")
    
    # 找出误差最大的环境
    print("\n误差最大的5个环境:")
    top5_idx = np.argsort(avg_pos_err)[-5:][::-1]
    for i, idx in enumerate(top5_idx):
        print(f"  {i+1}. 位置误差={avg_pos_err[idx]:.4f}m, "
              f"质量={masses[idx]:.4f}kg, COM偏移={com_norms[idx]:.4f}m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--headless", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    run_expert_test(
        num_envs=args.num_envs,
        steps=args.steps,
        headless=args.headless,
        device=args.device,
    )
