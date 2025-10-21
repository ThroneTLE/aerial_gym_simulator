# 导入自定义日志工具
from aerial_gym.utils.logging import CustomLogger

# 初始化日志记录器，使用当前模块名作为名称
logger = CustomLogger(__name__)

# 导入模拟环境构建器
from aerial_gym.sim.sim_builder import SimBuilder
import torch # 导入 PyTorch
from aerial_gym.utils.helpers import get_args # 导入获取命令行参数的函数

# 确保代码只在作为主程序运行时执行
if __name__ == "__main__":
    # 获取命令行参数（例如 --num_envs, --headless, --use_warp）
    args = get_args()
    
    logger.warning("This example demonstrates the use of geometric controllers for a quadrotor.")
    
    # 使用 SimBuilder 构建模拟环境
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",             # 模拟配置名称
        env_name="empty_env",            # 环境配置名称（空环境）
        robot_name="base_quadrotor",     # 机器人配置名称
        controller_name="lee_position_control", # 控制器名称（几何 Lee 位置控制器）
        args=None,
        device="cuda:0",                 # 设备设置为 GPU
        num_envs=args.num_envs,          # 并行环境数量（来自命令行参数）
        headless=args.headless,          # 是否无头模式（来自命令行参数）
        use_warp=args.use_warp,          # 是否使用 Warp 物理引擎（来自命令行参数）
    )
    
    # 初始化动作张量：形状为 (num_envs, 4)。4 通常代表推力指令和偏航率指令。
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    
    # 重置环境到初始状态
    env_manager.reset()
    
    # 运行模拟循环
    for i in range(10000):
        # 每 1000 步执行一次操作（此处是重置环境）
        if i % 1000 == 0:
            logger.info(f"Step {i}, changing target setpoint.")
            # 以下两行是被注释掉的，它们本来用于随机化动作指令或目标位置
            # actions[:, 0:3] = 2.0 * (torch.rand_like(actions[:, 0:3]) * 2 - 1)
            # actions[:, 3] = torch.pi * (torch.rand_like(actions[:, 3]) * 2 - 1)
            env_manager.reset() # 重置环境（包括机器人和目标）
            
        # 执行一步模拟
        env_manager.step(actions=actions)