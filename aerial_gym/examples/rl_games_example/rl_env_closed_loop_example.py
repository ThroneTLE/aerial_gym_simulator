import time
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)  # 初始化自定义日志记录器，用于打印调试信息
from aerial_gym.registry.task_registry import task_registry  # 导入任务注册表，用于创建任务
import torch  # 导入 PyTorch，用于张量操作和模型加载

from aerial_gym.examples.rl_games_example.rl_games_inference import MLP  # 导入 MLP 模型类，用于 RL 推理

import time  # 重复导入 time，可能为代码错误或冗余
import numpy as np  # 导入 NumPy，用于随机种子和数值操作

import matplotlib.pyplot as plt  # 导入 Matplotlib，用于绘图（但脚本中未实际使用，可能用于后续可视化 error_list 等）

if __name__ == "__main__":  # 主程序入口
    logger.print_example_message()  # 打印示例日志消息（可能是 Aerial Gym 的调试输出）
    start = time.time()  # 记录开始时间，用于基准测试
    seed = 42  # 设置随机种子，确保可重复性
    torch.manual_seed(seed)  # 设置 PyTorch 种子
    np.random.seed(seed)  # 设置 NumPy 种子
    torch.cuda.manual_seed(seed)  # 设置 CUDA 种子（用于 GPU 操作）

    plt.style.use("seaborn-v0_8-colorblind")  # 设置 Matplotlib 绘图风格（但未使用绘图，可能为未来扩展）

    rl_task_env = task_registry.make_task(  # 使用任务注册表创建任务实例
        "position_setpoint_task",  # 任务名称：位置设定点任务（文档中示例任务，用于无人机导航到目标位置）
        # "position_setpoint_task_acceleration_sim2real",  # 注释的备选任务，可能用于 sim-to-real 加速
        # other params are not set here and default values from the task config file are used
        seed=seed,  # 随机种子
        headless=False,  # 非 headless 模式：启用可视化查看器（文档中 viewer 配置）
        num_envs=24,  # 并行环境数量：24 个仿真实例（文档中 env.num_envs）
        use_warp=True,  # 使用 Warp（文档中 env.use_warp，用于加速渲染）
    )
    rl_task_env.reset()  # 重置所有环境，初始化状态（文档中 Task.reset）

    actions = torch.zeros(  # 创建动作张量，全零初始化
        (
            rl_task_env.sim_env.num_envs,  # 环境数量
            rl_task_env.task_config.action_space_dim,  # 动作维度（文档中 action_space_dim=4，对于四旋翼控制）
        )
    ).to("cuda:0")  # 移动到 GPU

    model = (  # 加载 MLP 模型
        MLP(
            rl_task_env.task_config.observation_space_dim,  # 输入维度：观测空间（文档中 observation_space_dim=13）
            rl_task_env.task_config.action_space_dim,  # 输出维度：动作空间
            # "networks/morphy_policy_for_rigid_airframe.pth"  # 注释的备选模型路径
            "networks/attitude_policy.pth"  # 加载姿态控制策略模型（attitude policy，可能用于控制无人机姿态）
            # "networks/morphy_policy_for_flexible_airframe_joint_aware.pth",  # 另一备选
        )
        .to("cuda:0")  # 移动到 GPU
        .eval()  # 设置为评估模式（无梯度计算）
    )
    actions[:] = 0.0  # 初始化动作全零
    counter = 0  # 计数器，未使用
    action_list = []  # 空列表，可能用于记录动作（但未填充）
    error_list = []  # 空列表，可能用于记录误差（但未填充）
    joint_pos_list = []  # 空列表，可能用于关节位置（但四旋翼无关节，未使用）
    joint_vel_list = []  # 空列表，可能用于关节速度（未使用）

    with torch.no_grad():  # 无梯度上下文，适合推理
        for i in range(10000):  # 循环 10000 步仿真
            if i == 100:  # 从第 100 步开始计时（跳过初始化开销）
                start = time.time()
            obs, reward, terminated, truncated, info = rl_task_env.step(actions=actions)  # 环境步进：输入动作，获取观测、奖励等（符合 Gymnasium API，文档中 Task.step）
            start_time = time.time()  # 记录推理开始时间
            #逻辑链step->simulate->pre_physics_step->self.robot.step(self.actions)->self.apply_disturbance()
            actions[:] = model.forward(obs["observations"])  # 使用模型生成新动作（MLP 前向传播）
            end_time = time.time()  # 记录结束时间（测量模型推理时间）

    end = time.time()  # 记录总结束时间