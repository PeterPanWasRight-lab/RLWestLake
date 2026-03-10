"""
连续矩形世界强化学习环境测试文件
测试ContinuousMountainWorld环境的各种功能
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from continuous_mountain_env import ContinuousMountainWorld

# 设置matplotlib支持中文显示，避免字体警告
try:
    # 设置中文字体，Windows系统常用字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    # 设置字体大小
    plt.rcParams['font.size'] = 12
    # 抑制字体警告
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
except Exception as e:
    print(f"字体设置警告: {e}")
    # 如果字体设置失败，至少抑制警告
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def test_random_agent():
    """测试随机智能体"""
    print("=" * 60)
    print("测试随机智能体")
    print("=" * 60)
    
    # 创建环境
    env = ContinuousMountainWorld(
        world_bounds=(-10, 10, -10, 10),
        max_step=1.5,
        max_steps=100,
        use_3rd_order=True,
        seed=42
    )
    
    # 运行多个episode
    n_episodes = 5
    
    for episode in range(n_episodes):
        print(f"\nEpisode {episode+1}/{n_episodes}")
        print("-" * 40)
        
        # 重置环境
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 随机动作
            action = env.action_space.sample()
            
            # 执行一步
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        print(f"总奖励: {total_reward:.2f}")
        print(f"最高到达海拔: {env.highest_value:.4f}")
        print(f"全局最高海拔: {env.global_max_value:.4f}")
        print(f"到达比例: {env.highest_value/env.global_max_value*100:.1f}%")
        
        # 可视化最后一个episode
        if episode == n_episodes - 1:
            print("\n正在可视化最后一个episode的轨迹...")
            env.render(show_trajectory=True, show_max_point=True, block=True)

def test_greedy_agent():
    """测试贪心智能体（总是向上爬）"""
    print("\n" + "=" * 60)
    print("测试贪心智能体")
    print("=" * 60)
    
    # 创建环境
    env = ContinuousMountainWorld(
        world_bounds=(-10, 10, -10, 10),
        max_step=1.0,
        max_steps=50,
        use_3rd_order=True,
        seed=42
    )
    
    # 重置环境
    state, _ = env.reset(start_pos=(0, 0))
    total_reward = 0
    done = False
    
    # 使用简单的梯度上升
    learning_rate = 0.5
    step_size = env.max_step
    
    while not done:
        x, y = state
        
        # 近似梯度（有限差分）
        eps = 0.1
        altitude_current = env.get_altitude(x, y)
        altitude_dx = env.get_altitude(x + eps, y)
        altitude_dy = env.get_altitude(x, y + eps)
        
        # 计算梯度
        grad_x = (altitude_dx - altitude_current) / eps
        grad_y = (altitude_dy - altitude_current) / eps
        
        # 归一化梯度
        grad_norm = np.sqrt(grad_x**2 + grad_y**2)
        if grad_norm > 0:
            grad_x /= grad_norm
            grad_y /= grad_norm
        
        # 向梯度方向移动
        action = np.array([grad_x * step_size, grad_y * step_size], dtype=np.float32)
        
        # 执行一步
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        if env.step_count % 10 == 0:
            print(f"步数 {env.step_count}: 海拔={info['altitude']:.2f}, 梯度=({grad_x:.3f}, {grad_y:.3f})")
    
    print(f"\n贪心智能体结果:")
    print(f"总奖励: {total_reward:.2f}")
    print(f"最高到达海拔: {env.highest_value:.4f}")
    print(f"全局最高海拔: {env.global_max_value:.4f}")
    print(f"到达比例: {env.highest_value/env.global_max_value*100:.1f}%")
    
    # 可视化
    env.render(show_trajectory=True, show_max_point=True, block=True)

def test_environment_interface():
    """测试环境接口"""
    print("=" * 60)
    print("测试环境接口")
    print("=" * 60)
    
    # 创建环境
    env = ContinuousMountainWorld(
        world_bounds=(-5, 5, -5, 5),
        max_step=0.5,
        max_steps=20,
        use_3rd_order=False,  # 使用二阶曲面
        seed=123
    )
    
    # 测试1: 环境属性
    print("1. 环境属性:")
    print(f"  动作空间: {env.action_space}")
    print(f"  状态空间: {env.observation_space}")
    print(f"  世界边界: x∈[{env.x_min}, {env.x_max}], y∈[{env.y_min}, {env.y_max}]")
    print(f"  最大步数: {env.max_steps}")
    
    # 测试2: 重置环境
    print("\n2. 重置环境:")
    state, _ = env.reset(start_pos=(0, 0))
    print(f"  初始状态: {state}")
    print(f"  初始海拔: {env.get_current_altitude():.2f}")
    
    # 测试3: 执行动作
    print("\n3. 执行动作:")
    test_actions = [
        np.array([0.5, 0.0], dtype=np.float32),  # 向右移动
        np.array([0.0, 0.3], dtype=np.float32),  # 向上移动
        np.array([-0.2, 0.2], dtype=np.float32), # 向左上移动
    ]
    
    for i, action in enumerate(test_actions):
        state, reward, terminated, truncated, info = env.step(action)
        print(f"  动作{i+1}: {action} -> 状态: {state}, 奖励: {reward:.2f}, 海拔: {info['altitude']:.2f}, 步数: {env.step_count}")
    
    # 测试4: 获取任意位置海拔
    print("\n4. 测试海拔计算:")
    test_points = [(-2, -2), (0, 0), (2, 2), (4, 4)]
    for x, y in test_points:
        if env.x_min <= x <= env.x_max and env.y_min <= y <= env.y_max:
            altitude = env.get_altitude(x, y)
            print(f"  点({x}, {y}): 海拔 = {altitude:.2f}")
        else:
            print(f"  点({x}, {y}): 在世界范围外")
    
    # 测试5: 检查边界
    print("\n5. 测试边界限制:")
    # 尝试移动到边界外
    env.reset(start_pos=(env.x_max - 0.1, env.y_max - 0.1))
    action = np.array([1.0, 1.0], dtype=np.float32)  # 尝试超出边界
    state, _, _, _, _ = env.step(action)
    print(f"  边界测试: 尝试移动到边界外 -> 最终位置: {state}")
    
    return env

def explore_surface_features():
    """探索曲面特征"""
    print("=" * 60)
    print("探索曲面特征")
    print("=" * 60)
    
    # 创建两个环境比较不同阶数的曲面
    env_2nd = ContinuousMountainWorld(
        world_bounds=(-8, 8, -8, 8),
        use_3rd_order=False,
        seed=42
    )
    
    env_3rd = ContinuousMountainWorld(
        world_bounds=(-8, 8, -8, 8),
        use_3rd_order=True,
        seed=42
    )
    
    # 分析曲面特征
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (env, title) in enumerate([(env_2nd, "二阶曲面"), (env_3rd, "三阶曲面")]):
        # 生成网格
        x = np.linspace(env.x_min, env.x_max, 100)
        y = np.linspace(env.y_min, env.y_max, 100)
        X, Y = np.meshgrid(x, y)
        
        # 计算海拔
        Z = np.zeros_like(X)
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                Z[row, col] = env.get_altitude(X[row, col], Y[row, col])
        
        # 绘制
        ax = axes[idx]
        contour = ax.contourf(X, Y, Z, 20, cmap='terrain')
        plt.colorbar(contour, ax=ax, label='海拔')
        
        # 标记极值点
        # 采样寻找局部极值
        sample_points = 500
        sample_x = np.random.uniform(env.x_min, env.x_max, sample_points)
        sample_y = np.random.uniform(env.y_min, env.y_max, sample_points)
        sample_z = [env.get_altitude(sx, sy) for sx, sy in zip(sample_x, sample_y)]
        
        # 找到最高和最低点
        max_idx = np.argmax(sample_z)
        min_idx = np.argmin(sample_z)
        
        ax.scatter(sample_x[max_idx], sample_y[max_idx], c='gold', s=150, 
                  marker='*', edgecolors='black', label='最高点')
        ax.scatter(sample_x[min_idx], sample_y[min_idx], c='blue', s=150, 
                  marker='v', edgecolors='black', label='最低点')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'{title}\n最高: {sample_z[max_idx]:.2f}, 最低: {sample_z[min_idx]:.2f}')
        ax.legend()
        ax.set_aspect('equal')
    
    plt.suptitle('不同阶数曲面对比')
    plt.tight_layout()
    plt.show(block=True)
    
    # 打印曲面方程
    print("\n二阶曲面方程 (近似):")
    a = env_2nd.surface_params
    print(f"Q(x,y) = {a[0]:.3f} + {a[1]:.3f}*x + {a[2]:.3f}*y + {a[3]:.3f}*x² + {a[4]:.3f}*y² + {a[5]:.3f}*x*y")
    
    print("\n三阶曲面方程 (近似):")
    a = env_3rd.surface_params
    print(f"Q(x,y) = {a[0]:.3f} + {a[1]:.3f}*x + {a[2]:.3f}*y + {a[3]:.3f}*x² + {a[4]:.3f}*y² + {a[5]:.3f}*x*y + {a[6]:.3f}*x³ + {a[7]:.3f}*y³ + {a[8]:.3f}*x²y + {a[9]:.3f}*xy²")

# 运行所有测试
if __name__ == "__main__":
    # 测试1: 环境接口
    env = test_environment_interface()
    
    # 测试2: 随机智能体
    test_random_agent()
    
    # 测试3: 贪心智能体
    test_greedy_agent()
    
    # 测试4: 探索曲面特征
    explore_surface_features()