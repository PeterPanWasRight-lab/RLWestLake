"""
连续矩形世界强化学习环境
Continuous Mountain World Reinforcement Learning Environment

实现一个连续动作和状态空间的强化学习环境，模拟有峰有谷的小山地形。
智能体可以在二维连续空间中任意移动，目标是到达高海拔区域。
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random

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


class ContinuousMountainWorld(gym.Env):
    """
    连续矩形世界环境
    
    环境特性：
    - 连续动作空间：二维位移向量(dx, dy)，步长限制在max_step内
    - 连续状态空间：当前位置坐标(x, y)
    - 地形：高阶曲面（二阶或三阶多项式），表示有峰有谷的小山
    - 奖励：当前海拔高度，Q(x, y)越大奖励越高
    - 终止条件：达到最大步数(max_steps)
    - 边界：智能体不能超出世界边界，移动会被截断
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 world_bounds: Tuple[float, float, float, float] = (-10, 10, -10, 10),
                 max_step: float = 1.0,
                 max_steps: int = 100,
                 use_3rd_order: bool = True,
                 seed: Optional[int] = None,
                 surface_params: Optional[np.ndarray] = None):
        """
        初始化连续矩形世界环境
        
        参数:
            world_bounds: 世界边界，格式为(x_min, x_max, y_min, y_max)
            max_step: 最大步长，智能体每一步可以移动的最大距离
            max_steps: 最大步数，一个episode的最大步数限制
            use_3rd_order: 是否使用三阶曲面（True为三阶，False为二阶）
            seed: 随机种子，用于可重复性
            surface_params: 自定义曲面参数，如果为None则随机生成
        """
        super().__init__()
        
        # 解析世界边界
        self.x_min, self.x_max, self.y_min, self.y_max = world_bounds
        
        # 环境参数
        self.max_step = max_step
        self.max_steps = max_steps
        self.use_3rd_order = use_3rd_order
        
        # 设置随机种子
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 动作空间：连续2维位移向量，范围在[-max_step, max_step]
        self.action_space = spaces.Box(
            low=-max_step, 
            high=max_step, 
            shape=(2,), 
            dtype=np.float32
        )
        
        # 状态空间：连续2维位置坐标，范围在世界边界内
        self.observation_space = spaces.Box(
            low=np.array([self.x_min, self.y_min], dtype=np.float32),
            high=np.array([self.x_max, self.y_max], dtype=np.float32),
            dtype=np.float32
        )
        
        # 生成曲面参数
        self.surface_params = self._generate_surface_params(use_3rd_order, seed, surface_params)
        
        # 状态变量（将在reset中初始化）
        self.position = None  # 当前位置 (x, y)
        self.step_count = 0   # 当前步数
        self.done = False     # episode是否结束
        self.highest_value = 0.0  # episode中到达的最高海拔
        self.global_max_value = 0.0  # 全局最高海拔（曲面上的最高点）
        self.trajectory = []  # 轨迹历史
        
        # 计算全局最高海拔（通过采样近似）
        self._calculate_global_max()
        
        # 可视化相关
        self.fig = None
        self.ax = None
        self.agent_marker = None
        self.trajectory_line = None
    
    def _generate_surface_params(self, use_3rd_order: bool, seed: Optional[int], 
                                surface_params: Optional[np.ndarray]) -> np.ndarray:
        """
        生成曲面参数
        
        二阶曲面：Q(x,y) = a0 + a1*x + a2*y + a3*x² + a4*y² + a5*x*y (6个参数)
        三阶曲面：Q(x,y) = a0 + a1*x + a2*y + a3*x² + a4*y² + a5*x*y + 
                         a6*x³ + a7*y³ + a8*x²y + a9*xy² (10个参数)
        """
        if surface_params is not None:
            # 使用自定义参数
            if use_3rd_order:
                if len(surface_params) != 10:
                    raise ValueError(f"三阶曲面需要10个参数，但提供了{len(surface_params)}个")
            else:
                if len(surface_params) != 6:
                    raise ValueError(f"二阶曲面需要6个参数，但提供了{len(surface_params)}个")
            return np.array(surface_params, dtype=np.float32)
        
        # 随机生成参数
        if use_3rd_order:
            # 三阶曲面参数：生成有意义的参数，确保曲面有峰有谷
            params = np.zeros(10, dtype=np.float32)
            # 常数项
            params[0] = np.random.uniform(-5, 5)
            # 一阶项（较小）
            params[1] = np.random.uniform(-0.5, 0.5)
            params[2] = np.random.uniform(-0.5, 0.5)
            # 二阶项（控制曲面形状）
            params[3] = np.random.uniform(-0.2, -0.05)  # x²项为负，形成峰
            params[4] = np.random.uniform(-0.2, -0.05)  # y²项为负，形成峰
            params[5] = np.random.uniform(-0.1, 0.1)    # 交叉项
            # 三阶项（增加复杂性）
            params[6] = np.random.uniform(-0.01, 0.01)
            params[7] = np.random.uniform(-0.01, 0.01)
            params[8] = np.random.uniform(-0.01, 0.01)
            params[9] = np.random.uniform(-0.01, 0.01)
        else:
            # 二阶曲面参数
            params = np.zeros(6, dtype=np.float32)
            # 常数项
            params[0] = np.random.uniform(-5, 5)
            # 一阶项
            params[1] = np.random.uniform(-0.5, 0.5)
            params[2] = np.random.uniform(-0.5, 0.5)
            # 二阶项
            params[3] = np.random.uniform(-0.2, -0.05)  # x²项为负
            params[4] = np.random.uniform(-0.2, -0.05)  # y²项为负
            params[5] = np.random.uniform(-0.1, 0.1)    # 交叉项
        
        return params
    
    def _calculate_altitude(self, x: float, y: float) -> float:
        """
        计算给定位置的海拔高度
        
        参数:
            x: x坐标
            y: y坐标
            
        返回:
            海拔高度
        """
        if self.use_3rd_order:
            # 三阶曲面计算
            return (self.surface_params[0] + 
                    self.surface_params[1] * x + self.surface_params[2] * y +
                    self.surface_params[3] * x**2 + self.surface_params[4] * y**2 + 
                    self.surface_params[5] * x * y +
                    self.surface_params[6] * x**3 + self.surface_params[7] * y**3 +
                    self.surface_params[8] * x**2 * y + self.surface_params[9] * x * y**2)
        else:
            # 二阶曲面计算
            return (self.surface_params[0] + 
                    self.surface_params[1] * x + self.surface_params[2] * y +
                    self.surface_params[3] * x**2 + self.surface_params[4] * y**2 + 
                    self.surface_params[5] * x * y)
    
    def _calculate_global_max(self):
        """计算全局最高海拔（通过网格采样近似）"""
        # 在边界内采样多个点
        sample_points = 1000
        sample_x = np.random.uniform(self.x_min, self.x_max, sample_points)
        sample_y = np.random.uniform(self.y_min, self.y_max, sample_points)
        
        max_altitude = -np.inf
        for x, y in zip(sample_x, sample_y):
            altitude = self._calculate_altitude(x, y)
            if altitude > max_altitude:
                max_altitude = altitude
        
        self.global_max_value = max_altitude
    
    def reset(self, start_pos: Optional[Tuple[float, float]] = None, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境到初始状态
        
        参数:
            start_pos: 可选起始位置(x, y)，如果为None则在边界内随机生成
            seed: 随机种子，用于可重复性
            
        返回:
            observation: 初始状态 [x, y]
            info: 包含额外信息的字典
        """
        # 如果提供了种子，设置随机种子
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.step_count = 0
        self.done = False
        self.highest_value = 0.0
        self.trajectory = []
        
        # 设置初始位置
        if start_pos is not None:
            x, y = start_pos
            # 确保起始位置在边界内
            x = np.clip(x, self.x_min, self.x_max)
            y = np.clip(y, self.y_min, self.y_max)
            self.position = np.array([x, y], dtype=np.float32)
        else:
            # 随机生成起始位置
            x = np.random.uniform(self.x_min, self.x_max)
            y = np.random.uniform(self.y_min, self.y_max)
            self.position = np.array([x, y], dtype=np.float32)
        
        # 记录初始位置
        self.trajectory.append(self.position.copy())
        
        # 计算初始海拔
        initial_altitude = self._calculate_altitude(self.position[0], self.position[1])
        self.highest_value = initial_altitude
        
        # 构建info字典
        info = {
            'altitude': initial_altitude,
            'highest_value': self.highest_value,
            'global_max_value': self.global_max_value,
            'position': self.position.copy(),
            'step_count': self.step_count
        }
        
        return self.position.copy(), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一个动作
        
        参数:
            action: 动作向量 [dx, dy]
            
        返回:
            observation: 新状态 [x, y]
            reward: 奖励值（当前海拔高度）
            terminated: 是否自然结束（如成功/失败）
            truncated: 是否因外部限制结束（如步数超限）
            info: 包含额外信息的字典
        """
        if self.done:
            raise ValueError("Episode已经结束，请先调用reset()")
        
        # 确保动作是numpy数组
        action = np.array(action, dtype=np.float32)
        
        # 动作限制：确保步长不超过max_step
        step_length = np.linalg.norm(action)
        if step_length > self.max_step:
            # 按比例缩放动作向量
            action = action * (self.max_step / step_length)
        
        # 计算新位置
        new_position = self.position + action
        
        # 边界处理：确保位置在世界边界内
        new_position[0] = np.clip(new_position[0], self.x_min, self.x_max)
        new_position[1] = np.clip(new_position[1], self.y_min, self.y_max)
        
        # 更新位置
        self.position = new_position
        self.step_count += 1
        
        # 记录轨迹
        self.trajectory.append(self.position.copy())
        
        # 计算海拔和奖励
        altitude = self._calculate_altitude(self.position[0], self.position[1])
        reward = float(altitude)
        
        # 更新最高到达海拔
        if altitude > self.highest_value:
            self.highest_value = altitude
        
        # 检查是否终止
        self.done = (self.step_count >= self.max_steps)
        
        # 构建info字典
        info = {
            'altitude': altitude,
            'step_count': self.step_count,
            'highest_value': self.highest_value,
            'global_max_value': self.global_max_value,
            'position': self.position.copy(),
            'max_steps_reached': self.done and (self.step_count >= self.max_steps)
        }
        
        return self.position.copy(), reward, False, self.done, info
    
    def get_altitude(self, x: float, y: float) -> float:
        """
        获取任意位置的海拔高度
        
        参数:
            x: x坐标
            y: y坐标
            
        返回:
            海拔高度
        """
        return self._calculate_altitude(x, y)
    
    def get_current_altitude(self) -> float:
        """
        获取当前位置的海拔高度
        
        返回:
            当前位置的海拔高度
        """
        return self._calculate_altitude(self.position[0], self.position[1])
    
    def render(self, mode: str = 'human', show_trajectory: bool = True, 
               show_max_point: bool = True, block: bool = False) -> Optional[np.ndarray]:
        """
        渲染环境
        
        参数:
            mode: 渲染模式，'human'显示图形，'rgb_array'返回RGB数组
            show_trajectory: 是否显示智能体轨迹
            show_max_point: 是否显示全局最高点
            block: 是否阻塞显示（仅当mode='human'时有效）。如果为True，程序会暂停直到用户关闭窗口
            
        返回:
            如果mode='rgb_array'，返回RGB数组；否则返回None
        """
        if mode not in self.metadata['render_modes']:
            raise ValueError(f"不支持的渲染模式: {mode}")
        
        # 创建图形
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        self.ax.clear()
        
        # 生成地形网格
        x = np.linspace(self.x_min, self.x_max, 100)
        y = np.linspace(self.y_min, self.y_max, 100)
        X, Y = np.meshgrid(x, y)
        
        # 计算海拔（向量化计算以提高效率）
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self._calculate_altitude(X[i, j], Y[i, j])
        
        # 绘制地形
        contour = self.ax.contourf(X, Y, Z, 20, cmap='terrain', alpha=0.7)
        plt.colorbar(contour, ax=self.ax, label='海拔高度')
        
        # 绘制等高线
        self.ax.contour(X, Y, Z, 10, colors='black', alpha=0.3, linewidths=0.5)
        
        # 显示智能体轨迹
        if show_trajectory and len(self.trajectory) > 1:
            trajectory_array = np.array(self.trajectory)
            self.ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                        'r-', linewidth=2, alpha=0.7, label='轨迹')
            # 标记起点
            self.ax.scatter(trajectory_array[0, 0], trajectory_array[0, 1], 
                          c='green', s=100, marker='o', edgecolors='black', 
                          label='起点', zorder=5)
        
        # 显示智能体当前位置
        if self.position is not None:
            self.ax.scatter(self.position[0], self.position[1], 
                          c='red', s=150, marker='*', edgecolors='black', 
                          label='当前位置', zorder=10)
        
        # 显示全局最高点
        if show_max_point:
            # 采样寻找最高点
            sample_points = 500
            sample_x = np.random.uniform(self.x_min, self.x_max, sample_points)
            sample_y = np.random.uniform(self.y_min, self.y_max, sample_points)
            sample_z = [self._calculate_altitude(sx, sy) for sx, sy in zip(sample_x, sample_y)]
            
            max_idx = np.argmax(sample_z)
            self.ax.scatter(sample_x[max_idx], sample_y[max_idx], 
                          c='gold', s=200, marker='*', edgecolors='black', 
                          label=f'全局最高点 ({sample_z[max_idx]:.2f})', zorder=10)
        
        # 设置图形属性
        self.ax.set_xlabel('X坐标', fontsize=12)
        self.ax.set_ylabel('Y坐标', fontsize=12)
        
        # 标题显示环境信息
        order_str = "三阶" if self.use_3rd_order else "二阶"
        title = f'连续矩形世界环境 ({order_str}曲面)\n'
        title += f'步数: {self.step_count}/{self.max_steps}, '
        title += f'最高到达: {self.highest_value:.2f}, '
        title += f'全局最高: {self.global_max_value:.2f}'
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        self.ax.set_aspect('equal')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        
        # 设置坐标轴范围
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        
        plt.tight_layout()
        
        if mode == 'human':
            plt.draw()
            if block:
                plt.show(block=True)
            else:
                plt.pause(0.01)
            return None
        elif mode == 'rgb_array':
            # 将图形转换为RGB数组
            self.fig.canvas.draw()
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data
    
    def close(self):
        """关闭环境，清理资源"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def __str__(self) -> str:
        """返回环境信息字符串"""
        order_str = "三阶" if self.use_3rd_order else "二阶"
        return (f"ContinuousMountainWorld(\n"
                f"  世界边界: x∈[{self.x_min}, {self.x_max}], y∈[{self.y_min}, {self.y_max}]\n"
                f"  最大步长: {self.max_step}, 最大步数: {self.max_steps}\n"
                f"  曲面类型: {order_str}多项式\n"
                f"  动作空间: {self.action_space}\n"
                f"  状态空间: {self.observation_space}\n"
                f")")
    
    def get_surface_equation(self) -> str:
        """返回曲面方程字符串"""
        a = self.surface_params
        if self.use_3rd_order:
            return (f"Q(x,y) = {a[0]:.3f} + {a[1]:.3f}*x + {a[2]:.3f}*y + "
                    f"{a[3]:.3f}*x² + {a[4]:.3f}*y² + {a[5]:.3f}*x*y + "
                    f"{a[6]:.3f}*x³ + {a[7]:.3f}*y³ + "
                    f"{a[8]:.3f}*x²y + {a[9]:.3f}*xy²")
        else:
            return (f"Q(x,y) = {a[0]:.3f} + {a[1]:.3f}*x + {a[2]:.3f}*y + "
                    f"{a[3]:.3f}*x² + {a[4]:.3f}*y² + {a[5]:.3f}*x*y")


# 简化导入
__all__ = ['ContinuousMountainWorld']