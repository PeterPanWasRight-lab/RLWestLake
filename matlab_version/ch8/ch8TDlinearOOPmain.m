%% main_td_value.m - 算法 8.1: 基于值函数的 TD 算法
clear; close all; clc;

% 1. 环境设置
% x_len = 5; y_len = 5;
% start = [1, 1]; final = [3, 4];
% obs = [2,2; 3,2; 3,3; 2,4; 4,4; 2,5];
%%
% x_len = 2; y_len = 2;
% start = [1, 1]; final = [2, 2];
% obs = [1,2];

% x_len = 1; y_len = 2;
% start = [1, 1]; final = [1,2];
% obs = [1,1];

x_len = 2; y_len = 3;
start = [1, 1]; final = [2,3];
obs = [2,2];

env = GridWorld(x_len, y_len, start, final, obs);
env.Reward_Forbidden = -1;
%%

% 2. 实验参数
episodes = 650000;
max_steps = 1;
alpha = 2;
gamma = 0.9;

% 3. 基函数配置 - 现在更灵活  dim越大，收敛越困难。由于用值函数的调整是整个值面的调整，比单点离散的值调整要难收敛很多
basis_type = 'linear'; % 可选: 'linear', 'quadratic', 'custom'

% 4. 创建 TDValueLinear 算法对象
agent = TDValueLinear(env, basis_type);

% 5. 设置算法参数
agent.set_alpha(alpha);
agent.set_gamma(gamma);
agent.set_epsilon(0.3); % ε-greedy策略的ε值

% 6. 训练智能体
fprintf('开始训练...\n');
agent.train(episodes, max_steps);
fprintf('训练完成！\n');

% 7. 获取训练结果
omega = agent.get_weights();
V_table = agent.get_value_table();

% 8. 绘制权重历史
agent.plot_omega_history();

% 9. 绘制值函数的三维网格图
fig_handle = agent.plot_value_function();

% 10. 在值函数图上叠加表格法的值函数
[m, n] = size(V_table);
% 创建坐标网格  注意坐标顺序
[Y, X] = meshgrid(1:n, 1:m);
% 绘制三维散点图
figure(fig_handle); 
hold on; 
scatter3(X(:), Y(:), V_table(:), 50, V_table(:), 'filled');
hold off;

% 11. 绘制环境的三维条形图
env.plot_3d_bar_chart([V_table(1,:), V_table(2,:)]');
% 
% 一直在这个周边浮动，上下好坏。但是总体来说比较正常。
% 最后几万步让alpha = 0.0001后收敛。
% 收敛数据记录在ch8TDlinear75e4.mat中