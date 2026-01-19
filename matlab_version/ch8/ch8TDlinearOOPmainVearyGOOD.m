
%% run_final_check.m
clear; close all; clc;

% 环境设置
x_len = 5; y_len = 5;
start = [1, 1]; final = [3, 4];
obs = [2,2; 3,2; 3,3; 2,4; 4,4; 2,5];

% x_len = 2; y_len = 2;
% start = [1, 1]; final = [2, 2];
% obs = [1,2];

% x_len = 1; y_len = 2;
% start = [1, 1]; final = [1,2];
% obs = [1,1];

% x_len = 2; y_len = 3;
% start = [1, 1]; final = [2,3];
% obs = [2,2];

% start对结果不起作用  这里只是为了统一接口。后面Sarsa算法会用到
env = GridWorld(x_len, y_len, start, final, obs);

% 在训练完成后，计算ε-greedy策略下的价值
epsilon = 1;   % 平均概率为0.2
gamma = 0.9;
episodes_len = 1000;
max_steps = 100;
num_iterations = 5000;

agent = PolicyIterationAgent(env.State_Space_Size, 5, gamma, episodes_len, max_steps);
env.Reward_Forbidden = -1;
% 注意注释掉env中step函数下的终点吸收规则
V_epsilon = agent.evaluate_epsilon_greedy_policy(env, epsilon, num_iterations);
env.plot_values(V_epsilon);
V_table = reshape(V_epsilon,x_len,y_len);

%%
env = GridWorld(x_len, y_len, start, final, obs);

% !注意: 脚本中将 forbidden reward 修改为了 -1，此处需手动设置以保持一致
env.Reward_Forbidden = -1;

% 2. 实例化 Agent
% 可选类型: 'linear', 'quadratic', 'custom'
agent = TDLinearValue(env, 'linear', 0.9);

% 3. 训练
agent.train(450000);

% 4. 绘图结果
agent.plot_omega_history();
fig_handle=agent.plot_surface();

[m, n] = size(V_table);
% 创建坐标网格  注意坐标顺序
[Y, X] = meshgrid(1:n, 1:m);
% 绘制三维散点图figure;
figure(fig_handle); hold on; 
scatter3(X(:), Y(:), V_table(:), 50, V_table(:), 'filled');

% env.plot_3d_bar_chart(abs([V_table(1,:),V_table(2,:)]')) % 取abs为了好看