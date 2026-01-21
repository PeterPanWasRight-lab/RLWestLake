clear; clc; close all;

%% 1. 环境设置
env = GridWorld(5, 5, [1,1], [4,4], [2,2; 3,2; 2,3]);

%% 2. Agent 设置 (注意 layer_config)
% 输入层 = 3 (x, y, action)
% 中间层 = 30, 30
% 输出层 = 1 (Value)
layer_config = [3, 30, 30, 1]; 
lr = 0.002;
gamma = 0.9;

agent = DQN_Agent_SA(env, layer_config, lr, gamma, 5000, 64);

%% 3. 训练
agent.train(1000); % 这种结构收敛可能稍慢，增加回合数

%% 4. 绘图
policy = agent.get_policy_matrix();
env.plot_policy_matrix(policy);