clear; clc; close all;

%% 1. 环境设置
x_len = 2; y_len = 3;
start = [1, 1]; final = [2,3];
obs = [2,2];

env = GridWorld(x_len, y_len, start, final, obs);

%% 2. Agent 设置 (注意 layer_config)
% 输入层 = 3 (x, y, action)
% 中间层 = 30, 30
% 输出层 = 1 (Value)
layer_config = [3, 50, 30, 1]; 
lr = 0.002;
gamma = 0.9;

agent = DQN_Agent_SA(env, layer_config, lr, gamma, 5000, 64);

%% 3. 训练
agent.train(1000); % 这种结构收敛可能稍慢，增加回合数

%% 4. 绘图
policy = agent.get_policy_matrix();
env.plot_policy_matrix(policy);
%%
% 清空工作区
clear; clc; close all;

% 1. 定义环境参数
X_Len = 6;
Y_Len = 6;
Start = [1, 1];
Target = [4, 4];
% 设置一些障碍物
Obstacles = [
    2, 2;
    2, 3;
    3, 2;
    4, 2;
    4, 5
];

% 2. 创建环境
env = GridWorld(X_Len, Y_Len, Start, Target, Obstacles);

% 3. 创建基于 Toolbox 的 Agent
% 注意：这里调用的是我们新写的类 DQN_Toolbox_Agent
agent = DQN_Toolbox_Agent(env);

% 4. 开始训练
% 建议训练 500-1000 回合以查看明显效果
MAX_EPISODES = 500; 
agent.train(MAX_EPISODES);

%% --- 训练后可视化 ---

% 1. 获取策略矩阵并绘图
policy_mat = agent.get_policy_matrix();
env.plot_policy_matrix(policy_mat);

% 2. 绘制 3D 价值图
% 我们需要先从 Agent 获取每个状态的最大 Q 值
state_values = agent.get_value_vector();
env.plot_3d_bar_chart(state_values);

% 3. 测试一次最终轨迹
curr_state = env.Start_State;
path_indices = [env.coord2idx(curr_state)];
for i = 1:30
    s_norm = agent.normalize_state(curr_state);
    % 贪婪选择 (设置 epsilon = 0)
    old_eps = agent.epsilon;
    agent.epsilon = 0;
    a_idx = agent.choose_action(s_norm);
    agent.epsilon = old_eps;
    
    [next_idx, ~, is_done] = env.step(env.coord2idx(curr_state), a_idx);
    curr_state = env.idx2coord(next_idx);
    path_indices(end+1) = next_idx;
    
    if is_done, break; end
end
env.plot_trajectory(path_indices);