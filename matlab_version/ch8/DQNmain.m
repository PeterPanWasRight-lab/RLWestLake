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
% 清理旧数据，防止类定义冲突%类里应该有动态申请内存，越跑越慢
clear classes; 
clear; clc; close all;

% 1. 环境参数
% x_len = 2; y_len = 3;
% start = [1, 1]; final = [2,3];
% obs = [2,2];

x_len = 5; y_len = 5;
start = [1, 1]; final = [3, 4];
obs = [2,2; 3,2; 3,3; 2,4; 4,4; 2,5];

% 2. 初始化
env = GridWorld(x_len, y_len, start, final, obs);
agent = DQN_Toolbox_Agent(env);

% 3. 训练 (建议 1000 回合)
% 你会看到 "Total Reward" 慢慢从负数变成正数
MAX_EPISODES = 1000; 
agent.train(MAX_EPISODES);

%% --- 可视化结果 ---
fprintf('正在绘制结果...\n');

% 1. 策略箭头图
policy_mat = agent.get_policy_matrix();
env.plot_policy_matrix(policy_mat);

% 2. 3D 价值分布图
state_values = agent.get_value_vector();
env.plot_3d_bar_chart(state_values);

% 3. 最终路径测试
path_indices = [env.coord2idx(env.Start_State)];
curr_coord = env.Start_State;
agent.epsilon = 0; % 关闭随机探索

for i = 1:50
    s_norm = agent.normalize_state(curr_coord);
    a_idx = agent.choose_action(s_norm);
    
    [next_idx, ~, is_done] = env.step(env.coord2idx(curr_coord), a_idx);
    curr_coord = env.idx2coord(next_idx);
    path_indices(end+1) = next_idx;
    
    if is_done, break; end
end
env.plot_trajectory(path_indices);

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