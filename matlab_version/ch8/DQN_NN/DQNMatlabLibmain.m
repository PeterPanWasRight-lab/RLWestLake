%% DQNMatlabLibmain: 使用MATLAB强化学习工具箱的DQN实现 (算法8.4架构)
% 按照《强化学习的数学原理》8.4节Deep Q-learning定义网络
% 输入: [行索引, 列索引] (状态)，输出: 5个动作的Q值
% 参考: MATLAB Reinforcement Learning Toolbox文档

clear; clc; close all;

% 添加必要的路径
addpath('.');              % 当前目录
addpath('./utils');        % 工具函数
addpath('../OOP');         % GridWorld类

fprintf('==========================================\n');
fprintf('    MATLAB强化学习工具箱DQN演示 (算法8.4架构)\n');
fprintf('==========================================\n\n');

%% 1. 创建GridWorld环境（与现有演示一致）
fprintf('1. 创建GridWorld环境...\n');
x_len = 3; y_len = 3;
start = [1, 1]; final = [3, 3];
obs = [3, 1];  % 一个障碍物

env_obj = GridWorld(x_len, y_len, start, final, obs);
env_obj.Reward_Target = 5;
env_obj.Reward_Forbidden = -10;
env_obj.Reward_Step = -0.01;

fprintf('   网格大小: %d x %d\n', x_len, y_len);
fprintf('   起点: (%d,%d), 终点: (%d,%d)\n', start(1), start(2), final(1), final(2));
fprintf('   障碍物数量: %d\n', size(obs, 1));
fprintf('   状态空间大小: %d\n', env_obj.State_Space_Size);
fprintf('   动作空间: 5个动作 (上、右、下、左、停留)\n\n');

%% 2. 为MATLAB强化学习工具箱创建环境接口
fprintf('2. 创建MATLAB RL工具箱环境接口...\n');

% 定义观察规范：状态坐标 [行索引, 列索引]
observationInfo = rlNumericSpec([2 1]);  % 2维向量：[行索引; 列索引]
observationInfo.Name = 'State Coordinates';
observationInfo.LowerLimit = [1; 1];
observationInfo.UpperLimit = [x_len; y_len];

% 定义动作规范（离散动作1-5）
actionInfo = rlFiniteSetSpec({1, 2, 3, 4, 5});  % 5个离散动作
actionInfo.Name = 'Action';

% 创建函数环境
env = rlFunctionEnv(observationInfo, actionInfo, ...
    @(action, loggedSignals) stepFunction(action, loggedSignals, env_obj), ...
    @() resetFunction(env_obj));

fprintf('   环境创建成功\n');
fprintf('   观察空间: 状态坐标 [行, 列] ∈ [1,%d]×[1,%d]\n', x_len, y_len);
fprintf('   动作空间: 离散动作 {1,2,3,4,5}\n\n');

%% 3. 创建深度Q网络 (按照范例风格)
fprintf('3. 创建深度Q网络 (按照范例风格)...\n');

% 按照范例创建dlnetwork：输入(状态,动作) -> 输出标量Q值
% 网络架构参考文档范例，但适配GridWorld的简单观察
% 状态路径：状态坐标 [行; 列] -> FC -> ReLU -> FC
% 动作路径：动作标量 -> FC
% 合并路径：加法 -> ReLU -> FC(1)

% 创建dlnetwork对象
net = dlnetwork();

% 状态输入路径（2维状态坐标）
tempLayers = [
    featureInputLayer(2, 'Normalization', 'none', 'Name', 'StateInput')
    fullyConnectedLayer(32, 'Name', 'state_fc1')
    reluLayer('Name', 'state_relu1')
    fullyConnectedLayer(32, 'Name', 'state_fc2')];
net = addLayers(net, tempLayers);

% 动作输入路径（1维动作标量）
tempLayers = [
    featureInputLayer(1, 'Normalization', 'none', 'Name', 'ActionInput')
    fullyConnectedLayer(32, 'Name', 'action_fc1')];
net = addLayers(net, tempLayers);

% 输出路径：合并状态和动作特征
tempLayers = [
    additionLayer(2, 'Name', 'addition')
    reluLayer('Name', 'addition_relu')
    fullyConnectedLayer(1, 'Name', 'QValue')];
net = addLayers(net, tempLayers);

% 连接层
net = connectLayers(net, "state_fc2", "addition/in1");
net = connectLayers(net, "action_fc1", "addition/in2");

% 查看网络结构
% figure
% plot(net)

% 初始化网络
net = initialize(net);

fprintf('   网络架构: 状态(2) -> 32 -> 32; 动作(1) -> 32; 合并 -> 1\n');
fprintf('   符合范例风格：输入(状态,动作) -> 输出标量Q值\n\n');

% 创建critic（按照范例使用rlQValueFunction）
% 按照范例设置critic优化选项
criticOpts = rlOptimizerOptions('LearnRate', 1e-3, 'GradientThreshold', 1);

% 创建Q值函数critic
critic = rlQValueFunction(net, observationInfo, actionInfo, ...
    "ObservationInputNames", "StateInput", ...
    "ActionInputNames", "ActionInput");

fprintf('   Critic创建成功 (使用rlQValueFunction)\n');
fprintf('   学习率: %.4f\n\n', criticOpts.LearnRate);

%% 4. 创建DQN Agent
fprintf('4. 创建DQN Agent...\n');

% 按照范例创建agent选项
agentOpts = rlDQNAgentOptions(...
    UseDoubleDQN=true,...           % 使用双DQN减少高估
    CriticOptimizerOptions=criticOpts,...
    ExperienceBufferLength=10000,...
    MiniBatchSize=64,...
    DiscountFactor=0.95,...
    TargetSmoothFactor=0.001,...
    TargetUpdateFrequency=100,...
    SampleTime=0.1);  % GridWorld环境的采样时间

% 探索策略设置（按照范例风格）
agentOpts.EpsilonGreedyExploration.Epsilon = 1.0;
agentOpts.EpsilonGreedyExploration.EpsilonMin = 0.01;
agentOpts.EpsilonGreedyExploration.EpsilonDecay = 1e-5;  % 按照范例设置

% 创建DQN agent
agent = rlDQNAgent(critic, agentOpts);

fprintf('   代理配置完成 (按照范例风格)\n');
fprintf('   使用双DQN: %s\n', string(agentOpts.UseDoubleDQN));
fprintf('   折扣因子: %.2f\n', agentOpts.DiscountFactor);
fprintf('   探索率衰减: %.3f -> %.3f\n', ...
    agentOpts.EpsilonGreedyExploration.Epsilon, ...
    agentOpts.EpsilonGreedyExploration.EpsilonDecay);
fprintf('   经验回放缓冲区: %d\n\n', agentOpts.ExperienceBufferLength);

%% 5. 训练配置
fprintf('5. 配置训练参数...\n');

trainOpts = rlTrainingOptions;
trainOpts.MaxEpisodes = 300;                 % 最大训练回合数
trainOpts.MaxStepsPerEpisode = 100;          % 每回合最大步数
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 2.5;           % 平均奖励达到2.5时停止
trainOpts.ScoreAveragingWindowLength = 20;   % 平均奖励计算窗口

% 保存选项
trainOpts.SaveAgentCriteria = "EpisodeReward";
trainOpts.SaveAgentValue = 2.0;              % 奖励达到2.0时保存代理
trainOpts.SaveAgentDirectory = pwd;          % 保存到当前目录

% 记录选项
trainOpts.Verbose = true;                    % 显示训练信息
trainOpts.Plots = "training-progress";       % 显示训练进度图

fprintf('   最大回合数: %d\n', trainOpts.MaxEpisodes);
fprintf('   每回合最大步数: %d\n', trainOpts.MaxStepsPerEpisode);
fprintf('   停止条件: 平均奖励 >= %.1f\n', trainOpts.StopTrainingValue);
fprintf('   训练进度图: 启用\n\n');

%% 6. 训练
fprintf('6. 开始训练...\n');
training_start_time = tic;

try
    trainingStats = train(agent, env, trainOpts);
    train_time = toc(training_start_time);
    fprintf('   训练完成！耗时: %.2f 秒\n', train_time);
catch ME
    fprintf('   训练过程中出现错误: %s\n', ME.message);
    fprintf('   尝试调整训练参数...\n');

    % 简化训练选项
    trainOpts.MaxEpisodes = 200;
    trainOpts.StopTrainingValue = 1.5;
    trainOpts.Verbose = false;

    fprintf('   使用简化配置重试...\n');
    training_start_time = tic;
    trainingStats = train(agent, env, trainOpts);
    train_time = toc(training_start_time);
    fprintf('   训练完成！耗时: %.2f 秒\n', train_time);
end

%% 7. 评估训练结果
fprintf('\n7. 评估训练结果...\n');

% 7.1 获取学习到的策略
fprintf('   获取学习到的策略...\n');
policy_matrix = get_policy_from_agent(agent, env_obj);
value_vector = get_value_from_agent(agent, env_obj);

% 7.2 绘制策略图
fprintf('   绘制策略图...\n');
figure('Name', 'MATLAB RL工具箱学习到的策略 (算法8.4架构)', 'Position', [100, 100, 800, 600]);
env_obj.plot_policy_matrix(policy_matrix);
title('MATLAB RL工具箱DQN学习到的策略 (算法8.4架构)');

% 7.3 绘制价值函数图
fprintf('   绘制价值函数图...\n');
figure('Name', 'MATLAB RL工具箱状态价值函数 (算法8.4架构)', 'Position', [100, 100, 1200, 400]);
subplot(1,2,1);
env_obj.plot_values(value_vector);
title('状态价值 (MATLAB RL工具箱, 算法8.4架构)');

subplot(1,2,2);
env_obj.plot_3d_bar_chart(value_vector);
title('3D状态价值 (MATLAB RL工具箱, 算法8.4架构)');

%% 8. 测试训练后的代理
fprintf('\n8. 测试训练后的代理...\n');

num_test_episodes = 10;
test_rewards = zeros(1, num_test_episodes);
test_steps = zeros(1, num_test_episodes);
test_success = false(1, num_test_episodes);

fprintf('   运行 %d 个测试回合:\n', num_test_episodes);

for test_ep = 1:num_test_episodes
    % 重置环境
    [obs, ~] = reset(env);
    episode_reward = 0;
    episode_steps = 0;
    is_done = false;

    while ~is_done && episode_steps < 100
        episode_steps = episode_steps + 1;

        % 选择动作（使用训练好的代理）
        action = getAction(agent, obs);

        % 执行动作
        [next_obs, reward, is_done, ~] = env.step(action);

        % 记录奖励
        episode_reward = episode_reward + reward;

        % 更新状态
        obs = next_obs;
    end

    test_rewards(test_ep) = episode_reward;
    test_steps(test_ep) = episode_steps;
    test_success(test_ep) = (episode_reward > 0); % 成功到达终点会有正奖励

    fprintf('     测试回合 %2d: 奖励=%6.2f, 步数=%3d, %s\n', ...
        test_ep, episode_reward, episode_steps, ...
        iif(test_success(test_ep), '成功到达终点', '未到达终点'));
end

% 测试结果统计
fprintf('   测试结果统计:\n');
fprintf('     平均奖励: %.2f ± %.2f\n', mean(test_rewards), std(test_rewards));
fprintf('     平均步数: %.1f ± %.1f\n', mean(test_steps), std(test_steps));
fprintf('     成功率: %.1f%% (%d/%d)\n', ...
    mean(test_success)*100, sum(test_success), num_test_episodes);

%% 9. 保存结果
fprintf('\n9. 保存结果...\n');
save_dir = './results/matlab_rl_toolbox/';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

save_path = fullfile(save_dir, sprintf('dqn_matlab_rl_algorithm84_%s.mat', datestr(now, 'yyyymmdd_HHMMSS')));
save(save_path, 'agent', 'env_obj', 'policy_matrix', 'value_vector', ...
    'test_rewards', 'test_steps', 'test_success', 'train_time');
fprintf('   结果已保存到: %s\n', save_path);

%% 10. 总结
fprintf('\n==========================================\n');
fprintf('MATLAB强化学习工具箱DQN演示完成！\n');
fprintf('网络架构 (算法8.4风格):\n');
fprintf('  输入: [行索引, 列索引] (2维)\n');
fprintf('  隐藏层: 32 → 32 (ReLU激活)\n');
fprintf('  输出: 5个动作的Q值\n');
fprintf('  符合《强化学习的数学原理》8.4节精神\n');
fprintf('==========================================\n');

%% 辅助函数

% 步进函数（用于rlFunctionEnv）
function [nextObs, reward, isDone, loggedSignals] = stepFunction(action, loggedSignals, env_obj)
    % 获取当前状态坐标
    current_state_idx = loggedSignals;
    current_coord = env_obj.idx2coord(current_state_idx);

    % 执行动作
    [next_state_idx, reward, is_done] = env_obj.step(current_state_idx, action);
    next_coord = env_obj.idx2coord(next_state_idx);

    % 返回结果：状态坐标 [行; 列]
    nextObs = next_coord(:);  % 转为列向量 [行; 列]
    reward = reward;
    isDone = is_done;
    loggedSignals = next_state_idx; % 更新内部状态索引
end

% 重置函数（用于rlFunctionEnv）
function [initialObs, loggedSignals] = resetFunction(env_obj)
    % 返回初始状态坐标
    initial_state_idx = env_obj.coord2idx(env_obj.Start_State);
    initial_coord = env_obj.idx2coord(initial_state_idx);

    initialObs = initial_coord(:);  % 转为列向量 [行; 列]
    loggedSignals = initial_state_idx;
end

% 从代理获取策略矩阵（适配rlQValueFunction）
function policy_matrix = get_policy_from_agent(agent, env_obj)
    state_space_size = env_obj.State_Space_Size;
    policy_matrix = zeros(state_space_size, 5);

    % 获取critic（rlQValueFunction对象）
    critic = getCritic(agent);

    for s_idx = 1:state_space_size
        % 获取状态坐标
        coord = env_obj.idx2coord(s_idx);
        % obs = coord(:);  % 转为列向量 [行; 列]

        % 为每个动作获取Q值（rlQValueFunction需要观察和动作作为元胞数组输入）
        q_vals = zeros(1, 5);
        for a_idx = 1:5
            % 获取单个动作的Q值：将观察和动作作为元胞数组传递
            q_val = getValue(critic, {coord}, {a_idx});

            % 提取标量值
            if iscell(q_val)
                q_val = q_val{1};
            end
            q_vals(a_idx) = q_val;
        end

        % 使用softmax将Q值转换为概率分布
        exps = exp(q_vals - max(q_vals));  % 数值稳定
        policy_matrix(s_idx, :) = exps / sum(exps);
    end
end

% 从代理获取价值向量（适配rlQValueFunction）
function value_vector = get_value_from_agent(agent, env_obj)
    state_space_size = env_obj.State_Space_Size;
    value_vector = zeros(state_space_size, 1);

    % 获取critic（rlQValueFunction对象）
    critic = getCritic(agent);

    for s_idx = 1:state_space_size
        % 获取状态坐标
        coord = env_obj.idx2coord(s_idx);
        % obs = coord(:);  % 转为列向量 [行; 列]

        % 为每个动作获取Q值，取最大值作为状态价值
        max_q = -inf;
        for a_idx = 1:5
            % 获取单个动作的Q值：将观察和动作作为元胞数组传递
            q_val = getValue(critic, {coord}, {a_idx});

            % 提取标量值
            if iscell(q_val)
                q_val = q_val{1};
            end

            % 更新最大值
            if q_val > max_q
                max_q = q_val;
            end
        end

        value_vector(s_idx) = max_q;
    end
end

% 内联if函数
function str = iif(condition, true_str, false_str)
    if condition
        str = true_str;
    else
        str = false_str;
    end
end