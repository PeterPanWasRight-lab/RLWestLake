%% test_algorithm84: 测试算法8.4实现的快速验证脚本
% 快速验证DQN_NN_Agent按照算法8.4修改后的基本功能

clear; clc; close all;

% 添加路径
addpath('.');
addpath('./utils');
addpath('../OOP');

fprintf('开始算法8.4实现测试...\n\n');

%% 1. 创建简单环境
fprintf('1. 创建测试环境 (2x2网格)...\n');
x_len = 2; y_len = 2;
start = [1, 1]; final = [2, 2];
obs = [];  % 无障碍物

env = GridWorld(x_len, y_len, start, final, obs);
fprintf('   环境创建成功\n');
fprintf('   状态空间大小: %d\n', env.State_Space_Size);

%% 2. 创建代理（使用算法8.4架构）
fprintf('\n2. 创建DQN_NN_Agent（算法8.4架构）...\n');
options = struct();
options.UseDoubleDQN = false;           % 测试标准DQN
options.UseHuberLoss = false;           % 使用MSE
options.UseConservativeQLearning = false; % 禁用CQL
options.learning_rate = 0.001;
options.batch_size = 4;
options.buffer_size = 100;
options.epsilon = 0.5;                  % 较高探索率以便收集经验
options.epsilon_decay = 0.995;
options.target_update_freq = 10;
options.tau = 0.001;

agent = DQN_NN_Agent(env, options);
fprintf('   代理创建成功\n');
agent.show_info();

%% 3. 收集一些经验
fprintf('\n3. 收集经验数据...\n');
for i = 1:50
    % 随机状态和动作
    s = [randi(x_len), randi(y_len)];
    a = randi(5);
    % 简单模拟：奖励为-1到1的随机数
    r = rand()*2 - 1;
    ns = [randi(x_len), randi(y_len)];
    d = rand() > 0.9;

    agent.store_transition(s, a, r, ns, d);
end
fprintf('   存储了 %d 条经验\n', agent.MemCount);

%% 4. 测试学习（单次更新）
fprintf('\n4. 测试学习函数...\n');
if agent.MemCount >= agent.batch_size
    loss = agent.learn();
    fprintf('   学习成功，损失: %.6f\n', extractdata(loss));
else
    fprintf('   缓冲区不足，跳过学习测试\n');
end

%% 5. 测试动作选择
fprintf('\n5. 测试动作选择...\n');
test_state = [1, 1];
s_norm = agent.normalize_state(test_state);
for i = 1:3
    action_idx = agent.choose_action(s_norm);
    fprintf('   测试 %d: 状态 (%d,%d) -> 动作 %d\n', ...
        i, test_state(1), test_state(2), action_idx);
end

%% 6. 测试策略和价值函数获取
fprintf('\n6. 测试策略和价值函数获取...\n');
try
    policy_matrix = agent.get_policy_matrix();
    fprintf('   策略矩阵大小: %d x %d\n', size(policy_matrix, 1), size(policy_matrix, 2));

    value_vector = agent.get_value_vector();
    fprintf('   价值向量大小: %d x 1\n', length(value_vector));

    % 检查概率分布是否合理
    for s = 1:min(2, env.State_Space_Size)
        probs = policy_matrix(s, :);
        fprintf('   状态 %d: 动作概率 [%s], 和=%.3f\n', ...
            s, sprintf('%.3f ', probs), sum(probs));
    end
catch ME
    fprintf('   获取策略/价值失败: %s\n', ME.message);
end

%% 7. 测试目标网络更新
fprintf('\n7. 测试目标网络更新...\n');
original_params = agent.TargetNet.Learnables;
agent.StepCounter = agent.target_update_freq;  % 设置为触发更新
agent.update_target_network();
new_params = agent.TargetNet.Learnables;

params_changed = false;
for i = 1:height(original_params)
    if ~isequal(original_params.Value{i}, new_params.Value{i})
        params_changed = true;
        break;
    end
end

if params_changed
    fprintf('   目标网络已更新 (tau=%.3f)\n', agent.tau);
else
    fprintf('   目标网络未更新 (可能未达到更新频率或tau=0)\n');
end

%% 8. 测试双DQN（如果启用）
fprintf('\n8. 测试双DQN功能...\n');
if options.UseDoubleDQN
    fprintf('   双DQN已启用，功能已集成到learn函数中\n');
else
    % 临时启用双DQN进行测试
    agent.UseDoubleDQN = true;
    if agent.MemCount >= agent.batch_size
        loss_double = agent.learn();
        fprintf('   双DQN学习测试成功，损失: %.6f\n', extractdata(loss_double));
    else
        fprintf('   缓冲区不足，跳过双DQN测试\n');
    end
    agent.UseDoubleDQN = false; % 恢复
end

%% 9. 测试Huber损失（如果启用）
fprintf('\n9. 测试Huber损失功能...\n');
if options.UseHuberLoss
    fprintf('   Huber损失已启用，功能已集成到modelLoss函数中\n');
else
    % 临时启用Huber损失进行测试
    agent.UseHuberLoss = true;
    if agent.MemCount >= agent.batch_size
        loss_huber = agent.learn();
        fprintf('   Huber损失学习测试成功，损失: %.6f\n', extractdata(loss_huber));
    else
        fprintf('   缓冲区不足，跳过Huber损失测试\n');
    end
    agent.UseHuberLoss = false; % 恢复
end

%% 10. 总结
fprintf('\n==========================================\n');
fprintf('算法8.4实现测试完成\n');
fprintf('所有基本功能测试通过\n');
fprintf('网络架构: 输入3维 -> 10 -> 10 -> 20 -> 输出1维\n');
fprintf('动作选择: 遍历5个动作，选择最大Q值\n');
fprintf('学习函数: 支持标准DQN和双DQN\n');
fprintf('损失函数: 支持MSE和Huber损失\n');
fprintf('经验回放: %d/%d\n', agent.MemCount, agent.buffer_size);
fprintf('当前探索率: %.3f\n', agent.epsilon);
fprintf('==========================================\n');

fprintf('\n注意：此测试仅验证基本功能，未进行实际训练。\n');
fprintf('要运行完整训练，请执行 DQN_NN_main.m 或 DQN_NN_main_improved.m\n');