%% test_basic: DQN_NN_Agent基本功能测试
% 不进行实际训练，只检查类是否可以正确实例化和调用基本方法

clear; clc; close all;

% 添加必要的路径
addpath('.');
addpath('./utils');
addpath('../OOP');

fprintf('开始基本功能测试...\n\n');

%% 1. 创建简单环境
fprintf('1. 创建测试环境 (2x2网格)...\n');
x_len = 2; y_len = 2;
start = [1, 1]; final = [2, 2];
obs = [];  % 无障碍物

env = GridWorld(x_len, y_len, start, final, obs);
fprintf('   环境创建成功\n');
fprintf('   状态空间大小: %d\n', env.State_Space_Size);

%% 2. 创建代理（禁用高级特性以简化测试）
fprintf('\n2. 创建DQN_NN_Agent（禁用高级特性）...\n');
options = struct();
options.UseDoubleDQN = false;
options.UseHuberLoss = false;
options.UseConservativeQLearning = false;
options.batch_size = 4;  % 小批次大小
options.buffer_size = 100;
options.epsilon = 0.1;   % 低探索率

try
    agent = DQN_NN_Agent(env, options);
    fprintf('   代理创建成功\n');
    agent.show_info();
catch ME
    fprintf('   代理创建失败: %s\n', ME.message);
    rethrow(ME);
end

%% 3. 测试经验回放
fprintf('\n3. 测试经验回放...\n');
% 存储一些测试经验
for i = 1:10
    s = [randi(x_len), randi(y_len)];
    a = randi(5);
    r = randn() * 10;
    ns = [randi(x_len), randi(y_len)];
    d = rand() > 0.8;

    agent.store_transition(s, a, r, ns, d);
end
fprintf('   存储了 %d 条经验 (缓冲区: %d/%d)\n', ...
    10, agent.MemCount, agent.buffer_size);

%% 4. 测试动作选择
fprintf('\n4. 测试动作选择...\n');
test_state = [1, 1];
s_norm = agent.normalize_state(test_state);
for i = 1:5
    action_idx = agent.choose_action(s_norm);
    fprintf('   测试 %d: 状态 (%d,%d) -> 动作 %d\n', ...
        i, test_state(1), test_state(2), action_idx);
end

%% 5. 测试学习函数（当缓冲区足够时）
fprintf('\n5. 测试学习函数...\n');
if agent.MemCount >= agent.batch_size
    try
        loss = agent.learn();
        fprintf('   学习成功，损失: %.6f\n', extractdata(loss));
    catch ME
        fprintf('   学习失败: %s\n', ME.message);
        fprintf('   错误堆栈:\n');
        for i = 1:length(ME.stack)
            fprintf('     %s:%d\n', ME.stack(i).name, ME.stack(i).line);
        end
    end
else
    fprintf('   缓冲区不足 (需要 %d，当前 %d)，跳过学习测试\n', ...
        agent.batch_size, agent.MemCount);
end

%% 6. 测试策略和价值函数获取
fprintf('\n6. 测试策略和价值函数获取...\n');
try
    policy_matrix = agent.get_policy_matrix();
    fprintf('   策略矩阵大小: %d x %d\n', size(policy_matrix, 1), size(policy_matrix, 2));

    value_vector = agent.get_value_vector();
    fprintf('   价值向量大小: %d x 1\n', length(value_vector));

    % 检查概率分布是否合理
    for s = 1:min(3, env.State_Space_Size)
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

%% 8. 测试工具函数
fprintf('\n8. 测试工具函数...\n');
try
    % 测试huber_loss
    pred = dlarray([1, 2, 3], 'CB');
    target = dlarray([1.1, 1.9, 3.2], 'CB');
    huber_loss_val = huber_loss(pred, target, 1.0);
    fprintf('   Huber损失测试通过: %.6f\n', extractdata(huber_loss_val));

    % 测试cql_regularizer (仅当启用时)
    if agent.UseConservativeQLearning
        q_vals = dlarray(randn(5, 4), 'CB');  % 5个动作，4个批次
        acts = randi(5, [1, 4]);
        cql_penalty = cql_regularizer(q_vals, acts, 0.1);
        fprintf('   CQL正则化测试通过: %.6f\n', extractdata(cql_penalty));
    else
        fprintf('   CQL未启用，跳过测试\n');
    end
catch ME
    fprintf('   工具函数测试失败: %s\n', ME.message);
end

%% 总结
fprintf('\n==========================================\n');
fprintf('基本功能测试完成\n');
fprintf('代理信息:\n');
agent.show_info();
fprintf('==========================================\n');

% 清理
fprintf('\n测试完成！所有基本功能似乎正常工作。\n');
fprintf('注意：此测试仅验证基本功能，未进行实际训练。\n');
fprintf('要运行完整训练，请执行 DQN_NN_main.m\n');