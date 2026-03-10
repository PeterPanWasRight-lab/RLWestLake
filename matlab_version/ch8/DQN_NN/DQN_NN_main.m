%% DQN_NN_main: 基于深度学习的DQN演示脚本
% 展示DQN_NN_Agent类的使用方法，包括双DQN、Huber损失、保守Q学习等高级特性
% 环境: GridWorld (5x5网格，包含障碍物)
% 作者: 基于用户提供的DQN伪代码和现有代码库

clear; clc; close all;
addpath('./DQN_NN/');  % 添加当前目录到路径
addpath('./DQN_NN/utils');  % 添加工具函数目录
addpath('./OOP');  % 添加OOP目录，包含GridWorld类

fprintf('==========================================\n');
fprintf('    DQN神经网络演示 (高级特性版) \n');
fprintf('==========================================\n\n');

%% 1. 环境设置
fprintf('1. 创建GridWorld环境...\n');
% 5x5网格世界，起点(1,1)，终点(5,5)，6个障碍物
x_len = 3; y_len = 3;
start = [1, 1]; final = [3, 3];
obs = [ 3,1];

% 创建环境对象
env = GridWorld(x_len, y_len, start, final, obs);
env.Reward_Target = 5;
env.Reward_Forbidden = -10;
env.Reward_Step = -0.01;
fprintf('   网格大小: %d x %d\n', x_len, y_len);
fprintf('   起点: (%d,%d), 终点: (%d,%d)\n', start(1), start(2), final(1), final(2));
fprintf('   障碍物数量: %d\n', size(obs, 1));
fprintf('   状态空间大小: %d\n', env.State_Space_Size);
fprintf('   动作空间: 5个动作 (上、右、下、左、停留)\n\n');

%% 2. 代理配置（启用所有高级特性）
fprintf('2. 配置DQN_NN_Agent（启用所有高级特性）...\n');
options = struct();

% 启用高级特性
options.UseDoubleDQN = false;           % 使用双DQN
options.UseHuberLoss = true;           % 使用Huber损失（对异常值更鲁棒）
options.UseConservativeQLearning = false; % 使用保守Q学习正则化
options.cql_alpha = 0.1;               % CQL正则化系数

% 训练超参数
options.gamma = 0.95;                  % 折扣因子
options.learning_rate = 0.0001;         % 学习率
options.batch_size = 64;               % 批次大小
options.buffer_size = 10000;           % 经验回放缓冲区大小
options.epsilon_decay = 0.999;         % 探索率衰减速度
options.target_update_freq = 100;      % 目标网络更新频率
options.tau = 0.001;                   % 软更新参数
options.max_step = 30;

% 创建代理
agent = DQN_NN_Agent(env, options);
fprintf('   双DQN: %s\n', string(options.UseDoubleDQN));
fprintf('   Huber损失: %s\n', string(options.UseHuberLoss));
fprintf('   保守Q学习: %s (alpha=%.2f)\n', ...
    string(options.UseConservativeQLearning), options.cql_alpha);
fprintf('   软更新参数 tau: %.4f\n', options.tau);
fprintf('   折扣因子 γ: %.2f, 学习率: %.4f\n\n', options.gamma, options.learning_rate);

%% 3. 训练
fprintf('3. 开始训练...\n');
episodes = 2000;  % 训练回合数
train_start_time = tic;
agent.train(episodes);
train_time = toc(train_start_time);

fprintf('   训练完成！耗时: %.2f 秒\n', train_time);
fprintf('   总步数: %d\n', agent.StepCounter);
fprintf('   平均每回合步数: %.1f\n\n', agent.StepCounter / episodes);

%% 4. 结果可视化
fprintf('4. 结果可视化...\n');

% 4.1 显示训练曲线
figure('Name', '训练曲线', 'Position', [100, 100, 1200, 800]);

subplot(2,2,1);
plot(agent.RewardHistory, 'b-', 'LineWidth', 2);
title('总奖励历史 (原始奖励)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('训练回合', 'FontSize', 10);
ylabel('总奖励', 'FontSize', 10);
grid on;
hold on;
% 添加移动平均线（窗口大小=20）
if length(agent.RewardHistory) >= 20
    moving_avg = movmean(agent.RewardHistory, 20);
    plot(moving_avg, 'r-', 'LineWidth', 1.5, 'DisplayName', '移动平均 (20回合)');
    legend('奖励', '移动平均', 'Location', 'best');
end

subplot(2,2,2);
if ~isempty(agent.LossHistory)
    plot(agent.LossHistory, 'r-', 'LineWidth', 1.5);
    title('训练损失历史', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('更新步数', 'FontSize', 10);
    ylabel('损失值', 'FontSize', 10);
    grid on;

    % 添加移动平均线（窗口大小=100）
    if length(agent.LossHistory) >= 100
        hold on;
        loss_moving_avg = movmean(agent.LossHistory, 100);
        plot(loss_moving_avg, 'k-', 'LineWidth', 1, 'DisplayName', '移动平均 (100步)');
        legend('损失', '移动平均', 'Location', 'best');
    end
else
    text(0.5, 0.5, '无损失历史记录', 'HorizontalAlignment', 'center');
    title('训练损失历史', 'FontSize', 12, 'FontWeight', 'bold');
end

subplot(2,2,3);
plot(agent.RewardHistory, 'b-', 'LineWidth', 1.5);
title('奖励历史（对数尺度）', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('训练回合', 'FontSize', 10);
ylabel('总奖励', 'FontSize', 10);
grid on;
set(gca, 'YScale', 'log');

subplot(2,2,4);
if ~isempty(agent.LossHistory)
    semilogy(agent.LossHistory, 'r-', 'LineWidth', 1.5);
    title('损失历史（对数尺度）', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('更新步数', 'FontSize', 10);
    ylabel('损失值', 'FontSize', 10);
    grid on;
else
    text(0.5, 0.5, '无损失历史记录', 'HorizontalAlignment', 'center');
    title('损失历史（对数尺度）', 'FontSize', 12, 'FontWeight', 'bold');
end

% 4.2 获取学习到的策略和价值函数
fprintf('   获取学习到的策略和价值函数...\n');
policy_matrix = agent.get_policy_matrix();
value_vector = agent.get_value_vector();

% 4.3 绘制策略图（概率策略）
fprintf('   绘制概率策略图...\n');
figure('Name', '学习到的策略', 'Position', [100, 100, 800, 600]);
env.plot_policy_matrix(policy_matrix);

% 4.4 绘制价值函数图
fprintf('   绘制状态价值图...\n');
figure('Name', '状态价值函数', 'Position', [100, 100, 1200, 400]);
subplot(1,2,1);
env.plot_values(value_vector);

subplot(1,2,2);
env.plot_3d_bar_chart(value_vector);

%% 5. 测试学习到的策略
fprintf('5. 测试学习到的策略...\n');

% 5.1 运行10个测试回合
num_test_episodes = 10;
test_rewards = zeros(1, num_test_episodes);
test_steps = zeros(1, num_test_episodes);
test_trajectories = cell(1, num_test_episodes);

fprintf('   运行 %d 个测试回合（纯利用，无探索）:\n', num_test_episodes);

% 保存原始探索率，并设置为0进行纯利用测试
original_epsilon = agent.epsilon;
agent.epsilon = 0;

for test_ep = 1:num_test_episodes
    % 重置环境
    curr_state_idx = env.coord2idx(env.Start_State);
    curr_coord = env.idx2coord(curr_state_idx);

    episode_steps = 0;
    episode_reward = 0;
    state_history = curr_state_idx;

    while episode_steps < 100  % 最大步数限制
        episode_steps = episode_steps + 1;

        % 纯利用：使用代理的choose_action方法（ε=0）
        s_norm = agent.normalize_state(curr_coord);
        action_idx = agent.choose_action(s_norm);

        % 执行动作
        [next_state_idx, reward, is_done] = env.step(curr_state_idx, action_idx);
        next_coord = env.idx2coord(next_state_idx);

        % 记录
        episode_reward = episode_reward + reward;
        state_history(end+1) = next_state_idx; %#ok<SAGROW>

        % 更新状态
        curr_state_idx = next_state_idx;
        curr_coord = next_coord;

        if is_done
            break;
        end
    end

    test_rewards(test_ep) = episode_reward;
    test_steps(test_ep) = episode_steps;
    test_trajectories{test_ep} = state_history;

    fprintf('     测试回合 %2d: 奖励=%6.2f, 步数=%3d, %s\n', ...
        test_ep, episode_reward, episode_steps, ...
        iif(is_done, '成功到达终点', '未到达终点'));
end

% 恢复原始探索率
agent.epsilon = original_epsilon;

% 5.2 绘制测试结果
fprintf('   绘制测试结果...\n');
figure('Name', '策略测试结果', 'Position', [100, 100, 1000, 400]);

subplot(1,2,1);
bar(test_rewards);
title('测试回合奖励', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('测试回合编号', 'FontSize', 10);
ylabel('总奖励', 'FontSize', 10);
grid on;
hold on;
plot(xlim, [mean(test_rewards) mean(test_rewards)], 'r--', 'LineWidth', 2);
text(1, mean(test_rewards)*0.9, sprintf('平均: %.2f', mean(test_rewards)), ...
    'FontSize', 10, 'Color', 'r');

subplot(1,2,2);
bar(test_steps);
title('测试回合步数', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('测试回合编号', 'FontSize', 10);
ylabel('步数', 'FontSize', 10);
grid on;
hold on;
plot(xlim, [mean(test_steps) mean(test_steps)], 'r--', 'LineWidth', 2);
text(1, mean(test_steps)*0.9, sprintf('平均: %.1f', mean(test_steps)), ...
    'FontSize', 10, 'Color', 'r');

% 5.3 绘制一个示例轨迹
fprintf('   绘制示例轨迹...\n');
figure('Name', '示例轨迹', 'Position', [100, 100, 600, 500]);
if ~isempty(test_trajectories)
    env.plot_trajectory(test_trajectories{1});
    title(sprintf('示例轨迹 (奖励=%.2f, 步数=%d)', ...
        test_rewards(1), test_steps(1)));
end

%% 6. 代理信息汇总
fprintf('\n6. 代理信息汇总:\n');
agent.show_info();

% 计算策略性能指标
if ~isempty(policy_matrix)
    % 计算策略确定性（最大概率动作的平均概率）
    max_probs = max(policy_matrix, [], 2);
    policy_confidence = mean(max_probs);
    fprintf('   策略确定性: %.3f (平均最大动作概率)\n', policy_confidence);

    % 计算状态价值范围
    fprintf('   状态价值范围: [%.3f, %.3f], 均值: %.3f\n', ...
        min(value_vector), max(value_vector), mean(value_vector));

    % 检查终点状态价值
    target_idx = env.coord2idx(env.Final_State);
    fprintf('   终点状态价值: %.3f\n', value_vector(target_idx));
end

%% 7. 高级特性对比分析（可选）
fprintf('\n7. 高级特性效果分析:\n');
fprintf('   当前配置:\n');
fprintf('     - 双DQN: 减少Q值高估，提高稳定性\n');
fprintf('     - Huber损失: 对异常值更鲁棒，防止梯度爆炸\n');
fprintf('     - 保守Q学习: 防止Q值过度高估，提高离线学习性能\n');
fprintf('     - 软更新: 平滑目标网络更新，提高训练稳定性\n');

%% 8. 保存结果（可选）
fprintf('\n8. 保存结果...\n');
save_dir = './results/';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% 保存代理对象和结果
save_path = fullfile(save_dir, sprintf('dqn_nn_results_%s.mat', datestr(now, 'yyyymmdd_HHMMSS')));
save(save_path, 'agent', 'env', 'options', 'policy_matrix', 'value_vector', ...
    'test_rewards', 'test_steps', 'train_time');
fprintf('   结果已保存到: %s\n', save_path);

%% 辅助函数
function str = iif(condition, true_str, false_str)
    % 内联if函数
    if condition
        str = true_str;
    else
        str = false_str;
    end
end

fprintf('\n==========================================\n');
fprintf('   演示完成！\n');
fprintf('==========================================\n');