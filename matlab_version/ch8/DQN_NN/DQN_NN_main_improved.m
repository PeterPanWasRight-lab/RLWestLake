%% DQN_NN_main_improved: 改进的DQN演示脚本
% 针对奖励不提升问题进行调整：奖励重塑、探索策略优化、网络简化
% 环境: GridWorld (3x3网格，包含障碍物)
% 目标: 提高训练效率，使奖励能够显著提升

clear; clc; close all;
addpath('./DQN_NN/');  % 添加当前目录到路径
addpath('./DQN_NN/utils');  % 添加工具函数目录
addpath('./OOP');  % 添加OOP目录，包含GridWorld类

fprintf('==========================================\n');
fprintf('    DQN神经网络改进版 (解决奖励不提升问题) \n');
fprintf('==========================================\n\n');

%% 1. 环境设置（使用更平衡的奖励）
fprintf('1. 创建GridWorld环境（调整奖励平衡）...\n');
x_len = 3; y_len = 3;
start = [1, 1]; final = [3, 3];
obs = [3, 2];  % 一个障碍物

% 创建环境对象
env = GridWorld(x_len, y_len, start, final, obs);

% 关键改进：奖励重塑，使奖励信号更平衡
% 原始：终点+3，撞墙-10，走路0 → 不平衡，智能体更关注避免撞墙
% 改进：终点+10，撞墙-1，走路-0.05（鼓励快速到达终点）
env.Reward_Target = 10;     % 增加终点奖励
env.Reward_Forbidden = -1;  % 减少撞墙惩罚
% env.Reward_Step = -0.05;   % 轻微负奖励鼓励快速行动（可选）

fprintf('   网格大小: %d x %d\n', x_len, y_len);
fprintf('   起点: (%d,%d), 终点: (%d,%d)\n', start(1), start(2), final(1), final(2));
fprintf('   障碍物数量: %d\n', size(obs, 1));
fprintf('   状态空间大小: %d\n', env.State_Space_Size);
fprintf('   动作空间: 5个动作 (上、右、下、左、停留)\n');
fprintf('   奖励设置: 终点=+%.1f, 撞墙=%.1f, 普通移动=%.2f\n\n', ...
    env.Reward_Target, env.Reward_Forbidden, env.Reward_Step);

%% 2. 代理配置（优化超参数）
fprintf('2. 配置DQN_NN_Agent（优化超参数）...\n');
options = struct();

% 高级特性配置
options.UseDoubleDQN = true;           % 保持双DQN（减少高估）
options.UseHuberLoss = false;          % 暂时使用MSE损失（更简单）
options.UseConservativeQLearning = false; % 暂时禁用CQL（防止过度保守）
% options.cql_alpha = 0.01;            % 如果启用CQL，使用更小的系数

% 训练超参数优化
options.gamma = 0.95;                  % 折扣因子（保持）
options.learning_rate = 0.001;         % 学习率（保持）

% 关键改进：更快的探索衰减
options.epsilon_decay = 0.995;         % 从0.999改为0.995，加速探索衰减
options.epsilon_min = 0.01;            % 最小探索率

% 批次和缓冲区大小（适合小网格）
options.batch_size = 32;               % 减少批次大小（从64到32）
options.buffer_size = 2000;            % 减少缓冲区大小（从10000到2000）

% 目标网络更新（保持）
options.target_update_freq = 100;      % 目标网络更新频率
options.tau = 0.001;                   % 软更新参数

% 最大步数（适合3x3网格）
options.max_step = 30;                 % 略微增加最大步数

% 关键改进：简化网络架构（防止过拟合）
% 对于9个状态的小网格，原网络128-64-32-5过于复杂
options.network_layers = [64, 32];     % 自定义网络层大小

% 创建代理（使用自定义构造函数）
agent = create_custom_agent(env, options);

fprintf('   双DQN: %s\n', string(options.UseDoubleDQN));
fprintf('   Huber损失: %s\n', string(options.UseHuberLoss));
fprintf('   保守Q学习: %s\n', string(options.UseConservativeQLearning));
fprintf('   探索衰减: %.3f → %.3f (500回合后ε≈%.3f)\n', ...
    1.0, options.epsilon_decay, 1.0 * options.epsilon_decay^500);
fprintf('   网络架构: 输入2 → %s → 输出5\n', strjoin(string(options.network_layers), ' → '));
fprintf('   折扣因子 γ: %.2f, 学习率: %.4f\n\n', options.gamma, options.learning_rate);

%% 3. 训练（添加详细监控）
fprintf('3. 开始训练（添加Q值监控）...\n');
episodes = 500;  % 训练回合数

% 添加监控变量
success_history = zeros(1, episodes);  % 记录是否成功到达终点
q_value_history = cell(1, episodes);   % 记录Q值统计
avg_reward_history = zeros(1, episodes); % 记录平均奖励

train_start_time = tic;

% 扩展训练循环以监控更多指标
figure(1); clf;
for ep = 1:episodes
    % 重置环境
    curr_state_idx = env.coord2idx(env.Start_State);
    curr_coord = env.idx2coord(curr_state_idx);

    total_reward = 0;
    steps = 0;
    reached_target = false;

    % 记录本回合的Q值
    episode_q_values = [];

    while steps < agent.max_step
        steps = steps + 1;

        % 1. 归一化当前状态
        norm_state = agent.normalize_state(curr_coord);

        % 2. 选择动作 (ε-greedy)
        action_idx = agent.choose_action(norm_state);

        % 3. 记录选择动作前的Q值（仅利用时）
        if rand >= agent.epsilon
            input_eval = dlarray(norm_state, 'CB');
            q_vals = extractdata(predict(agent.MainNet, input_eval));
            episode_q_values = [episode_q_values, max(q_vals)];
        end

        % 4. 执行动作
        [next_state_idx, raw_reward, is_done] = env.step(curr_state_idx, action_idx);
        next_coord = env.idx2coord(next_state_idx);

        % 5. 存储经验
        agent.store_transition(curr_coord, action_idx, raw_reward, next_coord, is_done);

        % 6. 学习（当缓冲区足够时）
        if agent.MemCount >= agent.batch_size
            loss = agent.learn();
            agent.LossHistory(end+1) = extractdata(loss); %#ok<SAGROW>
        end

        % 7. 更新目标网络
        agent.update_target_network();

        % 记录奖励和状态
        total_reward = total_reward + raw_reward;
        curr_state_idx = next_state_idx;
        curr_coord = next_coord;

        if is_done
            reached_target = true;
            break;
        end
    end

    % Epsilon 衰减
    if agent.epsilon > agent.epsilon_min
        agent.epsilon = agent.epsilon * agent.epsilon_decay;
    end

    % 记录本回合数据
    agent.RewardHistory(ep) = total_reward;
    success_history(ep) = reached_target;

    % 记录Q值统计（如果本回合有利用动作）
    if ~isempty(episode_q_values)
        q_value_history{ep} = struct(...
            'mean', mean(episode_q_values), ...
            'max', max(episode_q_values), ...
            'min', min(episode_q_values), ...
            'std', std(episode_q_values));
    end

    % 计算移动平均奖励（窗口=20）
    if ep >= 20
        avg_reward_history(ep) = mean(agent.RewardHistory(max(1, ep-19):ep));
    else
        avg_reward_history(ep) = mean(agent.RewardHistory(1:ep));
    end

    % 实时显示训练进度（每50回合）
    if mod(ep, 50) == 0 || ep == 1 || ep == episodes
        avg_loss = 0;
        if ~isempty(agent.LossHistory)
            recent_losses = agent.LossHistory(max(1, end-100):end);
            avg_loss = mean(recent_losses);
        end

        % 计算成功率（最近50回合）
        recent_success_rate = 0;
        if ep >= 50
            recent_success_rate = mean(success_history(max(1, ep-49):ep)) * 100;
        end

        fprintf('回合 %4d | 奖励: %6.2f (平均: %6.2f) | 步数: %3d | 损失: %.4f | ε: %.3f | 成功率: %.1f%%\n', ...
            ep, total_reward, avg_reward_history(ep), steps, avg_loss, agent.epsilon, recent_success_rate);

        % 实时绘图
        subplot(2,2,1);
        plot(1:ep, agent.RewardHistory(1:ep), 'b-', 'LineWidth', 1.5);
        hold on;
        plot(1:ep, avg_reward_history(1:ep), 'r--', 'LineWidth', 1.5);
        hold off;
        title('奖励历史 (蓝色: 回合奖励, 红色: 移动平均)', 'FontSize', 10);
        xlabel('回合'); ylabel('奖励'); grid on;
        legend('回合奖励', '移动平均(20)', 'Location', 'best');

        subplot(2,2,2);
        if ~isempty(agent.LossHistory)
            plot(agent.LossHistory, 'r-', 'LineWidth', 1.5);
            title('训练损失历史', 'FontSize', 10);
            xlabel('更新步数'); ylabel('损失'); grid on;
        end

        subplot(2,2,3);
        success_rates = zeros(1, ep);
        for i = 1:ep
            if i >= 20
                success_rates(i) = mean(success_history(max(1, i-19):i)) * 100;
            else
                success_rates(i) = mean(success_history(1:i)) * 100;
            end
        end
        plot(1:ep, success_rates, 'g-', 'LineWidth', 1.5);
        title('成功率 (移动平均, 窗口=20)', 'FontSize', 10);
        xlabel('回合'); ylabel('成功率 (%)'); grid on;
        ylim([0, 100]);

        subplot(2,2,4);
        % 绘制Q值趋势（如果有数据）
        valid_eps = find(~cellfun(@isempty, q_value_history(1:ep)));
        if ~isempty(valid_eps)
            q_means = arrayfun(@(i) q_value_history{i}.mean, valid_eps);
            plot(valid_eps, q_means, 'm-', 'LineWidth', 1.5);
            title('最大Q值平均值 (仅利用时)', 'FontSize', 10);
            xlabel('回合'); ylabel('平均最大Q值'); grid on;
        end

        drawnow;
    end
end

train_time = toc(train_start_time);
fprintf('   训练完成！耗时: %.2f 秒\n', train_time);
fprintf('   总步数: %d\n', agent.StepCounter);
fprintf('   平均每回合步数: %.1f\n', agent.StepCounter / episodes);
fprintf('   最终探索率: %.3f\n', agent.epsilon);
fprintf('   总成功率: %.1f%%\n', mean(success_history) * 100);

% 保存训练记录
agent.SuccessHistory = success_history;
agent.QValueHistory = q_value_history;
agent.AvgRewardHistory = avg_reward_history;

%% 4. 深入分析训练结果
fprintf('\n4. 训练结果深入分析...\n');

% 4.1 分析奖励提升情况
final_avg_reward = avg_reward_history(end);
initial_avg_reward = avg_reward_history(max(1, min(20, episodes)));
reward_improvement = final_avg_reward - initial_avg_reward;

fprintf('   奖励提升分析:\n');
fprintf('     初始平均奖励 (前20回合): %.2f\n', initial_avg_reward);
fprintf('     最终平均奖励 (最后20回合): %.2f\n', final_avg_reward);
fprintf('     奖励提升: %.2f (%.1f%%)\n', reward_improvement, ...
    reward_improvement/max(abs(initial_avg_reward), 1e-6)*100);

% 4.2 分析Q值变化
if ~all(cellfun(@isempty, q_value_history))
    early_q = [];
    late_q = [];
    for ep = 1:min(50, episodes)
        if ~isempty(q_value_history{ep})
            early_q = [early_q, q_value_history{ep}.mean];
        end
    end
    for ep = max(1, episodes-49):episodes
        if ~isempty(q_value_history{ep})
            late_q = [late_q, q_value_history{ep}.mean];
        end
    end

    if ~isempty(early_q) && ~isempty(late_q)
        fprintf('   Q值变化分析:\n');
        fprintf('     早期平均最大Q值: %.3f ± %.3f\n', mean(early_q), std(early_q));
        fprintf('     后期平均最大Q值: %.3f ± %.3f\n', mean(late_q), std(late_q));
        fprintf('     Q值变化: %.3f\n', mean(late_q) - mean(early_q));
    end
end

% 4.3 分析策略收敛性
fprintf('   策略收敛性分析:\n');
policy_matrix = agent.get_policy_matrix();
policy_confidence = mean(max(policy_matrix, [], 2));
fprintf('     策略确定性: %.3f (平均最大动作概率)\n', policy_confidence);

% 检查是否有明确策略（概率>0.9）
clear_policy_states = sum(max(policy_matrix, [], 2) > 0.9);
fprintf('     明确策略状态数: %d/%d (%.1f%%)\n', ...
    clear_policy_states, env.State_Space_Size, ...
    clear_policy_states/env.State_Space_Size*100);

%% 5. 结果可视化
fprintf('\n5. 结果可视化...\n');

% 5.1 获取学习到的策略和价值函数
policy_matrix = agent.get_policy_matrix();
value_vector = agent.get_value_vector();

% 5.2 绘制策略图
figure('Name', '学习到的策略 (改进版)', 'Position', [100, 100, 800, 600]);
env.plot_policy_matrix(policy_matrix);

% 5.3 绘制价值函数图
figure('Name', '状态价值函数 (改进版)', 'Position', [100, 100, 1200, 400]);
subplot(1,2,1);
env.plot_values(value_vector);
title('状态价值 (改进训练)');

subplot(1,2,2);
env.plot_3d_bar_chart(value_vector);
title('3D状态价值 (改进训练)');

% 5.4 绘制综合训练分析图
figure('Name', '综合训练分析', 'Position', [100, 100, 1400, 800]);

% 子图1: 奖励和成功率
subplot(2,3,1);
yyaxis left;
plot(agent.RewardHistory, 'b-', 'LineWidth', 1.5);
ylabel('回合奖励', 'Color', 'b');
yyaxis right;
success_rates = zeros(1, episodes);
for i = 1:episodes
    if i >= 20
        success_rates(i) = mean(success_history(max(1, i-19):i)) * 100;
    else
        success_rates(i) = mean(success_history(1:i)) * 100;
    end
end
plot(1:episodes, success_rates, 'g-', 'LineWidth', 1.5);
ylabel('成功率 (%)', 'Color', 'g');
xlabel('回合');
title('奖励和成功率趋势');
grid on;
legend('奖励', '成功率', 'Location', 'best');

% 子图2: 探索率衰减
subplot(2,3,2);
epsilon_values = 1.0 * options.epsilon_decay.^(0:episodes-1);
epsilon_values = epsilon_values .* (epsilon_values >= agent.epsilon_min) + ...
                 agent.epsilon_min .* (epsilon_values < agent.epsilon_min);
plot(1:episodes, epsilon_values, 'r-', 'LineWidth', 2);
xlabel('回合');
ylabel('探索率 (ε)');
title('探索率衰减曲线');
grid on;

% 子图3: Q值趋势
subplot(2,3,3);
if ~all(cellfun(@isempty, q_value_history))
    valid_eps = find(~cellfun(@isempty, q_value_history));
    q_means = arrayfun(@(i) q_value_history{i}.mean, valid_eps);
    q_stds = arrayfun(@(i) q_value_history{i}.std, valid_eps);

    errorbar(valid_eps, q_means, q_stds, 'm-', 'LineWidth', 1.5);
    xlabel('回合');
    ylabel('平均最大Q值 ± 标准差');
    title('Q值学习趋势');
    grid on;
else
    text(0.5, 0.5, '无Q值记录数据', 'HorizontalAlignment', 'center');
    title('Q值学习趋势');
end

% 子图4: 损失历史
subplot(2,3,4);
if ~isempty(agent.LossHistory)
    semilogy(agent.LossHistory, 'r-', 'LineWidth', 1.5);
    xlabel('更新步数');
    ylabel('损失 (对数尺度)');
    title('训练损失历史');
    grid on;
else
    text(0.5, 0.5, '无损失历史记录', 'HorizontalAlignment', 'center');
    title('训练损失历史');
end

% 子图5: 奖励分布直方图
subplot(2,3,5);
histogram(agent.RewardHistory, 20, 'FaceColor', 'b', 'EdgeColor', 'k');
xlabel('回合奖励');
ylabel('频次');
title('奖励分布直方图');
grid on;

% 子图6: 步数分布
subplot(2,3,6);
% 需要计算每回合步数（这里简化，实际需要记录）
% 使用成功率作为替代
histogram(success_history * 100, [0, 50, 100], 'FaceColor', 'g', 'EdgeColor', 'k');
xlabel('是否成功到达终点');
ylabel('回合数');
title('任务完成情况');
xticks([25, 75]);
xticklabels({'失败', '成功'});
grid on;

%% 6. 策略测试
fprintf('\n6. 测试学习到的策略...\n');

% 运行10个测试回合（纯利用）
num_test_episodes = 10;
test_rewards = zeros(1, num_test_episodes);
test_steps = zeros(1, num_test_episodes);
test_success = false(1, num_test_episodes);

% 保存原始探索率，并设置为0进行纯利用测试
original_epsilon = agent.epsilon;
agent.epsilon = 0;

fprintf('   运行 %d 个测试回合（纯利用，无探索）:\n', num_test_episodes);

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
            test_success(test_ep) = true;
            break;
        end
    end

    test_rewards(test_ep) = episode_reward;
    test_steps(test_ep) = episode_steps;

    fprintf('     测试回合 %2d: 奖励=%6.2f, 步数=%3d, %s\n', ...
        test_ep, episode_reward, episode_steps, ...
        iif(test_success(test_ep), '成功到达终点', '未到达终点'));
end

% 恢复原始探索率
agent.epsilon = original_epsilon;

% 测试结果统计
fprintf('   测试结果统计:\n');
fprintf('     平均奖励: %.2f ± %.2f\n', mean(test_rewards), std(test_rewards));
fprintf('     平均步数: %.1f ± %.1f\n', mean(test_steps), std(test_steps));
fprintf('     成功率: %.1f%% (%d/%d)\n', ...
    mean(test_success)*100, sum(test_success), num_test_episodes);

%% 7. 保存结果
fprintf('\n7. 保存结果...\n');
save_dir = './results/improved/';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% 保存代理对象和结果
save_path = fullfile(save_dir, sprintf('dqn_nn_improved_%s.mat', datestr(now, 'yyyymmdd_HHMMSS')));
save(save_path, 'agent', 'env', 'options', 'policy_matrix', 'value_vector', ...
    'test_rewards', 'test_steps', 'test_success', 'train_time', ...
    'success_history', 'avg_reward_history');
fprintf('   结果已保存到: %s\n', save_path);

%% 8. 总结
fprintf('\n==========================================\n');
fprintf('改进版训练完成！\n');
fprintf('关键改进:\n');
fprintf('  1. 奖励重塑: 终点+10, 撞墙-1 (原: +3, -10)\n');
fprintf('  2. 探索衰减: ε_decay=0.995 (原: 0.999)\n');
fprintf('  3. 网络简化: 64-32-5 (原: 128-64-32-5)\n');
fprintf('  4. 恢复终止状态break (防止终点后继续移动)\n');
fprintf('  5. 添加详细监控: Q值、成功率、奖励趋势\n');
fprintf('==========================================\n');

%% 辅助函数
function str = iif(condition, true_str, false_str)
    % 内联if函数
    if condition
        str = true_str;
    else
        str = false_str;
    end
end

function agent = create_custom_agent(env, options)
    % create_custom_agent: 创建自定义网络架构的代理
    % 继承DQN_NN_Agent但允许自定义网络层

    % 创建基础代理
    agent = DQN_NN_Agent(env, options);

    % 如果有自定义网络层，重新创建网络
    if isfield(options, 'network_layers')
        fprintf('   创建自定义网络架构...\n');

        % 构建自定义网络层
        layers = {};
        layers{1} = featureInputLayer(2, 'Normalization', 'none', 'Name', 'Input');

        % 添加自定义隐藏层
        for i = 1:length(options.network_layers)
            layer_size = options.network_layers(i);
            layers{end+1} = fullyConnectedLayer(layer_size, 'Name', sprintf('FC%d', i));
            layers{end+1} = reluLayer('Name', sprintf('Relu%d', i));
        end

        % 输出层（5个动作）
        layers{end+1} = fullyConnectedLayer(5, 'Name', 'Q_Output');

        % 创建网络
        lgraph = layerGraph(layers);
        agent.MainNet = dlnetwork(lgraph);
        agent.TargetNet = dlnetwork(lgraph);  % 目标网络初始相同

        fprintf('     网络架构: 输入2');
        for i = 1:length(options.network_layers)
            fprintf(' → %d', options.network_layers(i));
        end
        fprintf(' → 输出5\n');
    end
end