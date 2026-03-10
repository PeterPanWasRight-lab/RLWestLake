classdef DQN_NN_Agent < handle
    % DQN_NN_Agent: 基于深度学习的DQN代理（高级特性版）
    % 整合双DQN、Huber损失、保守Q学习等高级特性
    % 网络架构：输入状态(2维)，输出5个动作的Q值(5维)
    % 参考：DQN_Toolbox_Agent.m 和用户提供的伪代码

    properties
        env                    % GridWorld环境句柄

        % --- 网络相关 ---
        MainNet               % 主网络 (dlnetwork)
        TargetNet             % 目标网络
        UseDoubleDQN = false  % 是否使用双DQN
        UseConservativeQLearning = false  % 是否使用保守Q学习
        UseHuberLoss = false  % 是否使用Huber损失

        % --- 超参数 ---
        gamma = 0.95          % 折扣因子
        learning_rate = 0.001 % 学习率
        epsilon = 1.0         % 当前探索率
        epsilon_min = 0.01    % 最小探索率
        epsilon_decay = 0.995 % 衰减速度
        batch_size = 64       % 批次大小
        buffer_size = 10000   % 经验回放缓冲区大小
        target_update_freq = 500 % 目标网络更新频率
        tau = 0.001           % 软更新参数，0表示硬更新
        max_step = 200;       % 一条轨迹最大长度

        % --- 保守Q学习参数 ---
        cql_alpha = 0.1       % 正则化系数

        % --- 经验回放池 ---
        Memory                % [s_x, s_y, a, r, ns_x, ns_y, done]
        MemPtr = 1            % 缓冲区指针
        MemCount = 0          % 当前缓冲区大小

        % --- 训练记录 ---
        StepCounter = 0       % 总步数计数器
        LossHistory = []      % 损失历史
        RewardHistory = []    % 奖励历史

        % --- 优化器状态 (Adam) ---
        TrailingAvg = []      % 一阶矩估计
        TrailingAvgSq = []    % 二阶矩估计
    end

    methods
        function obj = DQN_NN_Agent(env, options)
            % 构造函数
            % 输入:
            %   env     - GridWorld环境对象
            %   options - 可选，包含超参数的结构体

            obj.env = env;

            % 初始化经验回放缓冲区
            obj.Memory = zeros(obj.buffer_size, 7);

            % 创建网络
            obj.create_network();

            % 处理可选参数
            if nargin >= 2 && isstruct(options)
                field_names = fieldnames(options);
                for i = 1:length(field_names)
                    field = field_names{i};
                    if isprop(obj, field)
                        obj.(field) = options.(field);
                    end
                end
            end

            fprintf('DQN_NN_Agent 初始化完成\n');
            fprintf('  使用双DQN: %s\n', string(obj.UseDoubleDQN));
            fprintf('  使用保守Q学习: %s (alpha=%.3f)\n', ...
                string(obj.UseConservativeQLearning), obj.cql_alpha);
            fprintf('  使用Huber损失: %s\n', string(obj.UseHuberLoss));
            fprintf('  软更新参数 tau: %.4f\n', obj.tau);
        end

        %% --- 网络创建 ---
        function create_network(obj)
            % create_network: 创建神经网络 (根据算法8.4: 输入状态-动作对，输出标量Q值)
            % 输入: 归一化状态坐标 [2, batch] 和动作 [1, batch] -> 拼接为 [3, batch]
            % 输出: 标量Q值 [1, batch]

            layers = [
                featureInputLayer(3, 'Normalization', 'none', 'Name', 'Input')  % 输入: x, y, action
                fullyConnectedLayer(100, 'Name', 'FC2')
                reluLayer('Name', 'Relu2')
                fullyConnectedLayer(1, 'Name', 'Q_Output')  % 标量Q值
            ];

            lgraph = layerGraph(layers);
            obj.MainNet = dlnetwork(lgraph);
            obj.TargetNet = dlnetwork(lgraph);  % 目标网络初始相同

            fprintf('网络创建完成 (算法8.4架构): 输入3维 -> 10 -> 10 -> 20 -> 输出1维\n');
        end

        %% --- 训练主循环 ---
        function train(obj, episodes)
            % train: 训练代理
            % 输入: episodes - 训练回合数

            fprintf('开始训练 DQN_NN_Agent (总回合: %d)...\n', episodes);

            % 预分配数组以提高性能
            max_loss_records = episodes * 50;  % 估计每个回合最多记录50次损失
            obj.LossHistory = zeros(1, max_loss_records);
            loss_counter = 0;  % 损失记录计数器

            reward_history = zeros(1, episodes);

            figure(1); clf;

            for ep = 1:episodes
                % 重置环境
                curr_state_idx = obj.env.coord2idx(obj.env.Start_State);
                curr_coord = obj.env.idx2coord(curr_state_idx);

                total_reward = 0;
                steps = 0;

                while steps < obj.max_step  % 最大步数限制
                    steps = steps + 1;
                    obj.StepCounter = obj.StepCounter + 1;

                    % 1. 归一化当前状态
                    norm_state = obj.normalize_state(curr_coord);

                    % 2. 选择动作 (ε-greedy)
                    action_idx = obj.choose_action(norm_state);

                    % 3. 执行动作
                    [next_state_idx, raw_reward, is_done] = obj.env.step(curr_state_idx, action_idx);
                    next_coord = obj.env.idx2coord(next_state_idx);

                    % 4. 存储经验 (使用原始奖励)
                    obj.store_transition(curr_coord, action_idx, raw_reward, next_coord, is_done);

                    % 5. 学习 (当缓冲区足够时)
                    if obj.MemCount >= obj.batch_size
                        loss = obj.learn();
                        loss_counter = loss_counter + 1;
                        if loss_counter <= max_loss_records
                            obj.LossHistory(loss_counter) = extractdata(loss);
                        end
                    end

                    % 6. 更新目标网络
                    obj.update_target_network();

                    % 记录奖励
                    total_reward = total_reward + raw_reward;
                    curr_state_idx = next_state_idx;
                    curr_coord = next_coord;

                    % if is_done
                    %     break;
                    % end
                end

                % Epsilon 衰减
                if obj.epsilon > obj.epsilon_min
                    obj.epsilon = obj.epsilon * obj.epsilon_decay;
                end

                reward_history(ep) = total_reward;

                % 实时显示训练进度
                if mod(ep, 20) == 0 || ep == 1 || ep == episodes
                    avg_loss = 0;
                    if loss_counter > 0
                        % 只计算实际记录的损失
                        valid_losses = obj.LossHistory(1:min(loss_counter, max_loss_records));
                        start_idx = max(1, length(valid_losses)-100);
                        avg_loss = mean(valid_losses(start_idx:end));
                    end
                    fprintf('回合 %4d | 步数: %3d | 总奖励: %6.2f | 平均损失: %.4f | ε: %.3f\n', ...
                        ep, steps, total_reward, avg_loss, obj.epsilon);

                    % 实时绘图
                    subplot(2,1,1);
                    plot(reward_history(1:ep), 'b-', 'LineWidth', 1.5);
                    title('总奖励历史 (原始奖励)');
                    xlabel('回合'); ylabel('总奖励'); grid on;

                    subplot(2,1,2);
                    if loss_counter > 0
                        valid_losses = obj.LossHistory(1:min(loss_counter, max_loss_records));
                        plot(valid_losses, 'r-', 'LineWidth', 1.5);
                    end
                    title('训练损失历史');
                    xlabel('更新步数'); ylabel('损失'); grid on;

                    drawnow;
                end
            end

            obj.RewardHistory = reward_history;

            % 截断损失历史，只保留实际记录的部分
            if loss_counter > 0
                obj.LossHistory = obj.LossHistory(1:min(loss_counter, max_loss_records));
            else
                obj.LossHistory = [];
            end

            fprintf('训练完成！总步数: %d\n', obj.StepCounter);
        end

        %% --- 核心学习函数 (根据算法8.4) ---
        function loss = learn(obj)
            % learn: 从经验回放中学习，更新主网络

            % 随机采样批次
            idx = randperm(obj.MemCount, obj.batch_size);
            batch_data = obj.Memory(idx, :);

            % 提取数据 [特征 x 批次]
            s_batch = batch_data(:, 1:2)';      % [2 x N]
            a_batch = batch_data(:, 3)';        % [1 x N] (动作索引 1~5)
            r_batch = batch_data(:, 4)';        % [1 x N]
            ns_batch = batch_data(:, 5:6)';     % [2 x N]
            dones = batch_data(:, 7)';          % [1 x N]

            % 归一化状态
            s_norm = obj.normalize_batch(s_batch);   % [2 x N]
            ns_norm = obj.normalize_batch(ns_batch); % [2 x N]

            % 归一化动作 (除以5，范围[0,1])
            a_norm = a_batch / 5.0;                  % [1 x N]

            % --- 计算目标Q值 (算法8.4: y = r + γ * max_a' Q(s', a'; w_T)) ---
            num_actions = 5;
            batch_N = obj.batch_size;

            % 构造目标网络的输入: 将每个Next State复制5份，分别对应5个动作
            ns_tiled = repmat(ns_norm, 1, num_actions); % [2 x 5N]

            actions_probe = [];
            for a = 1:num_actions
                actions_probe = [actions_probe, (a/5.0) * ones(1, batch_N)];
            end

            % 输入TargetNet: [3 x 5N]
            input_target = dlarray([ns_tiled; actions_probe], 'CB');

            % 预测所有下一个状态-动作对的Q值
            q_next_all = predict(obj.TargetNet, input_target); % [1 x 5N]

            % 重排为矩阵 [N x 5] (每行对应一个样本的5个动作Q值)
            q_next_mat = reshape(extractdata(q_next_all), batch_N, num_actions); % [N x 5]

            % 使用双DQN或标准DQN选择目标Q值
            if obj.UseDoubleDQN
                % 双DQN: 使用主网络选择动作，目标网络评估
                % 首先用主网络计算下一个状态的Q值
                q_next_main = predict(obj.MainNet, input_target); % [1 x 5N]
                q_next_main_mat = reshape(extractdata(q_next_main), batch_N, num_actions); % [N x 5]
                [~, best_actions] = max(q_next_main_mat, [], 2); % [N x 1]

                % 从目标网络的Q值矩阵中选择对应动作的Q值
                batch_indices = (1:batch_N)';
                q_next_selected = q_next_mat(sub2ind([batch_N, num_actions], batch_indices, best_actions)); % [N x 1]
                q_next_selected = q_next_selected'; % 转为 [1 x N]
            else
                % 标准DQN: 使用目标网络的最大Q值
                [max_q, ~] = max(q_next_mat, [], 2); % [N x 1]
                q_next_selected = max_q'; % 转为 [1 x N]
            end

            % 计算目标Q值: y = r + γ * maxQ * (1-done)
            r_batch_dl = dlarray(r_batch, 'CB');
            dones_dl = dlarray(dones, 'CB');
            y_target = r_batch_dl + obj.gamma * q_next_selected .* (1 - dones_dl);

            % --- 计算损失和梯度 ---
            % 准备训练输入：状态和动作拼接 [3 x N]
            X = dlarray([s_norm; a_norm], 'CB');

            % 使用dlfeval计算损失和梯度
            [loss, gradients] = dlfeval(@obj.modelLoss, ...
                obj.MainNet, X, y_target, true);

            % 使用Adam优化器更新网络
            [obj.MainNet, obj.TrailingAvg, obj.TrailingAvgSq] = ...
                adamupdate(obj.MainNet, gradients, ...
                obj.TrailingAvg, obj.TrailingAvgSq, 1, obj.learning_rate);
        end

        %% --- 损失函数 (用于dlfeval，根据算法8.4) ---
        function [loss, gradients] = modelLoss(obj, net, X, Y_target, is_training)
            % modelLoss: 计算损失和梯度 (算法8.4架构)
            % 输入:
            %   net        - 网络对象
            %   X          - 输入状态-动作对 [3, batch] (x, y, action)
            %   Y_target   - 目标Q值 [1, batch]
            %   is_training - 是否处于训练模式（影响CQL正则化）

            % 前向传播：获取当前状态-动作对的Q值
            q_pred = forward(net, X);  % [1, batch_size]

            % 计算TD误差
            td_error = q_pred - Y_target;  % [1, batch]

            % 选择损失函数：Huber损失或MSE
            if obj.UseHuberLoss
                % Huber损失 (delta=1.0)
                delta = 1.0;
                abs_error = abs(td_error);
                quadratic = min(abs_error, delta);
                linear = abs_error - quadratic;
                loss = 0.5 * quadratic.^2 + delta * linear;
                loss = mean(loss);  % 取平均
            else
                % 均方误差
                loss = 0.5 * mean(td_error.^2);
            end

            % 注意：保守Q学习(CQL)正则化在算法8.4架构中不直接适用，
            % 因为网络输出单个Q值而非所有动作的Q值。
            % 如果启用，可以给出警告或跳过。
            if obj.UseConservativeQLearning && is_training
                warning('保守Q学习正则化在算法8.4架构中不可用，已跳过。');
                % 可以选择使用其他正则化方法，但暂时跳过
            end

            % 计算梯度
            gradients = dlgradient(loss, net.Learnables);
        end

        %% --- 动作选择 (ε-greedy) ---
        function action_idx = choose_action(obj, s_norm)
            % choose_action: ε-greedy动作选择 (根据算法8.4)
            % 输入: s_norm - 归一化状态 [2x1]
            % 输出: action_idx - 动作索引 (1~5)

            if rand < obj.epsilon
                % 探索：随机选择动作
                action_idx = randi(5);
            else
                % 利用：选择最大Q值的动作
                % 需要遍历所有动作，计算每个动作的Q值
                num_actions = 5;
                s_repeat = repmat(s_norm, 1, num_actions); % [2 x 5]
                a_probe = (1:num_actions) / num_actions;   % [1 x 5] 归一化到[0,1]
                input_eval = dlarray([s_repeat; a_probe], 'CB'); % [3 x 5]
                q_values = predict(obj.MainNet, input_eval); % [1 x 5]
                [~, action_idx] = max(extractdata(q_values));
            end
        end

        %% --- 目标网络更新 ---
        function update_target_network(obj)
            % update_target_network: 更新目标网络
            % 支持软更新 (tau > 0) 和硬更新 (tau = 0)

            if mod(obj.StepCounter, obj.target_update_freq) == 0
                if obj.tau > 0 && obj.tau < 1
                    % 软更新: θ_target = τ*θ_main + (1-τ)*θ_target
                    target_params = obj.TargetNet.Learnables;
                    main_params = obj.MainNet.Learnables;

                    for i = 1:height(target_params)
                        target_params.Value{i} = ...
                            obj.tau * main_params.Value{i} + ...
                            (1 - obj.tau) * target_params.Value{i};
                    end
                    obj.TargetNet.Learnables = target_params;
                else
                    % 硬更新: 直接复制参数
                    obj.TargetNet.Learnables = obj.MainNet.Learnables;
                end
            end
        end

        %% --- 经验回放管理 ---
        function store_transition(obj, s, a, r, ns, d)
            % store_transition: 存储经验到回放缓冲区
            % 输入:
            %   s  - 当前状态坐标 [x, y]
            %   a  - 动作索引 (1~5)
            %   r  - 奖励
            %   ns - 下一个状态坐标 [x, y]
            %   d  - 是否终止 (逻辑值)

            idx = mod(obj.MemPtr - 1, obj.buffer_size) + 1;
            obj.Memory(idx, :) = [s, a, r, ns, double(d)];
            obj.MemPtr = obj.MemPtr + 1;
            obj.MemCount = min(obj.MemCount + 1, obj.buffer_size);
        end

        %% --- 辅助函数: 状态归一化 ---
        function sn = normalize_state(obj, s)
            % normalize_state: 归一化状态坐标
            % 输入: s - 状态坐标 [x, y]
            % 输出: sn - 归一化坐标 [x/X_Length; y/Y_Length]

            sn = [s(1)/obj.env.X_Length; s(2)/obj.env.Y_Length];
        end

        function sn = normalize_batch(obj, s_batch)
            % normalize_batch: 归一化批次状态
            % 输入: s_batch - 状态批次 [2 x N]
            % 输出: sn - 归一化批次 [2 x N]

            sn = s_batch; % [2 x N]
            sn(1,:) = s_batch(1,:) / obj.env.X_Length;
            sn(2,:) = s_batch(2,:) / obj.env.Y_Length;
        end

        %% --- 结果获取函数 (兼容现有可视化) ---
        function policy_matrix = get_policy_matrix(obj)
            % get_policy_matrix: 获取策略矩阵 (根据算法8.4架构)
            % 输出: policy_matrix - [状态数 x 5] 矩阵，每行是动作概率分布

            state_space_size = obj.env.State_Space_Size;
            policy_matrix = zeros(state_space_size, 5);
            num_actions = 5;

            for s_idx = 1:state_space_size
                coord = obj.env.idx2coord(s_idx);
                s_norm = obj.normalize_state(coord);

                % 遍历所有动作，计算每个动作的Q值
                s_repeat = repmat(s_norm, 1, num_actions); % [2 x 5]
                a_probe = (1:num_actions) / num_actions;   % [1 x 5] 归一化到[0,1]
                input_eval = dlarray([s_repeat; a_probe], 'CB'); % [3 x 5]
                q_vals = extractdata(predict(obj.MainNet, input_eval)); % [1 x 5]

                % 使用softmax将Q值转换为概率分布
                q_vals = q_vals(:)';  % 转换为行向量 [1 x 5]
                exps = exp(q_vals - max(q_vals));  % 数值稳定
                policy_matrix(s_idx, :) = exps / sum(exps);
            end
        end

        function v_vector = get_value_vector(obj)
            % get_value_vector: 获取状态价值向量 (根据算法8.4架构)
            % 输出: v_vector - [状态数 x 1] 向量，每个状态的最大Q值

            state_space_size = obj.env.State_Space_Size;
            v_vector = zeros(state_space_size, 1);
            num_actions = 5;

            for s_idx = 1:state_space_size
                coord = obj.env.idx2coord(s_idx);
                s_norm = obj.normalize_state(coord);

                % 遍历所有动作，计算每个动作的Q值，取最大值
                s_repeat = repmat(s_norm, 1, num_actions); % [2 x 5]
                a_probe = (1:num_actions) / num_actions;   % [1 x 5] 归一化到[0,1]
                input_eval = dlarray([s_repeat; a_probe], 'CB'); % [3 x 5]
                q_vals = extractdata(predict(obj.MainNet, input_eval)); % [1 x 5]

                v_vector(s_idx) = max(q_vals);
            end
        end

        %% --- 信息显示函数 ---
        function show_info(obj)
            % show_info: 显示代理信息

            fprintf('=== DQN_NN_Agent 信息 ===\n');
            fprintf('环境大小: %d x %d (状态数: %d)\n', ...
                obj.env.X_Length, obj.env.Y_Length, obj.env.State_Space_Size);
            fprintf('经验回放: %d/%d (%.1f%%)\n', ...
                obj.MemCount, obj.buffer_size, 100*obj.MemCount/obj.buffer_size);
            fprintf('当前探索率 ε: %.3f\n', obj.epsilon);
            fprintf('总步数: %d\n', obj.StepCounter);
            fprintf('损失历史长度: %d\n', length(obj.LossHistory));
            if ~isempty(obj.LossHistory)
                fprintf('最近平均损失: %.4f\n', mean(obj.LossHistory(max(1, end-100):end)));
            end
        end
    end
end