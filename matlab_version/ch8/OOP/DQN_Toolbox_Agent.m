classdef DQN_Toolbox_Agent < handle
    % DQN_Toolbox_Agent: 使用 MATLAB Deep Learning Toolbox 实现
    % 对应算法 8.4: 输入 (State, Action) -> 输出 Q(s,a)
    
    properties
        env                 % 环境句柄
        
        % --- 超参数 ---
        gamma = 0.9;        % 折扣因子
        epsilon = 1.0;      % 探索率
        epsilon_min = 0.05; % 最小探索率
        epsilon_decay = 0.995; % 衰减率
        batch_size = 64;    % 批次大小
        buffer_size = 5000; % 经验回放容量
        target_update_freq = 200; % 目标网络更新频率
        
        % --- 网络对象 (dlnetwork) ---
        MainNet             % 主网络 (用于计算当前 Q)
        TargetNet           % 目标网络 (用于计算 Target Q)
        
        % --- 优化器参数 ---
        LearnRate = 0.005;  % 学习率
        TrailingAvg = [];   % Adam 优化器的移动平均梯度
        TrailingAvgSq = []; % Adam 优化器的移动平均平方梯度
        
        % --- 经验回放池 ---
        % 存储格式: [s_x, s_y, a, r, ns_x, ns_y, is_done]
        Memory
        MemPtr = 1;
        MemCount = 0;
        
        StepCounter = 0;
        LossHistory = [];
    end
    
    methods
        function obj = DQN_Toolbox_Agent(env)
            obj.env = env;
            obj.Memory = zeros(obj.buffer_size, 7);
            
            % --- 1. 定义网络结构 ---
            % 根据图片描述：输入3个节点，隐藏层(这里用2层每层64个神经元效果更好)，输出1个节点
            layers = [
                featureInputLayer(3, 'Normalization', 'none', 'Name', 'Input')
                fullyConnectedLayer(64, 'Name', 'FC1')
                reluLayer('Name', 'Relu1')
                fullyConnectedLayer(64, 'Name', 'FC2')
                reluLayer('Name', 'Relu2')
                fullyConnectedLayer(1, 'Name', 'Q_Output')
            ];
            
            % --- 2. 初始化 dlnetwork ---
            lgraph = layerGraph(layers);
            obj.MainNet = dlnetwork(lgraph);
            obj.TargetNet = dlnetwork(lgraph); % 目标网络初始结构相同
        end
        
        %% --- 训练主循环 ---
        function train(obj, episodes)
            fprintf('开始使用 Deep Learning Toolbox 训练 (Input: s,a -> Output: Q)...\n');
            obj.LossHistory = [];
            reward_history = [];
            
            figure(1); clf; % 实时绘图窗口
            
            for ep = 1:episodes
                % 重置环境
                curr_state_idx = obj.env.coord2idx(obj.env.Start_State);
                curr_coord = obj.env.idx2coord(curr_state_idx);
                
                total_reward = 0;
                steps = 0;
                
                while steps < 200 % 限制每回合最大步数
                    steps = steps + 1;
                    obj.StepCounter = obj.StepCounter + 1;
                    
                    % 1. 归一化当前状态 (用于输入网络)
                    norm_state = obj.normalize_state(curr_coord);
                    
                    % 2. 选择动作 (Epsilon-Greedy)
                    action_idx = obj.choose_action(norm_state);
                    
                    % 3. 执行动作
                    [next_state_idx, reward, is_done] = obj.env.step(curr_state_idx, action_idx);
                    next_coord = obj.env.idx2coord(next_state_idx);
                    
                    % 4. 存储经验
                    obj.store_transition(curr_coord, action_idx, reward, next_coord, is_done);
                    
                    % 5. 经验回放与学习 (核心部分)
                    if obj.MemCount >= obj.batch_size
                        loss = obj.learn();
                        obj.LossHistory(end+1) = extractdata(loss);
                    end
                    
                    % 6. 更新目标网络 (Soft Update 或 Hard Update)
                    if mod(obj.StepCounter, obj.target_update_freq) == 0
                        obj.TargetNet.Learnables = obj.MainNet.Learnables;
                        % fprintf('目标网络已更新\n');
                    end
                    
                    % 状态转移
                    total_reward = total_reward + reward;
                    curr_state_idx = next_state_idx;
                    curr_coord = next_coord;
                    
                    if is_done, break; end
                end
                
                % Epsilon 衰减
                if obj.epsilon > obj.epsilon_min
                    obj.epsilon = obj.epsilon * obj.epsilon_decay;
                end
                
                reward_history(end+1) = total_reward;
                
                % 简单的实时绘图
                if mod(ep, 10) == 0
                    avg_loss = 0;
                    if ~isempty(obj.LossHistory), avg_loss = mean(obj.LossHistory(max(1, end-100):end)); end
                    fprintf('Episode %d | Steps: %d | Reward: %.2f | Loss: %.4f | Eps: %.2f\n', ...
                        ep, steps, total_reward, avg_loss, obj.epsilon);
                    
                    subplot(2,1,1); plot(reward_history); title('Total Reward'); grid on;
                    subplot(2,1,2); plot(obj.LossHistory); title('Training Loss'); grid on;
                    drawnow;
                end
            end
        end
        
        %% --- 核心学习函数 (使用 dlgradient) ---
        function loss = learn(obj)
            % 1. 从 Buffer 中随机采样
            idx = randperm(obj.MemCount, obj.batch_size);
            batch_data = obj.Memory(idx, :);
            
            % 提取数据并转为 dlarray (Deep Learning Array)
            % 输入格式需要是 [特征维度 x 批次大小]
            
            % S: [2 x N]
            s_batch = batch_data(:, 1:2)'; 
            % A: [1 x N]
            a_batch = batch_data(:, 3)';
            % R: [1 x N]
            r_batch = batch_data(:, 4)';
            % Next S: [2 x N]
            ns_batch = batch_data(:, 5:6)';
            % Done: [1 x N]
            dones = batch_data(:, 7)';
            
            % 归一化状态
            s_norm = obj.normalize_batch(s_batch);   % [2 x N]
            a_norm = a_batch / 5.0;                  % 简单归一化动作
            ns_norm = obj.normalize_batch(ns_batch); % [2 x N]
            
            % -----------------------------------------------------------
            % 计算 Target 值 (y)
            % 公式: y = r + gamma * max_a' Q(s', a'; TargetNet)
            % 难点: 网络输入是 (s, a)，我们需要对每个 s' 计算 5 个动作的 Q 值
            % -----------------------------------------------------------
            
            num_actions = 5;
            batch_N = obj.batch_size;
            
            % 技巧：将 next_states 复制 5 次，分别对应动作 1,2,3,4,5
            % ns_tiled: [2 x 5N] -> [s'1, s'2... | s'1, s'2...]
            ns_tiled = repmat(ns_norm, 1, num_actions);
            
            % actions_probe: [1,1... | 2,2... | ... | 5,5...] 归一化后
            actions_probe = [];
            for a = 1:num_actions
                actions_probe = [actions_probe, (a/5.0) * ones(1, batch_N)];
            end
            
            % 拼接输入给目标网络: [3 x 5N]
            input_target = dlarray([ns_tiled; actions_probe], 'CB'); 
            
            % 预测 Target Q (不需要梯度，所以不用 dlfeval)
            q_next_all = predict(obj.TargetNet, input_target); % [1 x 5N]
            
            % 重塑为 [5 x N] 矩阵，每列是一个样本的 5 个 Q 值
            % 注意 MATLAB 的 reshape 是列优先，我们需要转置一下逻辑
            % q_next_all 的排列是 [Action1_Batch, Action2_Batch...]
            q_next_matrix = reshape(q_next_all, batch_N, num_actions)'; 
            
            % 取最大值 [1 x N]
            [max_q_next, ~] = max(q_next_matrix, [], 1);
            
            % 计算 TD Target (如果是 Done，则没有下一状态的价值)
            y_target = r_batch + obj.gamma * extractdata(max_q_next) .* (1 - dones);
            y_target = dlarray(y_target, 'CB'); % 转回 dlarray
            
            % -----------------------------------------------------------
            % 计算当前梯度并更新 (需要梯度，使用 dlfeval)
            % -----------------------------------------------------------
            
            % 当前网络的输入: (s, a)
            input_train = dlarray([s_norm; a_norm], 'CB');
            
            % 使用 dlfeval 调用内部函数 modelGradients
            [loss, gradients] = dlfeval(@obj.modelGradients, obj.MainNet, input_train, y_target);
            
            % 使用 Adam 更新参数
            [obj.MainNet, obj.TrailingAvg, obj.TrailingAvgSq] = ...
                adamupdate(obj.MainNet, gradients, ...
                obj.TrailingAvg, obj.TrailingAvgSq, 1, obj.LearnRate);
        end
        
        %% --- 辅助函数：计算梯度 ---
        function [loss, gradients] = modelGradients(obj, net, X, Y_Target)
            % 前向传播
            Q_Predicted = forward(net, X);
            
            % 均方误差 Loss = Mean((y - Q)^2)
            loss = mse(Q_Predicted, Y_Target);
            
            % 自动求导
            gradients = dlgradient(loss, net.Learnables);
        end
        
        %% --- 动作选择 ---
        function action_idx = choose_action(obj, s_norm)
            % 1. 随机探索
            if rand < obj.epsilon
                action_idx = randi(5);
            else
                % 2. 贪婪策略：需要计算当前状态 s 对应所有动作的 Q
                % s_norm: [2 x 1]
                s_repeat = repmat(s_norm, 1, 5); % [2 x 5]
                a_probe = (1:5) / 5.0;           % [1 x 5] 归一化动作
                
                input_eval = dlarray([s_repeat; a_probe], 'CB');
                
                % 预测
                q_values = predict(obj.MainNet, input_eval);
                [~, action_idx] = max(extractdata(q_values));
            end
        end
        
        %% --- 辅助工具 ---
        function store_transition(obj, s, a, r, ns, d)
            idx = mod(obj.MemPtr - 1, obj.buffer_size) + 1;
            obj.Memory(idx, :) = [s, a, r, ns, double(d)];
            obj.MemPtr = obj.MemPtr + 1;
            obj.MemCount = min(obj.MemCount + 1, obj.buffer_size);
        end
        
        % 归一化: 将坐标除以地图边长，映射到 [0, 1]
        function sn = normalize_state(obj, s)
            sn = [s(1)/obj.env.X_Length; s(2)/obj.env.Y_Length]; % [2 x 1]
        end
        
        function sn = normalize_batch(obj, s_batch)
            sn = s_batch;
            sn(1,:) = s_batch(1,:) / obj.env.X_Length;
            sn(2,:) = s_batch(2,:) / obj.env.Y_Length;
        end
        
        % 获取策略矩阵 (用于 GridWorld 绘图)
        function policy_matrix = get_policy_matrix(obj)
            policy_matrix = zeros(obj.env.State_Space_Size, 5);
            for s_idx = 1:obj.env.State_Space_Size
                coord = obj.env.idx2coord(s_idx);
                s_norm = obj.normalize_state(coord); % [2 x 1]
                
                % 批量预测该状态下的5个动作
                s_repeat = repmat(s_norm, 1, 5);
                a_probe = (1:5) / 5.0;
                input_eval = dlarray([s_repeat; a_probe], 'CB');
                
                q_vals = extractdata(predict(obj.MainNet, input_eval)); % [1 x 5]
                
                % Softmax 归一化用于可视化概率箭头
                exps = exp(q_vals - max(q_vals));
                policy_matrix(s_idx, :) = exps / sum(exps);
            end
        end
        
        % 获取价值向量 (用于 3D 绘图)
        function v_vector = get_value_vector(obj)
            v_vector = zeros(obj.env.State_Space_Size, 1);
            for s_idx = 1:obj.env.State_Space_Size
                coord = obj.env.idx2coord(s_idx);
                s_norm = obj.normalize_state(coord);
                
                s_repeat = repmat(s_norm, 1, 5);
                a_probe = (1:5) / 5.0;
                input_eval = dlarray([s_repeat; a_probe], 'CB');
                
                q_vals = extractdata(predict(obj.MainNet, input_eval));
                v_vector(s_idx) = max(q_vals); % 状态价值 = max Q(s,a)
            end
        end
    end
end