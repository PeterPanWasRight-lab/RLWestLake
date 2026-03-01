classdef DQN_Toolbox_Agent < handle
    % DQN_Toolbox_Agent: 深度学习工具箱版 (修复版)
    % 核心改进：修复维度错误 + 增加奖励重塑机制
    
    properties
        env                 % 环境句柄
        
        % --- 超参数 (经过优化) ---
        gamma = 0.95;       % 折扣因子 (提高一点，让它看重远处的终点)
        epsilon = 1.0;      % 当前探索率
        epsilon_min = 0.01; % 最小探索率
        epsilon_decay = 0.995; % 衰减速度
        batch_size = 64;    % 批次大小
        buffer_size = 10000; % 增大经验池
        target_update_freq = 500; % 目标网络更新频率
        
        % --- 网络对象 ---
        MainNet             % 主网络
        TargetNet           % 目标网络
        
        % --- 优化器参数 ---
        LearnRate = 0.001;  % 学习率 (调小一点，防止震荡)
        TrailingAvg = [];   % Adam 参数
        TrailingAvgSq = []; % Adam 参数
        
        % --- 经验回放池 ---
        Memory              % [s_x, s_y, a, r, ns_x, ns_y, done]
        MemPtr = 1;
        MemCount = 0;
        
        StepCounter = 0;
        LossHistory = [];
    end
    
    methods
        function obj = DQN_Toolbox_Agent(env)
            obj.env = env;
            obj.Memory = zeros(obj.buffer_size, 7);
            
            % --- 1. 定义网络结构 (参考图片: 3输入 -> 100隐藏 -> 1输出) ---
            layers = [
                featureInputLayer(3, 'Normalization', 'none', 'Name', 'Input') % 输入: x, y, action
                fullyConnectedLayer(100, 'Name', 'FC1')                        % 隐藏层 100 节点
                reluLayer('Name', 'Relu1')
                fullyConnectedLayer(100, 'Name', 'FC2')                        % 隐藏层 100 节点
                reluLayer('Name', 'Relu2')
                fullyConnectedLayer(1, 'Name', 'Q_Output')                     % 输出: Q值
            ];
            
            % --- 2. 初始化网络 ---
            lgraph = layerGraph(layers);
            obj.MainNet = dlnetwork(lgraph);
            obj.TargetNet = dlnetwork(lgraph); % 目标网络初始相同
        end
        
        %% --- 训练主循环 ---
        function train(obj, episodes)
            fprintf('开始训练 (修复版: 增加奖励重塑)... \n');
            obj.LossHistory = [];
            reward_history = [];
            
            figure(1); clf;
            
            for ep = 1:episodes
                % 重置环境
                curr_state_idx = obj.env.coord2idx(obj.env.Start_State);
                curr_coord = obj.env.idx2coord(curr_state_idx);
                
                total_reward = 0;
                steps = 0;
                
                while steps < 200
                    steps = steps + 1;
                    obj.StepCounter = obj.StepCounter + 1;
                    
                    % 1. 归一化状态
                    norm_state = obj.normalize_state(curr_coord);
                    
                    % 2. 选择动作
                    action_idx = obj.choose_action(norm_state);
                    
                    % 3. 执行动作
                    [next_state_idx, raw_reward, is_done] = obj.env.step(curr_state_idx, action_idx);
                    next_coord = obj.env.idx2coord(next_state_idx);
                    
                    % --- 4. 关键：奖励重塑 (Reward Shaping) ---
                    % 原始奖励: 终点+1, 撞墙-10, 走路0
                    % 问题: 撞墙惩罚太大，终点奖励太小，网络学偏了。
                    % 修改为: 终点+10, 撞墙-1, 走路-0.05 (鼓励走直线)
                    shaped_reward = raw_reward;
                    if raw_reward > 0       % 到达终点
                        shaped_reward = 10.0;
                    elseif raw_reward < 0   % 撞墙/障碍
                        shaped_reward = -10.0;
                    else                    % 普通移动
                        shaped_reward = 0; 
                    end
                    
                    % 5. 存储经验 (存修改后的奖励)
                    obj.store_transition(curr_coord, action_idx, shaped_reward, next_coord, is_done);
                    
                    % 6. 学习
                    if obj.MemCount >= obj.batch_size
                        loss = obj.learn();
                        obj.LossHistory(end+1) = extractdata(loss);
                    end
                    
                    % 7. 更新目标网络
                    if mod(obj.StepCounter, obj.target_update_freq) == 0
                        obj.TargetNet.Learnables = obj.MainNet.Learnables;
                    end
                    
                    total_reward = total_reward + raw_reward; % 记录显示用原始奖励
                    curr_state_idx = next_state_idx;
                    curr_coord = next_coord;
                    
                    if is_done, break; end
                end
                
                % Epsilon 衰减
                if obj.epsilon > obj.epsilon_min
                    obj.epsilon = obj.epsilon * obj.epsilon_decay;
                end
                
                reward_history(end+1) = total_reward;
                
                % 实时绘图
                if mod(ep, 20) == 0
                    avg_loss = 0;
                    if ~isempty(obj.LossHistory), avg_loss = mean(obj.LossHistory(max(1, end-100):end)); end
                    fprintf('Episode %d | Steps: %d | Reward: %.2f | Loss: %.4f | Eps: %.2f\n', ...
                        ep, steps, total_reward, avg_loss, obj.epsilon);
                    
                    subplot(2,1,1); plot(reward_history); title('Total Reward (Original)'); grid on;
                    subplot(2,1,2); plot(obj.LossHistory); title('Training Loss'); grid on;
                    drawnow;
                end
            end
        end
        
        %% --- 核心学习函数 ---
        function loss = learn(obj)
            idx = randperm(obj.MemCount, obj.batch_size);
            batch_data = obj.Memory(idx, :);
            
            % 提取数据 [Dim x Batch]
            s_batch = batch_data(:, 1:2)'; 
            a_batch = batch_data(:, 3)';
            r_batch = batch_data(:, 4)';
            ns_batch = batch_data(:, 5:6)';
            dones = batch_data(:, 7)';
            
            % 归一化
            s_norm = obj.normalize_batch(s_batch);   % [2 x N]
            a_norm = a_batch / 5.0;                  % [1 x N]
            ns_norm = obj.normalize_batch(ns_batch); % [2 x N]
            
            % --- 计算 Target ---
            num_actions = 5;
            batch_N = obj.batch_size;
            
            % 构造 Target 网络的输入: 将每个 Next State 复制 5 份，分别对应 5 个动作
            ns_tiled = repmat(ns_norm, 1, num_actions); % [2 x 5N]
            
            actions_probe = [];
            for a = 1:num_actions
                actions_probe = [actions_probe, (a/5.0) * ones(1, batch_N)];
            end
            
            % 输入 TargetNet: [3 x 5N]
            input_target = dlarray([ns_tiled; actions_probe], 'CB'); 
            
            % 预测
            q_next_all = predict(obj.TargetNet, input_target); 
            
            % 重排并取最大值
            % Q_Next_Mat: [N x 5] (转置后)
            q_next_mat = reshape(q_next_all, batch_N, num_actions); 
            [max_q, ~] = max(q_next_mat, [], 2); % 对每行取最大, 结果 [N x 1]
            max_q = max_q'; % 转为 [1 x N]
            
            % y = r + gamma * maxQ * (1-done)
            y_target = r_batch + obj.gamma * extractdata(max_q) .* (1 - dones);
            y_target = dlarray(y_target, 'CB');
            
            % --- 计算梯度并更新 ---
            input_train = dlarray([s_norm; a_norm], 'CB'); % [3 x N]
            
            [loss, gradients] = dlfeval(@obj.modelGradients, obj.MainNet, input_train, y_target);
            
            [obj.MainNet, obj.TrailingAvg, obj.TrailingAvgSq] = ...
                adamupdate(obj.MainNet, gradients, ...
                obj.TrailingAvg, obj.TrailingAvgSq, 1, obj.LearnRate);
        end
        
        function [loss, gradients] = modelGradients(obj, net, X, Y_Target)
            Q_Pred = forward(net, X);
            loss = mse(Q_Pred, Y_Target);
            gradients = dlgradient(loss, net.Learnables);
        end
        
        %% --- 动作选择 (修复维度拼接问题) ---
        function action_idx = choose_action(obj, s_norm)
            % s_norm 必须是 [2x1] 列向量
            if rand < obj.epsilon
                action_idx = randi(5);
            else
                % 构造 5 个候选输入
                s_repeat = repmat(s_norm, 1, 5); % [2 x 5]
                a_probe = (1:5) / 5.0;           % [1 x 5]
                
                % 完美拼接: [2x5; 1x5] -> [3x5]
                input_eval = dlarray([s_repeat; a_probe], 'CB');
                
                q_values = predict(obj.MainNet, input_eval);
                [~, action_idx] = max(extractdata(q_values));
            end
        end
        
        %% --- 辅助函数 ---
        % 修正: 必须返回列向量 [2x1]
        function sn = normalize_state(obj, s)
            sn = [s(1)/obj.env.X_Length; s(2)/obj.env.Y_Length]; 
        end
        
        function sn = normalize_batch(obj, s_batch)
            sn = s_batch; % [2 x N]
            sn(1,:) = s_batch(1,:) / obj.env.X_Length;
            sn(2,:) = s_batch(2,:) / obj.env.Y_Length;
        end
        
        function store_transition(obj, s, a, r, ns, d)
            idx = mod(obj.MemPtr - 1, obj.buffer_size) + 1;
            obj.Memory(idx, :) = [s, a, r, ns, double(d)];
            obj.MemPtr = obj.MemPtr + 1;
            obj.MemCount = min(obj.MemCount + 1, obj.buffer_size);
        end
        
        % 绘图辅助: 获取策略矩阵
        function policy_matrix = get_policy_matrix(obj)
            policy_matrix = zeros(obj.env.State_Space_Size, 5);
            for s_idx = 1:obj.env.State_Space_Size
                coord = obj.env.idx2coord(s_idx);
                s_norm = obj.normalize_state(coord);
                
                s_repeat = repmat(s_norm, 1, 5);
                a_probe = (1:5) / 5.0;
                input_eval = dlarray([s_repeat; a_probe], 'CB');
                q_vals = extractdata(predict(obj.MainNet, input_eval));
                
                exps = exp(q_vals - max(q_vals));
                policy_matrix(s_idx, :) = exps / sum(exps);
            end
        end
        
        % 绘图辅助: 获取价值向量
        function v_vector = get_value_vector(obj)
            v_vector = zeros(obj.env.State_Space_Size, 1);
            for s_idx = 1:obj.env.State_Space_Size
                coord = obj.env.idx2coord(s_idx);
                s_norm = obj.normalize_state(coord);
                
                s_repeat = repmat(s_norm, 1, 5);
                a_probe = (1:5) / 5.0;
                input_eval = dlarray([s_repeat; a_probe], 'CB');
                q_vals = extractdata(predict(obj.MainNet, input_eval));
                v_vector(s_idx) = max(q_vals);
            end
        end
    end
end