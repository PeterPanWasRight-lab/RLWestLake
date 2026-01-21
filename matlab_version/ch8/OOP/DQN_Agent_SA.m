classdef DQN_Agent_SA < handle
    % DQN_Agent_SA: 基于状态+动作输入的深度Q网络
    % 严格对应算法 8.4: 输入 (s, a), 输出 Q(s,a)
    
    properties
        env             % 环境句柄
        
        % --- 超参数 ---
        gamma = 0.9;
        alpha = 0.001;  % 学习率
        epsilon = 1.0;
        epsilon_min = 0.001;
        epsilon_decay = 0.995;
        batch_size = 64;
        buffer_size = 5000;
        target_update_freq = 200; 
        
        % --- 网络参数 ---
        % layer_sizes 结构示例: [3, 24, 24, 1] 
        % 输入层必须是 3 (x, y, action)
        % 输出层必须是 1 (Scalar Q-Value)
        layer_sizes     
        W_main, B_main          % 主网络
        W_target, B_target      % 目标网络
        
        % --- 经验回放 ---
        memory          % [s_x, s_y, a, r, ns_x, ns_y, done]
        mem_ptr = 1;
        mem_cnt = 0;
        
        % --- 状态 ---
        loss_history = [];
        step_counter = 0;
    end
    
    methods
        %% 1. 构造函数
        function obj = DQN_Agent_SA(env, layer_cfg, lr, gam, buf_size, b_size)
            % 默认配置: 输入3节点 -> ... -> 输出1节点
            if nargin < 2, layer_cfg = [3, 30, 30, 1]; end
            
            obj.env = env;
            obj.layer_sizes = layer_cfg;
            obj.alpha = lr;
            obj.gamma = gam;
            obj.buffer_size = buf_size;
            obj.batch_size = b_size;
            
            % 初始化经验池 (7列数据)
            obj.memory = zeros(obj.buffer_size, 7);
            
            % 初始化参数 (Xavier)
            [obj.W_main, obj.B_main] = obj.init_weights();
            obj.W_target = obj.W_main;
            obj.B_target = obj.B_main;
        end
        
        function [W, B] = init_weights(obj)
            L = length(obj.layer_sizes);
            W = cell(L-1, 1); B = cell(L-1, 1);
            for i = 1:L-1
                % Xavier 初始化
                W{i} = randn(obj.layer_sizes(i+1), obj.layer_sizes(i)) * sqrt(2/obj.layer_sizes(i));
                B{i} = zeros(obj.layer_sizes(i+1), 1);
            end
        end
        
        %% 2. 训练主循环
        function train(obj, episodes)
            fprintf('开始训练 (Input: State+Action, Output: Value)...\n');
            obj.loss_history = [];
            reward_history = [];
            figure(10); clf;
            
            for ep = 1:episodes
                curr_state_idx = obj.env.coord2idx(obj.env.Start_State);
                curr_coord = obj.env.idx2coord(curr_state_idx);
                
                total_reward = 0;
                is_done = false;
                steps = 0;
                
                while ~is_done && steps < 200
                    steps = steps + 1;
                    obj.step_counter = obj.step_counter + 1;
                    
                    % 归一化状态
                    norm_state = obj.normalize_state(curr_coord);
                    
                    % --- A. 选择动作 ---
                    % 这里需要把 (s,1), (s,2)...(s,5) 放入网络，找最大的Q
                    action_idx = obj.choose_action(norm_state);
                    
                    % --- B. 执行动作 ---
                    [next_state_idx, reward, is_done] = obj.env.step(curr_state_idx, action_idx);
                    next_coord = obj.env.idx2coord(next_state_idx);
                    
                    % --- C. 存储经验 ---
                    obj.store_transition(curr_coord, action_idx, reward, next_coord, is_done);
                    
                    % --- D. 学习 ---
                    if obj.mem_cnt > obj.batch_size
                        loss = obj.learn_from_batch();
                        obj.loss_history(end+1) = loss;
                    end
                    
                    % --- E. 更新目标网络 ---
                    if mod(obj.step_counter, obj.target_update_freq) == 0
                        obj.W_target = obj.W_main;
                        obj.B_target = obj.B_main;
                    end
                    
                    total_reward = total_reward + reward;
                    curr_state_idx = next_state_idx;
                    curr_coord = next_coord;
                end
                
                reward_history(end+1) = total_reward;
                if obj.epsilon > obj.epsilon_min, obj.epsilon = obj.epsilon * obj.epsilon_decay; end
                
                if mod(ep, 50) == 0
                    fprintf('Episode %d, Reward: %.2f, Loss: %.4f\n', ep, total_reward, mean(obj.loss_history(max(1, end-100):end)));
                    subplot(2,1,1); plot(reward_history); title('Reward'); grid on;
                    subplot(2,1,2); semilogy(obj.loss_history); title('Loss'); grid on;
                    drawnow;
                end
            end
        end
        
        %% 3. 学习函数 (根据算法8.4调整)
        function loss = learn_from_batch(obj)
            % 1. 采样
            batch_indices = randi(min(obj.mem_cnt, obj.buffer_size), obj.batch_size, 1);
            batch_data = obj.memory(batch_indices, :);
            
            % 提取数据 [Batch x Dim]
            s_batch = batch_data(:, 1:2)';      % [2 x N]
            a_batch = batch_data(:, 3)';        % [1 x N]
            r_batch = batch_data(:, 4)';        % [1 x N]
            ns_batch = batch_data(:, 5:6)';     % [2 x N]
            dones = batch_data(:, 7)';          % [1 x N]
            
            N = obj.batch_size;
            
            % 归一化
            s_norm = obj.normalize_batch(s_batch);
            ns_norm = obj.normalize_batch(ns_batch);
            
            % --- 计算 Target y_T ---
            % 公式: y = r + gamma * max_a' Q(s', a'; w_T)
            % 难点: 我们的网络一次只能算一个(s,a)。
            % 方法: 构造一个大矩阵，包含 [s'; 1], [s'; 2] ... [s'; 5]，一次性算完
            
            num_actions = 5;
            % 扩展 Next States: 将每个 s' 复制 5 份
            % ns_tiled: [2 x 5N]
            ns_tiled = repmat(ns_norm, 1, num_actions); 
            
            % 构造对应的动作: [1,1.. 2,2.. 3,3..] (每个动作重复 N 次，方便 reshpae)
            % 或者 [1,2,3,4,5, 1,2,3,4,5...] 
            % 为了矩阵运算方便，我们需要构造 [Input_Next_State] = [3 x 5N]
            
            % 方法：先构造全 1 向量，全 2 向量...
            actions_probe = [];
            for act = 1:num_actions
                actions_probe = [actions_probe, act * ones(1, N)];
            end
            
            % 拼接输入: [ns_norm, ns_norm... ; 1..1, 2..2...]
            % 注意：repmat(ns_norm, 1, 5) 会把列横向铺开，顺序是 s'_1, s'_2 ... s'_N, s'_1 ...
            input_next = [repmat(ns_norm, 1, num_actions); actions_probe]; % [3 x 5N]
            
            % 通过 Target 网络
            Q_next_all = obj.forward(input_next, obj.W_target, obj.B_target); % [1 x 5N]
            
            % 重塑矩阵为 [5 x N]，每一列代表一个样本的 5 个动作 Q 值
            % 因为 actions_probe 是按块构造的 (N个1, N个2...)，我们需要调整 Q_next_all 的排列
            % 当前 Q_next_all = [Q(All s', a=1), Q(All s', a=2)...]
            Q_next_matrix = reshape(Q_next_all, N, num_actions)'; % 变为 [5 x N]
            
            % 取最大值
            [max_q_next, ~] = max(Q_next_matrix, [], 1); % [1 x N]
            
            % 计算 TD Target
            y_target = r_batch + obj.gamma * max_q_next .* (1 - dones);
            
            % --- 计算当前 Q(s, a) 并反向传播 ---
            % 输入就是 Batch 里的 (s, a)
            input_eval = [s_norm; a_batch]; % [3 x N]
            
            % 前向传播 (带缓存)
            [q_eval, Z_cache, A_cache] = obj.forward_full(input_eval, obj.W_main, obj.B_main);
            
            % 计算梯度
            % Loss = (q_eval - y_target)^2
            % dZ = 2 * (q_eval - y_target) / N
            dZ = 2 * (q_eval - y_target) / N;
            loss = mean((q_eval - y_target).^2);
            
            % 反向传播
            obj.backprop(dZ, Z_cache, A_cache);
        end
        
        %% 4. 反向传播与参数更新
        function backprop(obj, dZ, Z_cache, A_cache)
            L = length(obj.layer_sizes);
            dW = cell(L-1, 1); dB = cell(L-1, 1);
            deriv = @(x) double(x > 0);
            
            % 输出层
            dW{L-1} = dZ * A_cache{L-1}';
            dB{L-1} = sum(dZ, 2);
            
            % 隐藏层
            for i = L-2:-1:1
                dZ = (obj.W_main{i+1}' * dZ) .* deriv(Z_cache{i});
                dW{i} = dZ * A_cache{i}';
                dB{i} = sum(dZ, 2);
            end
            
            % 更新
            for i = 1:L-1
                obj.W_main{i} = obj.W_main{i} - obj.alpha * dW{i};
                obj.B_main{i} = obj.B_main{i} - obj.alpha * dB{i};
            end
        end
        
        %% 5. 动作选择 (需要轮询5个动作)
        function a_idx = choose_action(obj, s_norm)
            if rand < obj.epsilon
                a_idx = randi(5);
            else
                % 构造 5 个输入: [x; y; 1], [x; y; 2] ...
                input_batch = zeros(3, 5);
                for a = 1:5
                    input_batch(:, a) = [s_norm; a];
                end
                
                % 一次性预测
                q_values = obj.forward(input_batch, obj.W_main, obj.B_main);
                [~, a_idx] = max(q_values);
            end
        end
        

        %% 6. 通用辅助函数 (已修复)
        function [out, Zc, Ac] = forward_full(obj, X, W, B)
            % 初始化缓存
            Ac = cell(length(W)+1, 1); 
            Zc = cell(length(W), 1);
            
            Ac{1} = X; % 输入层
            act = @(x) max(0, x); % ReLU
            
            % 隐藏层前向传播
            for i = 1:length(W)-1
                Zc{i} = W{i} * Ac{i} + B{i};
                Ac{i+1} = act(Zc{i});
            end
            
            % 输出层 (线性输出)
            % 【修改点】：这里输入应该是 Ac{end-1}，即上一层的激活值
            Zc{end} = W{end} * Ac{end-1} + B{end}; 
            
            Ac{end} = Zc{end}; 
            out = Ac{end};
        end
        
        function out = forward(obj, X, W, B)
            A = X;
            act = @(x) max(0, x);
            for i = 1:length(W)-1
                Z = W{i} * A + B{i};
                A = act(Z);
            end
            out = W{end} * A + B{end};
        end
        
        function store_transition(obj, s, a, r, ns, d)
            idx = mod(obj.mem_ptr - 1, obj.buffer_size) + 1;
            obj.memory(idx, :) = [s, a, r, ns, double(d)];
            obj.mem_ptr = obj.mem_ptr + 1;
            obj.mem_cnt = min(obj.mem_cnt + 1, obj.buffer_size);
        end
        
        function sn = normalize_state(obj, s), sn = [s(1)/obj.env.X_Length; s(2)/obj.env.Y_Length]; end
        function sn = normalize_batch(obj, s), sn = s; sn(1,:) = s(1,:)/obj.env.X_Length; sn(2,:) = s(2,:)/obj.env.Y_Length; end
        
        % 获取策略矩阵 (用于绘图)
        function policy_matrix = get_policy_matrix(obj)
            policy_matrix = zeros(obj.env.State_Space_Size, 5);
            for s_idx = 1:obj.env.State_Space_Size
                coord = obj.env.idx2coord(s_idx);
                s_norm = obj.normalize_state(coord);
                % 轮询所有动作
                input_batch = zeros(3, 5);
                for a = 1:5, input_batch(:, a) = [s_norm; a]; end
                q_vals = obj.forward(input_batch, obj.W_main, obj.B_main);
                
                exps = exp(q_vals - max(q_vals));
                policy_matrix(s_idx, :) = (exps / sum(exps))';
            end
        end
    end
end