classdef REINFORCEAgent_NN < handle
    % REINFORCEAgent_NN: 神经网络版蒙特卡洛策略梯度算法
    
    properties (SetAccess = public)
        actorNet        % 策略神经网络 (替代原来的 theta 矩阵)
        alpha           % 学习率 α
        gamma           % 折扣因子 γ
        env_ref         % 环境变量引用 (用于获取网格尺寸和坐标)
        returns_history % 记录收敛曲线
        
        % 优化器状态
        velocity = [];  % 用于 SGDM 优化器
    end
    
    methods
        function obj = REINFORCEAgent_NN(env, alpha, gamma)
            obj.env_ref = env;
            obj.alpha = alpha;
            obj.gamma = gamma;
            obj.returns_history = [];
            
            % --- 1. 构建神经网络 ---
            % 输入层：2维连续特征 (归一化后的 x, y 坐标)
            % 输出层：5个动作的 Softmax 概率
            num_actions = length(env.Action_Space);
            
            layers = [
                featureInputLayer(2, 'Normalization', 'none', 'Name', 'state')
                fullyConnectedLayer(4, 'Name', 'fc1')
                reluLayer('Name', 'relu1')
                fullyConnectedLayer(4, 'Name', 'fc2')
                reluLayer('Name', 'relu2')
                fullyConnectedLayer(num_actions, 'Name', 'fc_out')
                softmaxLayer('Name', 'softmax')
            ];
            
            % 初始化为 dlnetwork 对象 (MATLAB 的深度学习网络对象)
            obj.actorNet = dlnetwork(layers);
        end
        
        function probs = get_action_probabilities(obj, state_idx)
            % 1. 将状态索引转为归一化的坐标 [x; y]
            norm_coord = obj.get_normalized_state(state_idx);
            
            % 2. 转换为 dlarray 格式 ('CB' 表示 Channel x Batch)
            state_dl = dlarray(norm_coord, 'CB');
            
            % 3. 前向传播预测概率 (使用 predict 而不是 forward, 因为不需要计算梯度)
            probs_dl = predict(obj.actorNet, state_dl);
            
            % 4. 提取为普通数组
            probs = extractdata(probs_dl)'; % 返回 1x5 array
        end
        
        function action_idx = select_action(obj, state_idx)
            probs = obj.get_action_probabilities(state_idx);
            cum_probs = cumsum(probs);
            action_idx = find(cum_probs >= rand(), 1);
        end
        
        function train(obj, num_episodes, max_steps)
            obj.returns_history = zeros(num_episodes, 1);
            
            for k = 1:num_episodes
                % 1. 生成轨迹
                [states, actions, rewards] = obj.generate_episode(max_steps);
                obj.returns_history(k) = sum(rewards);
                
                % 2. 策略更新 (神经网络反向传播)
                if ~isempty(states)
                    obj.update_network(states, actions, rewards);
                end
                
                if mod(k, 100) == 0
                    fprintf('Episode %d/%d, Total Reward: %.2f\n', k, num_episodes, obj.returns_history(k));
                end
            end
        end
        
        function policy_matrix = get_policy_matrix(obj)
            num_states = obj.env_ref.State_Space_Size;
            num_actions = length(obj.env_ref.Action_Space);
            policy_matrix = zeros(num_states, num_actions);
            for s = 1:num_states
                policy_matrix(s, :) = obj.get_action_probabilities(s);
            end
        end
    end
    
    methods (Access = private)
        % 辅助函数：将状态索引转化为归一化的 [x; y] 连续特征向量
        function norm_coord = get_normalized_state(obj, state_idx)
            coord = obj.env_ref.idx2coord(state_idx);
            % 归一化到 (0, 1] 之间，这对神经网络的稳定训练非常关键
            norm_coord = [coord(1) / obj.env_ref.X_Length; ...
                          coord(2) / obj.env_ref.Y_Length];
        end
        
        function [states, actions, rewards] = generate_episode(obj, max_steps)
            states_temp = zeros(max_steps, 1);
            actions_temp = zeros(max_steps, 1);
            rewards_temp = zeros(max_steps, 1);
            
            current_state = obj.env_ref.coord2idx(obj.env_ref.Start_State);
            step_count = 0;
            
            for t = 1:max_steps
                step_count = step_count + 1;
                action = obj.select_action(current_state);
                [next_state, reward, is_done] = obj.env_ref.step(current_state, action);
                
                states_temp(t) = current_state;
                actions_temp(t) = action;
                rewards_temp(t) = reward;
                
                % if is_done, break; end
                current_state = next_state;
            end
            
            states = states_temp(1:step_count);
            actions = actions_temp(1:step_count);
            rewards = rewards_temp(1:step_count);
        end
        
        function update_network(obj, states, actions, rewards)
            T = length(rewards);
            
            % 1. 计算每个时间步的累积回报 G_t (即原本的 q_values)
            q_values = zeros(T, 1);
            current_q = 0;
            for t = T:-1:1
                current_q = rewards(t) + obj.gamma * current_q;
                q_values(t) = current_q;
            end
            
            % [DRL 核心技巧]：对 G_t 进行标准化 (均值0，方差1)
            % 表格法不用标准化也能收敛，但神经网络对方差极大/极小的数值非常敏感。
            % 这等价于引入了一个动态的 Baseline，极大提升训练稳定性。
            if std(q_values) > 1e-5
                q_values = (q_values - mean(q_values)) / std(q_values);
            end
            
            % 2. 准备批次数据 (Batch Data)
            % 将整个轨迹的状态打包成一个矩阵传给网络
            coords_batch = zeros(2, T);
            for t = 1:T
                coords_batch(:, t) = obj.get_normalized_state(states(t));
            end
            states_dl = dlarray(coords_batch, 'CB');
            
            % 3. 使用 dlfeval 计算损失和网络参数梯度
            % 注意：必须调用静态或外部函数来计算自定义 Loss
            [loss, gradients] = dlfeval(@REINFORCEAgent_NN.compute_loss, ...
                                        obj.actorNet, states_dl, actions, q_values);
            
            % 4. 更新网络权重 (使用无动量的 SGD，严格对应你原始的表格法更新公式)
            [obj.actorNet, obj.velocity] = sgdmupdate(obj.actorNet, gradients, obj.velocity, obj.alpha, 0);
        end
    end
    
    methods (Static, Access = private)
        % --- 核心计算图：定义策略梯度损失 ---
        function [loss, gradients] = compute_loss(net, states_dl, actions, q_values)
            % 前向传播，获取该批次所有状态的动作概率
            % probs 的尺寸为: [Num_Actions, T]
            probs = forward(net, states_dl); 
            
            % 提取实际执行的动作对应的概率
            T = size(probs, 2);
            % 使用线性索引快速提取: sub2ind(矩阵尺寸, 行索引, 列索引)
            idx = sub2ind(size(probs), actions(:)', 1:T);
            selected_probs = probs(idx);
            
            % 计算策略梯度损失: Loss = - sum( G_t * log( pi(a_t|s_t) ) )
            % 添加 1e-8 防止对数运算溢出 (log(0) 报错)
            loss = -sum(q_values(:)' .* log(selected_probs + 1e-8));
            
            % 使用 MATLAB 自动微分引擎计算网络权重的梯度
            gradients = dlgradient(loss, net.Learnables);
        end
    end
end