classdef SarsaValueAgent < handle
    % SarsaAgent: 基于函数近似的 Sarsa 算法 (Algorithm 8.2)
    
    properties
        Alpha       % 学习率
        Gamma       % 折扣因子
        Epsilon     % 探索率
        Weights     % 权重矩阵 (Feature_Dim x Action_Dim)
        BasisType   % 基函数类型
        X_Max, Y_Max % 用于归一化
        Dim         % 特征维度
        NumActions  % 动作数量
    end
    
    methods
        function obj = SarsaValueAgent(x_len, y_len, num_actions, basis_type, alpha, gamma, epsilon)
            obj.X_Max = x_len;
            obj.Y_Max = y_len;
            obj.NumActions = num_actions;
            obj.BasisType = basis_type;
            obj.Alpha = alpha;
            obj.Gamma = gamma;
            obj.Epsilon = epsilon;
            
            % 初始化权重矩阵: 每一列代表一个动作的权重向量
            obj.Dim = obj.get_feature_dimension();
            obj.Weights = zeros(obj.Dim, num_actions);
        end
        
        function [q_val, phi] = get_q_value(obj, coord, action_idx)
            % 计算特定动作的 Q(s, a)
            phi = obj.get_features(coord);
            w_a = obj.Weights(:, action_idx);
            q_val = phi' * w_a;
        end
        
        function [q_values] = get_all_q_values(obj, coord)
            % 获取当前状态下所有动作的 Q 值
            phi = obj.get_features(coord);
            % 矩阵乘法: (1 x Dim) * (Dim x NumActions) -> 1 x NumActions
            q_values = phi' * obj.Weights;
        end
        
        function action_idx = choose_action(obj, coord)
            % Epsilon-Greedy 策略 (对应图中更新策略部分)
            
            % 1. 探索
            if rand < obj.Epsilon
                action_idx = randi(obj.NumActions);
            else
                % 2. 利用 (Greedy)
                q_values = obj.get_all_q_values(coord);
                
                % 找到最大值对应的索引 (处理多个最大值的情况)
                max_val = max(q_values);
                candidates = find(q_values == max_val);
                action_idx = candidates(randi(length(candidates)));
            end
        end
        
        function update(obj, s_coord, a_idx, reward, next_s_coord, next_a_idx, is_done)
            % Sarsa 更新规则 (对应图中更新值部分)
            % w <- w + alpha * [r + gamma * Q(s', a') - Q(s, a)] * grad
            
            % 1. 计算 Q(s, a) 和 特征
            [q_curr, phi_s] = obj.get_q_value(s_coord, a_idx);
            
            % 2. 计算 Q(s', a')
            if is_done
                q_next = 0; % 目标状态价值为 0
            else
                [q_next, ~] = obj.get_q_value(next_s_coord, next_a_idx);
            end
            
            % 3. 计算 TD 误差
            target = reward + obj.Gamma * q_next;
            td_error = target - q_curr;
            
            % 4. 更新权重 (只更新当前执行动作对应的权重列)
            % 梯度即为 phi_s
            obj.Weights(:, a_idx) = obj.Weights(:, a_idx) + obj.Alpha * td_error * phi_s;
        end
        
        function phi = get_features(obj, coord)
            % 特征归一化与生成
            x = coord(1); y = coord(2);
            nx = (x / obj.X_Max - 0.5) * 2; % 归一化到 [-1, 1]
            ny = (y / obj.Y_Max - 0.5) * 2;
            
            switch obj.BasisType
                case 'linear'
                    phi = [1; nx; ny];
                case 'quadratic'
                    phi = [1; nx; ny; nx^2; ny^2; nx*ny];
                case 'rbf'
                    phi = [1];
                    centers = [-0.5, -0.5; 0.5, 0.5; -0.5, 0.5; 0.5, -0.5; 0, 0];
                    sigma = 0.5;
                    for i = 1:size(centers, 1)
                        dist = norm([nx, ny] - centers(i,:));
                        phi = [phi; exp(-dist^2 / (2*sigma^2))];
                    end
            end
        end
        
        function dim = get_feature_dimension(obj)
            switch obj.BasisType
                case 'linear', dim = 3;
                case 'quadratic', dim = 6;
                case 'rbf', dim = 6;
                otherwise, dim = 1;
            end
        end
    end
end