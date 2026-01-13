classdef SarsaAgent < handle
    % SarsaAgent: 负责 Q 表维护、动作选择和参数衰减
    
    properties
        Q_Table         % State x Action matrix
        Alpha           % Learning rate
        Gamma           % Discount factor
        Epsilon         % Exploration rate
        Num_Actions
        
        % 新增属性用于衰减控制
        Epsilon_Min     % 最小探索率
        Epsilon_Decay   % 衰减系数 (例如 0.99)
    end
    
    methods
        function obj = SarsaAgent(num_states, num_actions, alpha, gamma, epsilon)
            obj.Q_Table = zeros(num_states, num_actions);
            obj.Num_Actions = num_actions;
            obj.Alpha = alpha;
            obj.Gamma = gamma;
            obj.Epsilon = epsilon;
            
            % 默认衰减参数 (如果不手动设置，默认每次衰减 0.5%)
            obj.Epsilon_Min = 0.01;
            obj.Epsilon_Decay = 0.995; 
        end
        
        function action_idx = choose_action(obj, state_idx)
            % Epsilon-Greedy 策略
            
            % 1. 生成基础概率分布
            policy_probs = ones(1, obj.Num_Actions) * (obj.Epsilon / obj.Num_Actions);
            
            % 2. 找到最优动作并增加其概率
            [~, best_a] = max(obj.Q_Table(state_idx, :));
            policy_probs(best_a) = policy_probs(best_a) + (1 - obj.Epsilon);
            
            % 3. 轮盘赌采样
            r = rand();
            cumulative_probs = cumsum(policy_probs);
            action_idx = find(r <= cumulative_probs, 1, 'first');
        end
        
        function update(obj, s, a, r, s_next, a_next)
            % Sarsa 更新: Q(s,a) += alpha * [r + gamma * Q(s',a') - Q(s,a)]
            current_q = obj.Q_Table(s, a);
            next_q = obj.Q_Table(s_next, a_next);
            
            td_target = r + obj.Gamma * next_q;
            td_error = td_target - current_q;
            
            obj.Q_Table(s, a) = current_q + obj.Alpha * td_error;
        end
        
        function update_terminal(obj, s, a, r)
             % 终点更新 (无 next state)
            current_q = obj.Q_Table(s, a);
            td_error = r - current_q;
            obj.Q_Table(s, a) = current_q + obj.Alpha * td_error;
        end

        % [修复] 补全了缺失的 decay_epsilon 方法
        function decay_epsilon(obj)
            % 将 epsilon 乘以衰减系数，但不低于最小值
            obj.Epsilon = max(obj.Epsilon_Min, obj.Epsilon * obj.Epsilon_Decay);
        end

        function [V, Policy] = get_results(obj)
            % 获取 V 值 (max Q) 和 确定性策略 (max index)
            [V, Policy] = max(obj.Q_Table, [], 2);
        end
    end
end