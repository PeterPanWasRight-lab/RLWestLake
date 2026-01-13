classdef SarsaAgent < handle
    % SarsaAgent: 负责Q表维护和动作选择
    
    properties
        Q_Table         % State x Action matrix
        Alpha           % Learning rate
        Gamma           % Discount factor
        Epsilon         % Exploration rate
        Num_Actions
    end
    
    methods
        function obj = SarsaAgent(num_states, num_actions, alpha, gamma, epsilon)
            obj.Q_Table = zeros(num_states, num_actions);
            obj.Num_Actions = num_actions;
            obj.Alpha = alpha;
            obj.Gamma = gamma;
            obj.Epsilon = epsilon;
        end
        
        function action_idx = choose_action(obj, state_idx)
            % 对应 stochastic_policy.m 的逻辑，但改为 Epsilon-Greedy
            
            % 1. 生成概率分布 (类似 stochastic_policy 中的 policy_i)
            policy_probs = ones(1, obj.Num_Actions) * (obj.Epsilon / obj.Num_Actions);
            
            % 找到最优动作
            [~, best_a] = max(obj.Q_Table(state_idx, :));
            
            % 增加最优动作的概率
            policy_probs(best_a) = policy_probs(best_a) + (1 - obj.Epsilon);
            
            % 2. 采样 (类似 randsrc 的功能)
            % randsrc(1, 1, [1:length; probs]) 的手动实现
            r = rand();
            cumulative_probs = cumsum(policy_probs);
            action_idx = find(r <= cumulative_probs, 1, 'first');
        end
        
        function update(obj, s, a, r, s_next, a_next)
            % Sarsa 更新: Q(s,a) = Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
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

        function [V, Policy] = get_results(obj)
            % 获取 V 值 (max Q) 和 确定性策略 (max index)
            [V, Policy] = max(obj.Q_Table, [], 2);
        end
    end
end