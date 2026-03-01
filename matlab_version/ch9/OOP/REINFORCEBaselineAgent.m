classdef REINFORCEBaselineAgent < handle
    % REINFORCEBaselineAgent: 带基线的蒙特卡洛策略梯度算法
    
    properties (SetAccess = public)
        theta           % 策略参数矩阵 (Num_States x Num_Actions)
        V               % 状态价值基线表 (Num_States x 1)
        
        alpha_theta     % 策略的学习率 (Actor)
        alpha_v         % 基线价值的学习率 (Critic)
        gamma           % 折扣因子
        
        num_states
        num_actions
        returns_history % 记录收敛曲线
    end
    
    methods
        function obj = REINFORCEBaselineAgent(num_states, num_actions, alpha_theta, alpha_v, gamma)
            obj.num_states = num_states;
            obj.num_actions = num_actions;
            obj.alpha_theta = alpha_theta;
            obj.alpha_v = alpha_v;
            obj.gamma = gamma;
            
            % 初始化策略参数 θ 和价值基线 V
            obj.theta = zeros(num_states, num_actions);
            obj.V = zeros(num_states, 1); 
            obj.returns_history = [];
        end
        
        % Softmax 策略分布
        function probs = get_action_probabilities(obj, state_idx)
            h_vals = obj.theta(state_idx, :); 
            max_h = max(h_vals); % 防指数爆炸
            exp_h = exp(h_vals - max_h);
            probs = exp_h / sum(exp_h);
        end
        
        % 按照策略采样
        function action_idx = select_action(obj, state_idx)
            probs = obj.get_action_probabilities(state_idx);
            cum_probs = cumsum(probs);
            action_idx = find(cum_probs >= rand(), 1);
        end
        
        % 训练主循环
        function train(obj, env, num_episodes, max_steps)
            obj.returns_history = zeros(num_episodes, 1);
            
            for k = 1:num_episodes
                [states, actions, rewards] = obj.generate_episode(env, max_steps);
                obj.returns_history(k) = sum(rewards);
                
                % 核心：策略与基线的同步更新
                obj.update_theta_and_baseline(states, actions, rewards);
                
                if mod(k, 100) == 0
                    fprintf('Episode %d/%d, Return: %.2f\n', k, num_episodes, obj.returns_history(k));
                end
            end
        end
        
        function policy_matrix = get_policy_matrix(obj)
            policy_matrix = zeros(obj.num_states, obj.num_actions);
            for s = 1:obj.num_states
                policy_matrix(s, :) = obj.get_action_probabilities(s);
            end
        end
    end
    
    methods (Access = private)
        function [states, actions, rewards] = generate_episode(obj, env, max_steps)
            states_temp = zeros(max_steps, 1);
            actions_temp = zeros(max_steps, 1);
            rewards_temp = zeros(max_steps, 1);
            
            current_state = env.coord2idx(env.Start_State);
            step_count = 0;
            
            for t = 1:max_steps
                step_count = step_count + 1;
                action = obj.select_action(current_state);
                [next_state, reward, is_done] = env.step(current_state, action);
                
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
        
        % 带有 Baseline 的更新逻辑
        function update_theta_and_baseline(obj, states, actions, rewards)
            T = length(rewards);
            
            % 1. 逆向计算累计回报 q_t (也即 G_t)
            q_values = zeros(T, 1);
            current_q = 0;
            for t = T:-1:1
                current_q = rewards(t) + obj.gamma * current_q;
                q_values(t) = current_q;
            end
            
            % 2. 遍历轨迹进行更新
            for t = 1:T
                s_t = states(t);
                a_t = actions(t);
                q_t = q_values(t);
                
                % 计算优势 (Advantage): 实际回报 - 预期回报
                delta_t = q_t - obj.V(s_t);
                
                % --- 更新 Critic (基线 V) ---
                % 目标是让 V(s) 逼近真实的 q_t
                obj.V(s_t) = obj.V(s_t) + obj.alpha_v * delta_t;
                
                % --- 更新 Actor (策略 \theta) ---
                probs = obj.get_action_probabilities(s_t);
                
                % 梯度的向量化计算: \nabla_\theta \ln \pi(a_t|s_t)
                grad_log_pi = -probs; 
                grad_log_pi(a_t) = grad_log_pi(a_t) + 1; 
                
                % 策略更新：用 delta_t (优势) 代替原来的 q_t
                % 公式: \theta <- \theta + \alpha_\theta * (q_t - b) * \nabla \ln \pi
                obj.theta(s_t, :) = obj.theta(s_t, :) + obj.alpha_theta * delta_t * grad_log_pi;
            end
        end
    end
end