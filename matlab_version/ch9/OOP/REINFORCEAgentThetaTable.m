classdef REINFORCEAgentThetaTable < handle
    % REINFORCEAgent: 严格依据提供的教材推导实现的蒙特卡洛策略梯度算法
    
    properties (SetAccess = public)
        theta           % 策略参数矩阵 (Num_States x Num_Actions)，即特征函数 h(s,a,θ)
        alpha           % 学习率 α
        gamma           % 折扣因子 γ
        num_states
        num_actions
        returns_history % 记录收敛曲线
    end
    
    methods
        function obj = REINFORCEAgentThetaTable(num_states, num_actions, alpha, gamma)
            obj.num_states = num_states;
            obj.num_actions = num_actions;
            obj.alpha = alpha;
            obj.gamma = gamma;
            % 初始化参数 θ 为 0  表格法theta
            obj.theta = zeros(num_states, num_actions);
            obj.returns_history = [];
        end
        
        % 对应教材式 (9.12)：Softmax 策略
        function probs = get_action_probabilities(obj, state_idx)
            h_vals = obj.theta(state_idx, :); % 特征函数 h(s, a, θ)
            
            % 数值稳定性处理 (防止指数爆炸，数值更稳定)，等价于原公式
            max_h = max(h_vals); 
            exp_h = exp(h_vals - max_h);
            
            probs = exp_h / sum(exp_h);   % 1x5 array
        end
        
        function action_idx = select_action(obj, state_idx)
            probs = obj.get_action_probabilities(state_idx);
            cum_probs = cumsum(probs);
            action_idx = find(cum_probs >= rand(), 1);
        end
        
        function train(obj, env, num_episodes, max_steps)
            obj.returns_history = zeros(num_episodes, 1); % 预分配内存
            
            for k = 1:num_episodes
                % 1. 算法9.1：根据 π(θ) 生成一个回合 {s0, a0, r1, ..., sT-1, aT-1, rT}
                [states, actions, rewards] = obj.generate_episode(env, max_steps);
                obj.returns_history(k) = sum(rewards);
                
                % 2. 策略更新
                obj.update_theta(states, actions, rewards);
                
                if mod(k, 100) == 0
                    fprintf('Episode %d/%d, Total Reward: %.2f\n', k, num_episodes, obj.returns_history(k));
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
            % 返回整个序列
            % offline 离线的算法
            % 内存预分配，避免 push_back 操作耗时
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
                rewards_temp(t) = reward; % 此处的 rewards_temp(t) 即为伪代码中的 r_{t+1}
                
                % if is_done, break; end
                current_state = next_state;
            end
            
            states = states_temp(1:step_count);
            actions = actions_temp(1:step_count);
            rewards = rewards_temp(1:step_count);
        end
        
        % 核心修改点：严格按照教材公式 9.32, 9.33 和 9.11 进行代码映射
        function update_theta(obj, states, actions, rewards)
            T = length(rewards);
            
            % 1. 算法9.1 中的 Value update: q_t(s_t, a_t) = \sum_{k=t+1}^{T} \gamma^{k-t-1} r_k
            q_values = zeros(T, 1);
            current_q = 0;
            for t = T:-1:1
                current_q = rewards(t) + obj.gamma * current_q;
                q_values(t) = current_q;
            end
            
            % 2. Policy update: 对于 t = 0, 1, ..., T-1 (MATLAB索引为 1 到 T)
            for t = 1:T
                s_t = states(t);
                a_t = actions(t);
                q_t = q_values(t);
                
                probs = obj.get_action_probabilities(s_t);
                pi_a_t = probs(a_t); % π(a_t | s_t, θ_t)
                
                % 计算 β_t = q_t / π(a_t | s_t, θ_t) (参考教材式 9.33 下方推导)
                beta_t = q_t / pi_a_t;
                
                % 计算真实概率梯度 ∇_θ π(a_t | s_t, θ_t) (基于Softmax的偏导)
                % 公式：对于 a = a_t，导数为 π(a_t)(1 - π(a_t))
                %       对于 a ≠ a_t，导数为 -π(a_t)π(a)
                % 让deepseek求一下这个表格theta的偏导数，会发现是这个结果。
                grad_pi = -pi_a_t * probs; % 得到 [ -π(a_t)π(1), -π(a_t)π(2), -π(a_t)π(3), -π(a_t)π(4), -π(a_t)π(5) ]
                grad_pi(a_t) = grad_pi(a_t) + pi_a_t; % 加上 π(a_t)，等价于 π(a_t) - π(a_t)^2  % 在 a_t 位置加上 π(a_t)，即变为 -π(a_t)π(a_t) + π(a_t) = π(a_t)(1-π(a_t))
                
                % 严格对应公式 9.33: θ_t+1 = θ_t + α * β_t * ∇_θ π(a_t | s_t, θ_t)
                obj.theta(s_t, :) = obj.theta(s_t, :) + obj.alpha * beta_t * grad_pi;
            end
        end
    end
end