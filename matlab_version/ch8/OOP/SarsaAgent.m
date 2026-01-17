classdef SarsaAgent < handle
    % SarsaAgent: 支持两种训练模式
    
    properties
        Q_Table         % Q值表
        Alpha; Gamma; Epsilon
        Num_Actions
        Epsilon_Decay; Epsilon_Min
        
        % 记录训练过程
        History_Rewards
    end
    
    methods
        function obj = SarsaAgent(num_states, num_actions, alpha, gamma, epsilon, decay)
            obj.Q_Table = zeros(num_states, num_actions);
            obj.Num_Actions = num_actions;
            obj.Alpha = alpha; obj.Gamma = gamma; obj.Epsilon = epsilon;
            if nargin < 6, obj.Epsilon_Decay = 0.999; else, obj.Epsilon_Decay = decay; end
            obj.Epsilon_Min = 0.01;
            obj.History_Rewards = [];
        end
        
        % --- 核心训练接口 ---
        function train(obj, env, episodes, max_steps, mode)
            % mode: 'reach_and_stop' (到达即停) OR 'fixed_length' (固定轨迹长度)
            
            fprintf('开始训练 - 模式: %s\n', mode);
            obj.History_Rewards = zeros(episodes, 1);
            
            % 如果是 "到达即停" 模式，我们需要手动修正终点价值以匹配策略迭代的结果(10.0)
            % 因为到达即停通常只能学到 R=1.0
            if strcmp(mode, 'reach_and_stop')
                target_idx = env.coord2idx(env.Final_State);
                theoretical_val = env.Reward_Target / (1 - obj.Gamma); % 1/(1-0.9) = 10
                obj.Q_Table(target_idx, :) = theoretical_val; 
            end
            
            for ep = 1:episodes
                curr_s = env.coord2idx(env.Start_State);
                curr_a = obj.choose_action(curr_s);
                total_r = 0;
                
                for step = 1:max_steps
                    % 1. 执行动作
                    [next_s, r, is_in_target] = env.step(curr_s, curr_a);
                    
                    % 2. 逻辑分支
                    if is_in_target && strcmp(mode, 'reach_and_stop')
                        % === 模式 1: 到达即停 ===
                        % 更新这一步，然后结束
                        obj.update_terminal(curr_s, curr_a, r);
                        
                        % 技巧：再次强制修正终点价值，防止被覆盖
                        obj.Q_Table(target_idx, :) = theoretical_val; 
                        
                        total_r = total_r + r;
                        break; % 退出 Step 循环
                    else
                        % === 模式 2: 固定长度 (包含到达终点后的持续循环) ===
                        % 或者是 reach_and_stop 还没到终点
                        
                        next_a = obj.choose_action(next_s);
                        obj.update(curr_s, curr_a, r, next_s, next_a);
                        
                        curr_s = next_s;
                        curr_a = next_a;
                        total_r = total_r + r;
                        
                        % 在 fixed_length 模式下，不 break，直到 step == max_steps
                    end
                end
                
                obj.History_Rewards(ep) = total_r;
                obj.decay_epsilon();
            end
            fprintf('训练结束。\n');
        end
        
        % 辅助方法
        function a_idx = choose_action(obj, s)
            if rand() < obj.Epsilon
                a_idx = randi(obj.Num_Actions);
            else
                [~, a_idx] = max(obj.Q_Table(s, :));
                % 随机打破平局
                best = find(obj.Q_Table(s, :) == obj.Q_Table(s, a_idx));
                if length(best)>1, a_idx = best(randi(length(best))); end
            end
        end
        
        function update(obj, s, a, r, s_next, a_next)
            td_target = r + obj.Gamma * obj.Q_Table(s_next, a_next);
            obj.Q_Table(s, a) = obj.Q_Table(s, a) + obj.Alpha * (td_target - obj.Q_Table(s, a));
        end
        
        function update_terminal(obj, s, a, r)
            % 传统的终止更新: Target = r
            obj.Q_Table(s, a) = obj.Q_Table(s, a) + obj.Alpha * (r - obj.Q_Table(s, a));
        end
        
        function decay_epsilon(obj)
            obj.Epsilon = max(obj.Epsilon_Min, obj.Epsilon * obj.Epsilon_Decay);
        end
        
        function [V, Pol] = get_results(obj)
            [V, Pol] = max(obj.Q_Table, [], 2);
        end
        
        function path = run_test(obj, env, max_steps)
            % 测试跑一条轨迹
            curr = env.coord2idx(env.Start_State);
            path = curr;
            for i=1:max_steps
                [~, a] = max(obj.Q_Table(curr, :));
                [next, ~, done] = env.step(curr, a);
                curr = next;
                path(end+1) = curr;
                if done && i > 10, break; end % 防止死循环
            end
        end
    end
end