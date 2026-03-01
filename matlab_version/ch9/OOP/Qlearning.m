classdef Qlearning < handle
    % Qlearning: 实现算法 7.2 (On-policy) 和 7.3 (Off-policy)
    % 配合 GridWorld 的可视化接口使用
    
    properties
        Q_Table         % Q值表 (State x Action)
        Alpha           % 学习率
        Gamma           % 折扣因子
        Epsilon         % 探索率 (用于 7.2 的行为策略)
        Epsilon_Decay   % 衰减系数
        Epsilon_Min     % 最小探索率
        
        Num_States
        Num_Actions
        Algorithm_Mode  % '7.2_on_policy' 或 '7.3_off_policy'
        
        % 训练记录 (用于绘制曲线)
        History_Reward
        History_Steps    % 每个epsilon走了多少步到达target
        History_Error
    end
    
    methods
        function obj = Qlearning(num_states, num_actions, alpha, gamma, epsilon, mode)
            obj.Q_Table = zeros(num_states, num_actions);
            obj.Num_States = num_states;
            obj.Num_Actions = num_actions;
            obj.Alpha = alpha; 
            obj.Gamma = gamma; 
            obj.Epsilon = epsilon;
            obj.Algorithm_Mode = mode;
            
            % 默认衰减参数
            obj.Epsilon_Decay = 0.999; 
            obj.Epsilon_Min = 0.01;
        end
        
        function train(obj, env, episodes, max_steps, true_V)
            % max_steps防止一直找不到target时无限循环。另外注释掉终点判断可以让所有回合都跑max_steps，用于世界探索
            % 初始化记录数组
            obj.History_Reward = zeros(episodes, 1);
            obj.History_Steps = zeros(episodes, 1);
            obj.History_Error = zeros(episodes, 1);
            
            fprintf('开始训练 Q-learning (%s)...\n', obj.Algorithm_Mode);
            
            % [Trick] 预设终点Q值，帮助算法在有限步数内理解吸收态 V=10 的概念
            % 理论上 Q-learning 可以自己学到，但预设能加速收敛并让可视化更准
            target_idx = env.coord2idx(env.Final_State);
            target_val = env.Reward_Target / (1 - obj.Gamma); 
            obj.Q_Table(target_idx, :) = target_val;
            
            for ep = 1:episodes
                curr_s = env.coord2idx(env.Start_State);
                
                % --- 1. 初始动作选择 ---
                if strcmp(obj.Algorithm_Mode, '7.3_off_policy')
                    % 算法 7.3: 使用行为策略 \pi_b 生成数据
                    % 这里假设行为策略是完全随机的 (Uniform Random)
                    curr_a = obj.choose_action_random();
                else
                    % 算法 7.2: 使用当前的 \epsilon-Greedy 策略
                    curr_a = obj.choose_action_epsilon_greedy(curr_s);
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % 这里采用的和环境一边交互一边拿数据的OffPolicy方法，实际上也可以先按照随机策略pi把整个轨迹都生成好，然后一次性跑完
                total_r = 0;
                
                for step = 1:max_steps
                    % --- 2. 与环境交互 ---
                    [next_s, r, is_target] = env.step(curr_s, curr_a);
                    
                    % --- 3. Q-Learning 更新公式 ---
                    % 核心公式一样: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',:) - Q(s,a)]
                    max_next_q = max(obj.Q_Table(next_s, :));
                    td_target = r + obj.Gamma * max_next_q;
                    
                    obj.Q_Table(curr_s, curr_a) = obj.Q_Table(curr_s, curr_a) + ...
                        obj.Alpha * (td_target - obj.Q_Table(curr_s, curr_a));
                    
                    % --- 4. 下一步动作选择 (策略差异) ---
                    if strcmp(obj.Algorithm_Mode, '7.3_off_policy')
                        % Off-policy: 下一步继续由行为策略 \pi_b (随机) 产生
                        next_a = obj.choose_action_random();
                    else
                        % On-policy: 下一步由更新后的 Q 表导出的 \epsilon-Greedy 产生
                        next_a = obj.choose_action_epsilon_greedy(next_s);
                    end
                    
                    curr_s = next_s;
                    curr_a = next_a;
                    total_r = total_r + r;
                    
                    if is_target
                        % 保持吸收态价值，防止被随机探索破坏
                        obj.Q_Table(target_idx, :) = target_val;
                        break; 
                    end
                end
                
                % --- 5. 记录与衰减 ---
                obj.History_Reward(ep) = total_r;
                obj.History_Steps(ep) = step;
                
                if nargin > 4 && ~isempty(true_V)
                    current_V = max(obj.Q_Table, [], 2);
                    obj.History_Error(ep) = mean(abs(current_V - true_V));
                end
                
                % 仅在 On-policy 模式下衰减 Epsilon
                % Off-policy 的行为策略通常保持固定的探索性(如纯随机)
                if strcmp(obj.Algorithm_Mode, '7.2_on_policy')
                    obj.Epsilon = max(obj.Epsilon_Min, obj.Epsilon * obj.Epsilon_Decay);
                end
            end
            fprintf('训练完成。\n');
        end
        
        % --- 辅助动作选择函数 ---
        
        function a_idx = choose_action_epsilon_greedy(obj, s)
            if rand() < obj.Epsilon
                a_idx = randi(obj.Num_Actions);
            else
                [~, a_idx] = max(obj.Q_Table(s, :));
                % 打破平局
                best = find(obj.Q_Table(s, :) == obj.Q_Table(s, a_idx));
                if length(best) > 1, a_idx = best(randi(length(best))); end
            end
        end
        
        function a_idx = choose_action_random(obj)
            a_idx = randi(obj.Num_Actions);
        end
        
        % --- 可视化数据适配接口 ---
        
        % 1. 获取概率矩阵 -> 用于 GridWorld.plot_policy_matrix
        function probs_matrix = get_policy_probs(obj)
            probs_matrix = zeros(obj.Num_States, obj.Num_Actions);
            
            % 根据不同算法，定义我们要可视化的"目标策略"
            if strcmp(obj.Algorithm_Mode, '7.3_off_policy')
                % 算法 7.3: 目标是学习最优策略 \pi_T (Greedy)
                % 所以这里 epsilon 设为 0，展示纯贪婪策略
                viz_epsilon = 0; 
            else
                % 算法 7.2: 目标是学习 \epsilon-Greedy 策略
                % 展示当前的探索策略
                viz_epsilon = obj.Epsilon; 
            end
            
            for s = 1:obj.Num_States
                % 基础概率 (Epsilon / |A|)
                probs_matrix(s, :) = viz_epsilon / obj.Num_Actions;
                
                % 贪婪动作获得剩余概率 (1 - Epsilon)
                [~, best_a] = max(obj.Q_Table(s, :));
                best_indices = find(obj.Q_Table(s, :) == obj.Q_Table(s, best_a));
                
                for idx = best_indices'
                    probs_matrix(s, idx) = probs_matrix(s, idx) + (1 - viz_epsilon) / length(best_indices);
                end
            end
        end
        
        % 2. 获取状态价值 -> 用于 GridWorld.plot_values
        function V = get_state_values(obj)
            V = max(obj.Q_Table, [], 2);
        end
        
        % 3. 获取确定性策略索引 -> 用于 GridWorld.plot_policy
        function Pol = get_deterministic_policy(obj)
            [~, Pol] = max(obj.Q_Table, [], 2);
        end
    end
end