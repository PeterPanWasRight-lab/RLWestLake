classdef TDValueAgent < handle
    % TDValueAgent: 实现算法 8.1 - 基于值函数的 TD 算法 (用于状态值估计)
    % 
    % 对应伪代码:
    % 初始化: 参数 w (即 V_Table)
    % 目标: 估计给定策略 pi 的状态值
    % 循环:
    %   对于 pi 生成的样本 (s_t, r_{t+1}, s_{t+1})
    %   更新: w = w + alpha * [r + gamma * v(s') - v(s)] * grad(v)
    
    properties
        V_Table         % 参数 w (表格型特征下，w 即状态价值)
        Alpha           % 学习率 alpha
        Gamma           % 折扣因子 gamma
        
        Num_States
        Num_Actions
        
        % 训练记录
        History_Value_Norm % 记录 V 值的变化范数，用于观察收敛
    end
    
    methods
        function obj = TDValueAgent(num_states, num_actions, alpha, gamma)
            % 初始化: 参数 w (V_Table) 初始化为 0 或任意值
            obj.V_Table = zeros(num_states, 1);
            
            obj.Num_States = num_states;
            obj.Num_Actions = num_actions;
            obj.Alpha = alpha; 
            obj.Gamma = gamma; 
        end
        
        function train(obj, env, episodes, max_steps, target_policy)
            
        end
        
        % --- 辅助函数: 根据概率矩阵采样动作 ---
        function a_idx = sample_action_from_policy(obj, s, policy_matrix)
            probs = policy_matrix(s, :);
            r = rand();
            cumulative = cumsum(probs);
            a_idx = find(r <= cumulative, 1, 'first');
        end
        
        % ==========================================
        % 适配 GridWorld 的可视化接口
        % ==========================================
        
        % 1. 获取 V 值 (用于 plot_values)
        function V = get_state_values(obj)
            V = obj.V_Table;
        end
        
        % 2. 获取用于绘图的策略概率 (用于 plot_policy_matrix)
        % 注意: TDValueAgent 本身只学习 V，不直接优化策略。
        % 为了可视化，我们通常展示 "基于当前 V 的贪婪策略" (Greedy w.r.t V)
        % 这需要利用环境模型进行一步前瞻 (Model-based 1-step lookahead)
        function probs_matrix = get_policy_probs(obj, env)
            probs_matrix = zeros(obj.Num_States, obj.Num_Actions);
            
            for s = 1:obj.Num_States
                if s == env.coord2idx(env.Final_State), continue; end
                
                % 计算 Q(s,a) = R + gamma * V(s')
                q_vals = zeros(1, obj.Num_Actions);
                for a = 1:obj.Num_Actions
                    [ns, r, ~] = env.step(s, a);
                    q_vals(a) = r + obj.Gamma * obj.V_Table(ns);
                end
                
                % 找出最大 Q 对应的动作 (Greedy)
                [~, best_a] = max(q_vals);
                best_indices = find(q_vals == q_vals(best_a));
                
                % 均匀分配概率给最优动作
                for idx = best_indices
                    probs_matrix(s, idx) = 1.0 / length(best_indices);
                end
            end
        end
        
        % 3. 获取确定性策略索引 (用于 plot_policy)
        function Pol = get_deterministic_policy(obj, env)
            Pol = ones(obj.Num_States, 1);
            for s = 1:obj.Num_States
                q_vals = zeros(1, obj.Num_Actions);
                for a = 1:obj.Num_Actions
                    [ns, r, ~] = env.step(s, a);
                    q_vals(a) = r + obj.Gamma * obj.V_Table(ns);
                end
                [~, best_a] = max(q_vals);
                Pol(s) = best_a;
            end
        end
    end
end