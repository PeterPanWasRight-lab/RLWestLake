classdef PolicyIterationAgent < handle
    % PolicyIterationAgent: 
    % 1. 不再跳过障碍物，计算其真实价值（通常为负值）
    % 2. 不再跳过终点，计算其吸收价值（收敛到 10）
    
    properties
        V_Table         % 状态价值表
        Policy          % 策略表
        Q_Table         % Q值缓存
        
        Gamma           % 折扣因子
        Episode_Length  % 总迭代步数
        PE_Length       % 评估步数
        
        Num_States
        Num_Actions
    end
    
    methods
        function obj = PolicyIterationAgent(num_states, num_actions, gamma, episode_len, pe_len)
            obj.Num_States = num_states;
            obj.Num_Actions = num_actions;
            obj.Gamma = gamma;
            obj.Episode_Length = episode_len;
            obj.PE_Length = pe_len;
            
            obj.V_Table = zeros(num_states, 1);
            obj.Q_Table = zeros(num_states, num_actions);   % only GridWorld可用的形式，其他带概率选择的环境不可用这种数据结构
            obj.Policy = ones(num_states, 1);   % 随便给的
        end
        
        function train(obj, env)
            fprintf('开始策略迭代 (计算所有状态)...\n');
            tic;
            
            for step = 1:obj.Episode_Length
                % 1. 策略评估
                obj.policy_evaluation(env);
                
                % 2. 策略改进
                changes = obj.policy_improvement();
                
                if mod(step, 10) == 0 || step == 1
                    fprintf('  -> Step %d/%d: 策略调整 %d\n', step, obj.Episode_Length, changes);
                end
            end
            
            elapsed = toc;
            fprintf('训练完成！总耗时: %.4f 秒\n', elapsed);
        end
        
        function policy_evaluation(obj, env)
            for k = 1:obj.PE_Length
                state_value_new = obj.V_Table;
                
                for si = 1:obj.Num_States
                    % [重要修改] 不再跳过任何状态！
                    % 无论是障碍物还是终点，都交给 Bellman 方程计算
                    
                    q_values = zeros(1, obj.Num_Actions);
                    for ai = 1:obj.Num_Actions
                        [s_next, r, ~] = env.step(si, ai);
                        
                        % Bellman Equation
                        q_values(ai) = r + obj.Gamma * obj.V_Table(s_next);
                    end
                    
                    % 更新 V
                    current_action = obj.Policy(si);
                    state_value_new(si) = q_values(current_action);
                    
                    % 缓存 Q
                    obj.Q_Table(si, :) = q_values;
                end
                
                obj.V_Table = state_value_new;
            end
        end
        
        function changes = policy_improvement(obj)
            changes = 0;
            for si = 1:obj.Num_States
                % [重要修改] 即使是障碍物内部，也要寻找逃离的最优策略
                % 唯独终点可以跳过策略更新（因为它已经是吸收态，策略无关紧要，或者默认为 Stay）
                % 但为了代码简单，这里对所有状态都做 update 也没问题
                
                old_action = obj.Policy(si);
                [~, best_action] = max(obj.Q_Table(si, :));
                obj.Policy(si) = best_action;
                
                if old_action ~= best_action
                    changes = changes + 1;
                end
            end
        end
        
        function [V, PolicyIdx] = get_results(obj)
            V = obj.V_Table;
            PolicyIdx = obj.Policy;
        end
    end
end