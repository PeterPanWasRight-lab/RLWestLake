classdef TDLinearValue < handle
    % TDLinearValue: 基于线性值函数近似的TD算法类
    
    properties
        env             % GridWorld 环境对象
        basis_type      % 基函数类型
        dim             % 特征维度
        phi_func        % 基函数句柄
        
        omega           % 权重向量
        gamma           % 折扣因子
        
        omega_history   % 历史记录
    end
    
    methods
        %% --- 构造函数 ---
        function obj = TDLinearValue(env, basis_type, gamma)
            if nargin < 3, gamma = 0.9; end
            if nargin < 2, basis_type = 'custom'; end
            
            obj.env = env;
            obj.basis_type = basis_type;
            obj.gamma = gamma;
            
            % 基函数配置
            switch basis_type
                case 'linear'
                    obj.phi_func = @(x, y) [1; x; y];
                    obj.dim = 3;
                case 'quadratic'
                    obj.phi_func = @(x, y) [1; x; y; x^2; y^2; x*y];
                    obj.dim = 6;
                    % obj.phi_func = @(x, y) [1; x; y; x^2; y^2; x*y; x^3; y^3; x^2*y; x*y^2];
                    % obj.dim = 10;
                case 'custom'
                    obj.phi_func = @(x, y) [1; x; y; sin(pi*x/5); sin(pi*y/5)];
                    obj.dim = 5;
                otherwise
                    error('未知的 basis_type');
            end
            
            obj.omega = zeros(obj.dim, 1);
            obj.omega_history = [];
        end
        
        %% --- 核心训练逻辑 ---
        function train(obj, episodes)
            if nargin < 2, episodes = 450000; end
            
            local_omega = obj.omega;  %局部变量的访问比访问obj.omega要快。在下面的循环中可以提速
            local_gamma = obj.gamma;
            phi_f = obj.phi_func;
            
            obj.omega_history = zeros(obj.dim, episodes);  % 预申请内存，切勿使用 obj.omega_history = [obj.omega_history; omega_new]导致内存申请缓慢
            siCoord = obj.env.Start_State;
            alpha = 0.01; 
            
            tic;
            for i = 1 : episodes
                % 1. 获取当前状态索引
                s_idx = obj.env.coord2idx(siCoord);
                
                % 2. 选择动作 (使用您要求的 epsilon-greedy 逻辑)
                % 默认 epsilon = 1 (纯随机)，所以这里无需传入 best_action
                a_idx = obj.choose_action(s_idx, 1.0); 
                
                % 3. 环境交互
                [next_state_idx, reward, ~] = obj.env.step(s_idx, a_idx);
                
                % 4. 提取特征
                x = siCoord(1); y = siCoord(2);
                phi_s = phi_f(x, y);
                
                next_coord = obj.env.idx2coord(next_state_idx);
                phi_s_next = phi_f(next_coord(1), next_coord(2));
                
                % 5. TD 更新
                v_curr = phi_s' * local_omega;
                v_next = phi_s_next' * local_omega;
                td_error = reward + local_gamma * v_next - v_curr;
                local_omega = local_omega + alpha * td_error * phi_s;
                
                % 6. 记录与更新
                obj.omega_history(:, i) = local_omega;
                siCoord = next_coord;
                
                % Alpha 衰减逻辑
                if i < 1e5
                    alpha = 0.01;
                elseif i < 3e5
                    alpha = max(0.001, alpha/i);
                else 
                    alpha = 0.0002;
                end
                if i == 50000 || i == 150000, alpha = 1; end
            end
            toc;
            obj.omega = local_omega;
        end
        
        %% --- 动作选择函数 ---
        function a_idx = choose_action(obj, s_idx, epsilon)
            % 默认 epsilon = 1 (随机游走)
            if nargin < 3, epsilon = 1.0; end
            
            num_actions = 5; % 上下左右停
            
            % 获取当前策略下的最优动作
            % 注意：在本算法(TD Prediction)中，并没有显式学习策略 Policy。
            % 当 epsilon=1 时，current_best_action 是谁完全不影响概率分布(都是1/N)。
            % 为了代码逻辑完整性，这里设为随机或 1。
            current_best_action = randi(num_actions); 
            
            % --- 核心逻辑: 构造概率分布 ---
            probs = zeros(1, num_actions);
            for ai = 1:num_actions
                if ai == current_best_action
                    % 贪婪动作概率: (1 - ε) + ε/|A|
                    probs(ai) = (1 - epsilon) + (epsilon / num_actions);
                else
                    % 其他动作概率: ε/|A|
                    probs(ai) = epsilon / num_actions;
                end
            end
            
            % --- 根据概率进行采样 ---
            r = rand();
            cumulative_prob = 0;
            a_idx = num_actions; % 默认防错
            
            for ai = 1:num_actions
                cumulative_prob = cumulative_prob + probs(ai);
                if r <= cumulative_prob
                    a_idx = ai;
                    return;
                end
            end
        end
        
        %% --- 绘图辅助 ---
        function fig_handle=plot_surface(obj)
            x_len = obj.env.X_Length; y_len = obj.env.Y_Length;
            V_grid = zeros(y_len, x_len);
            for i = 1:x_len
                for j = 1:y_len
                    V_grid(j, i) = obj.phi_func(i, j)' * obj.omega;
                end
            end
            [X, Y] = meshgrid(1:x_len, 1:y_len);
            fig_handle=figure(); surf(X, Y, V_grid, 'FaceColor', 'interp','FaceAlpha', 0.3);
            colormap(jet); colorbar; view(45, 30); grid on;
            title(sprintf('Value Surface (%s)', obj.basis_type));
        end
        
        function plot_omega_history(obj)
            figure; plot(obj.omega_history(1,:)); 
            title('Omega(1) Convergence'); grid on;
        end
    end
end