classdef SarsaValue < handle   
    % 
    % SarsaValue: 基于线性动作值函数近似的Sarsa算法
    % 支持：linear, quadratic, custom, fourier
    % 更新：使用 for 循环控制最大步数 (PE_lenMax)
    
    properties
        env             % GridWorld 环境对象
        basis_type      % 基函数类型
        state_dim       % 状态特征维度
        total_dim       % 总权重维度
        phi_action_func  % 状态基函数句柄
        
        omega           % 权重向量 w
        gamma           % 折扣因子
        
        omega_history   % 历史记录
        num_actions     % 动作数量
        
        fourier_coeffs  % 傅里叶系数矩阵
    end
    
    methods
        %% --- 构造函数 ---
        function obj = SarsaValue(env, basis_type, gamma)
            if nargin < 3, gamma = 0.9; end
            if nargin < 2, basis_type = 'fourier'; end 
            
            obj.env = env;
            obj.basis_type = basis_type;
            obj.gamma = gamma;
            obj.num_actions = length(obj.env.Action_Space); 
            
            % 1. 定义状态基函数 phi_s(x,y)
            switch basis_type
                % case 'linear'  % 这样定义基函数将会导致耦合 不一定能够搞出一个比较好的结果
                %     obj.phi_action_func = @(x, y, a) [1; x; y; a];
                %     obj.state_dim = 4;
                case 'quadratic'
                    obj.phi_action_func = @(x, y) [1; x; y; x^2; y^2; x*y];
                    obj.state_dim = 6;
                case 'custom'
                    obj.phi_action_func = @(x, y) [1; x; y; sin(pi*x/5); sin(pi*y/5)];
                    obj.state_dim = 5;
                case 'fourier'
                    order = 5; 
                    [c1, c2] = ndgrid(0:order, 0:order);
                    obj.fourier_coeffs = [c1(:), c2(:)]; 
                    obj.state_dim = (order + 1)^2;      
                    
                    x_len = obj.env.X_Length;
                    y_len = obj.env.Y_Length;
                    obj.phi_action_func = @(x, y) obj.fourier_basis_calc(x, y, x_len, y_len);
                otherwise
                    error('未知的 basis_type');
            end
            
            obj.total_dim = obj.state_dim * obj.num_actions;
            obj.omega = zeros(obj.total_dim, 1);
            obj.omega_history = [];
        end
        
        function phi = fourier_basis_calc(obj, x, y, Lx, Ly)
            norm_x = (x - 1) / (Lx - 1 + 1e-5); 
            norm_y = (y - 1) / (Ly - 1 + 1e-5);
            s_norm = [norm_x; norm_y];
            dot_prod = obj.fourier_coeffs * s_norm;
            phi = cos(pi * dot_prod);
        end
        
        function phi_sa = get_feature(obj, x, y, a_idx)
            phi_s = obj.phi_action_func(x, y);
            phi_sa = zeros(obj.total_dim, 1);
            start_idx = (a_idx - 1) * obj.state_dim + 1;
            end_idx = a_idx * obj.state_dim;
            phi_sa(start_idx:end_idx) = phi_s;
        end
        
        function q_val = get_q_value(obj, x, y, a_idx, current_omega)
            if nargin < 5, current_omega = obj.omega; end
            phi_sa = obj.get_feature(x, y, a_idx);
            q_val = phi_sa' * current_omega;
        end
        
        %% --- 核心训练逻辑 (算法 8.2) ---
        % 修改：使用 for 循环代替 while
        function train(obj, episodes, epsilon, max_steps)
            if nargin < 2, episodes = 10000; end
            if nargin < 3, epsilon = 0.1; end 
            if nargin < 4, max_steps = 200; end % 默认最大步数 PE_lenMax
            
            local_omega = obj.omega;
            local_gamma = obj.gamma;
            
            obj.omega_history = zeros(5, episodes); 
            
            alpha = 0.01; 
            
            tic;
            for i = 1 : episodes
                siCoord = obj.env.Start_State;
                s_idx = obj.env.coord2idx(siCoord);
                
                % Sarsa: 先选择动作
                a_idx = obj.choose_action(siCoord, local_omega, epsilon);
                
                % --- 修改部分开始 ---
                % 使用 for 循环控制最大步数，防止死循环
                for step = 1 : max_steps
                    
                    % 1. 执行动作
                    [next_s_idx, reward, is_done] = obj.env.step(s_idx, a_idx);
                    next_coord = obj.env.idx2coord(next_s_idx);
                    
                    % 2. 当前特征与Q值
                    x = siCoord(1); y = siCoord(2);
                    phi_sa_curr = obj.get_feature(x, y, a_idx);
                    q_curr = phi_sa_curr' * local_omega;
                    
                    % 3. 选择下一动作 (On-policy)
                    next_a_idx = obj.choose_action(next_coord, local_omega, epsilon);
                    
                    % 4. 计算 Target
                    if is_done
                        target = reward;
                    else
                        phi_sa_next = obj.get_feature(next_coord(1), next_coord(2), next_a_idx);
                        q_next = phi_sa_next' * local_omega;
                        target = reward + local_gamma * q_next;
                    end
                    
                    % 5. 更新权重
                    td_error = target - q_curr;
                    local_omega = local_omega + alpha * td_error * phi_sa_curr;
                    
                    % 6. 状态转移
                    siCoord = next_coord;
                    s_idx = next_s_idx;
                    a_idx = next_a_idx;
                    
                    % 7. 判断是否到达终点，如果是则跳出循环
                    if is_done
                        break;
                    end
                end
                % --- 修改部分结束 ---
                
                obj.omega_history(:, i) = local_omega(1:5);
                
                % 学习率衰减
                if i > 5000
                   alpha = max(0.0001, 0.01 * (5000/i));
                end
            end
            toc;
            obj.omega = local_omega;
        end
        
        %% --- 动作选择函数 ---
        function a_idx = choose_action(obj, coord, current_omega, epsilon)
            q_values = zeros(1, obj.num_actions);
            for a = 1:obj.num_actions
                q_values(a) = obj.get_q_value(coord(1), coord(2), a, current_omega);
            end
            
            max_q = max(q_values);
            best_actions = find(q_values == max_q);
            if length(best_actions) > 1
                best_action = best_actions(randi(length(best_actions)));
            else
                best_action = best_actions(1);
            end
            
            if rand() < epsilon
                a_idx = randi(obj.num_actions);
            else
                a_idx = best_action;
            end
        end
        
        %% --- 绘图 ---
        function fig_handle = plot_surface(obj)
            x_len = obj.env.X_Length; 
            y_len = obj.env.Y_Length;
            V_grid = zeros(y_len, x_len);
            
            for i = 1:x_len
                for j = 1:y_len
                    qs = zeros(1, obj.num_actions);
                    for a = 1:obj.num_actions
                        qs(a) = obj.get_q_value(i, j, a, obj.omega);
                    end
                    V_grid(j, i) = max(qs);
                end
            end
            
            [X, Y] = meshgrid(1:x_len, 1:y_len);
            fig_handle = figure(); 
            surf(X, Y, V_grid, 'FaceColor', 'interp', 'FaceAlpha', 0.8);
            colormap(parula); 
            colorbar; 
            view(45, 30); 
            grid on;
            title(sprintf('Learned Value Surface (Max Q) - %s', obj.basis_type));
        end
        
        function plot_omega_history(obj)
            figure; 
            plot(obj.omega_history'); 
            title('Weight Convergence (First 5 dimensions)'); 
            grid on;
        end
    end
end