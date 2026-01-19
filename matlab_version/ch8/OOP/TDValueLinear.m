%% TDValueLinear.m - 基于值函数的 TD 算法类
% 算法 8.1: 基于值函数的 TD 算法
classdef TDValueLinear < handle
    properties (Access = private)
        % 环境对象
        env
        
        % 基函数相关属性
        dim              % 基函数维度
        phi_func         % 基函数句柄
        basis_type       % 基函数类型
        
        % 算法参数
        omega           % 权重向量
        V_table         % 值函数表（用于比较）
        current_alpha   % 当前学习率
        current_gamma   % 当前折扣因子
        
        % 历史记录
        omega_history   % 权重历史
        s_history       % 状态历史
        
        % 当前状态
        current_state   % 当前状态坐标
        
        % 策略参数
        epsilon         % ε-greedy策略的ε值
        
        % 训练参数
        initial_alpha   % 初始学习率
    end
    
    properties (Constant, Access = private)
        % 动作数量
        NUM_ACTIONS = 5;
    end
    
    methods
        %% 构造函数
        function obj = TDValueLinear(env, basis_type)
            % TDValueLinear 构造函数
            % 输入:
            %   env: GridWorld环境对象
            %   basis_type: 基函数类型（'linear', 'quadratic', 'custom'）
            
            obj.env = env;
            obj.basis_type = basis_type;
            
            % 根据基函数类型设置基函数
            obj.setup_basis_functions();
            
            % 初始化参数
            obj.omega = zeros(obj.dim, 1);
            [x_len, y_len] = env.get_grid_size();
            obj.V_table = zeros(x_len, y_len);
            
            % 初始化历史记录
            obj.omega_history = [];
            obj.s_history = [];
            
            % 初始化当前状态
            obj.current_state = env.Start_State;
            
            % 默认参数
            obj.epsilon = 0.3;
            obj.current_alpha = 0.01;
            obj.current_gamma = 0.9;
            obj.initial_alpha = 2;
        end
        
        %% 设置基函数
        function setup_basis_functions(obj)
            % setup_basis_functions 根据基函数类型设置基函数
            
            switch obj.basis_type
                case 'linear'
                    % 线性基函数: [1, x, y]
                    obj.phi_func = @(x, y) [1; x; y];
                    obj.dim = 3;
                    
                case 'quadratic'
                    % 二次基函数: [1, x, y, x^2, y^2, x*y]
                    obj.phi_func = @(x, y) [1; x; y; x^2; y^2; x*y];
                    obj.dim = 6;
                    
                case 'custom'
                    % 自定义基函数 - 示例: [1, x, y, sin(pi*x/5), sin(pi*y/5)]
                    obj.phi_func = @(x, y) [1; x; y; sin(pi*x/5); sin(pi*y/5)];
                    obj.dim = 5;
                    
                otherwise
                    error('不支持的基函数类型: %s', obj.basis_type);
            end
        end
        
        %% 设置学习率
        function set_alpha(obj, alpha)
            % set_alpha 设置学习率
            % 输入:
            %   alpha: 学习率
            obj.current_alpha = alpha;
            obj.initial_alpha = alpha;
        end
        
        %% 设置折扣因子
        function set_gamma(obj, gamma)
            % set_gamma 设置折扣因子
            % 输入:
            %   gamma: 折扣因子
            obj.current_gamma = gamma;
        end
        
        %% 设置ε值
        function set_epsilon(obj, epsilon)
            % set_epsilon 设置ε-greedy策略的ε值
            % 输入:
            %   epsilon: ε值 (0-1之间)
            obj.epsilon = epsilon;
        end
        
        %% ε-greedy策略选择动作
        function a_idx = choose_action(obj, s_idx)
            % choose_action ε-greedy策略选择动作
            % 输入:
            %   s_idx: 当前状态索引
            % 输出:
            %   a_idx: 选择的动作索引
            
            % 以ε的概率随机选择动作，以1-ε的概率选择贪婪动作
            if rand() < obj.epsilon
                % 随机探索
                a_idx = randi(obj.NUM_ACTIONS);
            else
                % 贪婪策略
                a_idx = obj.choose_greedy_action(s_idx);
            end
        end
        
        %% 贪婪策略选择动作
        function a_idx = choose_greedy_action(obj, s_idx)
            % choose_greedy_action 贪婪策略选择动作
            % 输入:
            %   s_idx: 当前状态索引
            % 输出:
            %   a_idx: 选择的动作索引
            
            % 获取所有可能的动作
            state_coord = obj.env.idx2coord(s_idx);
            x = state_coord(1);
            y = state_coord(2);
            
            % 计算当前状态的值函数
            current_value = obj.compute_value(x, y);
            
            % 初始化最佳值和动作
            best_value = -inf;
            best_actions = [];
            
            % 遍历所有动作
            for a = 1:obj.NUM_ACTIONS
                % 获取下一个状态
                [next_idx, ~, ~] = obj.env.step(s_idx, a);
                next_coord = obj.env.idx2coord(next_idx);
                
                % 计算下一个状态的值函数
                next_value = obj.compute_value(next_coord(1), next_coord(2));
                
                % 更新最佳动作
                if next_value > best_value
                    best_value = next_value;
                    best_actions = a;
                elseif next_value == best_value
                    best_actions = [best_actions, a];
                end
            end
            
            % 从最佳动作中随机选择一个
            if ~isempty(best_actions)
                a_idx = best_actions(randi(length(best_actions)));
            else
                a_idx = randi(obj.NUM_ACTIONS); % 如果没有最佳动作，随机选择
            end
        end
        
        %% 计算状态值函数
        function value = compute_value(obj, x, y)
            % compute_value 计算给定状态的值函数
            % 输入:
            %   x, y: 状态坐标
            % 输出:
            %   value: 状态值函数估计
            
            % 计算特征向量
            phi_s = obj.phi_func(x, y);
            
            % 计算值函数估计
            value = phi_s' * obj.omega;
        end
        
        %% 执行一步TD更新
        function [td_error, reward] = td_update(obj, s_coord, a_idx, next_state_idx)
            % td_update 执行一步TD更新
            % 输入:
            %   s_coord: 当前状态坐标
            %   a_idx: 动作索引
            %   next_state_idx: 下一个状态索引
            % 输出:
            %   td_error: TD误差
            %   reward: 获得的奖励
            
            % 环境执行一步
            [next_state_idx, reward, ~] = obj.env.step(obj.env.coord2idx(s_coord), a_idx);
            next_state_coord = obj.env.idx2coord(next_state_idx);
            
            % 提取当前状态坐标
            x = s_coord(1);
            y = s_coord(2);
            
            % 计算特征向量
            phi_s = obj.phi_func(x, y);
            
            % 计算下一个状态的特征向量
            next_coord = obj.env.idx2coord(next_state_idx);
            phi_s_next = obj.phi_func(next_coord(1), next_coord(2));
            
            % TD更新
            td_error = reward + obj.current_gamma * (phi_s_next' * obj.omega) - (phi_s' * obj.omega);
            obj.omega = obj.omega + obj.current_alpha * td_error * phi_s;
            
            % 表格法TD更新（用于比较）
            next_state_coord_full = obj.env.idx2coord(next_state_idx);
            td_error_table = reward + obj.current_gamma * obj.V_table(next_state_coord_full(1), next_state_coord_full(2)) - obj.V_table(s_coord(1), s_coord(2));
            obj.V_table(s_coord(1), s_coord(2)) = obj.V_table(s_coord(1), s_coord(2)) + obj.current_alpha * td_error_table;
            
            % 更新当前状态
            obj.current_state = obj.env.idx2coord(next_state_idx);
        end
        
        %% 训练算法
        function train(obj, episodes, max_steps)
            % train 训练TD值函数算法
            % 输入:
            %   episodes: 训练回合数
            %   max_steps: 最大步数
            
            fprintf('开始TD值函数算法训练...\n');
            fprintf('回合数: %d, 初始学习率: %.4f, 折扣因子: %.2f\n', episodes, obj.initial_alpha, obj.current_gamma);
            
            % 计时开始
            tic;
            
            % 初始化当前状态
            siCoord = obj.env.Start_State;
            alpha = obj.initial_alpha; % 当前学习率
            
            for i = 1:episodes
                % 选择动作
                s_idx = obj.env.coord2idx(siCoord);
                a_idx = obj.choose_action(s_idx);
                
                % 执行TD更新
                [~, ~] = obj.td_update(siCoord, a_idx, s_idx);
                
                % 历史记录
                obj.omega_history = [obj.omega_history, obj.omega];
                obj.s_history = [obj.s_history, s_idx];
                
                % 更新学习率（衰减策略）
                if i < 1e5
                    obj.current_alpha = 0.01;
                else
                    obj.current_alpha = max(0.001, obj.initial_alpha/i);
                end
                
                % 中途重置学习率
                if i == 50000 || i == 100000 || i == 2e5 
                    obj.current_alpha = 2;  % 中途雄起
                end
                
                % 每10000回合打印进度
                if mod(i, 10000) == 0
                    fprintf('回合: %d, 学习率: %.6f\n', i, obj.current_alpha);
                end
            end
            
            % 计时结束
            training_time = toc;
            fprintf('训练完成！耗时: %.2f 秒\n', training_time);
        end
        
        %% 获取权重向量
        function omega = get_weights(obj)
            % get_weights 获取当前权重向量
            % 输出:
            %   omega: 权重向量
            omega = obj.omega;
        end
        
        %% 获取值函数表
        function V_table = get_value_table(obj)
            % get_value_table 获取值函数表
            % 输出:
            %   V_table: 值函数表
            V_table = obj.V_table;
        end
        
        %% 获取权重历史
        function omega_history = get_omega_history(obj, index)
            % get_omega_history 获取权重历史
            % 输入:
            %   index: 可选，指定获取第几个权重的历史
            % 输出:
            %   omega_history: 权重历史
            
            if nargin < 2
                % 返回所有权重历史
                omega_history = obj.omega_history;
            else
                % 返回指定权重的历史
                if index <= size(obj.omega_history, 1) && index > 0
                    omega_history = obj.omega_history(index, :);
                else
                    error('索引超出范围');
                end
            end
        end
        
        %% 获取状态历史
        function s_history = get_state_history(obj)
            % get_state_history 获取状态历史
            % 输出:
            %   s_history: 状态历史
            s_history = obj.s_history;
        end
        
        %% 获取当前状态
        function current_state = get_current_state(obj)
            % get_current_state 获取当前状态
            % 输出:
            %   current_state: 当前状态坐标
            current_state = obj.current_state;
        end
        
        %% 绘制权重历史
        function plot_omega_history(obj, index)
            % plot_omega_history 绘制权重历史
            % 输入:
            %   index: 可选，指定绘制第几个权重的历史
            
            figure('Position', [100, 100, 800, 400]);
            
            if nargin < 2
                % 绘制所有权重的历史
                for i = 1:min(size(obj.omega_history, 1), 10) % 最多绘制前10个权重
                    subplot(2, 5, i);
                    plot(obj.omega_history(i, :), 'LineWidth', 1.5);
                    xlabel('训练步数');
                    ylabel(sprintf('权重 ω_%d', i));
                    title(sprintf('权重 ω_%d 历史', i));
                    grid on;
                end
            else
                % 绘制指定权重的历史
                if index <= size(obj.omega_history, 1) && index > 0
                    plot(obj.omega_history(index, :), 'LineWidth', 2);
                    xlabel('训练步数');
                    ylabel(sprintf('权重 ω_%d', index));
                    title(sprintf('权重 ω_%d 历史', index));
                    grid on;
                else
                    error('索引超出范围');
                end
            end
            
            sgtitle('权重学习历史', 'FontSize', 14);
        end
        
        %% 绘制值函数三维网格图
        function fhandle = plot_value_function(obj)
            % plot_value_function 绘制值函数的三维网格图
            % 输出:
            %   fhandle: 图形句柄
            
            % 获取环境网格大小
            [x_len, y_len] = obj.env.get_grid_size();
            
            % 创建网格
            [X, Y] = meshgrid(1:x_len, 1:y_len);
            
            % 计算每个网格点的值函数
            V_grid = zeros(y_len, x_len);
            for i = 1:x_len
                for j = 1:y_len
                    % 计算特征向量
                    phi_ij = obj.phi_func(i, j);
                    % 计算值函数估计
                    V_grid(j, i) = phi_ij' * obj.omega;
                end
            end
            
            % 绘制三维表面图
            fhandle = figure('Position', [100, 100, 800, 600]);
            surf(X, Y, V_grid, 'FaceColor', 'interp', 'FaceAlpha', 0.3);
            
            % 美化图形
            colormap(jet); % 使用jet颜色映射
            colorbar; % 添加颜色条
            
            % 添加标签和标题
            xlabel('X坐标', 'FontSize', 12);
            ylabel('Y坐标', 'FontSize', 12);
            zlabel('值函数 V(s)', 'FontSize', 12);
            title(sprintf('值函数估计 (基函数: %s)', obj.basis_type), 'FontSize', 14);
            
            % 添加网格
            grid on;
            
            % 设置视角
            view(45, 30); % 调整视角以便更好地查看
            
            % 添加等高线投影
            hold on;
            contour3(X, Y, V_grid, 10, 'k:', 'LineWidth', 0.5);
            hold off;
        end
        
        %% 绘制值函数热图
        function plot_value_heatmap(obj)
            % plot_value_heatmap 绘制值函数热图
            
            % 获取环境网格大小
            [x_len, y_len] = obj.env.get_grid_size();
            
            % 计算每个网格点的值函数
            V_grid = zeros(y_len, x_len);
            for i = 1:x_len
                for j = 1:y_len
                    % 计算特征向量
                    phi_ij = obj.phi_func(i, j);
                    % 计算值函数估计
                    V_grid(j, i) = phi_ij' * obj.omega;
                end
            end
            
            % 绘制热图
            figure('Position', [100, 100, 600, 500]);
            imagesc(V_grid);
            colormap(jet);
            colorbar;
            
            % 添加标签和标题
            xlabel('X坐标', 'FontSize', 12);
            ylabel('Y坐标', 'FontSize', 12);
            title(sprintf('值函数热图 (基函数: %s)', obj.basis_type), 'FontSize', 14);
            
            % 添加数值标签
            for i = 1:x_len
                for j = 1:y_len
                    text(i, j, sprintf('%.2f', V_grid(j, i)), ...
                        'HorizontalAlignment', 'center', ...
                        'Color', 'white', 'FontWeight', 'bold');
                end
            end
        end
        
        %% 多项式基函数（静态方法）
        function phi = basis_polynomial(x, y, degree)
            % basis_polynomial 多项式基函数（静态方法）
            % 输入:
            %   x, y: 坐标
            %   degree: 多项式阶数
            % 输出:
            %   phi: 特征向量
            
            phi = [1]; % 常数项
            
            % 添加线性项
            if degree >= 1
                phi = [phi; x; y];
            end
            
            % 添加二次项
            if degree >= 2
                phi = [phi; x^2; y^2; x*y];
            end
            
            % 添加三次项
            if degree >= 3
                phi = [phi; x^3; y^3; x^2*y; x*y^2];
            end
        end
        
        %% 傅里叶基函数（静态方法）
        function phi = basis_fourier(x, y, max_freq)
            % basis_fourier 傅里叶基函数（静态方法）
            % 输入:
            %   x, y: 坐标
            %   max_freq: 最大频率
            % 输出:
            %   phi: 特征向量
            
            % 归一化坐标到 [0, 1] 区间
            x_norm = (x-1) / 4; % 假设网格是5x5
            y_norm = (y-1) / 4;
            
            phi = [1]; % 常数项
            
            for i = 0:max_freq
                for j = 0:max_freq
                    if i == 0 && j == 0
                        continue; % 常数项已添加
                    end
                    % 添加余弦项
                    phi = [phi; cos(pi*i*x_norm) * cos(pi*j*y_norm)];
                    % 添加正弦项
                    phi = [phi; sin(pi*i*x_norm) * sin(pi*j*y_norm)];
                end
            end
        end
        
        %% 重置算法状态
        function reset(obj)
            % reset 重置算法状态
            [x_len, y_len] = obj.env.get_grid_size();
            
            % 重置参数
            obj.omega = zeros(obj.dim, 1);
            obj.V_table = zeros(x_len, y_len);
            
            % 重置历史记录
            obj.omega_history = [];
            obj.s_history = [];
            
            % 重置当前状态
            obj.current_state = obj.env.Start_State;
            
            % 重置学习率
            obj.current_alpha = obj.initial_alpha;
            
            fprintf('算法状态已重置\n');
        end
    end
end