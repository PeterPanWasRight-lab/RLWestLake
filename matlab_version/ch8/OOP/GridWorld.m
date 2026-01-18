classdef GridWorld < handle
    % GridWorld: 最终整合版
    % 1. 物理: 目标状态吸收 (V->10), 障碍物可进入 (V->负值)
    % 2. 绘图: 修复遮挡问题，包含随机轨迹绘制功能
    
    properties (SetAccess = public)
        X_Length
        Y_Length
        Start_State      % [x, y]
        Final_State      % [x, y]
        Obstacles        % N x 2 matrix
        
        State_Space_Size
        Action_Space     % Cell array of actions
        
        % 奖励设置 (开放读取权限)
        Reward_Forbidden = -10;
        Reward_Target = 1;
        Reward_Step = 0;
    end
    
    properties (Access = private)
        % 颜色配置
        Color_Green = [0.4660 0.6740 0.1880] * 0.8;
        Color_Obstacle = [0.9290 0.6940 0.1250];
        Color_Final = [0.3010 0.7450 0.9330];
    end
    
    methods
        function obj = GridWorld(x_len, y_len, start, target, obs)
            obj.X_Length = x_len;
            obj.Y_Length = y_len;
            obj.Start_State = start;
            obj.Final_State = target;
            obj.Obstacles = obs;
            obj.State_Space_Size = x_len * y_len;
            obj.Action_Space = {[0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]};
        end
        
        %% --- 核心交互逻辑 ---
        function [next_state_idx, reward, is_done] = step(obj, current_state_idx, action_idx)
            current_state = obj.idx2coord(current_state_idx);
            
            % 1. 吸收态逻辑 (Absorbing State)  
            % 对于Sarsa类似的算法，用这个逻辑代替4.可以显著提升收敛效果和最终状态值的准确程度
            % 一个是到达重点后直接站着不动了，另外一个是到达终点后还会进行随机探索
            % if isequal(current_state, obj.Final_State)
            %     next_state_idx = current_state_idx;
            %     reward = obj.Reward_Target;
            %     is_done = true;
            %     return;
            % end
            
            action = obj.Action_Space{action_idx};
            new_state = current_state + action;
            is_done = false;

            % 2. 边界检查 (撞墙反弹)
            if new_state(1) < 1 || new_state(1) > obj.X_Length || ...
               new_state(2) < 1 || new_state(2) > obj.Y_Length
                new_state = current_state;
                reward = obj.Reward_Forbidden;
                
            % 3. 障碍物逻辑 (允许进入，不反弹)
            elseif ismember(new_state, obj.Obstacles, 'rows')
                % new_state 保持为进入障碍物后的坐标
                reward = obj.Reward_Forbidden;
                
            % 4. 到达终点
            elseif isequal(new_state, obj.Final_State)
                reward = obj.Reward_Target;
                is_done = true;
                
            % 5. 普通移动
            else
                reward = obj.Reward_Step;
            end
            
            next_state_idx = obj.coord2idx(new_state);
        end
        
        function idx = coord2idx(obj, coord)
            idx = obj.X_Length * (coord(2) - 1) + coord(1);
        end
        
        function coord = idx2coord(obj, idx)
            x = mod(idx - 1, obj.X_Length) + 1;
            y = floor((idx - 1) / obj.X_Length) + 1;
            coord = [x, y];
        end
        
        %% --- 可视化部分 ---
        
        % 1. 绘制价值图 (修复了遮挡问题)
        function plot_values(obj, state_values)
            figure('Name', 'Value Function', 'Color', 'w');
            obj.setup_axes();
            obj.draw_background_grid();
            obj.draw_static_elements(); % 先画色块
            
            % 后写文字
            for s_idx = 1:obj.State_Space_Size
                coord = obj.idx2coord(s_idx);
                [px, py] = obj.get_plot_coord(coord(1), coord(2));
                val = state_values(s_idx);
                
                txt_color = 'k'; % 默认黑色文字
                text(px + 0.5, py + 0.5, num2str(round(val, 2)), ...
                    'HorizontalAlignment', 'center', ...
                    'FontSize', 10, 'FontWeight', 'bold', 'Color', txt_color);
            end
            title('State Values');
        end

        % 2. 绘制策略图
        function plot_policy(obj, policy_indices)
            figure('Name', 'Policy', 'Color', 'w');
            obj.setup_axes();
            obj.draw_grid_and_arrows(policy_indices);
            obj.draw_static_elements();
            obj.draw_agent(obj.Start_State);
            title('Optimal Policy');
        end

        % 3. 绘制简单轨迹
        function plot_trajectory(obj, state_history)
            figure('Name', 'Trajectory', 'Color', 'w');
            obj.setup_axes();
            obj.draw_background_grid();
            obj.draw_static_elements();
            for k = 1:length(state_history)-1
                c1 = obj.idx2coord(state_history(k)); c2 = obj.idx2coord(state_history(k+1));
                [p1x, p1y] = obj.get_plot_coord(c1(1), c1(2)); [p2x, p2y] = obj.get_plot_coord(c2(1), c2(2));
                line([p1x+0.5, p2x+0.5], [p1y+0.5, p2y+0.5], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5);
            end
            obj.draw_agent(obj.Start_State);
            title('Single Trajectory');
        end

        % 4. [补全] 绘制随机抖动轨迹
        function plot_stochastic_trajectory(obj, history_input)
            figure('Name', 'Stochastic Exploration', 'Color', 'w');
            obj.setup_axes();
            obj.draw_background_grid();
            obj.draw_static_elements();
            
            if ~iscell(history_input)
                histories = {history_input};
            else
                histories = history_input;
            end
            
            noise_scale = 0.05; 
            hold on;
            
            for h = 1:length(histories)
                path = histories{h};
                for k = 1:length(path)-1
                    c_curr = obj.idx2coord(path(k)); 
                    c_next = obj.idx2coord(path(k+1));
                    
                    [px_c, py_c] = obj.get_plot_coord(c_curr(1), c_curr(2));
                    [px_n, py_n] = obj.get_plot_coord(c_next(1), c_next(2));
                    
                    cx = px_c + 0.5; cy = py_c + 0.5; 
                    nx = px_n + 0.5; ny = py_n + 0.5;
                    
                    if c_curr(2) ~= c_next(2) % 垂直移动
                        x_pts = [cx, cx+noise_scale*randn, nx+noise_scale*randn, nx];
                        y_pts = [cy, cy+(ny-cy)*0.25+noise_scale*randn, cy+(ny-cy)*0.75+noise_scale*randn, ny];
                    else % 水平移动
                        x_pts = [cx, cx+(nx-cx)*0.25+noise_scale*randn, cx+(nx-cx)*0.75+noise_scale*randn, nx];
                        y_pts = [cy, cy+noise_scale*randn, ny+noise_scale*randn, ny];
                    end
                    
                    line(x_pts, y_pts, 'Color', [0.4660 0.6740 0.1880 0.3], 'LineWidth', 1);
                end
            end
            obj.draw_agent(obj.Start_State);
            title(['Aggregated Transitions (' num2str(length(histories)) ' Episodes)']);
        end

        % 3. [核心重构] 根据 figure_policy.m 改写的概率策略绘图
        function plot_policy_matrix(obj, policy_matrix)
            % policy_matrix: (State_Size x Action_Size)
            
            figure('Name', 'Probabilistic Policy', 'Color', 'w'); 
            
            % 1. 绘制坐标轴标签 (addAxisLabels)
            obj.setup_axes(); 
            
            % 2. 绘制网格背景
            obj.draw_background_grid();
            
            % 3. 绘制静态元素 (障碍物/终点)
            obj.draw_static_elements();
            
            num_actions = length(obj.Action_Space);
            ratio = 0.5; % adjust the length of arrow
            
            for s = 1:obj.State_Space_Size
                coord = obj.idx2coord(s);
                [px, py] = obj.get_plot_coord(coord(1), coord(2));
                
                % 中心点坐标 (i_bias, j_bias)
                cx = px + 0.5; 
                cy = py + 0.5;
                
                % 写状态编号 (s1, s2...)
                text(px + 0.1, py + 0.8, ['s', num2str(s)], 'FontSize', 8, 'Color', [.5 .5 .5]);
                
                % 跳过终点和障碍物 (可选，如果不想在这些地方画箭头)
                if isequal(coord, obj.Final_State), continue; end
                
                % 获取当前状态的动作概率分布
                probs = policy_matrix(s, :);
                
                % 遍历所有动作 (kk)
                for kk = 1:num_actions
                    prob = probs(kk);
                    
                    % 只有概率不为0时才绘制 (参考: if policy(...) ~= 0)
                    if prob > 0.001 
                        % [Ref] kk_new calculation from figure_policy.m
                        % 将概率映射到 [0.5, 1.0] 的尺寸区间
                        kk_new = prob / 2 + 0.5;
                        
                        action_vec = obj.Action_Space{kk};
                        
                        % [Ref] drawPolicyArrow logic
                        if action_vec(1) == 0 && action_vec(2) == 0
                            % Stay: Draw Circle
                            % MarkerSize = 5 + kk_new * 6
                            marker_size = 5 + kk_new * 6;
                            plot(cx, cy, 'o', 'MarkerSize', marker_size, 'LineWidth', 2, 'Color', obj.Color_Green);
                        else
                            % Move: Draw Arrow
                            % Position: [cx, cy, ratio * kk_new * ax, -ratio * kk_new * ay]
                            % 注意 dy 的负号，这是为了适配 Y 轴反转
                            dx = ratio * kk_new * action_vec(1);
                            dy = -ratio * kk_new * action_vec(2); 
                            
                            ar = annotation('arrow', ...
                                'Position', [cx, cy, dx, dy], ...
                                'LineStyle', '-', ...
                                'Color', obj.Color_Green, ...
                                'LineWidth', 2);
                            ar.Parent = gca;
                        end
                    end
                end
            end
            
            % 4. 绘制 Agent
            obj.draw_agent(obj.Start_State);
            title('Policy Visualization');
        end
        %% --- 三维柱状图绘制函数（浅色主题版）---
        function plot_3d_bar_chart(obj, state_values)
            % plot_3d_bar_chart: 绘制状态价值的三维柱状图（浅色主题）
            % 输入:
            %   state_values - (State_Size x 1) 向量，每个状态的价值
            % 输出: 三维柱状图
            
            % 创建图形窗口
            figure('Name', '3D State Value Bar Chart', 'Color', 'w', ...
                   'Position', [100, 100, 800, 600]);
            
            % 将状态价值转换为矩阵形式 (Y_Length x X_Length)
            value_matrix = zeros(obj.Y_Length, obj.X_Length);
            
            for idx = 1:obj.State_Space_Size
                coord = obj.idx2coord(idx);
                % 注意：矩阵的行对应y坐标，列对应x坐标
                value_matrix(coord(2), coord(1)) = state_values(idx);
            end
            
            % 创建网格坐标
            [X, Y] = meshgrid(1:obj.X_Length, 1:obj.Y_Length);
            
            % 绘制三维柱状图 - 使用颜色表示高度
            bar_handles = bar3(value_matrix);
            
            % 设置浅色主题和颜色映射
            ax = gca;
            ax.Color = [0.95, 0.95, 0.95];  % 浅灰色背景
            ax.GridColor = [0.8, 0.8, 0.8]; % 浅灰色网格
            ax.GridAlpha = 0.3;              % 网格透明度

            mycolormap = [linspace(0.95, 0.5, 64)', ...  % R: 浅紫到紫
            linspace(0.85, 0.2, 64)', ...  % G: 浅粉到深紫
            linspace(1.0, 0.8, 64)'];      % B: 白到浅紫
            colormap(mycolormap);

            % colormap('summer');             % 使用parula颜色映射（更适合浅色主题）
            
            % 设置图形属性
            title('3D State Value Visualization', 'FontSize', 14, 'FontWeight', 'bold');
            xlabel('X Coordinate', 'FontSize', 12);
            ylabel('Y Coordinate', 'FontSize', 12);
            zlabel('State Value', 'FontSize', 12);
            
            % 设置坐标轴标签
            xticks(1:obj.X_Length);
            yticks(1:obj.Y_Length);
            
            % 设置视角
            view(45, 30); % 方位角45°，俯仰角30°
            grid on;
            
            % 调整颜色映射范围
            clim([min(state_values), max(state_values)]);
            
            % 添加颜色条
            cb = colorbar;
            cb.Label.String = 'State Value';
            cb.Label.FontSize = 11;
            
            % 设置柱子的颜色根据高度变化
            for i = 1:length(bar_handles)
                % 获取柱子的Z数据（高度）
                zdata = get(bar_handles(i), 'ZData');
                
                % 计算每个面的颜色（基于高度）
                cdata = zeros(size(zdata));
                for j = 1:size(zdata, 1)
                    for k = 1:size(zdata, 2)
                        if ~isnan(zdata(j, k))
                            % 归一化到[0,1]范围
                            normalized_value = (zdata(j, k) - min(state_values)) / ...
                                               (max(state_values) - min(state_values));
                            cdata(j, k) = normalized_value;
                        end
                    end
                end
                
                % 设置柱子的颜色数据
                set(bar_handles(i), 'CData', cdata, 'FaceColor', 'interp');
            end
            
            % 添加额外信息到标题
            subtitle_str = sprintf('Min: %.2f, Max: %.2f, Mean: %.2f', ...
                                  min(state_values), max(state_values), mean(state_values));
            subtitle(subtitle_str, 'FontSize', 10);
            
            % 标记特殊状态 - 保存句柄用于图例
            hold on;
            legend_handles = [];
            legend_labels = {};
            
            % 标记起始状态 - 使用不同的标记和颜色
            start_idx = obj.coord2idx(obj.Start_State);
            start_value = state_values(start_idx);
            h_start = plot3(obj.Start_State(1), obj.Start_State(2), start_value + 0.1, ...
                  'p', 'MarkerSize', 15, 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410], ...
                  'MarkerFaceColor', [0, 0.4470, 0.7410]);
            legend_handles = [legend_handles, h_start];
            legend_labels = [legend_labels, 'Start State'];
            
            % 标记目标状态
            target_idx = obj.coord2idx(obj.Final_State);
            target_value = state_values(target_idx);
            h_target = plot3(obj.Final_State(1), obj.Final_State(2), target_value + 0.1, ...
                  's', 'MarkerSize', 12, 'LineWidth', 2, 'Color', [0.4660, 0.6740, 0.1880], ...
                  'MarkerFaceColor', [0.4660, 0.6740, 0.1880]);
            legend_handles = [legend_handles, h_target];
            legend_labels = [legend_labels, 'Target State'];
            
            % 标记障碍物状态
            if size(obj.Obstacles, 1) > 0
                obs_handles = [];
                for i = 1:size(obj.Obstacles, 1)
                    obs_coord = obj.Obstacles(i, :);
                    obs_idx = obj.coord2idx(obs_coord);
                    obs_value = state_values(obs_idx);
                    h_obs = plot3(obs_coord(1), obs_coord(2), obs_value + 0.1, ...
                          '^', 'MarkerSize', 12, 'LineWidth', 2, 'Color', [0.8500, 0.3250, 0.0980], ...
                          'MarkerFaceColor', [0.8500, 0.3250, 0.0980]);
                    obs_handles = [obs_handles, h_obs];
                end
                % 只取一个障碍物作为图例代表
                if ~isempty(obs_handles)
                    legend_handles = [legend_handles, obs_handles(1)];
                    legend_labels = [legend_labels, 'Obstacle States'];
                end
            end
            
            hold off;
            
            % 添加图例 - 修正标注
            if ~isempty(legend_handles)
                legend(legend_handles, legend_labels, 'Location', 'best', 'FontSize', 10);
            end
            
            % 调整图形布局
            set(gca, 'FontSize', 10, 'FontWeight', 'bold', ...
                     'XColor', [0.3, 0.3, 0.3], 'YColor', [0.3, 0.3, 0.3], 'ZColor', [0.3, 0.3, 0.3]);
            
            % 添加数值标签到每个柱子的顶部（只在状态数量较少时）
            if obj.State_Space_Size <= 1000 % 调整为16个状态以下才显示标签
                for x = 1:obj.X_Length
                    for y = 1:obj.Y_Length
                        val = value_matrix(y, x);
                        if ~isnan(val)
                            text(x, y, val + 0.05, ...
                                 num2str(val, '%.2f'), ...
                                 'HorizontalAlignment', 'center', ...
                                 'VerticalAlignment', 'bottom', ...
                                 'FontSize', 8, ...
                                 'FontWeight', 'bold', ...
                                 'Color', 'k'); % 使用黑色文字保证可读性
                        end
                    end
                end
            end
            
            % 添加光照效果，增强三维感
            lighting gouraud;
            light('Position', [10, 10, 10], 'Style', 'infinite');
            light('Position', [-10, -10, 10], 'Style', 'infinite');
            material dull;
            
            % 设置边框样式
            box on;
            set(gca, 'LineWidth', 1);
            
            % 可选：添加网格地面
            hold on;
            % 绘制一个半透明的网格地面
            [Xg, Yg] = meshgrid(1:obj.X_Length, 1:obj.Y_Length);
            Zg = zeros(size(Xg));
            surf(Xg, Yg, Zg, 'FaceAlpha', 0.1, 'EdgeColor', [0.7, 0.7, 0.7], 'FaceColor', 'none');
            hold off;
            
            % 调整视角，确保所有元素可见
            axis tight;
        end

    end
 

    methods (Access = private)
        function [px, py] = get_plot_coord(obj, lx, ly), px = lx; py = obj.Y_Length + 1 - ly; end
        function setup_axes(obj), axis equal; axis off; hold on; axis([0.5, obj.X_Length + 1.5, 0.5, obj.Y_Length + 1.5]); end
        function draw_background_grid(obj)
            for j=1:obj.Y_Length
                for i=1:obj.X_Length, [px,py]=obj.get_plot_coord(i,j); rectangle('Position',[px,py,1,1]); 
                end
            end 
        end
        function draw_static_elements(obj)
            for i=1:size(obj.Obstacles,1), [px,py]=obj.get_plot_coord(obj.Obstacles(i,1),obj.Obstacles(i,2)); rectangle('Position',[px,py,1,1],'FaceColor',obj.Color_Obstacle); end
            [px,py]=obj.get_plot_coord(obj.Final_State(1),obj.Final_State(2)); rectangle('Position',[px,py,1,1],'FaceColor',obj.Color_Final);
        end
        function draw_agent(obj, c), [px,py]=obj.get_plot_coord(c(1),c(2)); plot(px+0.5,py+0.5,'*','MarkerSize',15,'LineWidth',2,'Color','b'); end
        function draw_grid_and_arrows(obj, pol)
            ratio=0.5; obj.draw_background_grid();
            for s=1:obj.State_Space_Size
                c=obj.idx2coord(s); [px,py]=obj.get_plot_coord(c(1),c(2));
                text(px+0.1,py+0.8,['s' num2str(s)],'FontSize',8,'Color',[.5 .5 .5]);
                if isequal(c,obj.Final_State), continue; end
                av=obj.Action_Space{pol(s)}; cx=px+0.5; cy=py+0.5;
                if av(1)==0&&av(2)==0, plot(cx,cy,'o','MarkerSize',8,'LineWidth',2,'Color',obj.Color_Green);
                else, ar=annotation('arrow','Position',[cx,cy,ratio*av(1),ratio*(-av(2))],'Color',obj.Color_Green,'LineWidth',2); ar.Parent=gca; end
            end
        end
    end
end