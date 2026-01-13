classdef GridWorld < handle
    % GridWorld: 封装环境动力学与可视化
    % 更新：增加了 plot_stochastic_trajectory 方法用于绘制带随机抖动的路径
    
    properties (SetAccess = private)
        X_Length
        Y_Length
        Start_State      % [x, y]
        Final_State      % [x, y]
        Obstacles        % N x 2 matrix
        
        State_Space_Size
        Action_Space     % Cell array of actions
    end
    
    properties (Access = private)
        % 奖励参数
        Reward_Forbidden = -10;
        Reward_Target = 1;
        Reward_Step = 0;
        
        % 绘图颜色配置
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
            
            % 动作定义: [dx, dy] (矩阵坐标系)
            obj.Action_Space = {[0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]};
        end
        
        %% --- 核心交互逻辑 ---
        function [next_state_idx, reward, is_done] = step(obj, current_state_idx, action_idx)
            current_state = obj.idx2coord(current_state_idx);
            action = obj.Action_Space{action_idx};
            
            new_x = current_state(1) + action(1);
            new_y = current_state(2) + action(2);
            new_state = [new_x, new_y];

            is_done = false;

            if new_x < 1 || new_x > obj.X_Length || new_y < 1 || new_y > obj.Y_Length
                new_state = current_state;
                reward = obj.Reward_Forbidden;
            elseif ismember(new_state, obj.Obstacles, 'rows')
                reward = obj.Reward_Forbidden;
            elseif isequal(new_state, obj.Final_State)
                reward = obj.Reward_Target;
                is_done = true;
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
        
        % 1. 绘制策略图 (Quiver/Arrow)
        function plot_policy(obj, policy_indices)
            figure('Name', 'Policy Visualization', 'Color', 'w');
            obj.setup_axes();
            
            obj.draw_grid_and_arrows(policy_indices);
            obj.draw_static_elements();
            obj.draw_agent(obj.Start_State);
            
            title('Sarsa Policy');
        end
        
        % 2. 绘制价值图 (Heatmap/Text)
        function plot_values(obj, state_values)
            figure('Name', 'Value Function', 'Color', 'w');
            obj.setup_axes();
            
            for j = 1:obj.Y_Length       
                for i = 1:obj.X_Length       
                    [plot_x, plot_y] = obj.get_plot_coord(i, j);
                    rectangle('Position', [plot_x, plot_y, 1, 1]);
                    
                    current_idx = obj.coord2idx([i, j]);
                    text(plot_x + 0.4, plot_y + 0.5, num2str(round(state_values(current_idx), 2)), ...
                        'FontSize', 10);
                    hold on;           
                end
            end
            
            obj.draw_static_elements();
            title('State Values');
        end

        % 3. 绘制简单轨迹 (虚线)
        function plot_trajectory(obj, state_history)
            figure('Name', 'Trajectory', 'Color', 'w');
            obj.setup_axes();
            
            obj.draw_background_grid();
            obj.draw_static_elements();
            
            for k = 1:length(state_history)-1
                coord_curr = obj.idx2coord(state_history(k));
                coord_next = obj.idx2coord(state_history(k+1));
                
                [curr_px, curr_py] = obj.get_plot_coord(coord_curr(1), coord_curr(2));
                [next_px, next_py] = obj.get_plot_coord(coord_next(1), coord_next(2));
                
                line([curr_px+0.5, next_px+0.5], [curr_py+0.5, next_py+0.5], ...
                     'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5);
            end
            
            obj.draw_agent(obj.Start_State);
            title('Agent Trajectory');
        end

        % 4. [新增] 绘制随机抖动轨迹 (对应 drawStateTransitions)
        function plot_stochastic_trajectory(obj, state_history)
            figure('Name', 'Stochastic Trajectory', 'Color', 'w');
            obj.setup_axes();
            
            obj.draw_background_grid();
            obj.draw_static_elements();
            
            % 移植 drawStateTransitions 的逻辑
            noise_scale = 0.03;
            
            for k = 1:length(state_history)-1
                coord_curr = obj.idx2coord(state_history(k));
                coord_next = obj.idx2coord(state_history(k+1));
                
                % 获取绘图坐标系下的左下角
                [px_curr, py_curr] = obj.get_plot_coord(coord_curr(1), coord_curr(2));
                [px_next, py_next] = obj.get_plot_coord(coord_next(1), coord_next(2));
                
                % 转换为格子中心坐标
                cx_curr = px_curr + 0.5; cy_curr = py_curr + 0.5;
                cx_next = px_next + 0.5; cy_next = py_next + 0.5;
                
                % 判断移动方向并添加随机抖动
                if coord_curr(2) ~= coord_next(2) 
                    % === 垂直移动 (Y改变) ===
                    % 逻辑：X 坐标添加纯噪声，Y 坐标分段添加噪声(0, 0.25, 0.75, 1)
                    x_pts = [cx_curr, ...
                             cx_curr + noise_scale * randn(1), ...
                             cx_next + noise_scale * randn(1), ...
                             cx_next];
                         
                    y_pts = [cy_curr, ...
                             cy_curr + (cy_next - cy_curr)*0.25 + noise_scale * randn(1), ...
                             cy_curr + (cy_next - cy_curr)*0.75 + noise_scale * randn(1), ...
                             cy_next];
                         
                    line(x_pts, y_pts, 'Color', 'green', 'LineWidth', 1);
                    
                elseif coord_curr(1) ~= coord_next(1)
                    % === 水平移动 (X改变) ===
                    % 逻辑：Y 坐标添加纯噪声，X 坐标分段添加噪声
                    x_pts = [cx_curr, ...
                             cx_curr + (cx_next - cx_curr)*0.25 + noise_scale * randn(1), ...
                             cx_curr + (cx_next - cx_curr)*0.75 + noise_scale * randn(1), ...
                             cx_next];
                             
                    y_pts = [cy_curr, ...
                             cy_curr + noise_scale * randn(1), ...
                             cy_next + noise_scale * randn(1), ...
                             cy_next];
                         
                    line(x_pts, y_pts, 'Color', 'green', 'LineWidth', 1);
                end
                
                hold on;
            end
            
            obj.draw_agent(obj.Start_State);
            title('Stochastic Transitions (Monte Carlo Style)');
        end
    end
    
    methods (Access = private)
        function [px, py] = get_plot_coord(obj, logic_x, logic_y)
            % 逻辑坐标转绘图坐标 (Y轴反转)
            px = logic_x;
            py = obj.Y_Length + 1 - logic_y;
        end

        function setup_axes(obj)
            axis equal; axis off; hold on;
            axis([0.5, obj.X_Length + 1.5, 0.5, obj.Y_Length + 1.5]);
            
            % X Labels
            for i = 1:obj.X_Length
                text(i + 0.5, obj.Y_Length + 1.2, num2str(i), 'HorizontalAlignment', 'center');
            end
            % Y Labels
            for j = 1:obj.Y_Length
                [~, py] = obj.get_plot_coord(1, j);
                text(0.8, py + 0.5, num2str(j), 'HorizontalAlignment', 'right');
            end
        end
        
        function draw_background_grid(obj)
            for j = 1:obj.Y_Length       
                for i = 1:obj.X_Length
                    [px, py] = obj.get_plot_coord(i, j);
                    rectangle('Position', [px, py, 1, 1]); 
                end
            end
        end
        
        function draw_static_elements(obj)
            for i = 1:size(obj.Obstacles, 1)
                ox = obj.Obstacles(i, 1); oy = obj.Obstacles(i, 2);
                [px, py] = obj.get_plot_coord(ox, oy);
                rectangle('Position', [px, py, 1, 1], 'FaceColor', obj.Color_Obstacle);
            end
            fx = obj.Final_State(1); fy = obj.Final_State(2);
            [px, py] = obj.get_plot_coord(fx, fy);
            rectangle('Position', [px, py, 1, 1], 'FaceColor', obj.Color_Final);
        end
        
        function draw_agent(obj, coord)
             [px, py] = obj.get_plot_coord(coord(1), coord(2));
             plot(px+0.5, py+0.5, '*', 'MarkerSize', 15, 'LineWidth', 2, 'Color', 'b');
        end
        
        function draw_grid_and_arrows(obj, policy_indices)
            ratio = 0.5; 
            for s_idx = 1:obj.State_Space_Size
                coord = obj.idx2coord(s_idx);
                [px, py] = obj.get_plot_coord(coord(1), coord(2));
                
                rectangle('Position', [px, py, 1, 1]);
                text(px + 0.1, py + 0.8, ['s', num2str(s_idx)], 'FontSize', 8, 'Color', [0.5 0.5 0.5]);
                
                if ismember(coord, obj.Obstacles, 'rows') || isequal(coord, obj.Final_State), continue; end
                
                best_action_idx = policy_indices(s_idx);
                action_vec = obj.Action_Space{best_action_idx};
                cx = px + 0.5; cy = py + 0.5;
                
                if action_vec(1) == 0 && action_vec(2) == 0
                    plot(cx, cy, 'o', 'MarkerSize', 8, 'LineWidth', 2, 'Color', obj.Color_Green);
                else
                    dx_plot = ratio * action_vec(1);
                    dy_plot = ratio * (-action_vec(2)); % Y轴反转修正
                    ar = annotation('arrow', 'Position', [cx, cy, dx_plot, dy_plot], ...
                        'Color', obj.Color_Green, 'LineWidth', 2);
                    ar.Parent = gca;
                end
            end
        end
    end
end