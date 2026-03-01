classdef RLVisualizer
    methods (Static)
        % 1. 绘制总奖励随回合变化
        function plot_rewards(hist, name)
            figure('Name',[name ' Rewards'],'Color','w');
            % 绘制平滑曲线
            plot(smoothdata(hist,'gaussian',20),'LineWidth',2,'Color','b'); hold on;
            % 绘制原始数据(浅色背景)
            plot(hist,'Color',[0.7 0.8 1], 'LineWidth', 0.5); 
            xlabel('Episode'); ylabel('Total Reward'); 
            title([name ' - Reward per Episode']); 
            legend('Smoothed', 'Raw'); grid on;
        end
        
        % 2. 绘制回合长度随回合变化
        function plot_steps(hist, name)
            figure('Name',[name ' Steps'],'Color','w');
            plot(smoothdata(hist,'gaussian',20),'LineWidth',2,'Color',[0.8500 0.3250 0.0980]); 
            xlabel('Episode'); ylabel('Steps'); 
            title([name ' - Steps per Episode']); grid on;
        end
        
        % 3. 绘制最优状态值估计误差
        function plot_error(hist, name)
            figure('Name',[name ' Error'],'Color','w');
            semilogy(hist,'LineWidth',1.5,'Color','k'); % 使用对数坐标
            xlabel('Episode'); ylabel('Mean Abs Error |V - V^*|'); 
            title([name ' - Value Estimation Error']); grid on;
        end
    end
end