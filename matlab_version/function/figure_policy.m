function figure_policy(x_length, y_length, agent_state,final_state, obstacle_state, state_number,policy, action)
    %% Inverse y coordinate
 
    
    xa_used = agent_state(:, 1) + 0.5;
    ya_used = y_length+1-agent_state(:, 2) + 0.5;
 
    
    
    xf = final_state(:, 1);
    yf = y_length+1-final_state(:, 2); 
    
    
    
    xo = obstacle_state(:, 1);
    yo = y_length+1-obstacle_state(:, 2);
                                                        
    
    
    %%
    
    greenColor=[0.4660 0.6740 0.1880]*0.8;
    
    
    
    % Initialize the figure
    figure();
    
    
    % Add labels on the axes
    addAxisLabels(x_length, y_length);
    
    % Draw the grid, state values, and policy arrows
    r = drawGridStateValuesAndPolicy(x_length, y_length, state_number, policy, greenColor, action);
    
    % Color the obstacles and the final state
    colorObstacles(xo, yo, r);
    colorFinalState(xf, yf, r);
    
    % Draw the agent
    agent = plot(xa_used, ya_used, '*', 'markersize', 15, 'linewidth', 2, 'color', 'b');  
    hold on;
    
    axis equal
    axis off
    
    
end
function addAxisLabels(x_length, y_length)
    for i = 1:x_length
        text(i + 0.5, y_length + 1.1, num2str(i));
    end
    for j = y_length:-1:1
        text(0.9, j + 0.5, num2str(y_length - j + 1));
    end
end

function r= drawGridStateValuesAndPolicy(x_length, y_length, state_number, policy, greenColor, action)
    ind = 0;
    ratio = 0.5; % adjust the length of arrow
    state_coordinate = zeros(x_length * y_length, 2); % Initialize state_coordinate
    for j = y_length:-1:1       
        for i = 1:x_length      
            r(i, j) = rectangle('Position', [i j 1 1]);
            ind = ind + 1;
            state_coordinate(state_number(ind), :) = [i, j];
            text(i + 0.4, j + 0.5, ['s', num2str(ind)]);
            hold on;
            
            % Calculate bias
            i_bias(ind) = state_coordinate(state_number(ind), 1) + 0.5;
            j_bias(ind) = state_coordinate(state_number(ind), 2) + 0.5;
            
            % Draw policy arrows or state markers
            for kk = 1:size(policy, 2)
                if policy(state_number(ind), kk) ~= 0
                    kk_new = policy(state_number(ind), kk) / 2 + 0.5;
                    drawPolicyArrow(kk, ind, i_bias, j_bias, kk_new, ratio, greenColor, action);                
                end
            end
        end
    end
end


function drawPolicyArrow(kk, ind, i_bias, j_bias, kk_new, ratio, greenColor, action)
    % Obtain the action vector
    action = action{kk};

    % For the non-moving action, draw a circle to represent the stay state
    if action(1) == 0 && action(2) == 0  % Assuming the fifth action is to stay
        MarkerSize = 5+kk_new*6;
        plot(i_bias(ind), j_bias(ind), 'o', 'MarkerSize', MarkerSize, 'linewidth', 2, 'color', greenColor);
        return;
    else
        % Draw an arrow to represent the moving action; note that '-' used when drawing the y-axis arrow ensures consistency with the inverse y-coordinate handling.
        arrow = annotation('arrow', 'Position', [i_bias(ind), j_bias(ind), ratio * kk_new * action(1), - ratio * kk_new * action(2)], 'LineStyle', '-', 'Color', greenColor, 'LineWidth', 2);
        arrow.Parent = gca;
    end
end

% Function to color the obstacles
function colorObstacles(xo, yo, r)
    for i = 1:length(xo)
        r(xo(i), yo(i)).FaceColor = [0.9290 0.6940 0.1250];
    end
end

% Function to color the final state
function colorFinalState(xf, yf, r)
    r(xf, yf).FaceColor = [0.3010 0.7450 0.9330];
end
