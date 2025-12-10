function figure_stateValue(x_length, y_length, agent_state,final_state, obstacle_state, state_value)
    %% Inverse y coordinate
 
    
    xa_used = agent_state(:, 1) + 0.5;
    ya_used = y_length+1-agent_state(:, 2) + 0.5;
    
    
    state_space=x_length*y_length;
    
    
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
    
    % Draw the grid and add state values
    r = drawGridAndStateValues(x_length, y_length, state_value);
    
    % Color the obstacles and the final state
    colorObstacles(xo, yo, r);
    colorFinalState(xf, yf, r);
    
    
    % Draw the agent
    % agent = plot(xa_used, ya_used, '*', 'markersize', 15, 'linewidth', 2, 'color', 'b');
    % hold on;
    
    
    % Set axis properties and export the figure
    axis equal;
    axis off;
    exportgraphics(gca, 'trajectory_Bellman_Equation.pdf');

end

% Function to draw the grid and add state values
function r = drawGridAndStateValues(x_length, y_length, state_value)
    ind = 0;
    for j = y_length:-1:1       
        for i = 1:x_length       
            r(i, j) = rectangle('Position', [i j 1 1]);
            ind = ind + 1;
            text(i + 0.4, j + 0.5, num2str(round(state_value(ind), 2)));
            hold on;           
        end
    end
end
function addAxisLabels(x_length, y_length)
    for i = 1:x_length
        text(i + 0.5, y_length + 1.1, num2str(i));
    end
    for j = y_length:-1:1
        text(0.9, j + 0.5, num2str(y_length - j + 1));
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