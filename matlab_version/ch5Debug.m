% by the Intelligent Unmanned Systems Laboratory, Westlake University, 2024
% 该框架只能用在书本的网格世界中。对于更一般形式的用树表示的拓扑关系不适用；尤其是该框架下状态转移是确定的。
clear 
close all
% Initialize environment parameters
agent_state = [1, 1];
final_state = [2, 2];
obstacle_state = [1,2];
x_length = 2;
y_length = 2;
gamma = 0.9;
state_space = x_length * y_length;     % 状态个数
state=1:state_space;         % 纯为了在网格上标注名字
state_value=zeros(state_space,1);

reward_forbidden = -10;
reward_target = 1;
reward_step = 0;  % 注意根据仿射变化不改变最优策略，通常情况没有用处

% Define actions: up, right, down, left, stay
actions = {[0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]};

% Initialize a cell array to store the action space for each state
action_space = cell(state_space, 1);

% Populate the action space
for i = 1:state_space       
    action_space{i} = actions;
end

number_of_action=5;

policy=zeros(state_space, number_of_action); % policy can be deterministic or stochastic, shown as follows:

% stochastic policy 

for i=1:state_space    %行和为1   即使行和不为1，也可能训练出结果。但是未必收敛 行和为1是确保了贝尔曼方程迭代的收敛性
    policy(i,:)=1/length(actions);         
end
% policy(3,2)=0; policy(3,4)=.4;
% policy(5,5)=0; policy(5,3)=.4;
% policy(7,3)=0 ; policy(7,4)= 0; policy(7,2)= 0; policy(7,1)= 0; policy(7,5)= 1;  % 确定的策略
% policy(6,2)=0; policy(6,3) = 1; policy(6,4) = 0; policy(6,5) = 0; policy (6,1) = 0;

% Initialize the episode
episode_length = 100;

state_history = zeros(episode_length, 2);   % 状态历史的坐标
reward_history = zeros(episode_length, 1);  

% Set the initial state
state_history(1, :) = agent_state;
%%
% 策略迭代(截断策略迭代)
PE_length = 10;

tic

for step = 1:episode_length
    state_value_history(:,step) = state_value;
    % 策略评价
    for k = 1:PE_length
        for si = 1:state_space
            siy = ceil(si/x_length);
            six = si-(siy-1)*x_length;
     
            for ai = 1:length(actions)   % 状态很少 使用parfor反而会降低速度，增大开销。
                q(ai)=q_pi([six,siy], action_space{si}{ai}, x_length, y_length, final_state, obstacle_state, reward_forbidden, reward_target, reward_step, gamma, state_value);
            end
           
            state_value_new(si) = policy(si,:)*q';
            si_q(si,:) = q;
        end
        state_value = state_value_new;
    end
    
    % 策略改进（贪婪）
    for si = 1:state_space
        % [qmax, action_index] = max(si_q(si,:));  % 也可以用迭代完后的状态值再算一遍每一个状态对应的动作值
        
        % siy = ceil(si/x_length);
        % six = si-(siy-1)*x_length;
        % for ai = 1:length(actions)
        %     q(ai)=q_pi([six,siy], action_space{si}{ai}, x_length, y_length, final_state, obstacle_state, reward_forbidden, reward_target, reward_step, gamma, state_value);
        % end
        % si_q(si,:) = q;

        [qmax, action_index] = max(si_q(si,:));
        policy(si,:) = 0;
        policy(si,action_index) = 1;
    end
end

toc

figure_policy(x_length, y_length, agent_state,final_state, obstacle_state, state, policy, actions) %[output:2b4ce430]
figure_stateValue(x_length,y_length,agent_state,final_state,obstacle_state,state_value) %[output:694b8c6b]
figure,plot([0:episode_length-1],state_value_history) %[output:0487adac]
% 第一步的策略是一个很差的策略，导致价值函数并不是单调的。
% 通常，不论是值迭代还是策略迭代，其前几步都不是按照指数收敛的。

%%
% MC basic
PE_length = 10;

for step = 1:episode_length
    state_value_history(:,step) = state_value;
    % 策略评价
    for k = 1:PE_length
        for si = 1:state_space
            siy = ceil(si/x_length);
            six = si-(siy-1)*x_length;
            for ai = 1:length(actions)
                % qan(ai)=q_pi([six,siy], action_space{si}{ai}, x_length, y_length, final_state, obstacle_state, reward_forbidden, reward_target, reward_step, gamma, state_value);
                q(ai)=q_pi_iter([six,siy], action_space{si}{ai}, x_length, y_length, final_state, obstacle_state, reward_forbidden, reward_target, reward_step, gamma, action_space,policy);
            end
            state_value_new(si) = policy(si,:)*q';
            si_q(si,:) = q;
        end
        state_value = state_value_new;
    end
    
    % 策略改进（贪婪）
    for si = 1:state_space
        % [qmax, action_index] = max(si_q(si,:));  % 也可以用迭代完后的状态值再算一遍每一个状态对应的动作值
        
        % siy = ceil(si/x_length);
        % six = si-(siy-1)*x_length;
        % parfor ai = 1:length(actions)
        %     q(ai)=q_pi_iter([six,siy], action_space{si}{ai}, x_length, y_length, final_state, obstacle_state, reward_forbidden, reward_target, reward_step, gamma, action_space,policy);
        % end
        % si_q(si,:) = q;

        [qmax, action_index] = max(si_q(si,:));
        policy(si,:) = 0;
        policy(si,action_index) = 1;
    end
end

figure_policy(x_length, y_length, agent_state,final_state, obstacle_state, state, policy, actions)
figure_stateValue(x_length,y_length,agent_state,final_state,obstacle_state,state_value)
figure,plot([0:episode_length-1],state_value_history)
% 第一步的策略是一个很差的策略，导致价值函数并不是单调的。
% 通常，不论是值迭代还是策略迭代，其前几步都不是按照指数收敛的。
%% DEbug


PE_length = 10;
tic
% 策略评价
for k = 1:PE_length
    for si = 1:state_space
        siy = ceil(si/x_length);
        six = si-(siy-1)*x_length;
        for ai = 1:length(actions)
            qan(ai)=q_pi([six,siy], action_space{si}{ai}, x_length, y_length, final_state, obstacle_state, reward_forbidden, reward_target, reward_step, gamma, state_value);
            q(ai)=q_pi_iter([six,siy], action_space{si}{ai}, x_length, y_length, final_state, obstacle_state, reward_forbidden, reward_target, reward_step, gamma, action_space,policy);
        end
        state_value_new(si) = policy(si,:)*q';
        si_q(si,:) = q;

        state_value_new_qan(si) = policy(si,:)*qan';
        si_q_qan(si,:) = qan;
    end
    state_value = state_value_new;
    state_value_qan = state_value_new_qan;
end
toc
state_value
state_value_qan
%%
    % 策略改进（贪婪）
    for si = 1:state_space
        % [qmax, action_index] = max(si_q(si,:));  % 也可以用迭代完后的状态值再算一遍每一个状态对应的动作值
        
        % siy = ceil(si/x_length);
        % six = si-(siy-1)*x_length;
        % parfor ai = 1:length(actions)
        %     q(ai)=q_pi_iter([six,siy], action_space{si}{ai}, x_length, y_length, final_state, obstacle_state, reward_forbidden, reward_target, reward_step, gamma, action_space,policy);
        % end
        % si_q(si,:) = q;

        [qmax, action_index] = max(si_q(si,:));
        policy(si,:) = 0;
        policy(si,action_index) = 1;
    end
%%

si = 1; ai = 3;
six = 1; siy = 1;
q_pi([six,siy], action_space{si}{ai}, x_length, y_length, final_state, obstacle_state, reward_forbidden, reward_target, reward_step, gamma, state_value)
q_pi_iter([six,siy], action_space{si}{ai}, x_length, y_length, final_state, obstacle_state, reward_forbidden, reward_target, reward_step, gamma, action_space,policy)

%%
function q=q_pi_iter(state, action, x_length, y_length, target_state, obstacle_state, reward_forbidden, reward_target, reward_step, gamma, action_space,policy)
    % 当前的state 选择 action后可以获得的回报的平均值
    [new_state_father, reward_intime] = next_state_and_reward(state, action, x_length, y_length, target_state, obstacle_state, reward_forbidden, reward_target, reward_step);
    % 走一步进入action下的某一个叶子节点（new_state）时，对该叶子节点下面进行蒙特卡洛搜索。如果有多个叶子节点，还需要将这些叶子节点加权做和
    
    n = 20;   % n >  250的时候差不多用parfor会快一点
    reward_future_recorder = 0;
    for iter_episode = 1:n  % 求reward的平均值用
        new_state = new_state_father;
        for iter_deepth = 1:30   % 按照某些策略可能永远都到不了终点，故最好不要用while=终点来结束循环。且越future，贡献越小，后面可以忽略
                                 % 到达一定次数后会收敛
            action = stochastic_policy(new_state, action_space, policy, x_length, y_length);
            [new_state, reward_future] = next_state_and_reward(new_state, action, x_length, y_length, target_state, obstacle_state, reward_forbidden, reward_target, reward_step);
            reward_future_recorder = reward_future_recorder + gamma^iter_deepth*reward_future;
            
            % action_history{iter_deepth} = action;  % debug
            % state_history{iter_deepth}= new_state;  % debug  
        end
    end
    ErewardFuture = reward_future_recorder/n;
    q = reward_intime + ErewardFuture;
end


function q=q_pi(state, action, x_length, y_length, target_state, obstacle_state, reward_forbidden, reward_target, reward_step, gamma, state_value)
        % 在state下执行action。检查是否到达target_state；是否到obstacle_state，如果是则扣分，并呆在该状态；检查是否撞墙，如果撞墙则弹回并扣分。
    new_x = state(1) + action(1);
    new_y = state(2) + action(2);
    new_state = [new_x, new_y];

    % Check if the new state is out of bounds
    if new_x < 1 || new_x > x_length || new_y < 1 || new_y > y_length
        new_state = state;
        reward = reward_forbidden;
    elseif ismember(new_state, obstacle_state, 'rows')
        % If the new state is an obstacle
        reward = reward_forbidden;
    elseif isequal(new_state, target_state)
        % If the new state is the target state
        reward = reward_target;
    else
         % If the new state is a normal cell
        reward = reward_step;
    end
    
    state_1d = x_length * (new_state(2)-1) + new_state(1);   % 转到s_i+1编号的状态
    Gt1 = gamma*state_value(state_1d);   % determinatstic 的转移

    q = reward + Gt1;
end

%%   useful function 
function [new_state, reward] = next_state_and_reward(state, action, x_length, y_length, target_state, obstacle_state, reward_forbidden, reward_target, reward_step)
    % 在state下执行action。检查是否到达target_state；是否到obstacle_state，如果是则扣分，并呆在该状态；检查是否撞墙，如果撞墙则弹回并扣分。
    new_x = state(1) + action(1);
    new_y = state(2) + action(2);
    new_state = [new_x, new_y];

    % Check if the new state is out of bounds
    if new_x < 1 || new_x > x_length || new_y < 1 || new_y > y_length
        new_state = state;
        reward = reward_forbidden;
    elseif ismember(new_state, obstacle_state, 'rows')
        % If the new state is an obstacle
        reward = reward_forbidden;
    elseif isequal(new_state, target_state)
        % If the new state is the target state
        reward = reward_target;
    else
         % If the new state is a normal cell
        reward = reward_step;
    end
end

function action = stochastic_policy(state, action_space, policy, x_length, y_length)
    % 模仿一步随机过程。在state下，从动作空间action_space中按照policy获取一个具体的action。
    % Extract the action space and policy for a specific state
    state_1d = x_length * (state(2)-1) + state(1);   % 转到s_i编号的状态
    actions = action_space{state_1d};                % 第i个status可以做的动作
    policy_i = policy(state_1d, :);

    % Ensure the sum of policy probabilities is 1
    assert(sum(policy_i) == 1, 'The sum of policy probabilities must be 1.');
    
    % Generate a random index based on policy probabilities
    action_index = randsrc(1, 1, [1:length(actions); policy_i]);
    
    % Select an action
    action = actions{action_index};
end


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":37.1}
%---
