% by the Intelligent Unmanned Systems Laboratory, Westlake University, 2024
% 该框架只能用在书本的网格世界中。对于更一般形式的用树表示的拓扑关系不适用；尤其是该框架下状态转移是确定的。
clear 
close all
% % Initialize environment parameters
% agent_state = [1, 1];
% final_state = [3, 4];
% obstacle_state = [2,2; 3,2; 3,3; 2,4; 4,4; 2,5];
% x_length = 5;
% y_length = 5;
% Initialize environment parameters
agent_state = [1, 1];
final_state = [2, 2];
obstacle_state = [1,2];
x_length = 2;
y_length = 2;
% Initialize environment parameters
% agent_state = [1, 1];
% final_state = [3, 3];
% obstacle_state = [1,2;3,2];
% x_length = 3;
% y_length = 3;

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
% MC basic  如果不收敛应该是迭代步数不够。越大的地图需要越多的步数
tic
PE_length = 10;

for step = 1:episode_length
    state_value_history(:,step) = state_value;
    % 策略评价
    for k = 1:PE_length
        for si = 1:state_space
            siy = ceil(si/x_length);
            six = si-(siy-1)*x_length;
            for ai = 1:length(actions)
                q(ai)=q_pi([six,siy], action_space{si}{ai}, x_length, y_length, final_state, obstacle_state, reward_forbidden, reward_target, reward_step, gamma, state_value);
                % q(ai)=q_pi_iter([six,siy], action_space{si}{ai}, x_length, y_length, final_state, obstacle_state, reward_forbidden, reward_target, reward_step, gamma, action_space,policy);
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
toc %[output:6facfa81]
figure_policy(x_length, y_length, agent_state,final_state, obstacle_state, state, policy, actions) %[output:1ad117a5]
figure_stateValue(x_length,y_length,agent_state,final_state,obstacle_state,state_value) %[output:8fd57dc5]
figure,plot([0:episode_length-1],state_value_history) %[output:86149756]
% 第一步的策略是一个很差的策略，导致价值函数并不是单调的。
% 通常，不论是值迭代还是策略迭代，其前几步都不是按照指数收敛的。
%%
%[text] ## MC egreedy
% MC egreedy 算法实现
% MC egreedy 算法实现
tic
% 初始化参数
episode_count = 100;  % 总回合数
Tlength = 3000;  % 最大回合长度
gamma = 0.9;
epsilon = 0;  % ε-greedy参数

% 这段代码有一个非常神奇的现象，如果epsilon=0，那么这段代码将无法收敛。
% 但是epsilon = 0.001时，这段代码将会收敛到一个非常好的结果。
% 这是因为epsilon接近贪婪策略时，导致算法固定选择一个动作，无法探索其他动作，从而不收敛到最优
% debug可以看到，当epsilon = 0时策略很快就收敛到了固定不变的情况
% 如果epsilon = 0.2时，又很难收敛。这是因为随机性太强了，扰乱了较好的训练结果

% epsilon=0时想实现收敛，必须把策略更新放到策略评估循环外边。即每一个episode更新一次策略。否则策略收敛过快

% 初始化数据结构
% 对于每个状态-动作对，记录回报和访问次数
Q = zeros(state_space, number_of_action);       % 动作价值函数
V = zeros(state_space, 1);                      % 状态价值函数

% 初始策略：均匀随机策略
policy = ones(state_space, number_of_action) * (1/number_of_action);

% 主循环
for episode = 1:episode_count
    % 临时存储当前回合的数据  每一个循环（生成新的路径之前）都要清空一次
    Return = zeros(state_space, number_of_action); % 累计回报
    Number = zeros(state_space, number_of_action); % 访问次数
    
    % 选择初始状态-动作对
    s = 1;  % 从第一个状态开始
    [a_idx, a] = stochastic_policy([1,1], action_space, policy, x_length, y_length);
    
    % 存储轨迹
    pairs = zeros(Tlength, 2);  % 存储状态-动作对
    rewards = zeros(Tlength, 1);  % 存储奖励
    visits = zeros(state_space, number_of_action);  % 访问计数
    
    s_temp = s;
    %%%%%%%%%%% 生成轨迹 %%%%%%%%%
    for i = 1:Tlength
        % 执行动作，获取奖励和下一个状态
        [next_state, reward] = next_state_and_reward([mod(s_temp-1, x_length)+1, floor((s_temp-1)/x_length)+1], ...
            action_space{s_temp}{a_idx}, x_length, y_length, final_state, ...
            obstacle_state, reward_forbidden, reward_target, reward_step);
        
        % 存储当前状态-动作对和奖励
        pairs(i, 1) = s_temp;
        pairs(i, 2) = a_idx;
        rewards(i) = reward;
        
        % 更新访问计数
        visits(s_temp, a_idx) = visits(s_temp, a_idx) + 1;
        
        % 检查是否到达终止状态
        next_state_1d = x_length * (next_state(2)-1) + next_state(1);
        
        % 选择下一个动作
        s_temp = next_state_1d;
        [a_idx, a] = stochastic_policy(next_state, action_space, policy, x_length, y_length);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % 实际轨迹长度
    actual_length = i;
    
    %%%%%%%%%%%%% 策略评估 - 每次访问方法
    g = 0;
    for t = actual_length:-1:1
        s_idx = pairs(t, 1);
        a_idx = pairs(t, 2);
        
        % 更新累积回报
        g = gamma * g + rewards(t);
        
        % 更新回报和访问次数
        Return(s_idx, a_idx) = Return(s_idx, a_idx) + g;
        Number(s_idx, a_idx) = Number(s_idx, a_idx) + 1;
        
        % 策略评估：更新Q值
        if Number(s_idx, a_idx) > 0
            Q(s_idx, a_idx) = Return(s_idx, a_idx) / Number(s_idx, a_idx);
        end
        
        % 策略改进：ε-greedy策略更新
        % 找到当前状态下的最优动作
        [~, a_star] = max(Q(s_idx, :));
        
        % 计算ε-greedy策略
        for a = 1:number_of_action
            if a == a_star
                policy(s_idx, a) = 1 - epsilon * (number_of_action - 1) / number_of_action;
            else
                policy(s_idx, a) = epsilon / number_of_action;
            end
        end
        
        % 更新状态价值函数  当然这里不严谨，应该用policy乘以Q然后相加.但是用动态epsilon（收敛到ε=0的情况时是可以这样的）
        V(s_idx) = max(Q(s_idx, :));   % 如果用egreedy策略，最终收敛到的价值和贪婪策略最终收敛到的肯定不一样
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
toc
% 显示最终结果
fprintf('\n=== MC Exploring Starts 算法完成 ===\n');
fprintf('总回合数: %d\n', episode_count);
fprintf('最终策略:\n');

% 显示每个状态的策略
for s = 1:state_space
    s_y = ceil(s/x_length);
    s_x = s - (s_y-1)*x_length;

    % 找到最优动作
    [~, best_action] = max(policy(s, :));

    % 将动作索引转换为方向
    action_names = {'上', '右', '下', '左', '停'};

    fprintf('状态(%d,%d): 最优动作 = %s, Q值 = [', s_x, s_y, action_names{best_action});

    for a = 1:number_of_action
        fprintf('%.3f ', Q(s, a));
    end
    fprintf(']\n');
end

% 可视化最终策略和价值函数
figure_policy(x_length, y_length, agent_state, final_state, obstacle_state, state, policy, actions);

% 将Q值转换为状态值函数（选择最大Q值）
state_value_final = max(Q, [], 2);
figure_stateValue(x_length, y_length, agent_state, final_state, obstacle_state, state_value_final);

% 绘制学习曲线：显示每个状态-动作对的访问次数  这局限于1条路径的展示。
figure;
imagesc(Number);
colorbar;
xlabel('动作');
ylabel('状态');
title('状态-动作对访问次数');
set(gca, 'YTick', 1:state_space);
set(gca, 'XTick', 1:number_of_action);
xticklabels({'上', '右', '下', '左', '停'});

% 绘制Q值热图
figure;
imagesc(Q);
colorbar;
xlabel('动作');
ylabel('状态');
title('动作价值函数 Q(s,a)');
set(gca, 'YTick', 1:state_space);
set(gca, 'XTick', 1:number_of_action);
xticklabels({'上', '右', '下', '左', '停'});


%%
function q=q_pi_iter(state, action, x_length, y_length, target_state, obstacle_state, reward_forbidden, reward_target, reward_step, gamma, action_space,policy)
    % 当前的state 选择 action后可以获得的回报的平均值
    [new_state_father, reward_intime] = next_state_and_reward(state, action, x_length, y_length, target_state, obstacle_state, reward_forbidden, reward_target, reward_step);
    % 走一步进入action下的某一个叶子节点（new_state）时，对该叶子节点下面进行蒙特卡洛搜索。如果有多个叶子节点，还需要将这些叶子节点加权做和
    
    n = 50;
    reward_future_recorder = 0;
    for iter_episode = 1:n  % 求reward的平均值用
        new_state = new_state_father;
        for iter_deepth = 1:30   % 按照某些策略可能永远都到不了终点，故最好不要用while=终点来结束循环。且越future，贡献越小，后面可以忽略
            [~,action] = stochastic_policy(new_state, action_space, policy, x_length, y_length);
            [new_state, reward_future] = next_state_and_reward(new_state, action, x_length, y_length, target_state, obstacle_state, reward_forbidden, reward_target, reward_step);
            reward_future_recorder = reward_future_recorder + gamma^iter_deepth*reward_future;
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

function [action_index,action] = stochastic_policy(state, action_space, policy, x_length, y_length)
    % 模仿一步随机过程。在state下，从动作空间action_space中按照policy获取一个具体的action。
    % Extract the action space and policy for a specific state
    state_1d = x_length * (state(2)-1) + state(1);   % 转到s_i编号的状态
    actions = action_space{state_1d};                % 第i个status可以做的动作
    policy_i = policy(state_1d, :);

    % Ensure the sum of policy probabilities is 1   经常有数值误差，所以注释掉
    % assert(sum(policy_i) == 1, 'The sum of policy probabilities must be 1.');
    
    % Generate a random index based on policy probabilities
    action_index = randsrc(1, 1, [1:length(actions); policy_i]);
    
    % Select an action
    action = actions{action_index};
end




%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":32.2}
%---
%[output:6facfa81]
%   data: {"dataType":"text","outputData":{"text":"历时 0.075240 秒。\n","truncated":false}}
%---
%[output:1ad117a5]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXQAAADgCAYAAAAT452yAAAAAXNSR0IArs4c6QAAHYpJREFUeF7tnQ+QVWd1wM+SBNj8gbBCDX9CEwyGqakZJcI6yVg1nWpr1kZoS8JMjAyxi1HEmZJhmTFq1SlLQzomaOo6YZiYlogOJoqxtTPWFLVlMElHGy0OJpB1WRJLViEmkH\/7Ot+L9\/H28d7ee757v\/O+C783k1Gy53znu7\/v8uPkvPseHZVKpSK8IAABCECg9AQ6EHrpz5ALgAAEIFAlgNC5EYIRGBkZkZUrV8q6deuku7s7WJ3TeeGNGzfKwMBADUFvb2+VN6\/TkwBCPz3PPfhVHzt2TPr6+mTnzp2ybds2hF4w8YTv4OCgbNmyRbq6uiT5A3Tu3LnS398vnZ2dBVdludgJIPTYT6iE+9u3b5+sWLFChoeHq7tH6MUfYsJ406ZNY\/6w3L17t6xdu1a2bt0q8+fPL74wK0ZNAKFHfTzl21wimp6eHlmyZElV7I3SKd9VlWfHTujLly\/nD9HyHFmhO0XoheJksXoCrbpIKIUjsH37dtm8eTMdejjEUa+M0KM+nnJvDqHbnl\/9fx3xxqgt+1iqIfRYTuIU3AdCtzvU5A1RVzF5k9SuOpViIYDQYzmJU3AfCN3mUJG5DecyVEHoZTilku4RoYc\/uITxjBkz6MzD446+AkKP\/ojKu0GEHvbsEr4LFy7kufOwqEuzOkIvzVGVb6MIPdyZ8SGicGzLvDJCL\/PpRb53hB7ugNzjievXr29ZYMOGDbJs2bJwG2DlKAkg9CiPhU1BAAIQ0BNA6HpmZEAAAhCIkgBCj\/BY3Me3d+zYEeHO2BIE\/AmsWbNG5syZ478AmakEEHoqIvuAefPmVW\/82bNn2xc\/jSoePHhQhoaGZPHixafRVbfnUh1r99q1a1d7NnCaVEXoER60E\/ptt90mS5cujXB3p86Wki+ycpKhcwx7rnfccUf1vzoReljOCD0sX6\/VEboXNnUSQlcj805A6N7oVIkIXYXLJhih23BG6DacXRWEbsMaodtwVlVB6Cpc3sEI3RudOhGhq5F5JSB0L2xhkxB6WL7J6gjdhjMduh1nhG7HOnMlhJ4ZVa5AhJ4LnyqZDl2FyzsYoXujC5eI0MOxrV8ZodtwpkO344zQ7VhnroTQM6PKFYjQc+FTJdOhq3B5ByN0b3ThEhF6OLZ06DZsG6sgdBvuCN2Gs6oKQlfh8g6mQ\/dGp05E6GpkXgkI3Qtb2CSEHpZvsjpCt+HMDN2OM0K3Y525EkLPjCpXIELPhU+VTIeuwuUdjNC90YVLROjh2DJDt2HLDL09nBF6e7iPWxWh2xwKHboNZ0YudpwRuh3rzJUQemZUuQIRei58qmRGLipc3sEI3RtduESEHo4tIxcbtoxc2sMZobeHOyOXCLjTodsdAh26DWuEbsNZVYUOXYXLOxihe6NTJyJ0NTKvBITuhS1sEkIPyzdZHaHbcHZVELoNa4Ruw1lVBaGrcHkHI3RvdOpEhK5G5pWA0L2whU1C6GH50qHb8K2vgtBtmCN0G86qKghdhcs7mA7dG506EaGrkXklIHQvbGGTEHpYvnToNnzp0O05I3R75qkVEXoqokIC6NALwZhpETr0TJhyByH03AiLXwChF8+02YoI3Yazq4LQbVgjdBvOqioIXYXLOxihe6NTJyJ0NTKvBITuhS1sEkIPy5cZug1fZuj2nBG6PfPUigg9FVEhAXTohWDMtAgdeiZMuYMQem6ExS+A0ItnygzdhmmrKgjdhj9Ct+GsqoLQVbi8g+nQvdGpExG6GplXAkL3whY2CaGH5csM3YYvM3R7zgjdnnlqRYSeiqiQADr0QjBmWoQOPROm3EEIPTfC4hcom9APHBC5+OITHPbvF7noouK5FL0iQi+aaOv1ELoNa4Ruw1lVBaGrcHkHI3RvdOpEhK5G5pWA0L2whU2KVeiuE2\/WeWfp0F2Me8XUuSP0sPcxM3Q7vkklhG7PPLVijEJPpP32t4t873tjLyFN6O7nK1aIPPTQq7lujRheCN3uFOjQbVgjdBvOqiqxCb1R2I1SH0\/o9TJPIMQidYSuui1zBSP0XPgyJyP0zKjsAmMTurvyd7zj1Q47edVLvZXQm8ncjVzcm6ZFvA4fOSjunwVzF3kth9C9sHklIXQvbOokhK5GVnzCyMiIrFy5UtatWyfd3d0So9DHk3ozobv4q68elCeemFsDVqTM3aI\/eOx+ufvbfTJ96my56rL3ybVXrlYdTtmFnuw\/uejLL79ctmzZIl1dXSoOFsEI3YKyCEK34dyyyrFjx6Svr0927twp27Zti1roraS+devYxxbdSOXDH\/6V\/OxnvxdM5m5h1507oe\/95Z5aHY3cyyz07du3y\/r162v3jAOwcePG6n20detWmT9\/fpvv7LHlEbrNcSB0G85Nq+zbt09WrFghw8PD1Z+XQeitpF4\/jpkyZUSOHj3RJRbdmdfDdFJfO\/DOk\/gmYl9w4eKWI5myCj1pAmbNmlX9r7rklfyX3nXXXSfLli1r4519cmmEbnMcCN2G80lVEpn39PTIkiVLqmLftGlT1B16\/R9Ahw5tk+PHu1PpTZt2REZGpqbG5QloJfVkzVZde8xCb\/zDvv4P\/FasEqG7sV296POwLSoXoRdFcvx1ELoN53GrJL95YxN68uy42\/yBAwfkllvWyo03fkCuueaa6vUsXXpUHn10Su3azp128KTrXLfui7Jq1arglN3YxY1f0l71co9V6Mn9sHr16lqn7fa6du3acccpzfLSeFj9HKHbkEboNpxLKfSODh2cpX3vlGZS161iE+3EftH5i+Suv\/ua\/NuD\/yFz5syxKZyhShZ5Ny6TjGEGBwejfGMUoWc4+AJCEHoBEPMuEWuHfioLPTmzJx9+Re79h4eiEnoyOvnxj38svb29Y8Yn7smew0eGqtuvf3\/AvSE6MDAw5k3SvPdlkfkIvUiarddC6Dac6dAj4Fy\/Bdehn9sxW77x5f+UnV+Jq0NP9plIOvm1k3tl7mO1p3quvfIj1Uc1Y5e52z9Ct\/kNgNBtOJdS6PUz9MYLcD9rfDSxcdzinnb50Ie+ajJDd2+M9n\/lhtTTXHDhIrnqD5dUn1uPdYbe7CKSxxTfvepCeV6eqoa8Z9Eq+cH9B8Y88poKoE0BCN0GPEK34VxKobfadLNPgLaKnTdvUB5\/\/MSHi0LgTmTu\/rfZK3kj9KrLllQ\/hJS8yiT0ZEb+3PQ98vKkkeolTDp6ifz0oSNRPnfeeA4IPcSdf\/KaCN2Gc+mFnsjv9tt3yJ13vmnM1wA0Xpx77ry+u2\/2hV5FYW8lcyfuajd+2ZLSPYeesE4+l+BYJe+zLPqrc2Tk+KtfXTny87Pl9r6vRvchomZni9CLuuPHXweh23AuvdDdBdx\/\/3\/L8uUvjHn+fMaM52TPnnNO+gsu3v\/+Ufn+9yfUrjuE1JvJvH6kkna0MXfojR\/rd9eyYcMG2T\/6rdoM3b2h++TDoyddpvtsQ39\/v3R2dqYhMPs5QrdBjdBtOKuqxPhdLuN90VarL+ca7wu9VEBaBD\/ww83ywA8\/X\/sul8aRSlqNmIXeau\/9991w0puiadcZw88Rus0pIHQbzqoqsQk97VsTx\/v63NBS3zu457T6tkWErvqtdNoFI\/QIjzw2obvvaXFiTl6N382S9hdc1Evd5bov84rhL7mgQ7e7+enQbVgjdBvOqiqxCd1tPpF6sy\/aShO6y3dSd3GxyNztCaGrbstcwQg9F77MyQg9Myq7wBiFnki9WWedReguv9XfSWpHdmwlhG5HHqHbsEboNpxVVWIVequLyCp0FQSDYIRuAPl3JRC6DWuEbsNZVQWhq3B5ByN0b3TqRISuRuaVgNC9sIVNKpvQw9IItzpCD8e2cWWEbsMaodtwVlVB6Cpc3sEI3RudOhGhq5F5JSB0L2xhkxB6WL7J6gjdhrOrgtBtWCN0G86qKghdhcs7GKF7o1MnInQ1Mq8EhO6FLWwSQg\/Llw7dhm99FYRuwxyh23BWVUHoKlzewXTo3ujUiQhdjcwrAaF7YQubhNDD8qVDt+FLh27PGaHbM0+tiNBTERUSQIdeCMZMi9ChZ8KUOwih50ZY\/AIIvXimzVZE6DacXRWEbsMaodtwVlVB6Cpc3sEI3RudOhGhq5F5JSB0L2xhkxB6WL7M0G34MkO354zQ7ZmnVkToqYgKCaBDLwRjpkXo0DNhyh2E0HMjLH4BhF48U2boNkxbVUHoNvwRug1nVRWErsLlHUyH7o1OnYjQ1ci8EhC6F7awSQg9LF9m6DZ8maHbc0bo9sxTKyL0VESFBNChF4Ix0yJ06Jkw5Q5C6LkRFr8AQi+eaRln6Hd\/u08OHzkofdffW9t+\/303yN5f7qn++torPyLXXrm6+v9dXP9Xbqj++qrL3mcDUFEFoStg5QhF6DnghUpF6KHIjl039g7dCf0Hj90vCy5cVJN6M6E7ma8deGf14m76s36EbnP7RFkFoUd4LAjd5lBiF7qTuZO6e02fOls29f67NAr9qsuW1GSexNjQ01WhQ9fx8o1G6L7kAuYh9IBw65aOXehuq67zdh14IvVkvJL8OvmZ+3V9J29DMHsVhJ6dVZ5IhJ6HXqBchB4IbMOyZRB6MnbJQqTvuntlwdxFWULNYxC6DXKEbsNZVQWhq3B5B5dB6HsH91Tf7Ex7xTxucXtH6GknWMzPEXoxHAtdBaEXirPlYmUQeuPYpdXFuCdb3Buisb4Qus3JIHQbzqoqCF2Fyzu4LEKvf3O01cXGPG6hQ\/e+RdWJCF2NLHwCQg\/P2FUoi9DdXj\/w95e2hBLzm6HJpunQbe5phG7DWVUFoatweQeXSejjvTka67Pn9QeD0L1vU1UiQlfhsglG6DacyyT0VmOX2N8MpUO3uZeTKgjdlnemagg9E6bcQWUSurvY+mfSk4svw7iFGXruWzXzAgg9Myq7QIRuw7psQm82don9zVA6dJt7mQ7dlrOqGkJX4fIOLpvQG59JL8u4hQ7d+xZVJ9Khq5GFT0Do4Rm7CmUTeuPYJfZnz+tPkTdFbe5phG7DWVUFoatweQeXUej1b46WZdxCh+59i6oTEboaWfgEJ\/TFixfLnDlzwhc7zSvs2LFDli5dWioK\/zfjm3JkuCKXnPXnpdn30NCQHDx4UHbt2lWaPZdxowg9wlNzQp\/ZJTKrK8LNnUJbGh4ROTQics6lC0t1VZPP65Djz1ZKtecXnzkkr508AaEHPjWEHhiwz\/JO6J9cXpGeOL84z+eSosx55BcivZ\/vkNdv3CkTp8+Kco+nyqZ+9c0BmfjItxF64ANF6IEB+yyP0H2o6XMQup6ZbwZC9yWny0PoOl4m0QjdBLMgdBvOrgpCt2GN0G04q6ogdBUu72CE7o1OnYjQ1ci8EhC6F7awSQg9LN9kdYRuw5kO3Y4zQrdjnbkSQs+MKlcgQs+FT5VMh67C5R2M0L3RhUtE6OHY1q+M0G0406HbcUbodqwzV0LomVHlCkToufCpkunQVbi8gxG6N7pwiQg9HFs6dBu2jVUQug13hG7DWVUFoatweQfToXujUycidDUyrwSE7oUtbBJCD8s3WR2h23Bmhm7HGaHbsc5cCaFnRpUrEKHnwqdKpkNX4fIORuje6MIlIvRwbJmh27Blht4ezgi9PdzHrYrQbQ6FDt2GMyMXO84I3Y515koIPTOqXIEIPRc+VTIjFxUu72CE7o0uXCJCD8eWkYsNW0Yu7eGM0NvDnZFLBNzp0O0OgQ7dhjVCt+GsqkKHrsLlHYzQvdGpExG6GplXAkL3whY2CaGH5ZusjtBtOLsqCN2GNUK34ayqgtBVuLyDEbo3OnUiQlcj80pA6F7YwiYh9LB86dBt+NZXQeg2zBG6DWdVFYSuwuUdTIfujU6diNDVyLwSELoXtrBJCD0sXzp0G7506PacEbo989SKCD0VUSEBdOiFYMy0CB16Jky5gxB6boTFL4DQi2fabEWEbsPZVUHoNqwRug1nVRWErsLlHYzQvdGpExG6GplXAkL3whY2CaGH5csM3YYvM3R7zgjdnnlqRYSeiqiQADr0QjBmWoQOPROm3EEIPTfC4hdA6MUzZYZuw7RVFYRuwx+h23BWVUHoKlzewXTo3ujUiQhdjcwrAaF7YQubhNDD8mWGbsOXGbo9Z4Ruzzy1IkJPRVRIAB16IRgzLUKHnglT7iCEnhth8Qsg9OKZMkO3YcoMvb2cEXp7+TetjtBtDqVsHfqEFw7JhBcP1eC8fN6bbUAVUIUOvQCIGZZA6BkgWYcgdBviZRC6k\/ikww\/Kmc8+Kmc9++hJYEYnzZQXXvMeOTb7JhtonlUQuic4ZRpCVwKzCEfoFpRFYhZ6IvLO4bszwzg266ZoxY7QMx9jrkCEngtfmGSEHoZr46qxCt3J3Incdeba10vnvVmeXXCXNi14PEIPjrhaAKHbcFZVQejZcI281CEjL4lccnYlW0JDVIxCb9aZu7GKE7Wbmbv\/db9O5ulnHX20Kv\/6V4xSR+het6g6CaGrkRWXsHHjRhkYGKgt2NvbK+vWrZMyC\/2Jpyry0YEOeerXHdXretebRuXW6ztk8sTiuCUr\/ejIBLnv0JnSdVZF3jJ1VN41\/RVVkRiF7rryc\/Z\/pnYdTt5HL72rKvFWLyf38\/53lZzx0tO1kNjGLwhddWt6ByN0b3T+iceOHZO+vj4ZHByULVu2SFdXl4yMjMjKlStl7ty5snPnTvnk8or0LPKv0Y7MROY3\/UlFrn1rhxx\/UeQz91Vk6JkO+dxfV2Taua9KvqiX69DvO3SGPP78hNqSGrnHJnQn5vN\/8j6VzJPgZx7YIBdOeUAmn3eC8W\/eeP+4fxAUdQ5Z1kHoWSjlj0Ho+RmqV9i3b5+sWLFCNm3aJN3d3bX83bt3y9q1a2V4eLiUQt+8U+SpkcqYjjyR\/KeWV+SK+cUK3YFzUv\/s42eddAaJ2F939mjLkUxsQu88eHdtfOI68t9efGt1zJL2+u3eh+XAbb0ydVaHXP7eM2vhL0x\/jzx38a1p6SY\/R+gmmJmh22DOVsUJffny5dXgGDv0xnGK2+cXPzw6rqhDC308qSfUW3XtsQn9vL031x5NfGHiH8gj\/\/S0vDzyVO3mueiWATl3wRVjbqaXn\/21PHnHGpn+7vfL4X\/9slz6x5Nkyhn\/U4sZecvubDdf4CiEHhjw75anQ7fhnKnK9u3bZfPmzVF26I3jFHdBD++ryKe2dcidvRWZd4F7g3JsB37kuYp8+r4OmTmtIqvf2yGTTm6kM3HJEvT48278cqI7bZVTL\/eYhN44bnnse1NkUvfN8pq3XVu9FNeFD235pFz0sc0yefa86r8bffG4DG39tEx8zQUy\/V03VMU+9x3z5bUTTzwdE8vYBaFnuYvzxyD0\/AwLWSEZw\/T09FTfKI2tQ2+Ud7OLdqOPRqkXAifAIk7ss18ZlS33VmTux78pE6fPClAl+5Lug0NT9t5cS3jkwWky84NfqMm72UrP7HpAfrPr6\/L7a+6o\/tgJ\/fy3LZH5nf218KML7so0tsm+U79IhO7HTZuF0LXEAsQnb4i6pd2bpFdccUV0Qv\/1byvysS91yE8HO+TGqyuyuudkEGUSerL7Jx9+RSb95TeiE\/qPvvN6Obb\/pzL9T2+UC\/7ioyfBPn7wCTnwudUyZ+XfVscwyegFoQf4DVqiJRF6mw+rUebuiZeYH1t0b3ze890To5V6uZdJ6K5DP\/t4RR58cFReuya+Dt2NSoZ3fk0O\/8s9tTs0kXv9qCWRfSL0GX90tVw0+R\/p0Nv8+7pd5RF6u8iLSDJmmTFjRu3xRbedmIVej+uB\/6rIZ7dPqHXsrcYt93xX5Cf7RT5xfUWmnlP8ky5uT+4DRncNpg\/p3VMvi6aOVp9bj2mG7q6h60cnnnhyT6e4p1SSlxuvHLrnM9WO\/fy3XlPtzuvfME3iGp90YYbext\/gbSiN0NsA3ZVMZL5w4ULp7++Xzs7O2k7KIvTkOXO38b9ZUpHbv94hF3R1jBnH1MeE+oCR+4PkrsEzW87vkzdCncTd\/09esQndPYPu3hx1r8ZHDpOu3P1szopPyISJk8fcuUmHXv+Ui3v00Qk9hhczdJtTQOg2nMdUqf8QUaPMY+3Q3Zuiq74wYcxjio1PvjSLSUY0aY83+h5DK5k7cb\/ubPcJ0ldK8xy6+4Ro\/fe3PH32GjnrDddX0SQz8xk9H6w9+VLPzAn9qS\/dLG982\/7av+Y5dN+7qrx5CL0NZ+ceT1y\/fv24lWN7ysVtNhF2\/cY\/vmy0+qnQ5NX4rPob5laCfErU1Wsm8\/qRStrRxtahu\/3Wj12OP1uRPf\/8cu0yZt54a1OZu4DKyF4597EPyqTOl6rxWb4yII1PkT+nQy+SZuu1ELoNZ1WVsoxcVBcVIPg7h88Q90+rkUpayRiFXv9p0UTMaZ8YdWOaKT+\/uTaucXl8l0va6Z+aP0foEZ4rQs9+KL94vuOU+rZFd+X1nxhNSLjxifsn+SoAJ3H3F16459cbv2aXb1vMfv+capEIPcITReg2hxJjh55ceTOpZ6ES09y8fr+MXLKcXv4YhJ6fYeErIPTCkTZdMGahuw03jl\/GoxL7X0WH0G3uaYRuw1lVBaGrcHkHxy50d2FpfxVdInLXmY\/3nenekApKROgFgUxZBqHbcFZVQegqXN7BZRB6\/cUlf0tR8u9GJ86MWuKMXLxvTe9EhO6NLlwiQg\/Htn7lsgndhkqYKnToYbg2rorQbTirqiB0FS7vYITujU6diNDVyLwSELoXtrBJCD0s32R1hG7D2VVB6DasEboNZ1UVhK7C5R2M0L3RqRMRuhqZVwJC98IWNgmhh+VLh27Dt74KQrdhjtBtOKuqIHQVLu9gOnRvdOpEhK5G5pWA0L2whU1C6GH50qHb8KVDt+eM0O2Zp1ZE6KmICgmgQy8EY6ZF6NAzYcodhNBzIyx+AYRePNNmKyJ0G86uCkK3YY3QbTirqiB0FS7vYITujU6diNDVyLwSELoXtrBJCD0sX2boNnyZodtzRuj2zFMrIvRURIUE0KEXgjHTInTomTDlDkLouREWvwBCL54pM3Qbpq2qIHQb\/gjdhrOqCkJX4fIOpkP3RqdOROhqZF4JCN0LW9gkhB6WLzN0G77M0O05I3R75qkVEXoqokIC6NALwZhpETr0TJhyByH03AiLXwChF8+UGboNU2bo7eWM0NvLv2l1hG5zKHToNpxdFTp0G9YI3YazqgpCV+HyDkbo3ujUiQhdjcwrAaF7YQubhNDD8k1WR+g2nOnQ7TgjdDvWmSsh9MyocgUi9Fz4VMl06Cpc3sEI3RtduESEHo5t\/coI3YYzHbodZ4RuxzpzJYSeGVWuQISeC58qmQ5dhcs7GKF7owuXiNDDsaVDt2HbWAWh23BH6DacVVUQugqXdzAdujc6dSJCVyPzSkDoXtjCJiH0sHyT1RG6DWdm6HacEbod68yVEHpmVLkCEXoufKpkOnQVLu9ghO6NLlwiQg\/Hlhm6DVtm6O3hjNDbw33cqgjd5lDo0G04M3Kx44zQ7VhnroTQM6PKFYjQc+FTJTNyUeHyDkbo3ujCJSL0cGwZudiwZeTSHs4IvT3cGblEwJ0O3e4Q6NBtWCN0G86qKnToKlzewQjdG506EaGrkXklIHQvbGGTnNAXXiIysytsHVYX+dYekfOv7AFFYAIvHR6Wac89Lbt27Qpc6fReHqFHeP5DQ0Nyyy23RLgztgQBfwJr1qyR7u5u\/wXITCWA0FMREQABCECgHAQQejnOiV1CAAIQSCWA0FMREQABCECgHAQQejnOiV1CAAIQSCWA0FMREQABCECgHAQQejnOiV1CAAIQSCWA0FMREQABCECgHAQQejnOiV1CAAIQSCWA0FMREQABCECgHAQQejnOiV1CAAIQSCWA0FMREQABCECgHAQQejnOiV1CAAIQSCWA0FMREQABCECgHAQQejnOiV1CAAIQSCWA0FMREQABCECgHAQQejnOiV1CAAIQSCWA0FMREQABCECgHAQQejnOiV1CAAIQSCWA0FMREQABCECgHAQQejnOiV1CAAIQSCWA0FMREQABCECgHAQQejnOiV1CAAIQSCWA0FMREQABCECgHAQQejnOiV1CAAIQSCXw\/8klHo3F+cbFAAAAAElFTkSuQmCC","height":179,"width":298}}
%---
%[output:8fd57dc5]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXQAAADgCAYAAAAT452yAAAAAXNSR0IArs4c6QAAFqtJREFUeF7tnE9ondl1wK+G1oxoCjOamEw8Qk1N3QSyGKjdWEMh2yxaQWoHPPVOCKpF8TgLDZIWbZp0YWlGiziaRdVWeOeOoG5DVQrpUpuYkFBmV1AJrZEVt8loStugYRqi8mn4lCfxnr57zr3nvHuln1cT6fy53+8+\/3R83lNGDg4ODgJ\/IAABCECgegIjCL36O+QBIAABCBwSQOi8EMwI7O3thZmZmTA\/Px8mJyfN+pznwsvLy2Ftbe0Iwezs7CFv\/pxPAgj9fN67+VPv7++HhYWFsLm5GR4+fIjQMxNv+T558iSsr6+HsbGx0P4AnZiYCEtLS2F0dDRzV8qVTgChl35DFZ5ve3s7TE9Ph93d3cPTI\/T8l9gyXllZOfbD8vHjx2Fubi48ePAgXLlyJX9jKhZNAKEXfT31Ha4VzdTUVLhx48ah2E9Kp76nqufEjdBv377ND9F6rizrSRF6VpwU6yUwaIqEkh2BjY2NsLq6yoRuh7joygi96Oup+3AI3ff+ev91xBujvuxL6YbQS7mJM3gOhO53qe0bok3H9k1Sv+50KoUAQi\/lJs7gORC6z6Uicx\/ONXRB6DXcUqVnROj2F9cyvnjxIpO5Pe7iOyD04q+o3gMidNu7a\/levXqVz53boq6mOkKv5qrqOyhCt7szfonIjm3NlRF6zbdX+NkRut0FNR9PXFxcHNjg3r174datW3YHoHKRBBB6kdfCoSAAAQjICSB0OTMyIAABCBRJAKEXeC3Nr28\/evSowJNxJAjoCdy9ezeMj4\/rC5DZSQChdyLyD7h8+fLhC\/+VV17xb36OOj59+jTs7OyE69evn6OnHs6jNqybP1tbW8M5wDnpitALvOhG6G+\/\/Xa4efNmgac7O0dq\/4+sGskwOdre6\/379w\/\/1YnQbTkjdFu+quoIXYVNnITQxcjUCQhdjU6UiNBFuHyCEboPZ4Tuw7npgtB9WCN0H86iLghdhEsdjNDV6MSJCF2MTJWA0FXYbJMQui3ftjpC9+HMhO7HGaH7sY7uhNCjUSUFIvQkfKJkJnQRLnUwQlejs0tE6HZseysjdB\/OTOh+nBG6H+voTgg9GlVSIEJPwidKZkIX4VIHI3Q1OrtEhG7Hlgndh+3JLgjdhztC9+Es6oLQRbjUwUzoanTiRIQuRqZKQOgqbLZJCN2Wb1sdoftwZofuxxmh+7GO7oTQo1ElBSL0JHyiZCZ0ES51MEJXo7NLROh2bNmh+7Blhz4czgh9ONxP7YrQfS6FCd2HMysXP84I3Y91dCeEHo0qKRChJ+ETJbNyEeFSByN0NTq7RIRux5aViw9bVi7D4YzQh8OdlUsB3JnQ\/S6BCd2HNUL34SzqwoQuwqUORuhqdOJEhC5GpkpA6CpstkkI3ZZvWx2h+3BuuiB0H9YI3YezqAtCF+FSByN0NTpxIkIXI1MlIHQVNtskhG7Llwndh29vF4Tuwxyh+3AWdUHoIlzqYCZ0NTpxIkIXI1MlIHQVNtskhG7Llwndhy8Tuj9nhO7PvLMjQu9ElCWACT0LxqgiTOhRmJKDEHoywvwFEHp+pv0qInQfzk0XhO7DGqH7cBZ1QegiXOpghK5GJ05E6GJkqgSErsJmm4TQbfmyQ\/fhyw7dnzNC92fe2RGhdyLKEsCEngVjVBEm9ChMyUEIPRlh\/gIIPT9Tdug+TAd1Qeg+\/BG6D2dRF4QuwqUOZkJXoxMnInQxMlUCQldhs01C6LZ82aH78GWH7s8Zofsz7+yI0DsRZQlgQs+CMaoIE3oUpuQghJ6MMH8BhJ6fKTt0H6bs0IfLGaEPl3\/f7gjd51KY0H04N12Y0H1YI3QfzqIuCF2ESx2M0NXoxIkIXYxMlYDQVdhskxC6Ld+2OkL34cyE7scZofuxju6E0KNRJQUi9CR8omQmdBEudTBCV6OzS0Todmx7KyN0H85M6H6cEbof64Gd9vb2wszMTJifnw+Tk5OhZKG3Emwf5t69e+HWrVsFUJQfoRahn3x99D7p\/v5+WFhYCJubm4dffvXVV8P6+noYGxuTAzHMYEI3hNtTGqH7cB7Ypfcv5MOHD4sW+sbGRlhcXAztOVvRND+Emh9Gtf2pQej9Xh8t5\/Z7ly5dOuK\/vLwcmucqTeoI3edvB0L34dy3y\/b2dpieng67u7uH3y9Z6IPk3chjbm4uPHjwIFy5cmWINOWtSxf6oNdH+6TND9jV1dVj7Nt7ev3114v6lxNCl78+NRkIXUMtQ077l3VqaircuHHjUOwrKyvFTujtedsztghKFUjMFZUs9NNeH+2zNdN4MwwsLS2F0dHRo0ce9PUYJlYxCN2K7PG6CN2H86ldTsqyxB16l9BrXLuULPTeF0w\/9v3WLb2iL23tgtB9RIPQfThXL\/RBk3jvJFnbHv2sCr1Zxbz77rtF7dERuo9oELoP5+qF3jzAoDdF33vvvTA7O1vdG6MI3e\/Fj9B9WCN0H85nQui9Um\/+u\/mI3DvvvBPeeuut0PtJiwKQRh3hrAq9xE+6IPSol2RyEEJPRpheoIYd+qCnPO0z0ulkbCvULPSGDG+K2r4+aqyO0Au4tRqEPmiHzscW7V9Ag96Q7rcrL\/VTR0zo9q+TpgNC9+F8JlYuJwXSiubOnTtFfeY59kprn9BbeU9MTBx9dLHEdUtzHwg99lWZFofQ0\/hlya5hQm8ftH1jtP3f7S9DZQHhXKR2oTe4+NV\/5xdN4e0QeoEXVOLn0AvElHykWoSe\/KAFFGBC97kEhO7DWdQFoYtwqYMRuhqdOBGhi5GpEhC6CpttEkK35dtWR+g+nNmh+3FG6H6sozsh9GhUSYEIPQmfKJkJXYRLHYzQ1ejsEhG6HdveygjdhzMTuh9nhO7HOroTQo9GlRSI0JPwiZKZ0EW41MEIXY3OLhGh27FlQvdhe7ILQvfhjtB9OIu6IHQRLnUwE7oanTgRoYuRqRIQugqbbRJCt+XbVkfoPpzZoftxRuh+rKM7IfRoVEmBCD0JnyiZCV2ESx2M0NXo7BIRuh1bdug+bNmhD4czQh8O91O7InSfS2FC9+HMysWPM0L3Yx3dCaFHo0oKROhJ+ETJrFxEuNTBCF2Nzi4RoduxZeXiw5aVy3A4I\/ThcGflUgB3JnS\/S2BC92GN0H04i7owoYtwqYMRuhqdOBGhi5GpEhC6CpttEkK35dtWR+g+nJsuCN2HNUL34SzqgtBFuNTBCF2NTpyI0MXIVAkIXYXNNgmh2\/JlQvfh29sFofswR+g+nEVdELoIlzqYCV2NTpyI0MXIVAkIXYXNNgmh2\/JlQvfhy4Tuzxmh+zPv7IjQOxFlCWBCz4IxqggTehSm5CCEnowwfwGEnp9pv4oI3Ydz0wWh+7BG6D6cRV0QugiXOhihq9GJExG6GJkqAaGrsNkmIXRbvuzQffiyQ\/fnjND9mXd2ROidiLIEMKFnwRhVhAk9ClNyEEJPRpi\/AELPz5Qdug\/TQV0Qug9\/hO7DWdQFoYtwqYOZ0NXoxIkIXYxMlYDQVdhskxC6LV926D582aH7c0bo\/sw7OyL0TkRZApjQs2CMKsKEHoUpOQihJyPMXwCh52fKDt2HKTv04XJG6MPl37c7Qve5FCZ0H85NFyZ0H9YI3YezqAtCF+FSByN0NTpxIkIXI1MlIHQVNtskhG7Lt62O0H04M6H7cUbofqyjOyH0aFRJgQg9CZ8omQldhEsdjNDV6OwSEbod297KCN2HMxO6H2eE7sc6uhNCj0aVFIjQk\/CJkpnQRbjUwQhdjc4uEaHbsWVC92F7sgtC9+GO0H04i7ogdBEudTATuhqdOBGhi5GpEhC6CpttUiP069evh\/HxcdtGVA+PHj0KN2\/ehIQxgZ2dnfD06dOwtbVl3Ol8l0foBd5\/I\/RPj4VwaazAw52hI+3uhfCjvRB+5bNXz9BTlfkoH73\/o\/Cp559D6MbXg9CNAWvKN0L\/2u2DMPUFTTY5sQR+8K8hzL4zEn5zeTNc+OSl2DTiFAT+8+\/XwoUf\/CNCV7CTpCB0CS2nWITuAxqh+3BuuiB0H9YI3YezqAtCF+FSByN0NTpxIkIXI1MlIHQVNtskhG7Lt62O0H04M6H7cUbofqyjOyH0aFRJgQg9CZ8omQldhEsdjNDV6OwSEbod297KCN2HMxO6H2eE7sc6uhNCj0aVFIjQk\/CJkpnQRbjUwQhdjc4uEaHbsWVC92F7sgtC9+GO0H04i7ogdBEudTATuhqdOBGhi5GpEhC6CpttEkK35dtWR+g+nNmh+3FG6H6sozsh9GhUSYEIPQmfKJkJXYRLHYzQ1ejsEhG6HVt26D5s2aEPhzNCHw73U7sidJ9LYUL34czKxY8zQvdjHd0JoUejSgpE6En4RMmsXES41MEIXY3OLhGh27Fl5eLDlpXLcDgj9OFwZ+VSAHcmdL9LYEL3YY3QfTiLujChi3CpgxG6Gp04EaGLkakSELoKm20SQrfl21ZH6D6cmy4I3Yc1QvfhLOqC0EW41MEIXY1OnIjQxchUCQhdhc02CaHb8mVC9+Hb2wWh+zBH6D6cRV0QugiXOpgJXY1OnIjQxchUCQhdhc02CaHb8mVC9+HLhO7PGaH7M+\/siNA7EWUJYELPgjGqCBN6FKbkIISejDB\/AYSen2m\/igjdh3PTBaH7sEboPpxFXRC6CJc6GKGr0YkTEboYmSoBoauw2SYhdFu+7NB9+LJD9+eM0P2Zd3ZE6J2IsgQwoWfBGFWECT0KU3IQQk9GmL8AQs\/PlB26D9NBXRC6D3+E7sNZ1AWhi3Cpg5nQ1ejEiQhdjEyVgNBV2GyTELotX3boPnzZoftzRuj+zDs7IvRORFkCmNCzYIwqwoQehSk5CKEnI8xfAKHnZ8oO3YcpO\/Thckbow+XftztC97kUJnQfzk0XJnQf1gjdh7OoC0IX4VIHI3Q1OnEiQhcjUyUgdBU22ySEbsu3rY7QfTgzoftxRuh+rKM7IfRoVEmBCD0JnyiZCV2ESx2M0NXo7BIRuh3b3soI3YczE7ofZ4Tuxzq6E0KPRpUUiNCT8ImSmdBFuNTBCF2NLj1xeXk5rK2tHRWanZ0N8\/PzoQahf\/C\/B+GrfzES7kwdhGtXRo7B+PCjEP7srw\/Cd\/75ucOvf37iIHzzDw\/Ci584HpdOMK1CLUL\/2f98EP79\/t3wqa+8ET7xuWvHHvrnH30Ydh58I\/z3975z+PXRX\/98+LW798Mv\/eqLaXAyZyP0zEAHlEPoPpyPddnf3w8LCwvhyZMnYX19PYyNjYW9vb0wMzMTJiYmwubmZvja7YMw9YUhHC6iZa+w\/\/yPfn5M6O33Xh5rZP9xsdXNEL6\/HYqTeg1C7xX2Z95cOyb09nsXXno5vPyVNw5ZP\/ubb4Wf\/sv3i5M6Qo\/4i5UhBKFngCgtsb29Haanp8PKykqYnJw8Sn\/8+HGYm5sLu7u7xQr9h88OwhtrI+HZBx9P2yeF\/u3vHoS\/+qeR8K3Zg3D55Y9j2mn+9187CF9+rZwpvXShf\/j0h+Hfvnkn\/Gzv2SHHk0J\/f+vb4cebfxk+89XV8Pwrlw9j2mn+hS\/eCC998cvSl6ZZPEI3Q3usMEL34RzVpRH67du3D2NLnNBbmX\/pt0L43d\/+WOx\/evv4yqWZxp\/tHYQ\/\/oOR8PyFXzz2oK9HgTEKKlnorcxfuP6l8MJrv3co9vGZrx+b0Jtp\/KP3n4Xx6T8Jz114\/ojSoK8bYYwqi9CjMCUHIfRkhPkKbGxshNXV1aIn9PZpW7n3Cr3fuqWNL3HtUrLQe19Vrdx7hd5v3dLmlLh2Qej5PHFaJYTuw7mzS7uGmZqaOnyjtMQJvfchpEJvVjF\/992RovboZ1XozSrmv7b+tqg9OkLvVECWAISeBWNakfYN0aZK8ybptWvXEHoa0qhshB6FKUsQQs+CsbMIQu9EZBtwUubNJ15q+NiidEJn5aJ\/HbFy0bM7b5kIfYg33q5ZLl68ePTxxeY4tQq9OTtviuZ\/QfUTetOFN0Xzs669IkIf0g22Mr969WpYWloKo6OjRyepWej9duV8bDHtRTZI6P125XxsMY117dkIfQg32PtLRCdlXvuE3sp7\/KVffHSxxHVLw7nmHXpz\/lbev3xx\/OijiyV+wqU5Kzt0H9EgdB\/Ox7o0H09cXFw8tXONn3JpH4hf\/c\/7oho0oTdd+NX\/vKxrr4bQC7zBGlYuBWITH6mWCV38YAUmMKH7XApC9+Es6oLQRbjUwQhdjU6ciNDFyFQJCF2FzTYJodvybasjdB\/O7ND9OCN0P9bRnRB6NKqkQISehE+UzIQuwqUORuhqdHaJCN2ObW9lhO7DmQndjzNC92Md3QmhR6NKCkToSfhEyUzoIlzqYISuRmeXiNDt2DKh+7A92QWh+3BH6D6cRV0QugiXOpgJXY1OnIjQxchUCQhdhc02CaHb8m2rI3QfzuzQ\/TgjdD\/W0Z0QejSqpECEnoRPlMyELsKlDkboanR2iQjdji07dB+27NCHwxmhD4f7qV0Rus+lMKH7cGbl4scZofuxju6E0KNRJQUi9CR8omRWLiJc6mCErkZnl4jQ7diycvFhy8plOJwR+nC4s3IpgDsTut8lMKH7sEboPpxFXZjQRbjUwQhdjU6ciNDFyFQJCF2FzTYJodvybasjdB\/OTReE7sMaoftwFnVB6CJc6mCErkYnTkToYmSqBISuwmabhNBt+TKh+\/Dt7YLQfZgjdB\/Ooi4IXYRLHcyErkYnTkToYmSqBISuwmabhNBt+TKh+\/BlQvfnjND9mXd2ROidiLIEMKFnwRhVhAk9ClNyEEJPRpi\/AELPz7RfRYTuw7npgtB9WCN0H86iLghdhEsdjNDV6MSJCF2MTJWA0FXYbJMQui1fdug+fNmh+3NG6P7MOzsi9E5EWQKY0LNgjCrChB6FKTkIoScjzF8Aoednyg7dh+mgLgjdhz9C9+Es6oLQRbjUwUzoanTiRIQuRqZKQOgqbLZJCN2WLzt0H77s0P05I3R\/5p0dEXonoiwBTOhZMEYVYUKPwpQchNCTEeYvgNDzM2WH7sOUHfpwOSP04fLv2x2h+1wKE7oP56YLE7oPa4Tuw1nUBaGLcKmDEboanTgRoYuRqRIQugqbbRJCt+XbVkfoPpyZ0P04I3Q\/1tGdEHo0qqRAhJ6ET5TMhC7CpQ5G6Gp0dokI3Y5tb2WE7sOZCd2PM0L3Yx3dCaFHo0oKROhJ+ETJTOgiXOpghK5GZ5eI0O3YMqH7sD3ZBaH7cEfoPpxFXRC6CJc6mAldjU6ciNDFyFQJCF2FzTapEfrV3wjh02O2fagewj98L4QXfmcKFMYE\/u8nu+HFn\/5H2NraMu50vssj9ALvf2dnJ7z55psFnowjQUBP4O7du2FyclJfgMxOAgi9ExEBEIAABOoggNDruCdOCQEIQKCTAELvREQABCAAgToIIPQ67olTQgACEOgkgNA7EREAAQhAoA4CCL2Oe+KUEIAABDoJIPRORARAAAIQqIMAQq\/jnjglBCAAgU4CCL0TEQEQgAAE6iCA0Ou4J04JAQhAoJMAQu9ERAAEIACBOggg9DruiVNCAAIQ6CSA0DsREQABCECgDgIIvY574pQQgAAEOgkg9E5EBEAAAhCogwBCr+OeOCUEIACBTgIIvRMRARCAAATqIIDQ67gnTgkBCECgkwBC70REAAQgAIE6CCD0Ou6JU0IAAhDoJIDQOxERAAEIQKAOAgi9jnvilBCAAAQ6CSD0TkQEQAACEKiDAEKv4544JQQgAIFOAv8Pzs+8YA\/bwakAAAAASUVORK5CYII=","height":179,"width":298}}
%---
%[output:86149756]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXQAAADfCAYAAADmzyjKAAAAAXNSR0IArs4c6QAAHhlJREFUeF7tnQ1wFdW9wP+RqkRE4MqHxIQPaaAtgoD2EaOBV59f05JWS6chkfE1xdZSqtgB8sE4ff2wmAR8bwK1Fm2K9clHpsPwND6fba0jsaZmXpmntFMtKQiSpgUkKAUiUpo3Z+Vc997szd1779n7sfvbmQxk7+7\/7Pmd\/\/7uybnnns3r7+\/vFzYIQAACEMh5AnkIPefbkApAAAIQsAggdBIBAhCAgE8IIHSfNCTVgAAEIIDQyQEIQAACPiGA0H3SkFQDAhCAAEInByAAAQj4hABC90lDUg0IQAACCJ0cgAAEIOATAgjdJw1JNSAAAQggdHIAAhCAgE8IZEzoXV1dsmbNGnnooYckFAqFcfb19UldXZ20tbVZ+8rLy6WhoUHy8\/N9gpxqQAACEPCGQEaErmReXV0tY8aMkZaWlgihNzY2Sk9PjyVxtSm5FxQUSG1trTcEiAoBCEDAJwTSLvTW1lapr6+XsrIyOX78eITQlehramqkqalJiouLLcRO+3zCnmpAAAIQMEogrULv7e2VFStWyOrVq+Xo0aOieuP2Hvorr7wyYJ8egqmsrJSSkhKjlScYBCAAAT8RSKvQ7eCc5K167x0dHRFj5lropaWlUlFREcG+u7vbT21BXSAAgRwkUFhYmDVXnbNCVzJftWqVdHZ2GoF5cf9YK874s1fK8H+MleHnflf79GtGCiIIBCCQ8wR2XrheTuQdtuoxd+5cWbt2rWSD2HNW6KqHX1VVZYG8\/PLLk06Qdw+ekZfWHZKL9k+1YowouiAca0TR+db\/R9r2JV1Qkid2\/7lbOl\/plIULFyYZITdOo5650U5ur9Lv7Rm6\/pScN+GY1aFsbm6W9vZ2hJ7KGLoWerIg3z34vuzedsyS+YTSYTJzUUiUwCeWXuw2Z9NyXKr1TMtFGiiEehqAmEUhaM\/MNEZW9dATmeWSasI8efteUb3zmRWjpGzVuEHpv9X7XmZaR0TU0NL27dtl+fLlGbuGdBRMPdNBOX1l+L09J4SGWjBT9ZDpFskqoavKuZ2HngrIZ+49KAc6TsriHVdYQyxK2C\/vfUd+\/adjFl\/1\/0xK3HQjEw8CEDBL4Omvz5brPzoSoWusTrNc1GtuvymarNDVUMvD17whZSvHycQlI2Tr\/\/5FGn++37os9a573ZSR1r8TQh9+M7Vo1AfvxmwQgAAEtCuUJ5L1kFcUM9ZDT7VCyYJ8ae0h2d16TMp\/MUk++8P\/sy6j8pOXSe0tk1O9JM6HAAQCRiBZD3mFKVBC173zkTfny6PT3raYqj+d9HiYV5CJCwEI+JMAQjfUrsmAVD1z1UN\/rrrPuoq2ZbMNXQ1hIACBIBJIxkNecgpMD133zo9fk2f1zvWHGl7CJTYEIOBvAgjdUPsmCvJAxwl55t5uS+Z5Hx8ir95\/raErIQwEIBBUAol6yGtOgemhq6GWZx75izx22zF6515nFfEhEBACCN1QQycKUs09\/+8XD8tvvvg+vXNDbUAYCASdQKIe8ppXYHro6puhz\/61V8rXF1nTFNkgAAEIpEoAoadK8Nz5iYJcM263dMw4Jc88z5rqhpqAMBAIPIFEPeQ1sED00PUMFzXc0rbhGq+ZEh8CEAgIAYRuqKETAalmuGy+fZ\/8YYXIkzUzDV0BYSAAgaATSMRD6WAViB66nuESemQEX\/FPR1ZRBgQCQgChG2roREDqGS7zWi7nA1FD\/AkDAQiwfK6xHEhE6HqGy9f\/s9ha8pINAhCAgAkCiXjIRHnxYgRiyIUZLvHSgNchAIFkCCD0ZKg5nOMWJDNcDAEnDAQgMICAWw+lC53ve+jMcElXKlEOBIJHAKEbanO3IJnhYgg4YSAAAXroXuWAW6Ezw8WrFiAuBCDg1kPpIuX7IRdmuKQrlSgHAsEjgNANtblbkMxwMQScMBCAAEMuXuWAG6Ezw8Ur+sSFAAQUATceSicpXw+59P5xn\/xo3gnp\/tp58sPvXJlOrpQFAQgEgABCN9TIbkDqKYujHhkpSz8\/wVDJhIEABCDwAQE3HkonK1\/30LXQr9p2mXzmU2PTyZWyIACBABBA6HEaua+vT+rq6qStrc06sry8XBoaGiQ\/Pz\/iTDcgd7ceEzVtcd7LE1jDJQA3F1WEQLoJuPFQOq8p63rojY2N0tPTY0lcbUruBQUFUltbm7DQX930O9n6vbOy7LcfkwmhoenkSlkQgEAACCD0QRq5q6tLampqpKmpSYqLi60jnfa5HbvS3xJF6AG4s6giBDJAAKEPAl3BUT30lpYWCYVC1pF6CKayslJKSj58HqgbkFrojftmZaCpKRICEPA7ATceSieDrBpyaW1tlY6Ojogxcy300tJSqaioCLNxA1J\/7f+R3VenkyllQQACASHgxkPpRJHzQt+yZYsUFhZaP9HbU1\/5jfxhzyVSv3N6OplSFgQg4HMC3d3dVg3Vv1VVVdLe3u7ooHRjyGmhb\/78PjmRd1h2Xrheli9fbv3YN7WOy8He9xB6urOK8iDgcwLNzc2ifvSG0B0aPNExdCX06ddOllmr8hx76Uroxy\/+h6hHz7FBAAIQMEVA9czVT2dnpyV2hO5ANtFZLkrot3yhTBasL3JspyduaZc\/Tx4n9T+aZqodiQMBCEAgTIAx9DjJkMg89HhC3\/CJ5+XEvCKEzg0IAQh4QgChx8GayDdF4wldLZ07cckIuWPNRE8ak6AQgECwCSB0Q+2vQA4m9H+c6paGyb0y89tjZMHS8YZKJQwEIACBDwkgdEPZoEBuXPiK\/Ms\/lcviHVMGRNVroU9aG5KqOwdOaTR0GYSBAAQCTAChG2r8eELXa6HPaymQ6xeMNlQqYSAAAQjQQzeeAwjdOFICQgACCRKgh54gsFiHxxO6Xgu9\/BeTZMZVlxgqlTAQgAAE6KEbzwEt9PmTZ8qXfn3DgPhvPv9H2XrHaUHoxtETEAIQOEeAHrqhVNBCv2b8rdZ659GbWgv92bp+Wfz6VNZCN8ScMBCAQCQBhG4oIxTIdV94Wj512Z0I3RBTwkAAAokRQOiJ8Yp5dDyhsxa6IdCEgQAEBvUQqy0aSJB4Qt\/53V3y7ONDhIdbGIBNCAhAwJEAPXRDiaGFXjZkniw\/cOuAqKyFbgg0YSAAAXroXueAFvqcM4tk9aGZCN1r4MSHAAQGEKCHbigpFMh\/++KPZf7pex2FrtZCV5vTsgCGLoEwEIBAwAkgdEMJEE\/oai30E2PH83ALQ7wJAwEIDCSA0A1lRTyhPz5vi\/zlE1ezFroh3oSBAAQQumc5YBd6ze8vkY+MmRRRlnq4Rei2YtZC96wFCAwBCNBDN5QDgwldrYX+8DVvyKzqGVK2apyhEgkDAQhAIJIAQjeUEXahq6\/+jyi6IByZh1sYgkwYCEBgUAII3VCCWNMWl66ROT0PWF\/9twtdr4WuHh49s2KUoRIJAwEIQIAeuic5oIS+omKNfOa92ELn4RaeoCcoBCBwjgA9dEOp4Ebod+y4QiaWXmyoRMJAAAIQoIfuSQ7Yhf7VZ4fK6KunhsthLXRPkBMUAhCIIkAP3VBKDCb0vU\/\/Slq\/MmbA2LqhogkDAQhAwCKA0A0lggL57WX3ybzuRyV6aEULnYdbGIJNGAhAwJEAQjeUGArkVyvvk0WnBgpdP63IadEuQ8UTBgIQgAA9dJ0DXV1dsmbNGnnooYckFAqFU6Ovr0\/q6uqkra3N2ldeXi4NDQ2Sn58fkT52oVdsHCJTbpsefp210LnTIACBdBCghy4iSubV1dUyZswYaWlpiRB6Y2Oj9PT0WBJXm5J7QUGB1NbWuhb6C3VPyevPT3N8NF06GpkyIACBYBAIvNBbW1ulvr5eysrK5Pjx4xFCV6KvqamRpqYmKS4utjLCaZ\/aP1gPHaEH42ailhDINIFAC723t1dWrFghq1evlqNHj4rqjdt76ApO9D49BFNZWSklJSXh9rNmuXyjSj5z8L8k+huh6mlFf3t7LGuhZzrbKR8CPicQaKHb29ZJ3qr33tHRETFmroVeWloqFRUVEUJXD2e96+RAoau10M+76HKE7vObiepBINMEEPq5FjAp9Fk1efLpFTPCbavWQj818ZM83CLT2U75EPApge7ubqtm6l\/VsWxvb5fCwsKM1zavv7+\/PxNXYVLofxz+7zLnntmyfPlyqypK6KNnlVlDMWwQgAAETBNobm4W9aM3Xwtdz2JRs1XU5jT10EnoiY6h6yGXK790wBK6fodUD7dgLXTTKUw8CEBAE1A9c\/XT2dlpid3XQnfT7E7yTnSWixb6rfeftYSuNh5u4YY+x0AAAiYIMIZ+jqKT0NVLicxDV0L\/5tBHI3rj+uEWrIVuIl2JAQEIDEYAoccReiLfFFVCrzj1qNx853D5VNPccA+9YXLvgKmMpCUEIAAB0wQQuiGiGmS00N89+L71PFHWQjcEmjAQgEBMAgjdUHLEErp+\/BxCNwSaMBCAAEL3OgcQuteEiQ8BCMQjQA89HiGXr9uFft0Xp4XnnNNDdwmQwyAAgZQJIPSUEX4QQIO866IHZOYNH36J6EDHCdl8+z7G0A1xJgwEIBCbAEI3lB0a5N2XPi5TP5ont28tsyJroS\/77cdkRNEFhkojDAQgAIGBBBC6oayIJXT9gGiEbgg0YSAAgZgEELqh5EDohkASBgIQSJoAQk8aXeSJ8YTO80QNgSYMBCBAD93rHNBCXzmtScaPHBceQ+cB0V6TJz4EIKAJ0EM3lAsaZM3szTL0UJ986dc3WJERuiHAhIEABOISQOhxEbk7YDChb\/3eWWncN8tdII6CAAQgkCQBhJ4kuOjTYgn9pbWHZHfrMVGzXNggAAEIeEkAoRuii9ANgSQMBCCQNAGEnjS6yBPtQv\/I716Xu\/Z82Tpg53d3ye+fHkYP3RBnwkAAArEJIHRD2aFBfu\/G\/5C3f36hLD9wK0I3xJYwEICAOwII3R2nuEdpkN\/\/7M\/k8I4jYaHvqHxJTr5XIIt3TIkbgwMgAAEIpEIAoadCz3ZuLKE\/9ZXfyN\/eHovQDXEmDAQgwJCL5zmA0D1HTAEQgEAcAvTQDaVItNDVNMWPjJkkT96+1yqBIRdDoAkDAQjEJIDQDSWHBrm++mHZ\/YPLpeb3l1hCf\/z6F2T0nOLwAy8MFUcYCEAAAgMIIHRDSYHQDYEkDAQgkDQBhJ40usgTNcgfLHtKXm3qD\/fQn7ilXULTJtNDN8SZMBCAQGwCCN1QdsQS+k\/mPCHFlbdI2apxhkoiDAQgAAFnAgjdUGbEEnrzxOdkzjdmI3RDnAkDAQjQQ\/c8B6KF\/s0XeiV\/+j8LQvccPQVAAALnCAS+h97Y2CgbN24MJ8SWLVukpKQk\/HtfX5\/U1dVJW1ubta+8vFwaGhokPz8\/Iok0yJ9+6zF5qXaMfPXZoTJywgXyyPw\/yazqGfTQueUgAAHPCQRa6ErmPT09YUF3dXVJdXW1rFu3Lix1+zGqNZTcCwoKpLa21pXQm648LjO\/PUYWLB3veWNSAAQgEGwCgRV6b2+vLFmyxBKzvUeuBK42tV8JvqamRpqamqS4uNja77RP7dcgn2z4H3nxntPhHroS+oL1RTKzYlSwM43aQwACnhMIrNBjkbULXcFRv7e0tEgoFLJO0UMwlZWVEW8E0UL\/8k\/\/KmPnfUIaJvcidM\/TmAIgAAF7x7K9vV0KCwszDiWvv7+\/P1NXoXvtixYtkoqKCmltbZWOjo6IMXMt9NLSUusYvTkJXX1T9NFPvyfzWgrk+gWjM1UtyoUABHxOoLu726qh+reqqkoCL3QtagVFf+iZjNCnjhor87oflZu\/v0smzb7OEvodO66QiaUX+zylqB4EIJApAs3NzaJ+9BZooWuZv\/XWWxHDK8kIvf6upXKk+Rap2DhERhSdj9AzleGUC4EAEVA9c\/XT2dlpid3XQtezV9SMFrXZpx7Gkrkej0p0DL1t8\/Py1G0nLKEPG\/pn+cm\/XkYPPUA3FlWFQCYJBPpDUadhFntjJDPLxS704aMPyY8XjkbomcxwyoZAgAgEWujR89Cd2j3Reeha6AsfOGgtn9t691nrAdEjii4IUFpRVQhAIBMEAiv06GEYO3ynIRm33xR98bkt8mTZRXLr\/Wdl2NAe2X5\/EULPRGZTJgQCSCCwQjfd1hokQjdNlngQgIBbAgjdLak4xzkJffjow\/Kz+8bL4tenyoTQUEMlEQYCEICAMwGEbigz7CCfuLpXbrx7rzWG\/twDQ2T1oZmGSiEMBCAAgdgEELqh7HASugr9\/MYpCN0QY8JAAAKDE0DohjIEoRsCSRgIQCBpAgg9aXSRJ9pB7rj5DblywXA5c2S\/dP3uKmuWCxsEIAABrwkgdEOEEbohkISBAASSJoDQk0ZHD90QOsJAAAKGCCB0D0CqtVyKZ7xmRWbIxRBgwkAAAnEJIPS4iNwdYAephf7uwfflT2fmSP3O6e6CcBQEIACBFAgg9BTg2U+NJfTTw0tl8Y4phkohDAQgAIHYBBC6oeywg3xhyT4ZHsqTvx\/ZLwjdEGDCQAACcQkg9LiI3B3gJPSju7pk6PT59NDdIeQoCEAgRQIIPUWA+vRooeefPSDvvnVGxsz\/uHzusWsNlUIYCEAAAgy5eJ4DdqG\/eM9pufBvHZbQL7v1Jlmwvsjz8ikAAhCAAD10QzkQLfQhe7fKyfcKELohvoSBAATiE0Do8Rm5OsIOctd3Dsqpzl9ZQi9efJXM\/9bVrmJwEAQgAIFUCCD0VOjZzo0W+pGdr8vJ0wUy5xuzpWzVOEOlEAYCEIAAY+ie5wBC9xwxBUAAAnEI0EM3lCJ2kK829ctfn\/slPXRDbAkDAQi4I4DQ3XGKe5ST0A+\/e418uiFPZlXPiHs+B0AAAhBIlQBCT5XgufMRuiGQhIEABJImgNCTRhd5oh3k3p8ckldbjlizXNQc9JkVowyVQhgIQAACsQkgdEPZgdANgSQMBCCQNAGEnjS6wXvoLz98vnVA5eYLZfKN0wyVQhgIQAAC9NBjEtDvaOqAgoIC2bRpkxQXF4eP7+vrk7q6Omlra7P2lZeXS0NDg+Tn50fEtL8zvrn1fHlp3SHr9Tt2XCETSy8mByEAAQh4TiDQPXRV+ZUrV4Ylrn5vbGyUlpYWCYVCFnz1e09PjyVxtSm5K\/HX1tYidM\/TkwIgAIFECARa6ErWkyZNkoqKCouZ7o1XVlZKSUmJdHV1SU1NjTQ1NYV77U771Ll2kG\/\/8pg8W9dPDz2RTORYCEAgZQKBFno0PS300tJSS\/JOPfZo6esYsYT+tfaLJTTtipQbigAQgAAE4hFA6DZCra2tsm3btvCQi\/q9o6MjYsw8WvrRQt+yZYsMfX1YuIeO0OOlIK9DAAKpEuju7rZCqH+rqqqkvb1dCgsLUw2b8vl5\/f39H4xVpHGzfzD64IMPhodgkhG6uuziv98g80\/fa9Vg2W8\/JiOKLkhjbSgKAhAIGoHm5mZRP3oLtNA1BN371h96JiP0tWvXSv7+qaLWc1Fb3ZshOe+izL9TBi3BqS8EgkRA9czVT2dnpyV2XwtdfZBZXV1tzVZRW6yph+o1+7j5nj17Bsx6cTOG3vvyMHnm3oMIPUh3FHWFQBYQCOwY+mBj4Xrq4tGjR5Oa5XLmjZOy9Y7TCD0LEpxLgECQCARW6KqR1ZDKhg0bwvPQe3t7ZcmSJbJo0aLwOHoy89DtQl99aGaQ8om6QgACGSQQaKFrqdfX14ebwP6hqNqZzDdFtdCHX3pY7vnDjRlsXoqGAASCRCDwQjfV2HaQCN0UVeJAAAKJEEDoidAa5Fg7yLNvjZTNt++zpiuqaYtsEIAABNJBAKEboozQDYEkDAQgkDQBhJ40usgT7SAvOvm+\/GjeCSmc9Y7c+fN5hkogDAQgAIHBCSB0QxmC0A2BJAwEIJA0AYSeNDp66IbQEQYCEDBEAKF7AHJ4\/1h5+Jo3ZELpMFm8Y4qhEggDAQhAgCGXtOSA\/Z1RC336Z0\/K5x67Ni3lUwgEIAABeuiGcsAOsiAk0jC5VxC6IbiEgQAEXBFA6K4wxT8IocdnxBEQgIC3BBC6Ib5OQi+pflNuaPicoRIIAwEIQIAx9LTkQPQ745pxu+W6ZWdk\/reuTkv5FAIBCECAHrqhHEDohkASBgIQSJoAQk8aXeSJCN0QSMJAAAJJE0DoSaMbXOgHNs2VsaWVkj\/tPkMlEAYCEIAAY+hpyYHod8bepydL\/rTlCD0t9CkEAhBQBOihG8oDhG4IJGEgAIGkCSD0pNENPuRy\/OVKOX\/0XHrohvgSBgIQiE8Aocdn5OqIbAPp6qI5CAIQ8BWBbPNQXn9\/f38uEs42kLnIkGuGAARSI5BtHkLoqbUnZ0MAAgEmgNANNX62gTRULcJAAAI5RCDbPEQPPYeSh0uFAASyiwBCN9Qe2QbSULUIAwEI5BCBbPMQPfQcSh4uFQIQyC4CCN3WHo2NjdZvtbW14b19fX1SV1cnbW1t1r7y8nJpaGiQ\/Pz8iJbMNpDZlWZcDQQgkA4C2eahjPXQNYi77747QuhK8j09PZbE1abkXlBQEHGM2p9tINORPJQBAQhkF4Fs81BGhN7b2ysrVqyQd955R0pKSsKy7urqkpqaGmlqapLi4mKr5Zz2IfTsSmquBgJBJRB4oeshldLSUtm\/f3\/EkIuCo3roLS0tEgqFrNf08ZWVlZb89ZZtIL1K6O7ubtm+fbssXLhQCgsLvSom43GpZ8abwOgFBKU9s81Dae+ht7a2SkdHhzWksn79+gih21\/TY+b2N4CKiorACT3bEsboXW8LRj29IpuZuLRnZrinVejRwyfRH4omI\/QtW7b4vudaVVUl1DMzN4jpUlXPlfY0TTVz8XR7tre3Z4WH0iZ0p552KkJXIFetWiWdnZ2Za01KhgAEAk9g7ty5snXr1qzg4InQVU+8urramq2iNjX18K677pKlS5eG99lrf9VVV1nj5nv27HE9hq7OV1JXP2wQgAAEMkVAfbaVLZ9veSJ0t2Cje+iJzHJxWwbHQQACEAgKgawSuoLudh56UBqIekIAAhBwSyDrhO72m6JuK8hxEIAABIJCIKNCDwpk6gkBCEAgHQQQejooUwYEIACBNBBA6GmATBEQgAAE0kEgJ4Xu13F2p+me0StNqg+NN27caOWGnu6pl0lIR8KYLiPWWj1+qGe8OsR73TRrL+K5uRdzuZ563anVq1eH15dSHN3UW39bVnNXXw60L1\/iRXvkpND9OBNGy3zdunXhRrfXUy2FoL5Ju23btvBaN9Gve5EgXsbUN8WuXbtk06ZN4Rsm1+up66XY6Tdkv7ZlvHsxl9tSyXzJkiVy5MiRiPxU7Rqv3tH3s5L7ypUrB8QxfX\/lnND9OlfdadkDe10vvfRSK7nU2vH6XV4nnH2f6QTxMp6+2VUZeoVNpzrlWj2dbl4\/tqVTu\/ilnrp3rf4Ktuen+r8bBzk968Fpn+n7K+eEnsiKjKZhpTuePXFU2dFLC+uewqRJk8S+cFm6rzOZ8nQ7qjcjleha6IMNweRKPZ3enO2M\/FBHVZ94Qs\/VnLWv8Ko6UtH3XTwHqTcB9RwHtaJs9IKCaokApwf2JHMPOZ2Tc0JPZAEvU5AyFcf+56rTsgha6Opf+1OfMnW9bsu1iyD6hnG6WXKtnronpt6A6uvrLSz2zzv8UEfd1tFDSfbfX3vttQFLeeRaWzq9+cZz0E033TTgr2lV71jt7va+cXMcQndDKQPH6D\/59AcpfpOAfhOKvmH8UE\/9IeCDDz4Y7qGpfapusdYsyjXR2W8J+4ee9ieQ+aEtEbrH8ov37phrQw9OuLTM7ULww83h1Evxq9Cj\/2rSf5UsWrRIJk6cmPM9V\/uQi6qTvu\/U\/blhwwbrw7+jR4\/mfD0RusdCjzd+5fW0II+rF35Wql3mqky\/jLvae3PRLFWd58yZk\/OfFQz28HM1ruqHOqq2i9e58kM9ne67eA5iDD0BS7r5hDmBcFl1aPQwi\/3i\/DD7wwl2dHv6oZ6qHaM\/\/LLXa+rUqb6YsRRP6E5jybk2Y8mtb+I9vCddQ2o5N4auwai11tWnxWpTnygXFBTk1AeD0XJzmocefUwuz+mN9c4Z60\/aXJ5vH+thLnoMXX0RzA9taR9GchpyUQ96z\/V6DvaX8WAOYh56An1lN9\/SSiBcVhw62FCE\/RtmufytOzc9dH1MrtczOkedvtWb63W0j6OrGS1q81s9YwndjYP4pmhWqJWLgAAEIJCbBHJyyCU3UXPVEIAABLwlgNC95Ut0CEAAAmkjgNDThpqCIAABCHhLAKF7y5foEIAABNJGAKGnDTUFQQACEPCWwP8DVqsJ6mCWl9kAAAAASUVORK5CYII=","height":179,"width":298}}
%---
