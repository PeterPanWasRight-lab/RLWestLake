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
% Initialize environment parameters  30分钟
% agent_state = [1, 1];
% final_state = [3, 4];
% obstacle_state = [2,2; 3,2; 3,3; 2,4; 4,4; 2,5];
% x_length = 5;
% y_length = 5;

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

toc %[output:793c4188]

figure_policy(x_length, y_length, agent_state,final_state, obstacle_state, state, policy, actions) %[output:29e1488a]
figure_stateValue(x_length,y_length,agent_state,final_state,obstacle_state,state_value) %[output:37f283b8]
figure,plot([0:episode_length-1],state_value_history) %[output:66bc7284]
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
%[output:793c4188]
%   data: {"dataType":"text","outputData":{"text":"历时 0.086341 秒。\n","truncated":false}}
%---
%[output:29e1488a]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbUAAAEHCAYAAAApqNijAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnX+wXVV1x9cLEAkRAo+8CglEEDKmoy0VhMSRWqu2qDQdzHQayJRqfNg4VNAZk0nSEUuLM0lMnBGCShxihloTM9MC01hb\/\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\/5zstjwzm8vDxm5YVt3X0itbqJ1n8+pFY\/UxNn9DZRtkqrVWyTSa1ZaAV8T2LzlpWJGzxRE0gtEfgeyiK1HmB5OtTjRPmHf\/jySqt4NYutk9TaCS1sP4YHSWK9HhrZLfPmXKIu5zEr9cU6H4jU7AeI1OxnpOrQ60TZSWztpBbAFFuOBaTYQgt1l296R6P8pW98n8w7e37PgvOalerGdD4IqdkPEKnZz2jSDkdHR2V4eFhWrlwpCxYsGD\/W80TZTmxbtkx8pD9sLw4Pj8jDD88Zv+YUQgvF7753o9x9763jfcycMbshuCveel2pu8tzVpNd4M6dO2XJkiXjh1xwwQWyefNmGRwcLMXF4kFIzWIqE3tCavYz6tjhwYMHZdWqVbJjxw7ZunVrNlILF9xObM1bk0NDv5Jf\/GJ6cqEVDdz+9VVyz4N3HZVVGcHlKLXt27fL6tWrJ9yX69ata9yrW7Zskblz57r8zUNq9mNDavYzatvhnj17ZOnSpbJ\/\/\/7Gz3OTWjuxdYoq1QqttZ+1266Wh362u22bhdzabU\/mJrXiD1uzZs1q7CAUr2JX4corr5TFixe7\/M1DavZjQ2r2Mzqqw0JoCxculEWLFjXktmHDBpcrteKzZeEiH330UVmxYrn8\/OdPjF\/z0NDX5P77Txn\/71eftu8oHjEfCul2u4QVWyexFWNbV28epdb6h6p2f7BqZVVILWyTN8uuG1NLP0dqltJo3wtSs5\/RpB0Wk4tXqQ0M9BbA771ro1zwR6+8f9XbaHtHB8H9\/MdH5IzjLpb169fba7BNR8U9d911142vuML7Z8uXL590a7HdOBcX3NQkUrOfGFKznxFSayKQm9SKSxv6xZ+6kVo3gT35zD6558E7x1O79I2LZPrUwcb7vyMjI64fFkFq9idMpGY\/I6SWsdTCSu373xyRMPHf9MlPu7gbi23EBx54QJYtW3bUVmKQWvExh3BBG5Z9Szbf9o+yadOmo977dXHBrNRcxYTUXMV1dLPetx+b31MLV3fbbbc1Jr7wOnx4tkybtm7CY\/vhPbXm99VOO+0ZufPOGWZSDI\/3l3lPbd7ZlzREFj607fE9tQA8PM0YRFW8CsG1Su38sT+XL236inuhhetkpWbmV61jI0jNfkZZr9Q6XVyQ3eWXPyE\/\/OFvdU2o3XdFdh3UhwNaP6\/WXCKsyGaeMlsu\/Z1Fjc+wNb+8Sq35GopH+IPYhj\/8FxNWaru\/crghtebPUfYBf5RTIrUomCsVQWqV8KUf7H2l1o5gu6++aj3uta8dk8cee+Upk9Ri6yS04knHsCoL\/97ulYPUisf4G6uZv7lePnHHe8cv9SOXbZY3X3Bp+l+WGjpAajVA7PMpkFqfAff79LlIrfj2ic985p\/lllveNOE7IMMHrXfvnj7hLwn90Y8OybvetVf27Tt\/HHEqsbX7RpHm7cVu94A3qRVZNX82svnJxne++9Kj3lPrJPRubKz9HKlZS+TofpCa\/Ywm7TAXqYWLvOuu\/5QlS16QQ4de+bqvILQnnpgurd\/9GKT2hS+slK997eMT3nNLIbbioYhO24vdbjFvUgvX0\/oVWOH\/W7NmTeMR\/\/seuEdu\/cbw+GWH7cdDz42N\/3f4fOXatWv\/\/\/3Sad3QmPs5UjMXyVENITX7Gak69DZRdvu2\/cn+6pnJvt1fBa\/HQeHBiPDSrka8ZdUNT7unH7VsutWK\/XOkFpt47\/WQWu\/MXIzwNlGG73UMciperV991e0vCW0WWxgbvgDZy18U6i2rbr8ASK0bIX7eTwJIrZ90E57b40RZiK3ddzl2k1pAHcQWjvMktNC3x6wmu7WRWsJffEoLUsv0JvA6UQaxtVthlZFaiDIcF6To6eU1q06MkZqnuy+\/XpFafpk2rii3ibKs1DzGmVtWSM3jXZhPz0gtnywnXEluEyVS83OjIjU\/WeXYKVLLMdUMV2qZxpTlqhqp5Xy32r82pGY\/I1WHua3UVBCcDMotK6Tm5MbLtE2klmmwuU2UmcbESs1ZsHxOzX5gSM1+RqoOkZoKW5JBuWXFSi3JbUTR3xBAapneCrlNlJnGxErNWbCs1OwHhtTsZ6TqEKmpsCUZlFtWrNSS3EYUZaWW9z2Q20SZc1q5ZYXUcr5b7V8bKzX7Gak6zG2iVEFwMii3rJCakxsv0zaRWqbB5jZRZhoT76k5C5b31OwHhtTsZ6TqEKmpsCUZlFtWrNSS3EYU5T21vO+B3CbKnNPKLSuklvPdav\/aWKnZz0jVYW4TpQqCk0G5ZYXUnNx4mbaJ1DINNreJMtOYeE\/NWbC8p2Y\/MKRmPyNVh0hNhS3JoNyyYqWW5DaiKO+p5X0P5DZR5pxWblkhtZzvVvvXxkrNfkaqDnObKFUQnAzKLSuk5uTGy7RNpJZpsLlNlJnGxHtqzoLlPTX7gSE1+xmpOkRqKmxJBuWWFSu1JLcRRXlPLe97ILeJMue0cssKqeV8t9q\/NlZq9jNSdZjbRKmC4GRQblkhNSc3XqZtIrVMg81tosw0Jt5TcxYs76nZDwyp2c9I1SFSU2FLMii3rFipJbmNKMp7annfA7lNlDmnlVtWSC3nu9X+tbFSs5+RqsPcJkoVBCeDcssKqTm58TJtE6llGmxuE2WmMbl\/Ty0IbO1Xr5Yr3nqdXPrG9zWuZzKp3X3vRrnnwbvkmveslXlzLnEXK++p2Y8MqdnPSNUhUlNhSzLIe1Yf+PTrG9yueOtHGnLrJLUgtLvvvVVmzpgtq678cuN\/vb2Qmv3EkJr9jFQdep8oVRftdJD3rG7\/+qrG6iu8wmotiG35pneMp7Fh2bfkngfvbAgtvOadfYmsuurLLtNCavZjQ2r2M1J16H2iVF2000HeswpCC2IrXkFaD\/1s9\/h\/B9EV0gv\/Z1iledx6DL0jNfu\/ZEjNfkaqDr1PlKqLdjooh6zCyixsO3Z7hS3HsHLz+kJq9pNDavYzUnWYw0SpunCHg3LIqnkLcrIIwqrtmveudZjSyy0jNfvRITX7Gak6zGGiVF24w0E5ZNX6cEinGDxvPSI1H79cSM1HTj13mcNE2fNFOx2QS1Zrt1094b201ji8bz0iNR+\/YEjNR049d5nLRNnzhTsckEtWrQ+MtEZRPPLvMKLxltl+tJ8eUrOfkarDXCZK1cU7G5RTVsVn1tqt0rx+Nq35WpCa\/V8upGY\/I1WHOU2UKgCOBuWUVacHRjx\/Ng2pOfplEhGk5iuv0t3mNFGWvminB+aUVactSO8PiBS3Fis1+79kSM1+RqoOc5ooVQAcDcotq9bPrOXwgAhS8\/MLhdT8ZNVTp7lNlD1dvLODc8uqdQvS+2fT2H709QuF1HzlVbrb3CbK0hfu8MDcsmr9zFouW4\/h1mL70f4vGFKzn5Gqw9wmShUEJ4NyzKr4zFpOW49IzccvFFLzkVPPXYaJct++fTJ\/\/vyexzIgLoFdu3Y1CuaU1eHp++Xpk+6TXx84U8447uK4QPtYLWQ1e\/Zs2bZtWx+rcOoqBJBaFXqGxwapjezZJbMGDTdJaw0C3\/+JyAkzZ8nU08\/MisiJJw80rufQc2PZXNeLTz0uF553NlIznChSMxxOldaC1IbGdsmNS\/KZUKrwsDx22a0D8tPXLJSzPnij5TbpTUT2fulGef3hJ5Ca4bsBqRkOp0prSK0KvbhjkVpc3lWqIbUq9OKMRWpxOEevgtSiI1cXRGpqdNEHIrXoyHsuiNR6RuZjAFLzkVPoEqn5yQqp2c8KqdnPSNUhUlNhSzIIqSXBriqK1FTYog5CalFxxyuG1OKxrloJqVUlGG88UovHWlsJqWnJGR+H1IwH1NQeUvOTFVKznxVSs5+RqkOkpsKWZBBSS4JdVRSpqbBFHYTUouKOVwypxWNdtRJSq0ow3nikFo+1thJS05IzPg6pGQ+I7Uc\/ATV1itTsx4bU7Gek6hCpqbAlGcRKLQl2VVGkpsIWdRBSi4o7XjGkFo911UpIrSrBeOORWjzW2kpITUvO+DikZjwgth\/9BMT2o6uskJqruMo3i9TKs0p9JCu11AmUr89KrTyrVEcitVTk+1wXqfUZcI2nR2o1wuzzqZBanwHXcHqkVgNEi6dAahZTad8TUvOTFVKznxVSs5+RqkOkpsKWZBBSS4JdVRSpqbBFHYTUouKOVwypxWNdtRJSq0ow3nikFo+1thJS05IzPg6pGQ+oqT2k5icrpGY\/K6RmPyNVh0hNhS3JIKSWBLuqKFJTYYs6CKlFxR2vGFKLx7pqJaRWlWC88UgtHmttJaSmJWd8HFIzHhDbj34CauoUqdmPDanZz0jVIVJTYUsyiJVaEuyqokhNhS3qIKQWFXe8YkgtHuuqlZBaVYLxxiO1eKy1lZCalpzxcUjNeEBsP\/oJiO1HV1khNVdxlW8WqZVnlfpIVmqpEyhfn5VaeVapjkRqqcj3uS5S6zPgGk+P1GqE2edTIbU+A67h9EitBogWT4HULKbSviek5icrpGY\/K6RmPyNVh0hNhS3JIKSWBLuqKFJTYYs6CKlFxR2vGFKLx7pqJaRWlWC88UgtHmttJaSmJWd8HFIzHlBTe0jNT1ZIzX5WSM1+RqoOkZoKW5JBSC0JdlVRpKbCFnUQUouKO14xpBaPddVKSK0qwXjjkVo81tpKSE1Lzvg4pGY8ILYf\/QTU1ClSsx8bUrOfkapDpKbClmQQK7Uk2FVFkZoKW9RBSC0q7njFkFo81lUrIbWqBOONR2rxWGsrITUtOePjkJrxgNh+9BMQ24+uskJqruIq3yxSK88q9ZGs1FInUL4+K7XyrFIdidRSke9zXaTWZ8A1nh6p1Qizz6dCan0GXMPpkVoNEC2eAqlZTKV9T0jNT1ZIzX5WSM1+RqoOkZoKW5JBSC0JdlVRpKbCFnUQUouKO14xpBaPddVKuUltyguPy5QXHx\/HcvjkC6siMjMeqZmJomMjSM1+RqoOkZoKW5JBOUgtiOxVT\/6rHP\/c\/XLCc\/cfxfHIq86UF06\/XA7OviYJ47qKIrW6SPbvPEitf2yTnhmpJcXfU3HPUitkNm3\/7aWv+eCsa9zKDamVjjnZgUgtGfr+FkZq\/eVb59m9Si0ILcgsrNB6fb108oXy3LzP9zos+fFILXkEXRtAal0R+TwAqcXL7Sf\/OyDnnzSmLuhRau1WaGGLMcgqvIcW\/jf8d\/H+2gnP3t8QYPPLo9iQmvo2jzYQqUVDXW+hdevWyaZNm8ZPumzZMlm5cuX4f+cqtYcPjMn1mwbkwNMDjWu97E1H5IarBuTEqfXy7eVsn\/rpCTL60oCcd9IRuWTGEbl4xpFehotHqYXV2fRHbhq\/ziCwZ1\/\/+YbI2r0OP\/e0HPjitfLbvz8qJww8M36It61IpNbTrZ3kYKSWBLu+6MGDB2XVqlUyMjIimzdvlsHBQRkdHZXh4WGZM2eOrF27VqZNmyY5Sq0Q2jV\/PCZXvGVADr0octO2Mdn71IB89q\/G5LRXvyy62K\/vPTNFtj1+\/HjZwRPGGmIL\/4R\/7\/byJrWw+jr1B+8rLbRw4IF\/ukWe\/Lc7ZM4Hrpc5M+5urOCK1y9\/966OMuzGLvbPkVps4r3XQ2q9M0s6Ys+ePbJ06VLZsGGDLFiwYLyXnTt3yvLly2XLli0yd+7cLKW2cYfIgdGxCSuzQnQ3LhmTN89NI7UQwjeePK7xT+urENxlM3\/d8b7xJrVp+24f30oMK7Pnz72hseXY6fX8Q\/fJo+uXNX585vtvkNe8aY6c8tC144e\/MPNy+dW5NyT9vSpbHKmVJZXuOKSWjn2tlYPUlixZIlu3bm3IzuNKrXVrMQC67a+PTCorK1ILvYbVWli1dXp12p70JrWTH7p2\/LH9Z\/aPyQP\/crhxyees2CSvnvfmCZcfth0fu\/mjMvPdfylP\/vs\/yKlvWySnv+2KxtZl8wMmoxfvrPX3oV8nQ2r9IlvfeZFafSyTnmn79u2yceNGtyu11q3FAPO+PWNy49YBuWXZmLzujJdXYeG9q9GXXkb9\/EGRW3eIDM0QufodIlNPSBpBo\/hXHz++0eNkr9btSU9Sa916fHbe5xurtLAa27v5b+Wcj22UE2e\/rnH5R148JHu3\/L1MPf0MmXnZ1Q25FVJrfU\/OyxYkUkv\/O9atA6TWjZCDnxdbkgsXLhx\/WMTbSq2dwNqh77TN5yCmti0GwR348RF56FeXy1kfvNH8ZYQPVzdvHU4mo6e+e7f88rt3yms\/enPjupqlFv578HuvbJ8XcrQOAKlZT0gEqdnPaNIOi4dEwkHFgyPh371J7ennx+RjXxyQ\/xkZkPe\/c0yuW9j+snOTWnGVP\/jvd7uU2g8fWyJn\/Nn1R4V1aN\/D8uhnr5Ozhv+usSVZbEMWKzWk5nziMdw+UjMcTrfWOgnNo9SKaw0Pg9zxzVe271oFl5vUwkrtsfuOyCNTfK7Udn\/lsBx67uUnPGe+5\/0NwTVvOxbCa5Vap23Mbvd86p+zUkudQPf6SK07I5NHFFuOQ0NDE1ZoRbPeVmrtIN\/9H2Pyqe1TJqzc2r1fdcc3RX7wiMgnrxqTGdPTPQEZ3uv73jPHTfqwSGOF8ptH\/sODI+FD257eU2tdYYWnFsPTi2Gr8fE7bmqI7dS3\/EljlXZ49EDb351p575Bzv\/QsJw28srnKnlPzeQ047IppOYwtkJoF1100fjn0lovIwepFZ9DC9f28UVj8pk7B+SMwYEJW5PNx6T+EPZkq8ggsvNOCp9f+\/VR3z7iTWrhM2rF58yKx\/GL1VnI6qyln5QpU0+ccEu2rtSan34MHwsIUvPwYqVmPyWkZj+jCR22+6B1u0vwJrXwoMiHPzdlwiP8rU9Etjum2K7s9uh\/v2Oe7HNq4TNqk33LiDepveq\/PirTX9o1jjQ85PH8s6c2VmdDCz\/UeGS\/9dUstaH58yd8eJvPqfX77jy2zo\/UnOUdHt1fvXp1x67XrFkjixcvdvegSLigQlrNF\/eJxUca3x5SvFo\/y\/aGOWNJv00k9NUqtNbtxW63mDeptW5BhvfUwntr4YPV7YQWji+kNvQH75zwjSLdvl6rG7vYP2elFpt47\/WQWu\/MXIzwtlJzAbVDk58bOV6ebnz3Y\/vtxW7X5lFqzd8qEq6vzDeLhC3LU3587YSvyOK7H7vdHfy8VwJIrVdiTo5HavGCCg+vlPmOx04deZRauJbmbxYpri1sJYZ\/iq\/NCiILf2lo+Hxb619Rw7f0x7tHj6VKSC3TtJGan2C9Sq2T2MqQ9\/Q+WvP1sP1YJt20xyC1tPz7Vh2p9Q1t7Sf2LLUAo3UrcjJAYZvyhdMv52++rv0u4oQFAaSW6b2A1PwE611qgXS7vzS0OYFCZmGF1unvXPOQGCs1+ykhNfsZqTpEaipsSQblILVmcMXfdl38f0emnulaZGw\/Jvm1UBdFamp0tgciNdv5NHeXm9T8kO+9U1ZqvTOLPQKpxSYeqR5SiwS6hjJIrQaIkU6B1CKBrlAGqVWAZ3koUrOczsTekJqfrJCa\/ayQmv2MVB0iNRW2JIOQWhLsqqJITYUt6iCkFhV3vGJILR7rqpWQWlWC8cYjtXistZWQmpac8XFIzXhATe0hNT9ZITX7WSE1+xmpOkRqKmxJBiG1JNhVRZGaClvUQUgtKu54xZBaPNZVKyG1qgTjjUdq8VhrKyE1LTnj45Ca8YDYfvQTUFOnSM1+bEjNfkaqDpGaCluSQazUkmBXFUVqKmxRByG1qLjjFUNq8VhXrYTUqhKMNx6pxWOtrYTUtOSMj0NqxgNi+9FPQGw\/usoKqbmKq3yzSK08q9RHslJLnUD5+qzUyrNKdSRSS0W+z3WRWp8B13h6pFYjzD6fCqn1GXANp0dqNUC0eAqkZjGV9j0hNT9ZITX7WSE1+xmpOkRqKmxJBiG1JNhVRZGaClvUQUgtKu54xZBaPNZVKyG1qgTjjUdq8VhrKyE1LTnj45Ca8YCa2kNqfrJCavazQmr2M1J1iNRU2JIMQmpJsKuKIjUVtqiDkFpU3PGKIbV4rKtWQmpVCcYbj9TisdZWQmpacsbHITXjAbH96Cegpk6Rmv3YkJr9jFQdIjUVtiSDWKklwa4qitRU2KIOQmpRcccrhtTisa5aCalVJRhvPFKLx1pbCalpyRkfh9SMB8T2o5+A2H50lRVScxVX+WaRWnlWqY9kpZY6gfL1WamVZ5XqSKSWinyf6yK1PgOu8fRIrUaYfT4VUusz4BpOj9RqgGjxFEjNYirte0JqfrJCavazQmr2M1J1iNRU2JIMQmpJsKuKIjUVtqiDkFpU3PGKIbV4rKtWQmpVCcYbj9TisdZWQmpacsbHITXjATW1h9T8ZIXU7GeF1OxnpOoQqamwJRmE1JJgVxVFaipsUQchtai44xVDavFYV62E1KoSjDceqcVjra2E1LTkjI9DasYDYvvRT0BNnSI1+7EhNfsZqTpEaipsSQaxUkuCXVUUqamwRR2E1KLijlcMqcVjXbUSUqtKMN54pBaPtbYSUtOSMz4OqRkPiO1HPwGx\/egqK6TmKq7yzSK18qxSH8lKLXUC5euzUivPKtWRSC0V+T7XRWp9Blzj6ZFajTD7fCqk1md0X4ZQAAACCklEQVTANZweqdUA0eIpkJrFVNr3hNT8ZIXU7GeF1OxnpOoQqamwJRmE1JJgVxVFaipsUQchtai44xVDavFYV62E1KoSjDceqcVjra2E1LTkjI8LUhvZs0tmDRpvlPZk\/6jIk1NmydTTz4SGcQIvPvW4XHje2bJt2zbjnR677SG1jLNfsWJFxlfHpUEgDYH169enKUzVUgSQWilMHAQBCEAAAh4IIDUPKdEjBCAAAQiUIoDUSmHiIAhAAAIQ8EAAqXlIiR4hAAEIQKAUAaRWChMHQQACEICABwJIzUNK9AgBCEAAAqUIILVSmDgIAhCAAAQ8EEBqHlKiRwhAAAIQKEUAqZXCxEEQgAAEIOCBAFLzkBI9QgACEIBAKQJIrRQmDoIABCAAAQ8EkJqHlOgRAhCAAARKEUBqpTBxEAQgAAEIeCCA1DykRI8QgAAEIFCKAFIrhYmDIAABCEDAAwGk5iEleoQABCAAgVIEkFopTBwEAQhAAAIeCCA1DynRIwQgAAEIlCKA1Eph4iAIQAACEPBAAKl5SIkeIQABCECgFAGkVgoTB0EAAhCAgAcCSM1DSvQIAQhAAAKlCCC1Upg4CAIQgAAEPBBAah5SokcIQAACEChFAKmVwsRBEIAABCDggQBS85ASPUIAAhCAQCkCSK0UJg6CAAQgAAEPBJCah5ToEQIQgAAEShH4P1uj0Ce2rbXAAAAAAElFTkSuQmCC","height":0,"width":0}}
%---
%[output:37f283b8]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbUAAAEICAYAAADY\/mp2AAAAAXNSR0IArs4c6QAAGwtJREFUeF7tnE9oJ+d5x1+FxFiQgiNXiXEXNQ2IFlIwZFNWvuSaQiuSrgOydVMF0aGsNwcZSYeWbXuQ5NUhG\/lQNVF0kyyo25Q9pcdcIpVA8a0gGtplu1nSWi75g4xjojIyo4zWv99q9nlnnj8znz3Zu\/PM87yf7+z70bz6aUdOT09PE78gAAEIQAACHSAwgtQ6kCJLgAAEIACBMwJIjQcBAhCAAAQ6QwCpdSbKfi7k+Pg4zc\/Pp6WlpTQ1NdVPCAFWvb6+nra2ts4nXVhYOMuMXxBomgBSa5oo91MjcHJykpaXl9Pdu3fT7u4uUlMjX79RmdG9e\/fS9vZ2GhsbS+UXIhMTE2ltbS2Njo7WvyFXQuASAkiNRyQkgaOjozQ3N5cePHhwNj9S8xljmdPGxsaFLzoODg7S4uJi2tnZSZOTkz6HZ6qQBJBayNj6PXS5UU5PT6fr16+fye3RTbPfhPyvvpDa7OwsX4z4jyrchEgtXGQMXCUw7E0ASr4J7O\/vp83NTd7UfMcUcjqkFjI2hi4JILV4z0L1TZsPi8TLz\/vESM17Qsz3WAJILdYDUn5IpJi6\/OBIrBUwrXcCSM17QsyH1DryDCC0jgTpfBlIzXlAjPd4ArypxXhCypzGx8d5Q4sRWdgpkVrY6Bi8IIDU\/D8HZUZXr17l59L8xxV+QqQWPsJ+LwCp+c6fH7T2nU8Xp0NqXUy1R2tCar7DLj66v7KyMnTI1dXVNDMz43sRTBeKAFILFRfDQgACEIDA4wggNZ4PCEAAAhDoDAGk1pkoWQgEIAABCCA1ngEIQAACEOgMAaTWmSgvLuT+\/fvprbfe6ujqWBYE7AjcvHnTrjmdLyWA1C5FFPOC1157LR0eHsYcvmdTF1+AXLlypWerjrncIquXXnop3b59O+YCejA1UutoyK+88srZRslfPv8Bk5X\/jMoJiy8WC7Ht7e3FGbpnkyK1jgbORhknWLKKkxVS858VUvOfkWhCNkoRNpMisjLBLmqK1ETYVIuQmipuvWZslHqsczuRVS5BvXqkpsda2gmpSck5r2OjdB5QZTyyipMVUvOfFVLzn5FoQjZKETaTIrIywS5qitRE2FSLkJoqbr1mbJR6rHM7kVUuQb16pKbHWtoJqUnJOa9jo3QeEMePcQKqTIrU\/MeG1PxnJJoQqYmwmRSRlQl2UVOkJsKmWoTUVHHrNWOj1GOd24mscgnq1SM1PdbSTkhNSs55HRul84A4fowTEMePobJCaqHiqj8sUqvPyvpKsrJOoH5\/3tTqs7K6EqlZkW+5Lxtly4AbvD1ZNQiz5VshtZYBN3B7pNYARI+3YKP0mMrgmcgqTlZIzX9WSM1\/RqIJ2ShF2EyKyMoEu6gpUhNhUy1Caqq49ZqxUeqxzu1EVrkE9eqRmh5raSekJiXnvI6N0nlAlfHIKk5WSM1\/VkjNf0aiCdkoRdhMisjKBLuoKVITYVMtQmqquPWasVHqsc7tRFa5BPXqkZoea2knpCYl57yOjdJ5QBw\/xgmoMilS8x8bUvOfkWhCpCbCZlJEVibYRU2RmgibahFSU8Wt14yNUo91bieyyiWoV4\/U9FhLOyE1KTnndWyUzgPi+DFOQBw\/hsoKqYWKq\/6wSK0+K+sryco6gfr9eVOrz8rqSqRmRb7lvmyULQNu8PZk1SDMlm+F1FoG3MDtkVoDED3ego3SYyqDZyKrOFkhNf9ZITX\/GYkmZKMUYTMpIisT7KKmSE2ETbUIqani1mvGRqnHOrcTWeUS1KtHanqspZ2QmpSc8zo2SucBVcYjqzhZITX\/WSE1\/xmJJmSjFGEzKSIrE+yipkhNhE21CKmp4tZrxkapxzq3E1nlEtSrR2p6rKWdkJqUnPM6NkrnAXH8GCegyqRIzX9sSM1\/RqIJkZoIm0kRWZlgFzVFaiJsqkVITRW3XjM2Sj3WuZ3IKpegXj1S02Mt7YTUpOSc17FROg+I48c4AXH8GCorpBYqrvrDIrX6rKyvJCvrBOr3502tPiurK5GaFfmW+7JRtgy4wduTVYMwW74VUmsZcAO3R2oNQPR4CzZKj6kMnoms4mSF1PxnhdT8ZySakI1ShM2kiKxMsIuaIjURNtUipKaKW68ZG6Ue69xOZJVLUK8eqemxlnZCalJyzuvYKJ0HVBmPrOJkhdT8Z4XU\/GckmpCNUoTNpIisTLCLmiI1ETbVIqSmiluvGRulHuvcTmSVS1CvHqnpsZZ2QmpScs7r2CidB8TxY5yAKpMiNf+xITX\/GYkmRGoibCZFZGWCXdQUqYmwqRYhNVXces3YKPVY53Yiq1yCevVITY+1tBNSk5JzXsdG6Twgjh\/jBMTxY6iskFqouD467PHxcZqfn09LS0tpamrq\/IKoUjs4OEizs7Pn61hdXU0zMzPBU3r8+FGzKlc17Bks\/vzk5CQtLy+nu3fvnl3+wgsvpO3t7TQ2NhYyU97U\/MeG1PxnNHTC6oaxu7sbXmr7+\/tpZWUllWspN8tC1oW0u\/orstQe9wyWf\/b888+f57e+vp6KL1yiig2p+f9biNT8ZzRwwqOjozQ3N5cePHhw9ufRpTZMYMUGuLi4mHZ2dtLk5GTQtLr5pnbZM1h8kbK5uXkhuzLnl19+OeQbOFLz\/1cQqfnP6CMTlpvJ9PR0un79+pncNjY2Qr+plWt6dB3RN8E6j1fEN7U6z2DxVlZ80bW2tpZGR0fPUQz7\/TqsrK9BatYJXN4fqV3OyPUVw2QQbaO8TGpdPoKMltWjfyEGZTfo6LGsi3wEidRcb4dnwyE1\/xk9dsKuSG3YG1n1jaCr31frm9SKY8k333wz5PfVkJr\/DROp+c+oF1IrFjnsgyJvv\/12WlhY6OyHRZBanL+ESM1\/VkjNf0a9kVpVbMV\/Fx\/\/fuONN9Lrr7+eqp+gCx7ZR8bvm9Q4fuzaE+xrPUjNVx5PPE1Xjh+HLfxxPwP1xLCcFnRRagVqPiji9IHr+FhILXjAXZHasO+p8ZF+\/w\/osGdw0PfOon+aleNH\/88jUvOfUW+OHx\/dBMvN8saNGyF\/pqnuo9XVN7VSYBMTE+cf64989FjkidTqPtV21yE1O\/aNdO7Km1oJo\/ywSPn\/j\/5QeSPQnN2kq1IrMPPPZDl72HowDlLraMjRN8qOxjJwWWQVJ23e1PxnhdT8ZySakI1ShM2kiKxMsIuaIjURNtUipKaKW68ZG6Ue69xOZJVLUK8eqemxlnZCalJyzuvYKJ0HVBmPrOJkhdT8Z4XU\/GckmpCNUoTNpIisTLCLmiI1ETbVIqSmiluvGRulHuvcTmSVS1CvHqnpsZZ2QmpScs7r2CidB8TxY5yAKpMiNf+xITX\/GYkmRGoibCZFZGWCXdQUqYmwqRYhNVXces3YKPVY53Yiq1yCevVITY+1tBNSk5JzXsdG6Twgjh\/jBMTxY6iskFqouOoPi9Tqs7K+kqysE6jfnze1+qysrkRqVuRb7stG2TLgBm9PVg3CbPlWSK1lwA3cHqk1ANHjLdgoPaYyeCayipMVUvOfFVLzn5FoQjZKETaTIrIywS5qitRE2FSLkJoqbr1mbJR6rHM7kVUuQb16pKbHWtoJqUnJOa9jo3QeUGU8soqTFVLznxVS85+RaEI2ShE2kyKyMsEuaorURNhUi5CaKm69ZmyUeqxzO5FVLkG9eqSmx1raCalJyTmvY6N0HhDHj3ECqkyK1PzHhtT8ZySaEKmJsJkUkZUJdlFTpCbCplqE1FRx6zVjo9RjnduJrHIJ6tUjNT3W0k5ITUrOeR0bpfOAOH6MExDHj6GyQmqh4qo\/LFKrz8r6SrKyTqB+f97U6rOyuhKpWZFvuS8bZcuAG7w9WTUIs+VbIbWWATdwe6TWAESPt2Cj9JjK4JnIKk5WSM1\/VkjNf0aiCdkoRdhMisjKBLuoKVITYVMtQmqquPWasVHqsc7tRFa5BPXqkZoea2knpCYl57yOjdJ5QJXxyCpOVkjNf1ZIzX9GognZKEXYTIrIygS7qClSE2FTLUJqqrj1mrFR6rHO7URWuQT16pGaHmtpJ6QmJee8jo3SeUAcP8YJqDIpUvMfG1Lzn5FoQqQmwmZSRFYm2EVNkZoIm2oRUlPFrdeMjVKPdW4nssolqFeP1PRYSzshNSk553VslM4D4vgxTkAcP4bKCqmFiqv+sEitPivrK8nKOoH6\/XlTq8\/K6kqkZkW+5b5slC0DbvD2ZNUgzJZvhdRaBtzA7ZFaAxA93oKN0mMqg2ciqzhZITX\/WSE1\/xmJJmSjFGEzKSIrE+yipkhNhE21CKmp4tZrxkapxzq3E1nlEtSrR2p6rKWdkJqUnPM6NkrnAVXGI6s4WSE1\/1khNf8ZiSZkoxRhMykiKxPsoqZITYRNtQipqeLWa8ZGqcc6txNZ5RLUq0dqeqylnZCalJzzOjZK5wFx\/BgnoMqkSM1\/bEjNf0aiCZGaCJtJEVmZYBc1RWoibKpFSE0Vt14zNko91rmdyCqXoF49UtNjLe2E1KTknNexUToPiOPHOAFx\/BgqK6QWKq76wxZSK37dvn27fhFXmhAovvonKxP0T9z0zp076f79+2lvb++JaynQIYDUdDirdymkdnh4qN6XhhDoOoFr164hNcchIzXH4eSMVkjtg3cO09f\/+DTnNtQqELi1O5J+Nn41fforCwrdaJFD4Kf\/vJX+8JmPI7UciC3XIrWWAVvdvpDa+OlhujWL1KwyqNt34Y2R9B+fmU5X\/vxW3RKuMyJw\/7u30u9\/8FOkZsS\/TlukVodSwGuQWpzQkFqcrJCa\/6yQmv+MRBMiNRE2kyKkZoJd1BSpibCpFiE1Vdx6zZCaHuvcTkgtl6BePVLTYy3thNSk5JzXITXnAVXGQ2pxskJq\/rNCav4zEk2I1ETYTIqQmgl2UVOkJsKmWoTUVHHrNUNqeqxzOyG1XIJ69UhNj7W0E1KTknNeh9ScB8TxY5yAKpMiNf+xITX\/GYkmRGoibCZFvKmZYBc1RWoibKpFSE0Vt14zpKbHOrcTUsslqFeP1PRYSzshNSk553VIzXlAHD\/GCYjjx1BZIbVQcdUfFqnVZ2V9JW9q1gnU78+bWn1WVlciNSvyLfdFai0DbvD2SK1BmC3fCqm1DLiB2yO1BiB6vAVS85jK4JmQWpyskJr\/rJCa\/4xEEyI1ETaTIqRmgl3UFKmJsKkWITVV3HrNkJoe69xOSC2XoF49UtNjLe2E1KTknNchNecBVcZDanGyQmr+s0Jq\/jMSTYjURNhMipCaCXZRU6QmwqZahNRUces1Q2p6rHM7IbVcgnr1SE2PtbQTUpOSc16H1JwHxPFjnIAqkyI1\/7EhNf8ZiSZEaiJsJkW8qZlgFzVFaiJsqkVITRW3XjOkpsc6txNSyyWoV4\/U9FhLOyE1KTnndUjNeUAcP8YJiOPHUFkhtVBx1R8WqdVnZX0lb2rWCdTvz5tafVZWVyI1K\/It90VqLQNu8PZIrUGYLd8KqbUMuIHbI7UGIHq8BVLzmMrgmZBanKyQmv+skJr\/jEQTIjURNpMipGaCXdQUqYmwqRYhNVXces2Qmh7r3E5ILZegXj1S02Mt7YTUpOSc1yE15wFVxkNqcbJCav6zQmr+MxJNiNRE2EyKkJoJdlFTpCbCplqE1FRx6zVDanqsczshtVyCevVITY+1tBNSk5JzXofUnAfE8WOcgCqTIjX\/sSE1\/xmJJkRqImwmRbypmWAXNUVqImyqRUhNFbdeM6Smxzq3E1LLJahXj9T0WEs7ITUpOed1SM15QBw\/xgmI48dQWSG1UHHVHxap1WdlfSVvatYJ1O\/Pm1p9VlZXIjUr8i33RWotA27w9kitQZgt3wqptQy4gdsjtQYgerwFUvOYyuCZkFqcrJCa\/6yQmv+MRBMiNRE2kyKkZoJd1BSpibCpFiE1Vdx6zZCaHuvcTkgtl6BePVLTYy3thNSk5JzXITXnAVXGQ2pxskJq\/rNCav4zEk2I1ETYTIqQmgl2UVOkJsKmWoTUVHHrNUNqeqxzOyG1XIJ69UhNj7W0E1KTknNeh9ScB8TxY5yAKpMiNf+xITX\/GYkmRGoibCZFvKmZYBc1RWoibKpFSE0Vt14zpKbHOrcTUsslqFeP1PRYSzshNSk547r19fW0tbV1PsXCwkJaWlo6\/\/\/oUnv3F6fpG38\/km5Mn6YvTo5coP3e+yn97d5p+v6\/fezs9z8\/cZq++fXT9KlPXrzOOKLa7aNL7YOfv5v+687N9JmvvZo++QdfvLDuX7\/\/Xrq\/8zfpZ\/\/6\/bPfH\/29z6ffvXknffy3PlWbj6cLkZqnNAbPgtT8Z3RhwpOTk7S8vJzu3buXtre309jYWDo+Pk7z8\/NpYmIira2tpdHR0RRZalVp\/d1f\/PqC1Mo\/e26sEN6HaDbvpvSjoxRWbJGlVpXWZ1\/buiC18s+eeva59NzXXj3L6uE\/fCv98t9\/FFZsSM3\/honU\/Gd0YcKjo6M0NzeXNjY20tTU1PmfHRwcpMXFxbSzs5MmJyfDSu3HD0\/Tq1sj6eG7H751PSq17\/3wNH3nX0bStxZO0+ee+\/Ca8q3uz148TV99Md7bWlSpvfffP07\/+c0b6YPjh2c5PCq1d37wvfQ\/d7+dPvuNzfT073zu7Jryre6ZL11Pz37pq8H+9qWE1PxHhtT8Z1RrwkJqs7OzaXd390x2Ed\/USqF9+Qsp\/ckffSi3W7MXjx+Lt7KHx6fpL18ZSU8\/9Rs0w36\/FjzjiyJKrRTaM9e+nJ558U\/P5HZl\/q8vvKkVb2Xvv\/MwXZn7q\/Sxp54+pzzs941jqNUeqdXCZHoRUjPF31zz\/f39tLm5Gf5NrSRSCq4qtUFHj+X1kY8gI0qt+uSWgqtKbdDRY1kT+QgSqTW3Z7V1J6TWFlnF+5ZHktPT0+cfFon4plZF9qRSK44l\/+mHIyG\/r9Y3qRXHkv\/3g38M+X01pKa4sQlbITUhOC9l5YdEinnKD44U\/43UvCR0+RxI7XJGXq5Aal6SGD4HUvOf0dAJhwmtj1Lj+NHuQeb40Y49nT9KAKkFfSrKI8fx8fELb2jlcrr4plasjQ+K+HtgB0mtmJIPivjLqg8TIbWAKZdCu3r16vnPpT26jK5KbdD3zvhIv+1DPExqg753xkf6bbPqQ3ekFizlQT9oPWgJXZVaKbArz\/7mY\/2Rjx6L7Lr4PbViXaXAPjF+5fxj\/ZE\/+Visie+p+d8wkZr\/jC5MWHx0f2VlZejUq6uraWZmppMfFCkXzT+T5euhHfamVkzJP5PlK6s+TIPUOppy9De1jsYycFnR39T6lBVvav7TRmr+MxJNiNRE2EyKkJoJdlFTpCbCplqE1FRx6zVDanqsczshtVyCevVITY+1tBNSk5JzXofUnAdUGQ+pxckKqfnPCqn5z0g0IVITYTMpQmom2EVNkZoIm2oRUlPFrdcMqemxzu2E1HIJ6tUjNT3W0k5ITUrOeR1Scx4Qx49xAqpMitT8x4bU\/GckmhCpibCZFPGmZoJd1BSpibCpFiE1Vdx6zZCaHuvcTkgtl6BePVLTYy3thNSk5JzXITXnAXH8GCcgjh9DZYXUQsVVf1ikVp+V9ZW8qVknUL8\/b2r1WVldidSsyLfcF6m1DLjB2yO1BmG2fCuk1jLgBm6P1BqA6PEWSM1jKoNnQmpxskJq\/rNCav4zEk2I1ETYTIqQmgl2UVOkJsKmWoTUVHHrNUNqeqxzOyG1XIJ69UhNj7W0E1KTknNeh9ScB1QZD6nFyQqp+c8KqfnPSDQhUhNhMylCaibYRU2RmgibahFSU8Wt1wyp6bHO7YTUcgnq1SM1PdbSTkhNSs55HVJzHhDHj3ECqkyK1PzHhtT8ZySaEKmJsJkU8aZmgl3UFKmJsKkWITVV3HrNkJoe69xOSC2XoF49UtNjLe2E1KTknNchNecBcfwYJyCOH0NlhdRCxVV\/WKRWn5X1lbypWSdQvz9vavVZWV2J1KzIt9wXqbUMuMHbI7UGYbZ8K6TWMuAGbo\/UGoDo8RZIzWMqg2dCanGyQmr+s0Jq\/jMSTYjURNhMipCaCXZRU6QmwqZahNRUces1Q2p6rHM7IbVcgnr1SE2PtbQTUpOSc16H1JwHVBkPqcXJCqn5zwqp+c9INCFSE2EzKUJqJthFTZGaCJtqEVJTxa3XDKnpsc7thNRyCerVIzU91tJOSE1KznkdUnMeEMePcQKqTIrU\/MeG1PxnJJoQqYmwmRTxpmaCXdQUqYmwqRYhNVXces2Qmh7r3E5ILZegXj1S02Mt7YTUpOSc1yE15wFx\/BgnII4fQ2WF1ELFVX9YpFaflfWVvKlZJ1C\/P29q9VlZXYnUrMi33BeptQy4wdsjtQZhtnwrpNYy4AZuj9QagOjxFkjNYyqDZ0JqcbJCav6zQmr+MxJNiNRE2EyKkJoJdlFTpCbCplqE1FRx6zVDanqsczshtVyCevVITY+1tBNSk5JzXofUnAdUGQ+pxckKqfnPCqn5z0g0IVITYTMpQmom2EVNkZoIm2oRUlPFrdcMqemxzu2E1HIJ6tUjNT3W0k5ITUrOeR1Scx4Qx49xAqpMitT8x4bU\/GckmhCpibCZFPGmZoJd1BSpibCpFiE1Vdx6zZCaHuvcTkgtl6BePVLTYy3thNSk5JzXITXnAXH8GCcgjh9DZYXUQsVVf1ikVp+V9ZW8qVknUL8\/b2r1WVldidSsyLfct5DavaPDlrtw+yYI\/OQ4pU\/89vNN3Ip7tEzgV\/\/7IF27di3t7e213InbSwkgNSm5AHV37twJMCUjQiAWgZs3b8YauGfTIrWeBc5yIQABCHSZAFLrcrqsDQIQgEDPCCC1ngXOciEAAQh0mQBS63K6rA0CEIBAzwggtZ4FznIhAAEIdJkAUutyuqwNAhCAQM8IILWeBc5yIQABCHSZAFLrcrqsDQIQgEDPCCC1ngXOciEAAQh0mQBS63K6rA0CEIBAzwggtZ4FznIhAAEIdJkAUutyuqwNAhCAQM8IILWeBc5yIQABCHSZAFLrcrqsDQIQgEDPCCC1ngXOciEAAQh0mQBS63K6rA0CEIBAzwggtZ4FznIhAAEIdJkAUutyuqwNAhCAQM8IILWeBc5yIQABCHSZAFLrcrqsDQIQgEDPCCC1ngXOciEAAQh0mQBS63K6rA0CEIBAzwggtZ4FznIhAAEIdJkAUutyuqwNAhCAQM8IILWeBc5yIQABCHSZAFLrcrqsDQIQgEDPCCC1ngXOciEAAQh0mQBS63K6rA0CEIBAzwj8PwP1VwaiQJpKAAAAAElFTkSuQmCC","height":0,"width":0}}
%---
%[output:66bc7284]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbUAAAEICAYAAADY\/mp2AAAAAXNSR0IArs4c6QAAIABJREFUeF7tnQ10FOW5x5\/4UQig4MqHhmD4aMBK+RBticHg0esHR4lXStuQYL2NqJRaoR4gCVyv16sWkoCnDZQq2IhfBFLrTSUWP0q9EjWa21KVWgVSQCE3FZGgiERsae55B2ed3UyS3ewm+8zub87hJFl2Z575\/d99f\/vOvDuT1NLS0iIsEIAABCAAgTggkITU4iBFdgECEIAABCwCSI2GAAEIQAACcUMAqcVNlOwIBCAAAQggNdoABCAAAQjEDQGkFjdRsiMQgAAEIIDUaAMQgAAEIBA3BJBa3ETJjkAAAhCAAFKjDUAAAhCAQNwQQGpxEyU7AgEIQAACSI02AAEIQAACcUMAqcVNlOwIBCAAAQggNdoABCAAAQjEDQGkFjdRsiMQgAAEIKBaavX19bJkyRK57777xOfz+dNqbm6WoqIiqa6uth7Lzs6W4uJiSU5OJlEIQAACEEhgAmqlZoSWn58vAwYMkPLy8gCplZSUSGNjoyUysxjBpaSkSGFhYQJHya5DAAIQgIBKqVVWVsqiRYskKytLDh8+HCA1I7uCggIpLS2V9PR0K0G3x4gWAhCAAAQSj4A6qTU1Ncn8+fNl8eLFcvDgQTGjMudI7bXXXmv1mH04Mjc3VzIyMhIvRfYYAhCAAAQsAuqk5szFTWBmFFdbWxtwDs2WWmZmpuTk5BAtBCAAAQgkKIG4l1pDQ0OXRXtay0Br3e\/Vfmr9\/Hjv5yd+7jvxkwUCEICAVwmclPaRjM05w19+amqqJ3YlrqVmhLZw4UKpq6uLahjp\/7hMRv7jMjn7+NcD1vtJ0gfW30dOOvGTBQIQgIBXCRxJ+kC29FjhL3\/ixImybNky0S43z0ktnHNq5rl5eXlWEIMHD464be2t\/VReL22xQu075FRJy+wj50zqLX876a2I190dKzByLysrixqP7qi5K7cBjy\/pwiKwpcHDnUdNTQ1Si6RTchNYOLMfbalFGoQ5nLhtwyF5afl+OSezt1xfNSKS3YrZa6PFI2Y7EOUNw+NLoLAIbFzw8C4Pz43UDOpQv6cWrYb50rL9sq3ykHV8OWvhINeudW\/TZ9bj9k\/z+75DJx7TspjDsU8++aTMmzdPS0kxrQMeX+KHRWBThIfIkDN6ysVf7WeBiVZf2h1veE9KLdQrikQjiPdqj8i6abtlZtVw63CjLa5Xdn0kL\/\/1kCUx8zsLBCAAgXgiMGlEP6m+9XykpinUaEjt8Wm7rF0yhxyNwNb\/4W9S8ty7co6vp\/+TzDm+ZOt3s5jH7cX5uyYu1AIBCEAgHALR6EvD2V4kz1U9Uotkx6IxZHaO0pLOPUWu\/cXrVkmFVw2T3G+cFWl5vB4CEICAJwggNSUxRRqEGaV9vO\/v4ru\/n9y6\/h1rFLbxh+cHjMaU7CplQAACEOgyApH2pV1WmMuKGam1QdvMeFx14XYZe9cAuePgu9bIzIzQWCAAAQgkGgGkpiTxSIJ4eu4+60ohb89vsSaCvHHHRUr2ijIgAAEIdC+BSPrS7q1U+bUfI4URSRBLBm2zRmk31G+XVblf4xxapGHweghAwLMEIulLu3unOfzoQtyeIPJ\/PzuVUVp3t0i2BwEIqCOA1JRE0tkgzJetn77\/b\/LgdYesEZoZqbFAAAIQSFQCne1LY8GLkZoLdTPr0ZxHe\/W7n3MuLRatkm1CAAKqCCA1JXF0NggjtQ2fHJC0WX0ZpSnJkjIgAIHYEehsXxqLihmpBVG3p\/I\/c9ERWbr8PP+1z2IRDtuEAAQgoIEAUtOQQicvwmlLzRx6rF55oZI9oQwIQAACsSOA1GLHPmDLnQnCnvn46p3H\/RfzVLI7lAEBCEAgJgQ605fGpFDhe2qtuNszH3339+UKIrFqlWwXAhBQRQCpKYmjM0HYMx+HLvMhNSU5UgYEIBBbAp3pS2NVMRNFgsjbMx+LHjiXSSKxapVsFwIQUEUAqSmJI9wgnDMf1z00gavxK8mRMiAAgdgSCLcvjWW1jNQc9Jn5GMumyLYhAAGtBJCakmTCDYKZj0qCowwIQEAVgXD70lgWz0jNQd+e+XjkJz24kkgsWyXbhgAEVBFAakriCDcIZj4qCY4yIAABVQTC7UtjWTwjNQd9Zj7GsimybQhAQCsBpKYkmXCDMDcGNdd8ZOajkgApAwIQUEEg3L40lkUzUvuCPjMfY9kM2TYEIKCZAFJTkk44QdgzH9+eL\/J4wVgle0AZEIAABGJPIJy+NNbVMlL7IgF75uPb81u4kHGsWyXbhwAEVBFAakriCCcII7WKxxqEaz4qCY8yIAABNQTC6UtjXTQjtS8SMDMfN73fJD98LJ1rPsa6VbJ9CEBAFQGkpiSOcIKwpVb82zFc81FJfpQBAQjoIBBOXxrrihmpfZHAqgu3y1+GH5Nf\/GpcrDNh+xCAAARUEUBqSuIIJwgjtcMXJsmiB0YpqZ4yIAABCOggEE5fGuuKGal9kYD54nXStB5ILdYtku1DAALqCCC1Lo6kublZioqKpLq62tpSdna2FBcXS3JycsCWwwnCSO2MuX1kzr8P7+LqWT0EIAABbxEIpy+N9Z55cqRWUlIijY2NlsjMYgSXkpIihYWFnZKafTURM50\/74bUWGfC9iEAAQioIoDUujCO+vp6KSgokNLSUklPT7e25PaYeTzUIJBaFwbGqiEAAc8TCLUv1bCjnhupGbhmpFZeXi4+n89iaB+OzM3NlYyMDD\/XUINo2rFbHph8xPriNSM1Dc2SGiAAAU0EQu1LNdTsOalVVlZKbW1twDk0W2qZmZmSk5MTttTs6z6O23CWXHPpQA25UAMEIAABNQSQWhdG0RmpVVRUSGrqiXNl9k9nibbUsp8fKmPGnd6F1bNqCEAAAt4g0NDQ4C\/U\/J6Xlyc1NTWufaimPYr7kdoLN+2W+kMfyJYeKyzu8+bNs\/45l22Vh+Tpuftk8ivncIksTa2TWiAAgZgRKCsrE\/PPuSC1Logj3HNq6761W0ZfNEzGL0zyj9SCR2tvrP2zbCpqQWpdkBerhAAEvEnAjM7s0VpdXZ0lOKTWBVmGO\/vRSO2qb2fJ1BVD2qzGvu3MrX88l+s+dkFmrBICEPA2Ac6pdXF+4XxPLRSpbbl7q2x6+GQp2T2+iytn9RCAAAS8RwCpdXFm4VxRJBSpmfNpv33xA7l\/2wVdXDmrhwAEIOA9AkhNSWYmiI03vyaZl05v9\/DjUze\/Km\/vPF0WbRmtpHLKgAAEIKCHAFJTkoUJYvX01+Rfvpkt11eNaLMqcy+1fU2fITUluVEGBCCgiwBSU5JHOFJ76\/OjYm4QygIBCEAAAoEEkJqSFhGq1B69qkaODDxbfvjYiWtJskAAAhCAwJcEkJqS1hCq1Faet1mOTB7CvdSU5EYZEICALgJITUkettQuGTZWvv\/yZW1WZaTmuy5dZi5JU1I5ZUAAAhDQQwCpKcnCltqFZ08R88Vqt+WfRxukeFiTjL1rgEydc7aSyikDAhCAgB4CSE1JFiaI5d\/eKJeedQNSU5IJZUAAAt4jgNSUZBaK1LiXmpKwKAMCEFBLAKkpiSYcqU0uT5GLp\/ZXUjllQAACENBDAKkpycKWWtbJk2Xee1Ncq7LvpTazarikZfZRUjllQAACENBDAKkpySIUqe3ZvEPWzzwm3CBUSWiUAQEIqCOA1JREYoL4z+\/+Ui45NlcW7x\/rWtWujb+XypsHcC81JZlRBgQgoI8AUlOSSShSs28Qev07I7mXmpLcKAMCENBFAKkpySMUqXGDUCVhUQYEIKCWAFJTEo1TagVvnS6nDBjaqjJuEKokLMqAAATUEkBqSqIJRWovFD0l72we1eaXs5XsCmVAAAIQiBkBpBYz9IEbDkVq3CBUSViUAQEIqCWA1JREY03pn7NEJjTea43E+g75SqvKkJqSsCgDAhBQSwCpKYnGBDE\/Z4lc81nbUuNeakrCogwIQEAtAaSmJJpQpPbw5Ao5mvYNbhCqJDPKgAAE9BFAakoycUrtlk09pf8FI1tVZqR26sWTuJeakswoAwIQ0EcAqSnJpCOpmXuprbpwOzcIVZIXZUAAAjoJIDUluZgg7rr1xzK5YY24jdTsG4RmLRgkWQsHKamaMiAAAQjoIoDUlORhgrgl98cy4+gacbsKP3e9VhIUZUAAAqoJIDUl8TillrP6ZBlx3eiAyuwbhHIvNSWBUQYEIKCSAFJTEgtSUxIEZUAAAp4mgNSUxNeR1D58vUbWTOnnemhSyS5QBgQgAIGYE0BqMY\/gRAHW7Mcf5ck1+34jbocf7XupcYNQJYFRBgQgoJIAUlMSix3ETZ\/+RqauGCJjc84IqMyWWluX0FKyG5QBAQhAIKYEkFpM8X+58Y6kxg1ClQRFGRCAgGoCSE1JPE6pTbnjuEy47fyAyv606hl59u7Bsnj\/WCUVUwYEIAABfQSQWpQyqa+vlyVLlsh9990nPp\/Pv9bm5mYpKiqS6upq67Hs7GwpLi6W5OTkgC13JDVzL7XnfpUmJbvHR6liVgMBCEAg\/gggtShkaoSWn58vAwYMkPLy8gCplZSUSGNjoyUysxjBpaSkSGFhYdhS4wahUQiLVUAAAnFNAKlFGG9lZaUsWrRIsrKy5PDhwwFSM7IrKCiQ0tJSSU9Pt7bk9ph53A7i9p5rZNKcM1sdfjQjtbpXvyqLtgR+KTvC8nk5BCAAgbgigNQiiLOpqUnmz58vixcvloMHD4oZlTlHagZu8GP24cjc3FzJyMjwb90ptfH5Y1pd37Fq5hPy6dEJcn3ViAgq5qUQgAAE4psAUotSvm4CM6O42tragHNottQyMzMlJyenldRyjq6RK284TS4tnRhQmblB6Em9BiO1KOXFaiAAgfgkgNSilGs0pXbeZR9bUktNTfVXZ+6l1n98lvUdNhYIQAACEPiSQENDg\/8P83teXp7U1NQE9KEaeSW1tLS0aCzM1BRNqR3u\/bw80\/JrmTdvnvXPLEhNa\/LUBQEIxJpAWVmZmH\/OBam1k4o9u9HMYjSL27R8N6l15pyaOfxoRmrpcwdbnzLMP\/sGoW7n2mLdmNg+BCAAgVgTMKMze7RWV1dnCQ6pRZiKm8A6M\/vxpl73ytjLAg8zIrUIw+HlEIBAwhDgnFqUonaTmll12N9T63WvfG3MEJm2PstfmX2DULdrQkapfFYDAQhAIC4IILUoxdiW1MK9osjsMx+WkV9NQmpRyoXVQAACiUUAqSnJ2w7CTWr2Xa9nVg2XtMw+SiqmDAhAAAL6CCA1JZkgNSVBUAYEIOBpAkhNSXx2EAtGlcrZ\/QYFHH5kpKYkJMqAAATUE0BqSiJySq3XJz75\/suX+StDakpCogwIQEA9AaSmJCI7iILz10nP\/c0BUnuv9oism7ZbuOu1krAoAwIQUEsAqSmJpj2p7dm8Q9bPPIbUlGRFGRCAgF4CSE1JNkhNSRCUAQEIeJoAUlMSH1JTEgRlQAACniaA1JTEZwdxz+U\/lQ+f6yHz3pvir+yNtX+WTUUtsnj\/WCXVUgYEIAABnQSQmpJckJqSICgDAhDwNAGkpiQ+O4ifXPuEfFB1gJGaklwoAwIQ8BYBpKYkr46ktv6e41Kye7ySaikDAhCAgE4CSE1JLu1J7aVl+2Vb5SFrSj8LBCAAAQi0TQCpKWkdTqntWX+qFLx1upwyYKhV3Za7t8pbG3sjNSVZUQYEIKCXAFJTko0dxIr8VbLt54ORmpJcKAMCEPAWAaSmJC+kpiQIyoAABDxNAKkpic8O4ue3PiVvlLYEjNSeuvlV+eTDgXJ91Qgl1VIGBCAAAZ0EkJqSXJCakiAoAwIQ8DQBpKYkPqSmJAjKgAAEPE0AqSmJL1hqt2zqKf0vGGlV9\/i0XdZPDj8qCYsyIAABtQSQmpJo7CAeufNBealwgDil9vDFL0j\/CekydcUQJdVSBgQgAAGdBJCaklzak9qjV9WIb9QwpKYkK8qAAAT0EkBqSrKxg3i8+Bl58bZjASM1pKYkJMqAAATUE0BqSiIKltqNj7wvZ0250qruoQmPSo8pl8jMJWlKqqUMCEAAAjoJIDUlubQntbK0Z2XCj86XrIWDlFRLGRCAAAR0EkBqSnKxg3jqwUekOq+v5Kw+WUZcN9qqbuV5m2V8\/hikpiQryoAABPQSQGpKsmlLav848K7cf8lfkZqSnCgDAhDQTQCpKcnHDqJ63WZ56roj\/pGakVrp1w\/L2LsGyNQ5ZyupljIgAAEI6CSA1JTk0pHUzHfUxuacoaRayoAABCCgkwBSU5JLsNSm37tPRt18jfzzaIMUD2uyvqOG1JSERRkQgIBaAkhNSTR2EC8+WyGPZ\/WSKXcclwm3nS8fbt0pa67+DKkpyYkyIAAB3QSQmpJ8OpLazKrhkpbZR0m1lAEBCEBAJwGkFmEuJSUlsnr1av9aKioqJCMjw\/93c3OzFBUVSXV1tfVYdna2FBcXS3JycsCWkVqEQfByCEAAAiKC1CJoBkZojY2NfknV19dLfn6+LF++3C8253PMpozgUlJSpLCw0FVqNTU18ugFTXL57F3yzbunSdOO3fLA5CPCSC2CoHgpBCCQMASQWiejbmpqklmzZllyco7MjMTMYh43kisoKJDS0lJJT0+3Hnd7zDzuDMIptQ9efkF+Ob0\/UutkTrwMAhBILAJILcp5O6Vm4Jq\/y8vLxefzWVuyD0fm5uYGyLAtqe36zV+kcvZxyX5+qIwZd3qUq2V1EIAABOKLAFKLYp726G3GjBmSk5MjlZWVUltbG3AOzZZaZmam9Rx7cQZRdeV2+frU0+TS0omy48HfypN3DJFb\/3iu9B3ylShWy6ogAAEIxB8BpBalTG1ZmdXZE0E6IzUz0eQPNx6RfuM\/lu9VTEdqUcqH1UAAAvFLoKGhwb9z5ve8vDwx8xNSU1NV73RSS0tLi8YKbaHt3bs34FBjZ6Rm9u\/2nmtk+7HnZeTcwXL1wHPliR+fzUhNY\/DUBAEIqCBQVlYm5p9zQWrtRGPPajQzHc3inJbfltDM8zpzTm3ZsmXy\/tLRkj7mTTnvnknyQdUBefbek2Xx\/rEqGg9FQAACENBGwIzO7NFaXV2dJTik1omU3A45OlfT2dmP5oLGRmpXrs2R\/72zSjavHoHUOpEPL4EABBKPAOfUIsg8+HtqbqvqzPfUkFoEofBSCEAgoQkgtU7GH3xI0rkat8OToV5RxAyZX5i1W07zJcm09VnyPwV18uyve0jJ7vGdrJSXQQACEEgcAkhNSdbOIJxSez6\/Uur\/PM6aKMICAQhAAALtE0BqSlpIsNSSj78n39n8PUFqSgKiDAhAwBMEkJqSmJxBvHjbMenxSS1SU5INZUAAAt4hgNSUZNWW1KpyX5LtjT5ZtGW0kkopAwIQgIBeAkhNSTbOILb+1z75fM9frZHaE5c\/JsdOy5Trq0YoqZQyIAABCOglgNSUZBMstQNb3pGbdt6I1JTkQxkQgIA3CCA1JTkhNSVBUAYEIOBpAkhNSXzOIN4obZH3n\/2dNVL75ciHpOfoSzj8qCQnyoAABHQTQGpK8mlPamdNuUKmrhiipFLKgAAEIKCXAFJTkg1SUxIEZUAAAp4mgNSUxOcMYtdD++WN8gPWVUQenvSCMFJTEhJlQAAC6gkgNSURuUntlk09Zc3Vn0n\/mV+TmUvSlFRKGRCAAAT0EkBqSrJpT2oTfnS+ZC0cpKRSyoAABCCglwBSU5KNM4g960+VP\/38dbFHakhNSUiUAQEIqCeA1JRE5Ca1Gx95Xx76t7MEqSkJiTIgAAH1BJCakoiCpfbS8v1yW+WfZGXOBLm6OEnG549RUillQAACENBLAKkpycYZxIe\/OySbilpkzsr\/lvtv+5b1HbWxOWcoqZQyIAABCOglgNSUZOMmtZvv\/Kk8ePftSE1JRpQBAQjoJ4DUlGTkJrW8gkqpKM2R3HU9ZNjlo5RUShkQgAAE9BJAakqycQbR9EpveXruPrGlNrNquKRl9lFSKWVAAAIQ0EsAqSnJxk1q38kvkSfWFgpSUxISZUAAAuoJIDUlEblJ7crL75HnN\/8HUlOSEWVAAAL6CSA1JRk5g\/j79k9l\/cxjgtSUhEMZEICAZwggNSVRuUlt0sXl8srLs+QHNX3EN2q4kkopAwIQgIBeAkhNSTZuUps46j+kbsc91tX6+w75ipJKKQMCEICAXgJITUk2ziCO7+0n66btFqSmJBzKgAAEPEMAqSmJqj2pFe3xyUm9UpVUShkQgAAE9BJAakqycQbR69PP5YHJR2TYoI2yZ\/+1gtSUhEQZEICAegJITUlE7Ult8f6xSqqkDAhAAAK6CSA1JfkgNSVBUAYEIOBpAkhNSXxuUhvY94\/SfMo5ctvblyupkjIgAAEI6CaA1JTk4wzitJaBsurC7YLUlIRDGRCAgGcIILUIo7IBmtWkpKTI2rVrJT093b\/W5uZmKSoqkurqauux7OxsKS4uluTk5IAtO4NI8YkUD2uS3j0b5ZQBQ63vqbFAAAIQgEDHBJBax4zafIaBt2DBAr\/IzN8lJSVSXl4uPp\/Pep35u7Gx0RKZWYzgjPwKCwuRWgTseSkEIAABNwJILYJ2YYQ1dOhQycnJsdZij8pyc3MlIyND6uvrpaCgQEpLS\/2jN7fHzGvdRmrm8dTxH8kNz02OoEpeCgEIQCBxCCC1KGZtSy0zM9MSndvILVh89uaRWhSDYFUQgEDCEkBqUYy+srJSNmzY4D\/8aP6ura0NOIcWLD43qaWmpsqSQdus\/zons7dcXzUiilWyKghAAALxSwCpRSFb52SRpUuX+g9HdkZqFRUVYqT26AVNSC0K2bAKCEAg\/gk0NDT4d9L8npeXJzU1NVZfqnlJamlpadFcoD0KsyeCdEZq9v7d3nONfHJwoIy+9lP51wcv0rzb1AYBCEAgpgTKysrE\/HMuSK2dSMzkjvz8fGsWo1nampZv\/s95Hm3nzp2tZkN2dE5t2bJlMnjwYPnDjUeQWkzfJmwcAhDwCgEzOrNHa3V1dZbgkFqY6bV3bsye1n\/w4MFOzX40Q+aV521GamFmwtMhAAEIcE4tgjZgDi+uXLnS\/z21pqYmmTVrlsyYMcN\/Xq0z31NzSm1szhkydcWQCKrkpRCAAAQShwBSizBrI7ZFixb51+KcKGIe7MwVRYzUzGWyPt73uUy69e9yyZ0XRFglL4cABCCQGASQmpKcg4NAakqCoQwIQMBTBJCakriCg3j0qhppeKOfZC0YJFkLBympkjIgAAEI6CaA1JTkg9SUBEEZEICApwkgNSXxtSW1q4uTZHz+GCVVUgYEIAAB3QSQmpJ8kJqSICgDAhDwNAGkpiS+4CAen7ZL9tZ+KozUlAREGRCAgCcIIDUlMQUH8dTNr8pfNvaW3HU9ZNjlo5RUSRkQgAAEdBNAakryQWpKgqAMCEDA0wSQmpL4kJqSICgDAhDwNAGkpiQ+pKYkCMqAAAQ8TQCpKYkvOIin5+6TbZWHZGbVcEnL7KOkSsqAAAQgoJsAUlOST1tS+0FNH\/GNGq6kSsqAAAQgoJsAUlOST3AQW+7eKq+sOlWQmpKAKAMCEPAEAaSmJCakpiQIyoAABDxNAKkpia8tqRXt8clJvVKVVEkZEIAABHQTQGpK8kFqSoKgDAhAwNMEkJqS+JCakiAoAwIQ8DQBpKYkvuAgdm38vby0fL\/c8OxkDj8qyYgyIAAB\/QSQmpKMgoNo3vEzad5RJr5r9yipkDIgAAEI6CeA1JRkhNSUBEEZEICApwkgNSXxITUlQVAGBCDgaQJITUl8SE1JEJQBAQh4mgBSUxJfcBD\/PNogx482yKn9M5RUSBkQgAAE9BNAakoy8lIQSpBRBgQgAIFWBLzUlya1tLS0xGuGXgoiXjNgvyAAAe8T8FJfitS8397YAwhAAAJdSgCpdSne0FfupSBC3yueCQEIQKB7CXipL2Wk1r1tg61BAAIQ8BwBpKYkMi8FoQQZZUAAAhBoRcBLfSkjNRowBCAAAQi0SwCpKWkgXgpCCTLKgAAEIMBITWsbQGpak6EuCEDASwS81JeqP\/xYUlJiZV9YWOhvA83NzVJUVCTV1dXWY9nZ2VJcXCzJyckB7cRLQXipgVMrBCCQWAS81JeqlpoNcvbs2QFSM6JrbGy0RGYWI7iUlJSA55jHvRREYr1F2FsIQMBLBLzUl6qVWlNTk8yfP18++ugjycjI8Aurvr5eCgoKpLS0VNLT06124fYYUmv9lmloaJAnn3xSpk+fLqmpqV56T3VJrfD4EissApsYPAJ5ILUIuyD78GJmZqa8++67AYcfDVwzUisvLxefz2f9n\/383NxcS4D24qUgIkQW0svh4d03akgBR\/Ak2gZto73m46X2oXKkVllZKbW1tdbhxRUrVgRIzfl\/9jk0pwRzcnJaSa2iooKRiYiYT595eXkCjxNNBB6BIzXaBjzaEpv9XqmpqVHfl6qTWvChxOCJIuFIzQSxcOFCqauri+AzLC+FAAQgAIGJEyfK+vXr1YNQJTW3EVckUrM\/jRu5sUAAAhCAQOcJmPPwXjgXHzOpmRFZfn6+NYvRLGZa\/k033SRz5szxP+bEP27cOOs82s6dO0M+p9b5+HglBCAAAQh4kUDMpBYqrOCRWjizH0PdBs+DAAQgAIH4IOA5qRnsoX5PLT4iYi8gAAEIQCBUAp6UWqhXFAkVAs+DAAQgAIH4IKBeavGBmb2AAAQgAIHuIIDUuoMy24AABCAAgW4hgNS6BTMbgQAEIACB7iAQt1JL9PNubl+ZCL6TgZlws3r1aqud2V+ZsC891h2NLxbbaOs6oYnGoqP97ej\/Y5FdV2wzlH4iEVjY19pdvHix\/5q6hncofOxLaNn5mCsWOS9X2BW5tbfOuJVaIs+QtIW2fPlyf+Ny8jCXFzNXZtmwYYP\/GprB\/9\/dDbE7tme\/Qbdu3Spr1671v3kTiYXNwPC2P+QkctvoqJ9IhLZhhDZr1iw5cOBAwPtPOlJ3AAADHklEQVTCtJGO+AT3NUZwCxYsaLWe7nh\/29uIS6kl+nfZ3C4l5mRy5plnWo3Y3KPO\/kRlN2znY93ZELtjW3YHZbZl3+XBbb\/jmYVbp5OobcMt50RjYY+yzJEa5\/vC\/B5KP+p2v0u3x7rj\/R3XUgvnSv7dCTuW23I2UFNH8O177E9lQ4cOFedFoWNZczS3bbcJI23zprOl1t7hyHhk4faBx8k5kXh0JLV4f584725iPugG9wkd9aNGhOZeluZuKsEXkjfXiHS7cXM039NtrSsuR2rhXPS4OyBr2IbzMIrbpcZsqZmfzruMa6g90hqcnVfwm9ftjRvPLOxP0UbYixYtstA6z6cmIg\/7hsPmsLzzcNubb77Z6pJ88do23D7MdNSPXnHFFa2O+Bg+bbWhSN\/Hob4eqYVKysPPsw8x2CdwE7HjsmUd\/OZNRBZmctDSpUv9n65NR244tHVt1XjtyO23tHMiyOzZs\/0f6hKpbSA15R18R58w4vHwWluR2EJzdmKJ9GYN3lekVmI1Fedo3B7JzpgxQ9LS0hJmdOLcb7tPMH3HypUrrYkOBw8eTBgWSE251Do6FhzL6abdic5NaGb7iXTexPkpPJi9Ef2ECRMS6vyi20l85y2fEolHRx9+E4mFW5\/QUT\/KObVu7M1DmbXTjeXEZFPBhxydRSTajD\/nvge3jURjYdpF8El8J4ORI0cmzMzYjqTmds4oXmfGhtpndnQTZw2HquPynJoN1j4BbP42s3RSUlLibhKEmzHdvqcW\/LxE+P5NW2yCZ3klEou2bsRrn1MzX75PFB4dHX5MT09PGBbtHb1prx\/le2rdOGYJ5Zvw3VhOt26qvUNuzm\/7J8KVEoLBc0WR1leJcLuaTKK0DVtsZqajWRKVRVvvi1D6Ua4o0q3dOxuDAAQgAIFEIhC3hx8TKUT2FQIQgAAEThBAarQECEAAAhCIGwJILW6iZEcgAAEIQACp0QYgAAEIQCBuCCC1uImSHYEABCAAAaRGG4AABCAAgbghgNTiJkp2BAIQgAAEkBptAAIQgAAE4oYAUoubKNkRCEAAAhD4f7gYg5xuCGedAAAAAElFTkSuQmCC","height":0,"width":0}}
%---
