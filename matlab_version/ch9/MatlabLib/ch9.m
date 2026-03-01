%% 1. 实例化自定义 GridWorld 环境并修改奖励 (Reward Shaping)
% 最原始的策略梯度效果就是很差的  效果没有自己写的算法好。。。
% x_len = 5; y_len = 5;
% start = [1, 1]; final = [3, 4];
% % obs = [2,2; 3,2; 3,3; 2,4; 4,4; 2,5; 2,3];
% obs = [2,2; 2,4; 4,4; 2,5; 2,3];

x_len = 3; y_len = 3;
start = [1, 1]; final = [3, 3];
obs = [1,2; 1,3; 2,2];

% 创建环境对象
env_obj = GridWorld(x_len, y_len, start, final, obs);

% 【关键调参1】给每步增加微小惩罚，逼迫 Agent 寻找最短路径，而不是原地徘徊
env_obj.Reward_Forbidden = -50;
env_obj.Reward_Step = -0.1;  
env_obj.Reward_Target = 2;

%% 2. 定义 RL 工具箱所需的 Observation 和 Action
% 【关键调参2】状态归一化。我们将坐标除以边界长度，映射到 (0, 1] 之间
obsInfo = rlNumericSpec([2 1], ...
    'LowerLimit', [0; 0], ...
    'UpperLimit', [1; 1]);
obsInfo.Name = 'GridStateNormalized';

% 动作空间: 离散动作 1 到 5
actInfo = rlFiniteSetSpec(1:5);
actInfo.Name = 'GridAction';

%% 3. 创建 rlFunctionEnv 包装器
resetHandle = @() customResetFcn(env_obj);
stepHandle = @(Action, LoggedSignals) customStepFcn(Action, LoggedSignals, env_obj);

rl_env = rlFunctionEnv(obsInfo, actInfo, stepHandle, resetHandle);

%% 4. 配置并创建 PG Agent
% 【关键调参3】修改 PG 算法核心参数
agentOpts = rlPGAgentOptions(...
    'UseBaseline', true, ...            % 使用 Critic 减少方差 (变成 Actor-Critic)
    'DiscountFactor', 0.95, ...         % 提高折扣因子，让它目光更长远
    'EntropyLossWeight', 0.05);         % 添加熵正则化，保持探索性，防止陷入死胡同

% 【关键调参4】降低学习率，使其更新更平滑
agentOpts.ActorOptimizerOptions.LearnRate = 1e-4;
agentOpts.CriticOptimizerOptions.LearnRate = 2e-4;

agent = rlPGAgent(obsInfo, actInfo, agentOpts);

%% 5. 配置训练参数
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 2000, ...             % 增加训练回合数
    'MaxStepsPerEpisode', 50, ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', 0.5, ...        % 因为加入了每步负惩罚，阈值调低
    'ScoreAveragingWindowLength', 30, ...
    'Plots', 'training-progress');

%% 6. 开始训练
disp('开始训练 PG Agent (请耐心等待，观察 Average Reward 曲线)...'); %[output:3000e9a8]
trainingStats = train(agent, rl_env, trainOpts);

%% 7. 训练后测试与轨迹可视化
disp('测试训练好的策略，绘制轨迹...'); %[output:1909fc40]
simOptions = rlSimulationOptions('MaxSteps', 30);
experience = sim(rl_env, agent, simOptions);

% 提取轨迹坐标并转换为 index (注意这里提取出的是归一化坐标，需要还原)
obs_history = experience.Observation.GridStateNormalized.Data;
steps = size(obs_history, 3);
state_history_idx = zeros(1, steps);

for k = 1:steps
    norm_coord = obs_history(:,:,k)';
    % 还原真实坐标
    real_coord = round([norm_coord(1) * env_obj.X_Length, norm_coord(2) * env_obj.Y_Length]);
    state_history_idx(k) = env_obj.coord2idx(real_coord);
end

% 绘制测试轨迹
env_obj.plot_trajectory(state_history_idx); %[output:3ae1d4d4]

%% 8. 提取 Actor 网络的动作概率分布，并绘制概率箭头图
disp('提取策略矩阵，绘制概率分布图...'); %[output:9d565e4c]
num_states = env_obj.State_Space_Size;
num_actions = length(env_obj.Action_Space);
policy_matrix = zeros(num_states, num_actions);

for s = 1:num_states %[output:group:50770f61]
    coord = env_obj.idx2coord(s);
    % 构造归一化的状态输入
    obs_input = [coord(1) / env_obj.X_Length; coord(2) / env_obj.Y_Length];
    
    % 从 Actor 网络中评估当前状态的输出分布
    % 注意: evaluate 返回的是一个 cell 数组，提取第一个元素即可获得概率向量
    action_probs_cell = evaluate(agent.Actor, {obs_input}); %[output:07b18c0a]
    probs = action_probs_cell{1}; 
    
    policy_matrix(s, :) = probs(:)';
end %[output:group:50770f61]

% 调用你写的超酷的概率可视化函数
env_obj.plot_policy_matrix(policy_matrix);

%% ================= 包装器函数定义 ================= %%
function [InitialObservation, LoggedSignals] = customResetFcn(env_obj)
    % 记录真实状态 Index
    LoggedSignals.StateIdx = env_obj.coord2idx(env_obj.Start_State);
    % 返回归一化的坐标作为观测值
    coord = env_obj.Start_State(:);
    InitialObservation = [coord(1) / env_obj.X_Length; coord(2) / env_obj.Y_Length];
end

function [Observation, Reward, IsDone, LoggedSignals] = customStepFcn(Action, LoggedSignals, env_obj)
    current_state_idx = LoggedSignals.StateIdx;
    
    % 调用你的核心交互逻辑
    [next_state_idx, Reward, IsDone] = env_obj.step(current_state_idx, Action);
    
    % 更新状态
    LoggedSignals.StateIdx = next_state_idx;
    
    % 返回归一化的坐标
    coord = env_obj.idx2coord(next_state_idx);
    Observation = [coord(1) / env_obj.X_Length; coord(2) / env_obj.Y_Length];
end

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":34}
%---
%[output:3000e9a8]
%   data: {"dataType":"text","outputData":{"text":"开始训练 PG Agent (请耐心等待，观察 Average Reward 曲线)...\n","truncated":false}}
%---
%[output:1909fc40]
%   data: {"dataType":"text","outputData":{"text":"测试训练好的策略，绘制轨迹...\n","truncated":false}}
%---
%[output:3ae1d4d4]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbIAAAEFCAYAAACGvGLRAAAAAXNSR0IArs4c6QAAGcBJREFUeF7tnU+MHdWVh28vJmoWLAbPYvB0oIOADZEYiYFuQYTaC3ZjZ9FCgFcYNGIBiNEICSNEMiwi\/hhFAoaFMwLDBsPCSLFHKDtbrTFqB1kCCTZgTSymMRGMrayGliajju6jq1OvUvWq6lTd2\/ec+7UUKYnr3jrn+92uz7eq3vPc1tbWluMHAhCAAAQgoJTAHCJTmhxlQwACEIDAhAAiYyFAAAIQgIBqAohMdXwUDwEIQAACiIw1kCSB9fV1d\/Dgwana9u\/f71544QV31VVXTf5\/f8yTTz7pjh075m666abR+hhj3hdffNEdPXp0Zk3Vfvo0cOXKFffwww+7+++\/39133319hs489osvvnBvv\/22e+aZZ3Y4jzY5E0EgEAFEFggs08oJvPfee+61116bEtR3333nDh8+7M6fPz+6uKqVjiGy6pxebJcuXZoSsZxQuJFa6gxHgJk1EkBkGlMzXPOsnUaoXQgi+zMBRGb4l8twa4jMcLgaWytktby87J566qmZLZR3TgsLC5Md2969e93i4qJ7+umnJ2P9\/67eevS3zw4dOjTZIfmfZ5991n388ceTY\/0563Zk1VudfW8LNgnC7z4\/\/PBDd\/XVV7t33nlnqt7q7clbb73VvfHGG+6aa65xTVL38xW9+94eeeSRv+BYPeb55593Bw4cmPA7derUDnNfj8\/B\/8yat6jFz+Hr81zvvfded\/bsWff4449P3fos2L\/88ss7c2tcp9ScFgFEllYeVOOcKy7g5Qt3HZg6kfkLsb8w++dGxe3IL7\/8ckcAhZCqx\/hxxUW\/KrLiIl5c2It5fU3lZ3azwpslMi+eqnCqx1fP6f939RmZH+P7KMRd95eC6jGFWArh1NXZNO9111036b+o5dtvv905dxMjz\/Ldd9\/dyYMFD4ExCCCyMSgyx6gEiotgeXfgT1DIpzhZnciqculyTHEx97us6o5sz549E2FUd4hVAbQBmCWy6vPApl1LuZeiruJlj6Z6CnF7Cfsxfida3Q0Vu0IvpVdffXXqWd6sWvzLOH7em2++uZZR9VlnkWux821jxp9DoCsBRNaVFMftCoFiV\/HJJ59Mzl++Vdh0a7F8S3LWxb9oqLpzKY+5fPny5O3J8m02P67vRXmWyGbtUKpSL\/qviqzuBRlfZ\/kW5PXXX9\/6lme1zqYXX8rz3nPPPbVvUFZvf3JbcVd+hbI4KSLLImYbTRa7i+L5lJdb8fp9+RlZVWSFiIodSfW5TVVKdSJrIlj3DKru2L4iKz8fK26xfv755zv91oms\/GysWoPfzfqf6u6vely1zi6CbBKZn7s838mTJ7mtaONXMbkuEFlykeRdUNszFH9h9KLxLxWUL+xdRFbcAqt+9qptRzbGZ9X6iKx8O7B42cKvilm7yzZufnyTlMorbswdWbnm119\/3b355ps7L9TkvcrpfmwCiGxsosw3iEDbs6fyhbbvjszvbPybef6n\/JJGl2dkTfLr+oHkPiJrEk75pYumZ2TV51\/l23lNz8jKgnz\/\/fdrn5FVd7Fl2Tb9BaF8a\/OWW25xn332mXvppZdG\/fD6oMXGYDMEEJmZKO00UrwlWH25o3qB7\/qMrPyMa9Zbi8X52t5aLG6ZFTtD\/zp8208fkVVrLHY2vo+mZ2RFTeW3FuveHGx7s7FuZ9f1rcUmqRd59v3IQhtT\/hwCBQFExlpIkkD1s16+yFlfUdXl1mJxm67pc2R33nnn5LX92J8jq3vZo+5zaw899JB79NFHJ5\/NanouJf0cWfE1V2U25b9I1H32rBjT9kH1tl12kguQolQRQGSq4qLYEASKC7F\/SaT8TCrEucaas00eY51njHlCfOXXGHUxhx0CiMxOlnTSQqDpQ7rlF0i63CZMAbSmV9k9X\/\/T9k0tKXClBp0EEJnO3KhaSKD6ubS6W5bCqaMNK17NT\/2Zk5Y6owXHiYIRQGTB0DIxBCAAAQjEIIDIYlDmHBCAAAQgEIwAIguGlokhAAEIQCAGAUQWgzLngAAEIACBYAQQWTC0TAwBCEAAAjEIILIYlDkHBCAAAQgEI4DIgqFlYghAAAIQiEEAkcWgzDkgAAEIQCAYAUQWDC0TQwACEIBADAKILAZlzgEBCEAAAsEIILJgaJkYAhCAAARiEEBkMShzDghAAAIQCEYAkQVDy8QQgAAEIBCDACKLQZlzQAACEIBAMAKILBhaJoYABCAAgRgEEFkMypwDAhCAAASCEUBkwdAyMQQgAAEIxCCAyGJQ5hwQgAAEIBCMACILhpaJIQABCEAgBgFEFoMy54AABCAAgWAEEFkwtEwMAQhAAAIxCCCyGJQ5BwQgAAEIBCOAyIKhZWIIQAACEIhBAJHFoMw5IAABCEAgGAFEFgwtE0MAAhCAQAwCiCwGZc4BAQhAAALBCCCyYGiZGAIQgAAEYhBAZDEocw4IQAACEAhGAJEFQ5vPxBsbG+7cuXP5NEynSRBYXV1Nog6K2H0CiGz3M1BfwQMPPIDI1Keor4GlpSV3\/PhxfYVT8egEENnoSPOb8IYbbnBHjhxxVv6G\/Morr7gTJ064tbU1M2GSkZkoaaSGACJjWQwmwEVyMMLgE5BRcMScYBcJILJdhG\/l1Fwk00+SjNLPiArlBBCZnB0jtwlwkUx\/KZBR+hlRoZwAIpOzYyQiU7MGEJmaqChUQACRCaAxZJoAF8n0VwQZpZ8RFcoJIDI5O0ayI1OzBhCZmqgoVEAAkQmgMYQdmbY1gMi0JUa9fQggsj60OLaWABfJ9BcGGaWfERXKCSAyOTtGcmtRzRpAZGqiolABAUQmgMYQbi1qWwOITFti1NuHACLrQ4tjubWodA0gMqXBUXYnAoisEyYOmkWAi2T664OM0s+ICuUEEJmcHSMjPSO7eNG5H\/3oz7h\/9zvnFhfD4edLg8OxHWtmixmNxSbHeRBZjqmP3HPov+0jsuGBhc5oeIX9ZkBk\/XhZPxqRWU84Qn9jXSS9sOp2Wl1E5o\/xP2Ps1CxeJMfKKMJy6nQKixl1apyDagkgMhbGYAJjXCQLWa2sOHf69HRJbSLzf37okHNnznw\/1s8x5MfiRXKMjIYwHXusxYzGZpTTfIgsp7QD9Tr0IlkVVVVms0RWlljR3lCZWbxIDs0o0NIRT2sxIzEMBjpExiIYTGCMi+S+fd\/vqIqfssyaRFYnMX9r0b8MMuTH4kVyjIyGMB17rMWMxmaU03yILKe0A\/U61kWySWZ1IvOtFLcTi7bGkJify+JFcqyMAi2h3tNazKg3BAbsEEBkLIbBBMa8SNbJ7Nix6dfv\/a3D556b3sGNJTFENng5RJkAkUXBrOYkiExNVOkWOqbIfJd1MivfdvTSKt5S9MePKTFElu46K1eGyHTkFKtKRBaLtOHzjC2yOpk14RtbYohMx0JFZDpyilUlIotF2vB5pCIr76rq8FSfgdUd0\/Zih+RzZRYvktKMUl22FjNKlbWGuhCZhpQSr1F6kZybC9\/Y1lbzOfb5e5g1PxsbG87\/Z3l5eedPT1c\/3NYwbdOc1cO7zjcWIWlGY51\/7HkQ2dhEdc+HyHTnl0T10ovkbotsrkcBW7OMWEqh65x+vrfeestdvHjRLS4uugcffDBoltKMghY1YHJENgCewaGIzGCosVuSXiR7eETcknRH5k+4sLAQdEfmd29nzpxxKysrLvQOTZqRGHzggYgsMGBl0yMyZYGlWK70IjnrGZn\/s+or9tXe\/fOvtjt+KT8jQ2Ty1YzI5OwsjkRkFlON3JNUZE1l1n1jR9Oxdd\/NOLT9WBdJRCZPKlZG8goZGZMAIotJ2+i5xhRZF4lVP0c2tsxiXSQRmfwXIlZG8goZGZMAIotJ2+i5xhJZ03cn+tuH1X9Ys\/pq\/pgyi3WRRGTyX4hYGckrZGRMAogsJm2j5xpDZLO+ALjpS4NnfdHwENSxLpKITJ5SrIzkFTIyJgFEFpO20XMNFVnbt9jP+mdcQsgs1kUSkcl\/IWJlJK+QkTEJILKYtI2ea6jI\/Pcolj+bXP3aqbZ\/WLMsMz\/Wf8nwkH9cM9ZFEpHJfyFiZSSvkJExCSCymLSNnmuoyDyWQmZ1353YJjI\/3svMHzdUYn6uWBdJRCb\/hYiVkbxCRsYkgMhi0jZ6rjFEVsisbifVRWR+vD9O8rmxaiyxLpKITP4LESsjeYWMjEkAkcWkbfRcY4msCU9XkY2F1+JFMnRGY7HvOo\/FjLr2znF\/SQCRsSoGEwh9kURkgyNyoTMaXmG\/GRBZP17Wj0Zk1hOO0B8XyQiQB56CjAYCZHjSBBBZ0vHoKI6LZPo5kVH6GVGhnAAik7Nj5DYBLpLpLwUySj8jKpQTQGRydoxEZGrWACJTExWFCgggMgE0hkwT4CKZ\/oogo\/QzokI5AUQmZ8dIdmRq1gAiUxMVhQoIIDIBNIawI9O2BhCZtsSotw8BRNaHFsfWEuAiKVsYfLOHjJsfxefI5OwsjkRkFlON3BMikwFHZDJuiEzOzepIRGY12Yh9ITIZbEQm44bI5NysjkRkVpON2Bcik8FGZDJuiEzOzepIRGY12Yh9ITIZbEQm44bI5NysjkRkVpON2Bcik8FGZDJuiEzOzepIRGY12Yh9eZEtLCxEPGPYU21sbExOELqnjz76yG1ubrr5+Xl3++23B23K9xS6n6ANVCYv+llbW4t5Ws6VKAFElmgwmsryIrvtRuduu3FLU9mNtZ6\/MOcuXXFu\/x1h+\/n3X\/\/e\/delTXfD3nn3Tz\/926DsfvWbObe0tOSWl5eDnifW5Ovr6+6rr75yiCwW8bTPg8jSzkdFdV5kPz+45fbfoaLc1iJ\/9RvnTv12zp36WViRHXjm9+7sp5vurh\/Pu5O\/CCuyf\/jnOXfkyBG3urra2r+GA\/gcmYaU4tWIyOKxNnsmRCaLFpHJuPlRiEzOzuJIRGYx1cg9ITIZcEQm44bI5NysjkRkVpON2BciiwhbeCpuLQrBMUwFAUSmIqa0i0Rkaefjq0Nk6WdEhXICiEzOjpHbBBBZ+ksBkaWfERXKCSAyOTtGIjI1awCRqYmKQgUEEJkAGkOmCbAjS39FILL0M6JCOQFEJmfHSHZkatYAIlMTFYUKCCAyATSGsCPTtgYQmbbEqLcPAUTWhxbH1hLg1mL6CwORpZ8RFcoJIDI5O0Zya1HNGkBkaqKiUAEBRCaAxhBuLWpbA4hMW2LU24cAIutDi2O5tTjiGuArquQw+a5FOTuLIxGZxVQj98QzMhlwRCbj5kchMjk7iyMRmcVUI\/eEyGTAEZmMGyKTc7M6EpFZTTZiX4hMBhuRybghMjk3qyMRmdVkI\/aFyGSwEZmMGyKTc7M6EpFZTTZiX4hMBhuRybghMjk3qyMRmdVkI\/aFyGSwEZmMGyKTc7M6EpFZTTZiX4hMBhuRybghMjk3qyMRmdVkI\/aFyGSwEZmMGyKTc7M6EpFZTTZiX4hMBhuRybghMjk3qyMRmdVkI\/aFyOSwv\/zmj+6\/v\/ljpwnu+vF863FnP92sPeaRf5tzTzzxhFtaWtr585WVldb5zpw503pMcUCX+TpP1nIgH4gei6SNeRCZjRx3tQtEJsf\/4vE\/uJfe\/UOnCS7\/erH1uD0\/vdh6THHA1tZW67Fzc3Otx\/SZr\/NkiGwsVFnMg8iyiDlsk4hMzteSyE6fPu327ds3geH\/e8gdGjsy+ZqzOBKRWUw1ck+ITA7c0q1FTwGRydcCI+UEEJmcHSO3CSCy9JdCjH\/GxT9PQ2TprwWLFSIyi6lG7gmRRQYuOB0iE0BjiBoCiExNVOkWisjSzaaoDJGlnxEVygkgMjk7RnJrUc0aQGRqoqJQAQFEJoDGkGkC7MjSXxGILP2MqFBOAJHJ2TGSHZmaNYDI1ERFoQICiEwAjSHsyLStAUSmLTHq7UMAkfWhxbG1BLi1mP7CQGTpZ0SFcgKITM6OkdxaVLMGYogsJgy+2SMm7fTPhcjSzyj5CtmRJR+RQ2TpZ0SFcgKITM6OkezI1KwBRKYmKgoVEEBkAmgMmSbAjiz9FYHI0s+ICuUEEJmcHSPZkalZA4hMTVQUKiCAyATQGMKOTNsaQGTaEqPePgQQWR9aHFtLgFuL6S8MRJZ+RlQoJ4DI5OwYya1FNWsAkamJikIFBBCZABpDuLWobQ0gMm2JUW8fAoisDy2O5dai0jUQQ2T8w5pKF4eBshGZgRB3uwWeke12Au3nR2TtjDhCLwFEpje7ZCpHZMlE0VgIIks\/IyqUE0BkcnaM3CaAyNJfCogs\/YyoUE4AkcnZMRKRqVkDiExNVBQqIIDIBNAYMk2AHVn6KwKRpZ8RFcoJIDI5O0aWdmT\/eIdzt924ZYLJ+Qtz7j9+69zPD9rox4fy3DtzbnV11S0vLwfL6MKFC+7w4cOT+R977DG3srIS7Fzr6+vu3Llzbm1tLdg5mFgPAUSmJ6tkK\/U7Mn4gsLm56b7++usJiGuvvdbNz88HhbKwsIDIghLWMzki05NVspVavLX4wac\/NHWR9Bn93UP\/6v76rv3B1tHlj\/\/TffgvBybz3\/nLk27P3\/8k2Lm+OXnU\/eD8B6YyCgYrg4kRWQYhh24RkYUmPHx+RDacITOkSwCRpZuNmsoQWfpRIbL0M6JCOQFEJmfHyG0CiCz9pYDI0s+ICuUEEJmcHSMRmZo1EENkMWHwjCwm7fTPhcjSzyj5CtmRJR+RQ2TpZ0SFcgKITM6OkezI1KwBRKYmKgoVEEBkAmgMmSbAjiz9FYHI0s+ICuUEEJmcHSPZkalZA4hMTVQUKiCAyATQGMKOTNsaQGTaEqPePgQQWR9aHFtLgFuL6S8MRJZ+RlQoJ4DI5OwYya1FNWsAkamJikIFBBCZABpDuLWobQ0gMm2JUW8fAoisDy2O5dai0jUQQ2R8abDSxWGgbERmIMTdboFnZLudQPv5EVk7I47QSwCR6c0umcoRWTJRNBaCyNLPiArlBBCZnB0jtwkgsvSXAiJLPyMqlBNAZHJ2jERkatYAIlMTFYUKCCAyATSGTBNgR5b+ikBk6WdEhXICiEzOjpHsyNSsAUSmJioKFRBAZAJoDGFHpm0NIDJtiVFvHwKIrA8tjq0lwK3F9BcGIks\/IyqUE0BkcnaM5NaimjWAyNRERaECAohMAI0h3FrUtgZiiCwmk29OHnU\/OP+BW1tbi3lazpUoAUSWaDCayuLWYvppIbL0M6JCOQFEJmfHSG4tqlkDiExNVBQqIIDIBNAYwq1FbWsAkWlLjHr7EEBkfWhxbC0Bbi2mvzAQWfoZUaGcACKTs2MktxbVrAFEpiYqChUQQGQCaAzh1qK2NYDItCVGvX0IILI+tDiWW4tK1wAiUxocZXcigMg6YeKgWQR4Rpb++kBk6WdEhXICiEzOjpE8I1OzBhCZmqgoVEAAkQmgMYRnZNrWACLTlhj19iGAyPrQ4liekSldA4hMaXCU3YkAIuuEiYN4RqZ7DSAy3flR\/WwCiIwVMpgAL3sMRhh8AkQWHDEn2EUCiGwX4Vs5NSJLP0lEln5GVCgngMjk7Bi5TQCRpb8UEFn6GVGhnAAik7NjJCJTswYQmZqoKFRAAJEJoDFkmgA7svRXBCJLPyMqlBNAZHJ2jGRHpmYNIDI1UVGogAAiE0BjCDsybWsAkWlLjHr7EEBkfWhxbC0Bbi2mvzAQWfoZUaGcACKTs2MktxbVrAFEpiYqChUQQGQCaAzh1qK2NYDItCVGvX0IILI+tDiWW4tK1wAiUxocZXcigMg6YeKgWQR4Rpb++kBk6WdEhXICiEzOjpE8I1OzBhCZmqgoVEAAkQmgMYRnZNrWACLTlhj19iGAyPrQ4liekSldA4hMaXCU3YkAIuuEiYNmEbj77rvd\/\/\/vhhlIX1\/5vpWFhQUzPW1sbLi\/+pu9Zvr5v\/+55JaWltzx48fN9EQjcgKITM6OkdsE\/EXyxIkT8IBAVAKrq6um\/rIRFZ6xkyEyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCCAyY4HSDgQgAIHcCCCy3BKnXwhAAALGCPwJdbb\/LcYKaIIAAAAASUVORK5CYII=","height":199,"width":330}}
%---
%[output:9d565e4c]
%   data: {"dataType":"text","outputData":{"text":"提取策略矩阵，绘制概率分布图...\n","truncated":false}}
%---
%[output:07b18c0a]
%   data: {"dataType":"error","outputData":{"errorType":"runtime","text":"未识别类 'rl.agent.rlPGAgent' 的方法、属性或字段 'Actor'。"}}
%---
