% 1. 初始化环境
clear

x_len = 3; y_len = 3;
start = [1, 1]; final = [3, 3];
obs = [1,2];
% obs = [1,2; 1,3; 2,2; 3,2];

env = GridWorld(x_len, y_len, start, final, obs);
env.Reward_Step = -0.1; % 记得加上步数惩罚

% 2. 初始化神经网络版 REINFORCE
alpha = 0.0001; % NN通常使用较小的学习率
gamma = 0.95;
agent = REINFORCEAgent_NN(env, alpha, gamma);

% 3. 开始训练
agent.train(1500, 200); %[output:15d62e41]

% 4. 画图验证
policy_matrix = agent.get_policy_matrix();
env.plot_policy_matrix(policy_matrix); %[output:9d75f181]

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":34}
%---
%[output:15d62e41]
%   data: {"dataType":"text","outputData":{"text":"Episode 100\/1500, Total Reward: -725.10\nEpisode 200\/1500, Total Reward: -574.40\nEpisode 300\/1500, Total Reward: -642.60\nEpisode 400\/1500, Total Reward: -489.70\nEpisode 500\/1500, Total Reward: -493.00\nEpisode 600\/1500, Total Reward: -406.10\nEpisode 700\/1500, Total Reward: -348.90\nEpisode 800\/1500, Total Reward: -175.10\nEpisode 900\/1500, Total Reward: -218.00\nEpisode 1000\/1500, Total Reward: -49.70\nEpisode 1100\/1500, Total Reward: -1703.00\nEpisode 1200\/1500, Total Reward: 91.10\nEpisode 1300\/1500, Total Reward: -396.20\nEpisode 1400\/1500, Total Reward: -29.90\nEpisode 1500\/1500, Total Reward: -49.70\n","truncated":false}}
%---
%[output:9d75f181]
%   data: {"dataType":"image","outputData":{"dataUri":"data:,","height":0,"width":0}}
%---
