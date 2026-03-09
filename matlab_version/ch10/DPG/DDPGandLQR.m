% ddpg_lqr_stable_optimized.m
% 引入目标网络(Target Networks)和经验回放池(Replay Buffer)的稳定版 DDPG
clear; clc; close all;

%% 1. 环境与基准 LQR 定义
A = [1.1, 0.5; 
     0.0, 0.9];
B = [0.1; 
     1.0];
Q = eye(2);
R = 1;
[~, ~, K_lqr] = dare(A, B, Q, R);

%% 2. 构建 Actor 和 Critic 及其 Target 网络
% --- Actor 网络 (保持纯线性无偏置，等效于 K 矩阵) ---
actor_layers = [
    featureInputLayer(2, 'Name', 'state_in')
    fullyConnectedLayer(1, 'Name', 'action_out') 
];
actorNet = dlnetwork(actor_layers);
idx_a_bias = find(actorNet.Learnables.Parameter == "Bias");
actorNet.Learnables.Value{idx_a_bias} = dlarray(0); % 强制 Bias 为 0

% --- Critic 网络 (多层感知机) ---
critic_layers = [
    featureInputLayer(3, 'Name', 'state_action_in')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(1, 'Name', 'q_value_out')
];
criticNet = dlnetwork(critic_layers);

% --- 克隆出 Target 网络 ---
actorNet_target = actorNet;
criticNet_target = criticNet;

%% 3. 超参数与经验回放池初始化
num_episodes = 600;      % 训练回合数
steps_per_episode = 50;  % 每回合步数
batch_size = 128;
gamma = 0.95;            % 折扣因子 (深度RL中必须 < 1 以防止 Q 值爆炸)
tau = 0.005;             % 软更新系数 (非常关键)

learning_rate_actor = 1e-4;
learning_rate_critic = 1e-3;

% 探索噪声 (随时间衰减)
initial_noise = 1.0;
noise_decay = 0.995;
current_noise = initial_noise;

% 经验回放池预分配内存
buffer_capacity = 20000;
buffer_X = zeros(2, buffer_capacity);
buffer_U = zeros(1, buffer_capacity);
buffer_C = zeros(1, buffer_capacity);
buffer_X_next = zeros(2, buffer_capacity);
buffer_idx = 1;
buffer_size = 0;

% 优化器状态
opt_actor = struct('avg', [], 'sq', []);
opt_critic = struct('avg', [], 'sq', []);

error_history = zeros(num_episodes, 1);
fprintf('--- 开始稳定版 DDPG 训练 ---\n'); %[output:05b26563]

%% 4. 训练主循环
for ep = 1:num_episodes %[output:group:67677e56]
    % 随机初始化状态 (限制在一定范围内，防止二次型代价过大导致梯度爆炸)
    x = (rand(2, 1) * 2 - 1) * 2; 
    
    for step = 1:steps_per_episode
        % --- 步骤 A: 与环境交互并存储数据 ---
        dlX_curr = dlarray(x, 'CB');
        u_policy = extractdata(predict(actorNet, dlX_curr));
        u_action = u_policy + current_noise * randn(); % 探索噪声
        
        % 环境步进
        x_next = A * x + B * u_action;
        cost = x' * Q * x + u_action' * R * u_action;
        
        % 存入 Replay Buffer
        buffer_X(:, buffer_idx) = x;
        buffer_U(1, buffer_idx) = u_action;
        buffer_C(1, buffer_idx) = cost;
        buffer_X_next(:, buffer_idx) = x_next;
        
        buffer_idx = mod(buffer_idx, buffer_capacity) + 1;
        buffer_size = min(buffer_size + 1, buffer_capacity);
        
        x = x_next;
        
        % --- 步骤 B: 如果 Buffer 数据足够，则进行一次批量训练 ---
        if buffer_size > batch_size
            % 1. 随机采样 Batch
            sample_indices = randperm(buffer_size, batch_size);
            X_batch = dlarray(buffer_X(:, sample_indices), 'CB');
            U_batch = dlarray(buffer_U(:, sample_indices), 'CB');
            C_batch = dlarray(buffer_C(:, sample_indices), 'CB');
            X_next_batch = dlarray(buffer_X_next(:, sample_indices), 'CB');
            
            % 2. 计算 Target Q 值 (使用 Target Networks)
            U_next_target = predict(actorNet_target, X_next_batch);
            XU_next_target = [X_next_batch; U_next_target];
            Q_next_target = predict(criticNet_target, XU_next_target);
            Y_target = C_batch + gamma * Q_next_target;
            
            % 3. 更新 Critic 网络
            [loss_critic, grad_critic] = dlfeval(@criticLoss, criticNet, X_batch, U_batch, Y_target);
            [criticNet, opt_critic.avg, opt_critic.sq] = adamupdate(criticNet, grad_critic, ...
                opt_critic.avg, opt_critic.sq, ep, learning_rate_critic);
                
            % 4. 更新 Actor 网络
            [loss_actor, grad_actor] = dlfeval(@actorLoss, actorNet, criticNet, X_batch);
            
            % 依然强制把 Actor Bias 的梯度清零，保持纯线性 K
            grad_actor.Value{idx_a_bias} = 0 * grad_actor.Value{idx_a_bias}; 
            
            [actorNet, opt_actor.avg, opt_actor.sq] = adamupdate(actorNet, grad_actor, ...
                opt_actor.avg, opt_actor.sq, ep, learning_rate_actor);
                
            % 5. 软更新 Target Networks
            actorNet_target = softUpdate(actorNet_target, actorNet, tau);
            criticNet_target = softUpdate(criticNet_target, criticNet, tau);
        end
    end
    
    % 每回合结束，衰减噪声
    current_noise = max(0.01, current_noise * noise_decay);
    
    % 记录当前的 K 矩阵误差
    W_current = extractdata(actorNet.Learnables.Value{find(actorNet.Learnables.Parameter == "Weights")});
    K_current = -W_current;
    error_history(ep) = norm(K_current - K_lqr, 'fro');
    
    if mod(ep, 50) == 0
        fprintf('Episode %d: Noise = %.2f | ||K_actor - K_lqr||_F = %.4f\n', ... %[output:85dae429]
            ep, current_noise, error_history(ep)); %[output:85dae429]
    end
end %[output:group:67677e56]

%% 5. 最终验证与输出
fprintf('\n--- 训练完成 ---\n');
disp('最终 Actor 学习到的增益矩阵 K_actor:');
disp(K_current);
disp('理论最优 LQR 增益矩阵 K_lqr:');
disp(K_lqr);

% === Actor 网络输出与理论 LQR 状态反馈对比测试 ===
num_test = 200;
X_test = randn(2, num_test) * 5; % 使用更大的范围测试泛化性
dlX_test = dlarray(X_test, 'CB');
U_lqr_test = -K_lqr * X_test;
U_actor_test = extractdata(predict(actorNet, dlX_test));

mse_test = mean((U_actor_test - U_lqr_test).^2, 'all');
fprintf('=> 测试集上 Actor 与 LQR 输出的均方误差 (MSE): %e\n', mse_test);

%% 6. 可视化
figure('Name', '稳定的 DDPG 收敛与对比', 'Color', 'w', 'Position', [100, 100, 1000, 400]);

% 子图 1: K 矩阵参数收敛
subplot(1,2,1);
plot(1:num_episodes, error_history, 'LineWidth', 2, 'Color', '#0072BD');
set(gca, 'YScale', 'log');
xlabel('回合数 (Episodes)'); ylabel('参数误差 || K_{Actor} - K_{LQR} ||_F');
title('Actor 增益矩阵收敛过程');
grid on;

% 子图 2: 测试集输出残差对比图
subplot(1,2,2);
plot(1:num_test, U_lqr_test, 'o', 'MarkerEdgeColor', '#0072BD', 'MarkerFaceColor', 'none', 'LineWidth', 1);
hold on;
plot(1:num_test, U_actor_test, 'x', 'Color', '#D95319', 'LineWidth', 1.5, 'MarkerSize', 6);
% 画误差线
for i = 1:num_test
    plot([i, i], [U_lqr_test(i), U_actor_test(i)], 'k-', 'Color', [0.5 0.5 0.5 0.3]);
end
xlabel('随机测试样本'); ylabel('控制动作输入 u');
title(sprintf('动作预测对比 (MSE: %.2e)', mse_test));
legend('理论 LQR (u=-Kx)', 'Actor 网络输出', 'Location', 'best');
grid on;

%% --- 辅助函数 ---

function [loss, gradients] = criticLoss(criticNet, dlX, dlU, Y_target)
    dlXU = [dlX; dlU];
    Q_pred = forward(criticNet, dlXU);
    loss = mse(Q_pred, Y_target);
    gradients = dlgradient(loss, criticNet.Learnables);
end

function [loss, gradients] = actorLoss(actorNet, criticNet, dlX)
    dlU_pred = forward(actorNet, dlX);
    dlXU_pred = [dlX; dlU_pred];
    Q_val = forward(criticNet, dlXU_pred);
    loss = mean(Q_val, 'all');
    gradients = dlgradient(loss, actorNet.Learnables);
end

function targetNet = softUpdate(targetNet, sourceNet, tau)
    % 将源网络权重以 tau 的比例软更新到目标网络中
    num_vars = size(targetNet.Learnables, 1);
    for i = 1:num_vars
        targetNet.Learnables.Value{i} = tau * sourceNet.Learnables.Value{i} + ...
                                        (1 - tau) * targetNet.Learnables.Value{i};
    end
end

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":34}
%---
%[output:05b26563]
%   data: {"dataType":"text","outputData":{"text":"--- 开始稳定版 DDPG 训练 ---\n","truncated":false}}
%---
%[output:85dae429]
%   data: {"dataType":"text","outputData":{"text":"Episode 50: Noise = 0.78 | ||K_actor - K_lqr||_F = 1.8651\n","truncated":false}}
%---
