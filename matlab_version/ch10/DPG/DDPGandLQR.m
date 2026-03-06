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
fprintf('\n--- 训练完成 ---\n'); %[output:148c73cb]
disp('最终 Actor 学习到的增益矩阵 K_actor:'); %[output:18a5c005]
disp(K_current); %[output:87f1b0a8]
disp('理论最优 LQR 增益矩阵 K_lqr:'); %[output:953f3624]
disp(K_lqr); %[output:21aafca9]

% === Actor 网络输出与理论 LQR 状态反馈对比测试 ===
num_test = 200;
X_test = randn(2, num_test) * 5; % 使用更大的范围测试泛化性
dlX_test = dlarray(X_test, 'CB');
U_lqr_test = -K_lqr * X_test;
U_actor_test = extractdata(predict(actorNet, dlX_test));

mse_test = mean((U_actor_test - U_lqr_test).^2, 'all');
fprintf('=> 测试集上 Actor 与 LQR 输出的均方误差 (MSE): %e\n', mse_test); %[output:5c2b802f]

%% 6. 可视化
figure('Name', '稳定的 DDPG 收敛与对比', 'Color', 'w', 'Position', [100, 100, 1000, 400]); %[output:9bc0025b]

% 子图 1: K 矩阵参数收敛
subplot(1,2,1); %[output:9bc0025b]
plot(1:num_episodes, error_history, 'LineWidth', 2, 'Color', '#0072BD'); %[output:9bc0025b]
set(gca, 'YScale', 'log'); %[output:9bc0025b]
xlabel('回合数 (Episodes)'); ylabel('参数误差 || K_{Actor} - K_{LQR} ||_F'); %[output:9bc0025b]
title('Actor 增益矩阵收敛过程'); %[output:9bc0025b]
grid on; %[output:9bc0025b]

% 子图 2: 测试集输出残差对比图
subplot(1,2,2); %[output:9bc0025b]
plot(1:num_test, U_lqr_test, 'o', 'MarkerEdgeColor', '#0072BD', 'MarkerFaceColor', 'none', 'LineWidth', 1); %[output:9bc0025b]
hold on; %[output:9bc0025b]
plot(1:num_test, U_actor_test, 'x', 'Color', '#D95319', 'LineWidth', 1.5, 'MarkerSize', 6); %[output:9bc0025b]
% 画误差线
for i = 1:num_test
    plot([i, i], [U_lqr_test(i), U_actor_test(i)], 'k-', 'Color', [0.5 0.5 0.5 0.3]); %[output:9bc0025b]
end
xlabel('随机测试样本'); ylabel('控制动作输入 u'); %[output:9bc0025b]
title(sprintf('动作预测对比 (MSE: %.2e)', mse_test)); %[output:9bc0025b]
legend('理论 LQR (u=-Kx)', 'Actor 网络输出', 'Location', 'best'); %[output:9bc0025b]
grid on; %[output:9bc0025b]

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
%   data: {"dataType":"text","outputData":{"text":"Episode 50: Noise = 0.78 | ||K_actor - K_lqr||_F = 2.1923\nEpisode 100: Noise = 0.61 | ||K_actor - K_lqr||_F = 2.2503\nEpisode 150: Noise = 0.47 | ||K_actor - K_lqr||_F = 2.2806\nEpisode 200: Noise = 0.37 | ||K_actor - K_lqr||_F = 2.2846\nEpisode 250: Noise = 0.29 | ||K_actor - K_lqr||_F = 2.2884\nEpisode 300: Noise = 0.22 | ||K_actor - K_lqr||_F = 2.2886\nEpisode 350: Noise = 0.17 | ||K_actor - K_lqr||_F = 2.2865\nEpisode 400: Noise = 0.13 | ||K_actor - K_lqr||_F = 2.2840\nEpisode 450: Noise = 0.10 | ||K_actor - K_lqr||_F = 2.2922\nEpisode 500: Noise = 0.08 | ||K_actor - K_lqr||_F = 2.2960\nEpisode 550: Noise = 0.06 | ||K_actor - K_lqr||_F = 2.2962\nEpisode 600: Noise = 0.05 | ||K_actor - K_lqr||_F = 2.2990\n","truncated":false}}
%---
%[output:148c73cb]
%   data: {"dataType":"text","outputData":{"text":"\n--- 训练完成 ---\n","truncated":false}}
%---
%[output:18a5c005]
%   data: {"dataType":"text","outputData":{"text":"最终 Actor 学习到的增益矩阵 K_actor:\n","truncated":false}}
%---
%[output:87f1b0a8]
%   data: {"dataType":"text","outputData":{"text":"   -1.4897    0.2654\n\n","truncated":false}}
%---
%[output:953f3624]
%   data: {"dataType":"text","outputData":{"text":"理论最优 LQR 增益矩阵 K_lqr:\n","truncated":false}}
%---
%[output:21aafca9]
%   data: {"dataType":"text","outputData":{"text":"    0.7064    0.9455\n\n","truncated":false}}
%---
%[output:5c2b802f]
%   data: {"dataType":"text","outputData":{"text":"=> 测试集上 Actor 与 LQR 输出的均方误差 (MSE): 1.118592e+02\n","truncated":false}}
%---
%[output:9bc0025b]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXMAAACVCAYAAACq0Bq5AAAAAXNSR0IArs4c6QAAIABJREFUeF7tnQm81VP3\/1dSmlUapDRrNJTMlCJREoVkLpl5TJmHxxAeDyKzTGUeUjJThh4lZIiUijTTTNEkDf\/\/e9\/fOvbd9zud+dx7z3697uvee873u+f9WWt\/1tprl9m6detWiTOtX79err76atlpp53kqquukp9++kluv\/12Of74483v4cOHyy677BJnrvnH8z2Q74F8D+R7INEeKJMImP\/2229y8803y4033ig1a9Y0Zb\/88svywAMP5IE80ZHIv5fvgXwP5HsgiR5IKZhTjxNOOCGJ6hSPVzdu3Chr166V7bffXrbZZhtT6RUrVkj58uWlWrVq5v8\/\/vhD\/vrrL6lVq5aUKVOmUMO2bNkiCMQddtihyHfLly+XihUrSpUqVQq9o\/nxjpaZbG9t2rRJFi5cKDvuuKMpMyhR59WrV0vlypVNO6kPP7y77bbb+r5KH6xbt06qV68eayvlzpw50+zsVBmIpy3kOXXqVGnSpInpX6\/EhpMf7SuvMXPfY8f5999\/x8aQvxm7oPbFU2\/7WepGP5QrVy40C56jD5kTUcde+4jx2XnnnWNl0MYNGzYUmrt86fd5aOXyD+RMD+TBPIGh+Pzzz+W\/\/\/2vPPnkkwaM1qxZI9dcc41ZNJdddpmULVtWHnzwQfnyyy\/lvvvukxo1ahQqhc8vvfRSufvuu2W\/\/fYrtNCgrwDsa6+9thCIfPTRR\/LSSy\/JPffcEwP6JUuWyOjRo43QcNNBBx0ke++9t0yZMkVuueWW2NfdunWT\/v37G\/BevHixDBw4UM477zw56qijAnsC4cOz0GrUeeLEiXLTTTeZnZgNFm4m77zzjukn+qNevXoxQXfhhRcawX\/kkUcWemXOnDny6KOPyr\/+9S\/ffN26kAF98Msvv8js2bNl0qRJ8t5778lee+1l6jx06FDzHe1t2rRprF\/\/\/e9\/S\/v27WPl\/+9\/\/5PBgwfLE088IY0aNTJ1njVrlhlrhJib3L7l+zZt2kiLFi1kzJgxnv3ZoEEDMx7U88orr5Tzzz\/f9Oebb75ZaBxRFPr27WvGWmnMIUOGRBZ+2kf9+vUrpGCxg2Ye6dzVSvp9nsDyyL+SpR4oFmDOYho2bJi88MILhcAvSp999913ZnGFaZ5R8tJnbDBH40RjZWEDcGeccYagXQPG\/Oy6665FtCA0rYceeki+\/vrrQmAPYAPwDz\/8sDRu3LhQlQAWyjjttNOMsEBb3G677QzoAQb169cv9LyCOXW9+OKL5eijjzbARD+g0fI5Gtq8efNMWRUqVIi9f9ZZZ0mPHj1MeU899ZQBHzRJG8ypD9qxLVzcPgRAAasFCxZI165d5YYbbjBa77Rp0+SSSy6R\/\/znP9KsWTPzGto+wIVgBLRmzJgh999\/v9SpU6fI0HiB+fPPP2\/yd9Odd95pKED6A8AcNGiQeYR+vu2222LzSe1AlIfAYqd1zjnnyJlnnukr6BA8r7\/+uskPzX\/s2LEGzGnzypUrzeeffPKJfPjhh3L55ZdL1apVzZjtvvvuRiMfP368oSoRsPQlc4dn6DfqisBj3lMPftMv9NMrr7xi5pwm5gKCG+Glu4hMgTnz6MUXX5Q77rhD3njjDU9BofXUOjFf6WOS9jvt1YRiQX66Zhk\/lCW\/lAguxCsgqTtzhzVt2wPdujGnU81OUNcBAwbIr7\/+arrALUPrkDCYs7ABSq+0xx57FJH88YCl\/SwNQYMhNWzYsNAgh+VJI9HS7IkR9k6U720w53mAqVWrVmbyAeho6yxqFuXPP\/9s+gKgYrJDa5AAC0DtgAMOMAsUGoP+ZFvfrl078wwLv0+fPkYYoLHzPO8BiGiYJ598sgFaV8O326B1feSRR4yGSr0ArOnTp5u6sm0H6OxkCwLdgfC9gjkCCnBCaPm9CygDVCR2IeQDAFHXjz\/+uMjiBLB0geu7zZs3l82bNxswtBPAiWaLlktf2Bq2C\/S6aE888UShDxgLuy26M0KQIgwATdoXBCDQQ2rk\/+GHH6RSpUry448\/mv7lh8X+559\/GhqFXQZ\/646NecB4a6KtzAmEDvOcdiHg7rrrrhiI22DO\/KC9ixYtiuXB37179zZ9qpResmD++++\/m\/l67LHHFqH8tGAX4FS7b9u2rVx33XVFFCgUAMZSwdyvjswV5q3uHlK9jhUca9euHQmntJ7Md9u5w93N+LUnCqb4PePmqXXXNW\/XISEw14JdiaEgPm7cOLNNtSmERBukA4lmySSOx1Mm1ZOArTyLDo2KBQTYAFRoFR06dDC0wOTJk42miyaL5gal4gXmUfpDwZyFBXjiKYSmjABA+mv\/Q3fsueeesSxZ0MrnK5ijhfI+AAvYQNFAr1xwwQXCpEYAuXSJK7QUzMkbjRUgRaO3wRXtZf\/99zdjxeRX7RrQop4s9C+++MIAJponAgUq6rHHHitERwGAUBtou2jpdkKbQyPu2LGj2ZEwNwB2ykBgXHTRRWY3wph89dVXZvdCHe2EBkt\/HHfccbJs2TLzTt26dY3gZwGdffbZ0r17d7ObQYCyW1EQ5l2EIm0fMWKE2WXRFwA5ZXppm1q2apHMJbRzhKkKHED7rbfeMl5h5KUgboO5a2MgH7RF5hyCwNZwtUxb+EShWegrxo9xYjfArsorkRc7OxXCut4oDyXE1mDp08cff9wINnYfvOPSlbaQsHeBqVzHKqSZO9h8XLrJbSd1POmkkwRsI7H+aZeOsa5FfS\/Ruvr1hVd++hljhDKndUgYzBVIWCi6rdCOSpVmbnfYYYcdZjRDlwN0F45O3G+++Sam\/dmT2RVAtkZI\/b\/\/\/nsz4VgU7nYGTQuNFoBgUQKwaFEs5nPPPddsf1lcaL4YrNj2qjZoL0KAHi0cQGJLDpiSN0BNQsviM00MNHwuAAMgumCu2y993u5\/nSSnnnqqfPDBB2Yyoh3SPrbHaIB85k5K8vIDcwTUM888YwQroIxmSftvvfVWw0EDvoAA4Ipw08TCoX8AThY7E\/G1114zzwCsCC+vhOBEQCmV5GrfCDuEEvW1Exoz9oFRo0YZ4WUn8qPdRxxxhBFw1IPt\/fXXX28AnTlAG6C86N+g3R1zhoWFMGDOAPLspNgNMBYYUgFIBAIg1rJlS7Obo73MIepC+6OCOcZ3+pE5haBhXtD\/lE+9SfxGezv00EPlkEMOMcIHgeO146DOJLh0+pF+ITE3FMTccfGiumzFC0Fu0w18RwL8SQrmtD9MQYsKkGj0mrfXPLJ3EtBgtt3L63nFFsYROxY7JwVzz4n6f159Nhvg4pMfNvqBuVeb\/CiihMDcbqSXAY+GpoLaoIH2YLuDqvWwy7M1D3YIdseqlFXtyOXwdKKHcXAuyAEGLE4ogaefftoY+tDq5s6da4BepT8gznYcTRSND+A4+OCDjXYGzcJnuHdCQ6A5sLB22203oyni\/QEgoN3wWadOnQw\/Sv\/YNIs7KfT\/e++914wJRjUAF80VnhVQRWvfd999Y9ywUgFeYA6\/+9xzzxlhxa4EDp\/n1HBI221PEvtvxou6As4IL+pDewAZtHR34gMybPPpE\/h7NSa7QILQQJAC2Iwh2jkCEVpOKSN77lC+Lsz58+cboQ\/YYf+gP1hAaMbYWvib+YDmq54k7IJ4n2egZdwEKCMAAAHyZw7yPG1UAzJgB+1AP55yyikxTTxMM2dXxC4G4UN\/MOfIm79RLBAcCIkgmoX3lcOn7uyiqSP9wJyiPsxn9czyAi4v8NE+1r5V12XGlTl2+umnG4O9Aq473rZiZZcZFcz9ADZq\/YPeV7o3CMzd8zdeY+C3M\/LqTz\/t368uCYG5l585HUGF0FopzPZBj6eT7WddqeTyRWEd7E4CLylndyLg72Xpd+tvv+PFYdrPq\/cCi5CJDJfIbgaDhpeHBO8CSmi+LHj6kXaixbpcMZoPgBAFzBEon376qaGIADooFjQ8r6TCzAvMMYziFQKdQP8CaBgfsQ9QR6gIAAVwBfTgSNllINygA9hd0SecUwCo0crQntEedecDENImNE7aSD+o8KHv2B7b23BtAxo+RkTapYBLW9iFICxJyinzG0GKRoxdg\/qr4LfdE5kztAGqQd0IoaN69eplxoMfdoGAKcIHwUOd2XUFgTl1oU78IPTj0czRyhl3PIXYGamQZDxQCBCsfOa1kw2iWag7Qp8dpetO686ToO0\/Qhp6jf5F2WP+ouQw9\/nc1Z5VybLLsHfFXrsJfdY1lkbFGj9N2O\/9MKzhPZfrt43Dasy1AZo5ahs27bLR4Bljr11zRsCcyvgBfdRO1ue8tnGu5AsbEHvCkS9bX5dOsLdeLMooBlO3XNurwW0nfC7GShKLEIMZgORSI+57aCnwz7aPM3QGFAWLlUWsws32ZrGNrmzD7bqyK2DCoZlBQSBM\/KS\/CmcvA+g+++xjtF8AFsEwcuRIY4y13QyhjNA60XihiNB4dXLiqsnf77\/\/vgEzjIQ2X++l0dAOjJT0HWDrBebQPLgEsrOA6gKkEYpskWkH48QCgtMHLKBQMFwDXH7aH++F0Szax+y00NShXKAowsAc4YVHEcIxHjBnbHgX8OWHchFi6j3EuGBEjxfMoygyOk\/DwBwakXFnJ6IUC0IujAqxtXVVKoqDZk67UBpcA6mfF45L4WZNM\/ejWRQA1FUpGXfAIGmsvJOCk58RIwqY21IuHjBnMOCvAQ1AVt3\/bAmMdobGZlNRlAcYo5kCNqqtw2fDJwJWLFB4VzUs6QJiwKE10Kp5Do0Oyc622t4S624A0NRJQl7smlx3wjAwR5vCXsGuhfJsCgqgpA+gawBnm+eHFmFcqB+\/oQDQ2qEaAE91s8QGQt1soRXmFeAl6AEyNH2Amn6knxBa9DW7IgQjwg23TJ6hjwFe7CmkIDB3NXPbB9wWepTFroH5yS4EMKcu2BjwZrFpFt5TP3UENH0ZlTPX+YBwgytHQKHx0v+Mg\/qwU14yfubMSYQd8ylezZx5xRyHcsFuBMWC4TAMzCnHVdpyHcy9gFw19TBFwF7bXhx+VM6c5xKiWaiAnwE0FUf63cG0J5Jdrm6l\/XisRGiWMM2cLT+aKDQA226oBThHQM\/mIdUAZfsya7+hhWFcQxjZPrrqGeA1gGhiCAbcE9F48cIANNDIorgm6tYfgytUC4KLhGZMuQgDPU15+OGHG2Md9eAADWANzYDmZrcH7QuvFrRyhJm9NUfb5Fm0QzhtttksaLh5QB0gg14ByGiD\/a4XmFNP8sBdDmoD33W0f+pGgu5BaFBvaB+EKs8giPCIgbunDH40HBGuiJQdBuYuZ24LS3cHg8BQryfKRljD3dNHaOyUh82COgC8CCE+Q6Nj14gxFSGAIMK+AV3FroZdhH1oSKknTniiTFAG4MncoK\/o4yiauZ5EpZ6uZh4EvGGaOf3P+7pDUldFzZNxt+M7ucLCLjuXwdylVux2+FFaRSSj42xgO0tE7WeUtoTBnArpolN\/81R5sQTxU7bRU11zNOCXgqUaYFyLdRQDaBCYs0gBABYcCxBPCNysyDcezTwRMIcLBYhZsNAjaOZwyiz6KGCuuxflpdXrwTWq0odoeHhcUCaggPEPjxGb2gCE0HYBFfoF7bpnz56FQJmxwnCLZgsQw2UjHNDQ0FQxKJIvCgDlkdBsAWaMrHC3CAS0aRLC7+233zZaPIDIjoe\/AS92SrQRAyz1YXyYF7g+0kZ2OmjUPI\/w\/eyzz4ocTPEa+3hoFnuHCEhizGRsEPKUB4ABviR1DcXQCLfMuDAv2E2xi8HQrLsV13tBKRY0Z8afPmQtAPbaV0EGUIQTSgg7NWgpDJ4kG8zVPRHBhYbvJi+PChd4dHdtUwo2SOt6dCkHd\/3nKpiHgbXXISnX7ucF7PZnGfMzD6tIot+HLSDby4XJhoRX\/1rbDdEWNkoPhLkmhmnmLBw7JotqZmFgrrEy4G2pC9tjtmAsSLRJ2oQ2iwZtc4xscwErgIBFzqIFYPEBBqBYaF4nQNVIhwYe5IIVRLN4TSpAG1qH34A8YIumT70QNoAo2iEANmHCBMNXYzRFs4fnx20RLa1z587GYwLtmlOS1BHXQ+gJ+gWwAYihDygHDRzBaSeENa6iuF2i8XMylt0SuwXqR3\/R1wAegk8PZSEw8FxhZ4BnkLrsxQvmgCpCkZO8ttcSnyG8+KFfunTpYoAWAzS7q9atW5sxpm5QPWiv7PagY6g3fUF72H2xk7CBE0EObcHuA2Ml3jrKvWMr0DqtWrUqtuvBrkHiO07sYqMgIfjwaWcHgNDFO4l5jBcPvvc8yzhgnHaTF9Xqgq6XUuZq+65CSDmuUhhEufK8CoMoFI62w4ujDrLBuW3xqrfmbdff67kwbzkvwel1AtTNOynN3AusU2UATVQQZOO9KJq5ak\/2yb0odQUMAB9AD6Oafbw9KDZLusAcqoStP0JG3ddUC0Xrpo5ovhh6qYMdTAwgBywAKfJBKOjxfUAKgUVMFd7TmClo\/tAkCCUEoZ3wq0ajt43MNp+OZqtapdo2MNwiDKElADM4f7TZKAZQxs5eVPDy7BYYf6gRQJrxIvE3oEs5JIQ5\/y9dutQYLCkP4ydChrHFloHnErsixhUtHdsI4E\/7EYYIBOYCAg\/aBhrH9Trhez2MhWKD6yd10ToAeAhb+p++U3dL+o2yqIcmlATcY92gb\/o9bbAPDUWZz\/ln0tcDeTBPQd96ReRzowwmU4wbATCevMKiBZI3miQabZDB2m6PRhJ0gQTNLyzCIPWxj7PTlmTa5\/aF2+9R80bTxEsG7d9uF8KGPPlONX5AToUGgoQfNFl2iNhveF7\/tk9B+tWFfgNUFVjt5yifsu3dYFA\/63jif05b3PGIMkZR55ftCZa\/vyBqr6XvuTyYp69v8zmX8B4IMtLxHfRQqoMu5VqXevlS51odS0t98mBeWka6lLYziNukS5Ix2ttg7vK18fC3pXRo8s1OcQ8kBObpXCApbl+h6HKpzjufX+Z7wMvnOZFaJEsRuF4JribuxyfHazNJpG35dzLXA6maj6mocSiYM+m\/\/fZbT4u2WwGs9xyvTeT2mFQ0xs2DhXPFFVcYz498ymwPzO39hMjWLVLhtzmyYYfmpvBt162QTZW8bwaKp3YLL23heZAlnjx4NlGKwMvlLAqY5+djvCOU+8\/jdYTRPxdAPRKYex2dtrtZJzefhYWUzOTwqB8rne1e3pDJepTksr78Zb1c+cZM6dS0utx8eBM58PllsqV84SvvUt3+VIJ5kNumV729gJznotAspWU+ElwMzyU8cMIM4qmeG5nMDyURDyBca4sFmNM56ufp5R+pE9T2785khwaVpXXLlc7OlX5Jph5lBn2UzOtxvbvghgNl85at0rhmwS1IqR7PsPMMbmWDwgy4tIqXATTV9Y+rMzP4sMabwcW0JIN5ro1nqGauc8Ar\/gCf6bVWbhyRDM4d36JyrbNzoU+SqcO5r86SYZ\/9klAWjWpUlDJltsq83zZIi9qV5Mfl60w+5+xfX64+pJHUrlJOypYpIxXKFVyQ7ZUSGc8g+048CkhQiAnqah8q4X+v2NeJ1D+hzs7yS3kwz84ARAZzqqdxCABuTvhxUi\/e00yZbKYuHgISsQ2ytQQvf1s+I2XyucrX\/E\/WbdxcqFvQRDs3qyFP9G0lZbcpY76btWydtPrv59J3jzryynfLjLbarUVNeah3c9nlvwUXQEy5bB9Z8ecG853bBq92xfNZv+dnyKipyyMP39YhBSFtWdgc\/FEtza+Po9Qlm2DonhzWjrBDsNonFb3ugsxm\/SMPXAoezIN5CjoxgSziAnMFdLTxZFy6EqhnQq\/o4uHoMzExNG4FmREB0P5fP+N3Jp5DQ21y26SE2hXPSx+d114al10lc1f9LV2fXyRb7j5E\/t68RRYtmG+ycdvqfjbhu5\/kqo9WyGeL1nsWi+CYe90BwnOjZv4pJ+3fXPZpWM08Sx9z8pGTngrmfBalXK\/niCNCTJfiSpvlwTyemZveZ1e+MkQqtt1fKrU9IFbQHx+\/IuumT5IdLxwaqfBcG8+4wdzrdp9ILc\/CQ9rZHIdGM7dvoEd7sP+nenxGysRzs1esl13+81kWeqWgyAZVt5Wfrt5HKl5XWKDw+cKbOsXqNfPX36X1kCmF6rn+toIF4PaT12fMF+KEKJj79XGU\/IgBQ+yXPJhnbdpEKjjXNXOAnB+zDm5+1QA6QL7koUvMZzv0HWR+wlKxB3MaGMYfhnVCpr7Ptc7Wdnd5+BsZ\/\/OqWDeg3aKpi2Dsq\/h\/f7u9tJV7csyH\/zyfjp78pxyv3D8+v72hgKKmVC7sXB3PqH1R3OsftZ2pHPOoZcbzHNr3ohuPi71SrXNf+WP8K+b\/crV3lmpd+pZMMA87IGR3Yq5RL9lePCO+XCw3j50rlcqVlR+Wel\/RRv\/BLyeaznh5hjEcPt63lcni\/Vm\/ybmvzpTGNSoUEhiJ5m+\/l0g9U7mwsz2eyfZhca9\/lPaj8ZZp1l5W12wS243FS19EKSfZZ1xAjwrkKF3zfi+gHBct+kX+PaBXzuwU46ZZku3ETL6ficWjgF2gWRekeDRn5ZzT1S+73vWFTF+yVl4+ra303aOuKcbLvVDd\/+x2JAvkvJ\/rYE6ALIJYweunO2ViPqa7DUH52\/TFtuc+KA0795J1E0bHTV+4ZaSC3\/aqN\/kuGPOYVN74p\/kaDd2PL2dd6Fq38+Ig3HXH7C03HV4Qcz+bqVSAeTq9WRL1uwY8h\/drbSiLVHrRRPUW0XpjIO3SvIA24V08Y2xAh0Ofe\/0BCXvH5LI3C+FrCTOLEZiDZVwMkc7kgnm6QMpuQybK0PJSRV+49U8Fv+2Oq3Lk39fpIHNrtJRes14wjyiHbj9vAznr9vS96hmF7dPvZxuA51Rz52bV5ePz90zn9AnNu1SAeTq8WRb9sUn+\/emfcVEZJ7WvK7ceWCmtXjTJeIuk491c9WbRm5I4qThixIhY\/PTQFZPEAzaYV5z0ckqMcFE15WQMffE0OVH6gjK8BI+t7av27PLbf3a9IEZ98EyYTcc2dgLmi1p2le4T\/htrpgvoAPaAl2ZI\/73rGQVMk45ns\/OHyQeLy8qN3ZpkVUMvFWCeDm+W8bN\/l+7DZxSa59Ou2FegNUhI7hmDCiR1JrxjtCLJeIuk491c82Yh3jexhjgrwYUheltPPICV6LM2mNdcvSAlRriguqRDU47SdgXgN1qeJLst+9q4MYe5+4V5mLjlYqj8bY9e8maTk4xdyk6qPftRH0sevMQYPMnj+7oFmvlZXfeOjYftzaIuxF50qD2eHZ9eYHa0idiVovRplGdKBZinw5XN9khJN+8dZSBz9Zlc4sw5vMQlylyi7d4vmon+c2mWZLRYlwZQo5yrmaaqjLD+UcNgze\/ekAovXGMef3G3c2S\/ReOlye+zPOkLO88wwcOzP26qJHNqtJBD57xpgPy9fa4yQG5TH9Tj6a8WG2ANoj6UukEz5zpBbsCiDuunf1bIk2X8z79Ll4eneGrd9niimaO9c+ZC7U9hfZbq7\/NgnmCP2lx5Ngcwwepn7LVcAHO9k5OLoznCf80115hLlDNh9LQ72ssACqjMePtF+bxBZzl07ptS74DuoVqs5ulnlHM1UzX0UQbgapeRLKdu16H3mk\/kjpWPmer9sm0t+aT1sTEw5zMvPjoI0PkO7Zn09\/KF4vLbp9a9Ttp0PKQQ9aH5AazQI2HUB2cXFMy9FoWC+au1PpLuvXoYsJ+8fKNsbNBGulT7S569ZZB0GXipTP16qpy4zWkSr+tuKhdiUmCejluzU9m4dHkPdBv2rYz78TdT1bxWHjxi2QRz+Po33njDXOXGVXXcWdqzZ8\/A6\/FSOf\/cvNz5qNztsso7yYdNjzJGODwrgkBPwXdZvT1j3hWAaN9K82TDSf8xGqmtmb7eerbxJllbvqpAe6DV1ln7qykDYAoyLsJF\/3nY+b4ctGsYfHHLM4JmXqZGPRldpaNsqF5Dfl21WQYtfMJ0hU1f+AkRPge4oT7oC+qqKQq\/bfc5J6zDqI8oYD7q9hvkX6tHx7JGKNKf9KWdHti+jxx77eBQzj5dc6xUgHmqvVlsrVwlca7EeonqzZKp57LlzcKlyFyu\/MEHH5i106VLFznrrLMMvZKtSH42mFf7aVLMZc8Fcz8t1uaVP+z5oJz\/fXWxtWEbLNFMV3\/8ckxTdsFcwVXBnP\/twzNo1qMrd5IHq\/eJ0RguB+1lGCQ\/xnxjx5PkmWeekTHrW0i3OlvkygZLYvRFFH4c4QaYs5NQTf\/7OnsZkC+\/fQMZOOmKIgLCBUmtX9DOOQzMyXPfCx6SZ5feFgjm9Nc3O3eTQXfdnS6sDs23VIB5Kr1ZdNulPasGj2zHetH6pMMjhbzDYrj4lZsNb5Y5c+bIoEGDjAbev39\/+eGHH+TNN980sWIaNWqUNZoFEBt46wPy4DsTZdsxd8dOHSqYn911H\/nz3oG+IOXyyq9V6Si910yIgR08MoCu3hwvXdpf9lw41lAVW9p3k+cWbyt992ohZYcXHFXnWeKT2Kch+VxPQWJcRLOl3pMrtJZKbfePud+xq3jk6VHy6C4XG57YTrobUzD\/9u+6hQyDMyZ8JGXvOyX2ii1EFLhnNty\/EJjzzLCKnWXVglkydM3ess+GGfLwbqukw7nXmXwwakOXnH766bF8FcyDqA8\/MLd3DuTzyIjRMUB3NXMVfPRnNv3NSwWYp9Kb5bBh38oHHhRLtmO96AxOh0cKeUeJw+L1XKa9WVasWGHolEqVKsltt90mtWvXNl2zZcsWmTVrlvFeQVvv0aOHXH\/99SYAWyaSlza68MZjDdWhYI4RrvwvPxQxwtn18zJo2lo0z9oGwa8fvc0ATNluA81FCoAd1IVt6NO6KbXRfO+DZMNJdxh3v92+fipGxcBRwxtfVv7L2K4CKkYB1Z6DixcvNpr5qp32laEzy8cAXw\/eAMa2tqvvIkigaKZVaCAjgOAzAAAgAElEQVTdKi6WDt88GxM8aqz8tm43QzHZroJeoHzT+3ONgTReMPcaq\/vvuk+O+OK\/hl75sMlRUvnvgoNGddb8KrPXVZKZR9zkyd1nYm5pGUmBeSYrmkhZ6eDMvSiWROpWWt7JBmeO73i1atU8b5fCNfHLL780ronlypUz3i0tWrRI+3AEeWvYYF69evXQugA2X417yxhNT\/x+mNnezzziRlk97X8y7\/cNhtqwE2DWbocyMTBv3Lhx7Gvb5xpOHc8T3AkBbrRxF3Td3UC1zicUAnOujuQqNYzLNpjjn42Wb3uf7PbNk9L0+5ExLn\/e5u2l9UGHyF4\/7S\/9K02VY\/ZqIoc1rx0TPDZg6zrUnbEL5pQ14KUfTH+4Owcar94386d+Id9++53ce\/NVsT6JMlYIxLXlqsb6K8y4GzqoKXggD+ZxdmIezOPrsGyAeZQaLly40ITTLVu2rNx\/\/\/1Sp06dKK8l9cxP74+UrY9fXCSP35rsK+9XaWfc4wBzr1gmrusfWrSCORluvvg5mTxnoYz5aX0RMEeDvfeIejEwrzp5lKFXNi1bFNOwycMGc\/6Hl+e5uounSLNnzixUbz9q4eabb5bDDz9c6tata8D824q7y5il2xst2tamVYi4XP6TB9wld\/5SLwbmRx99dKxcG7CVQlH3Q\/s72zDrerO4HkCdt5sv7cotkzHlO5qTnZ2bFwhTdi5um\/lcBa8L5nyXbUDPg3kcy1O3bbyS92KJ1nG5CubUXo\/zt2\/fXgYPHpxWLxcFEbTqEzZ9boBYj5Dbmvk2U8YWimUCjaHUhG3sdMGc9jzT4gxZtWU7qdWlgI9Wrxb+PmaXitJu6VjpWXGFAOYkKA08RzQB5uX\/\/E2O+PU98xG0RocrHxA0eXVv5HMMk3DY9eb2KrIOXDBnlwBnbq8ZezfgZZg1Loc1\/zaauYI5YZSVF0foKZhrvv13XGb49D16nma0f\/rbPbFZBMibVZfG636U6n\/ONzy8my5cNdqMFQeLMMTS7iAwjxo6N9rKif+pUgHmqfJmsQ8K9WxVQ948q32sx\/PeLN63NGXLmyXqUnjnnXcMl37++eenze9cQWTq6KeNd4kLxAoQp9TbJJvHPhkDWvuEI8qDuv7xgOaxYsHymNuceoC0bdU2djISAbDvXzPkjtqnyyVVvixkAHX7CDAHeH9aW8nQK9s22FmO7rKv7Ll1mad7o3rU2IdzFMz\/rriDfDTmBbNLWFW1kfSc84LxCT+vfx\/RE5iUv7F+GxlVs2OheuHit2an5oXAHPpm3rx5ZucCmKtipUHtVMMGlP1OgNoCQNtuvxf7rFl16b12guHIXU8jF8wrLf1R+rRrKuXq7BwpbG7UeZnIc0mBOeFxGbwbb7xRatasmUj5aX0n1TcN2RTLC73ryYkH\/ROnIe\/N4n2DUDa8WeKZVJs3F1zZB92SrgSIvPHEsJiboB+Yq5+57U3ixgRBQ\/5p1k+yesGsGM0Ct9247GrzP9v\/xas2y9U7nFPIbfHZGj0MOKsB1PVggTb5ulk3qb1XV5lbqaXR6o\/ZOEG6lJkvraaNMV3jp0Wb8v8vAFWZz5+Ryq0OlPtmlJczy3xswPyg5Z\/GBM4NrQcbXrzfVzfL\/HnzTX0RItRr\/cI58vCI0cYdEs68XdMdZY9OhxvvHAVztOPjBt1q+HD7TgAF5W\/rHlYo4JXSU+Nnrypy7J822WCOUIJjb79gbGys\/MBc58riGd\/K9Xc\/YC6\/yXbKOJjb9yTS+KA7RN17F8855xxjuCLpBRm4nGmyv+ezVN405F7zxm07mYy5Qnui3IRU2r1Z7AXF8X20OXzLy5QpuNgj00nnzR0rhxk3wqUby8jWQ06Xj1dvJ\/u039vwsgoYevQdCgNvEg69eNF5uMde+vh7ckzFH40BlKR5YMAkH9tQiXCY06iTTPyrigFNpU0QDOrBsqjmrvJFtdZSvWFLuXHgMYZWGf7dYgO25AfYY\/D7unm32MEjyoVaQJPW+Cg3VZtgABwNX\/+GurE9V6gbPLVSTQrmI2ZtNflAb7TZ4W9Dazz094GybKf2Btzlj6XmsyZVKshxKw4p5LWDIVPdFgHl4f3aFAlZS74IHvKDH0dgNV47S44sM12u2NAjNjV0rGhztT0PlrfW15L5SzbI1cufjvWzPrzr0q+k+wXXFLp+LtNzTMtLCswVMOfPny8nnHBCaBsAVy6FfvLJJ40mz\/9cDD18+HDZZZddCr2vl2IA3vvtt5\/o\/\/369TNl8T++xNdee22RdzWjVHqzjJm2XHoP\/95knefLQ4c69kA2OXMMcLghsnvE2Ikfslfq1q2b3H333VKlSpXoDYv4pB3bo\/f0++Xld8fL3z0HSZVVP8tNf3QUgOPArT+bE6AAp54qxPAIYOEKiGuhORm5rIDfJmhV+6tfNGCuh2oAOfJQbxSe4zPV8ic0O1a+e+tpA+Y1506O8fK20ZO\/397aVtr\/f1rl3J\/uMxoz2rjuGFQzb7XgMzm4fSvjJ4\/hdelOBXTj018ukcazXpEyK5bK21vaSo86K4xr4phl1QzNYp+i1N0J1BL+740OOlqenrXFaNt9K82Vzn9NjvmZA\/7V6lQ3QLpqSwXZtGihrDnsArl7m3di93iqAbTMRoyXk0zcFltzB8i1fDh5fOYvqb\/Q8PD0E+9dvuUfQCdvBHGlnZtKpTJ\/xvpAhaYN5nj\/ZNv4SX2SAnO\/W4ii3jjkAnbY+kAQkAB4tPbbb79dhgwZ4kvxpBLMbb786F1ryZgBu4dVN\/99Fi+nQCOHB4f+QllgTrKbsxMnQTdu3Cg33XSTXHfddVKvXr2Uj5mCOa55zSuslaOuelAa776vAWI96q4A8fdff8tpPz5VqA6AZZ3FU2K+3gD2Tgf2kPqVysrbk7424A3I7rb06xiY1127OEbB6IULQ8fNkNWTXpH2VctIq0mPmjKUWlEB4BpA\/cD8h9\/KycMHblfI\/xyNF35+j53KxOKxEGgLMJ\/\/1eeFTqLip43gAtDhuOHyR6zdXXBNJOGB0mTJJ1JmyvsxYUW7Fcx5Z8W+\/\/SJ8eRZsVGWf\/VBzKjMbmFahzNk16+firlY2sKE+m6pVdfUQQUEuwx2BowN3PyIxXXkQJkth1VbHArm2TZ+Jg3myc78ZMDc1fK96qJgrgbQZOpb7qpPYq9fd8jOWT3plUw7Mv1uKjXzr776Sk466aTAa7rwIyewFto4Nwj9+uuv5hQou0cCbNnpqKOOMorB1KlT5YADDjC+6alOtmbOjg4O\/Ji6q6Xd+qkGiNGsbaOaG++D+qCl4ypHUiOnAjZ5\/FipqRzxy3tFwFzzB+yunV7gzVLl19ly1MoPDZBzXB8e\/aDlE41vOeDNYRitgw3m2i98BvD2XPlhIU1baR07UiJ\/o6X3W1zAuZNUu6duAOmkFVXkgFprTJ7tV39njLXw\/dAqXf6cHOsfPEpsMHf9359seIrUrrglBuaUhf89J2C9EvXQwF2Aue3NQtmrtlYwdBG2iP6Vp\/qC+brVa2Xd6nXyVtOTDHVTbE+A0kmudh5VK+dd+POXXnopRrsELSTlz9kOQ7u43LtXuQrmp512WqFjvvEuWC6i6PTMPy5cGD\/3q18h3mxK5fMYQOGuOWmZbEwUjGBQJ0EhjadMmWLAG6DG6+HWW2+V448\/XqpWrWpA3ivtsMMO0qpVq7Tx6hjO7asEL2\/4q6FZ4JUbLJ5m3N9smgWgLdB0C47q28kFc\/joHys2kXV\/rDPufICkDfRKw\/yr3iADmuO320furv6NOd0JjYNWbIM5IKc8vA3mtjatAsGrL20AV2DXYFnQGS6YQ+FQzuplq+TcpS+bLNWbRcFc68F3NvBCUXXb+K3Mqd7C5AvYq7tn0GIzfb3mV5lWdy\/zWBCYd679l6F8oHYOp6waLYwQ0jT+r4Yy\/q9GsfHN5o1DSdEsaoREq1HOPGokRQXaIAOodpgKDEBcDaBQLmhdd9xxh\/EPdv\/nXS0DAMAIlmia99t6aT1kSuz1GYPaS+OaFRPNrlS9xxxZunSpsfYnC+avv\/660a6DwJzyOLKPts3c4G98lTGUT58+3QgWjvirgNlmm20EbX7YsGHmAuJ0JNclrl25pWYrD5ij\/fXd9LkBYpLy07j91fjujSKaJUBEUjBavn4bY5is8dt82b5OdeMnDngu2nHXmObP8wqiu25YJAfWr2x49z7n3GQA3qZZeE41egXRDrPHCrFSlBrRyIsIHVL9TSti3QaA0yaAmPogTNRjhffQ\/JWn1+va7GiOGvL2vartDDhr2dSLhEAYuapJjDoZsma4EYT6nReYU86aclXM7oNEHe2kYK6eLfjqq2auYE6+1AejrZ0UzPFpJ0UJu5uOOUaeSYG5l2tiFHfFZIHcqzPQ3K+88koT7lSNqanizPOHhRKffqmkWaKOJ7sBjunPnDlT+BtNHaoNDX3kyJGx3xgDa9WqJaNHj5bevXunxb3WK+a4DeZlli+Ryza8FQMkBUrtcdc1TrljpSmUM59XqaUx1Plp5q5boVI3rgHUC8wxDv6+bQVDgwCMCvZedSQ\/qJXGZVfFNFjVvm0wR+hsrFrTCC8bzL\/vcIYRDlP+3GrAEy8ShEL1bQouTAfM+Qxag51Ln19HFgFzhAx50BYE3dLK9YwWjucJgO6CuQqISttXMl42NpjbnDn1dJOCOZ+ra2NY2N3EV1Pwm0mBOVmjEbPI8FAhDRw40NAgqkG7xQd5sLjPuh4sQU3xMohGXfxhnWsbP7O5jQqrZy5+nw0wpx\/QvKFZ0MYfeughGTNmjNkZ8H\/btm3Nb05+cokzcUT69OmTlqBbqpXbNIuCOfVEa1XjJf\/7gbkCIZohIOeCuZcB1AZdLx9xyosC5qta9xKZPjYG5oCffXpVwVKBGW2cz5SO0M8BUygg9brhe56lTQq0UCwkXBNpJ1x2\/0rfx8BctehL\/3+wLzxuXO8SpYjIw3bX1LrwvReYIyCm1e1QBMwv3fCmESYNlkwzux03ccDq0+0PNL7wjLWOczYurEkazBXQ2aaSXF9vu\/Eu7x0EPkrhcDOMKxhcesfv2VSBuQa5p76vDdhNjtm1IBJfPoX3QLbAnJpNnDjReDwNHTrU0C7EY\/FK6QJz9THHLa7H0T2kx\/8qyLbrVsiYnpUFY64XmOPRgjFTeWYbkPgbmoLf0CsY\/IK8WbzAHPCEcgBUlf5waRYVKhrqFaBTGgihAJhzKxL1IC\/lnuGVMai6YI4gUirEFkKq5auvu02R6A7ED8zVF9zLVVDbZhuW+ZvkBeZ8TiRGDNN4xEzfpkERA6j2tzt\/5m3aXqo3bCX3nnWEORfAmJOyceNQSsA8fEkXPOEaLfW9\/\/znP4ZzB3y5nxGuc9GiRTJgwADDi9sJwxbfk66++mrDhZL0c\/hzTa43i83ZxnP83vZkQeI2qLatKSLR\/Fzu2O+iiESey9SlE17t9\/osW8f5H330URM8iwh+9evXNy6K99xzj\/E3J6oirovnnXde2oye9AWeLPYtNRqJkEMqe1ctuKmKpJSD\/o+HBPFRoAoUkABQkoK5n2aOhgvfq1qk+q7HOPOlXxn6QYFUtXz1ZlFuGgBWMNeytH56aYStYfOdauBoseQLjUJy3+cz6mXvMGxt39acERBoxlovNHM4eTxM4NOrb\/NXTFCQL+1HuKhHjgoRrXuznz6Rn3fpVAhT9B\/eNYLl7z9lbPn2hbxZsG+wo\/JK7dq1M3YZ2zZS7MA8Uc7cs0fS8GGqjvM3fbDgqLoeFvK7iMG+wIHns\/Fc\/nKKgokE\/cdJx7333lswnP7+++\/y7rvvSvfu3WXdunVGU+\/atasJg5suzRwwv+qeF4qcfqxXvWwhjwgXKBT8bEA1QL7xT18whxvWAz62oc6lcWyDpA1iLpgrKNrAGxXM8Z+nDgp+XmCueel3gPR50+8zH9tgjoDAk8YG8y\/X1DC+3ydsO1em\/7IytjPgXQRhpe0rFwFz+o42Up5Ls9j9MLH2QdJi\/Rzj4bLzkmmylN3H\/48T4wXma9asMQfNXDDP1qHChDRzr6P0NhYfeeSRxhBpa8lpwOrQLFNxnL\/j\/ZNl4vw1pizly6Mcq+f5bDxX2o\/zo3Vz6cSCBQvkuOOOE+YiQM4O77HHHpOzzz7b+JM\/\/PDDcvLJJ8e8bNq0aZNyP3P1Mb+y\/uLYNWfMC9vVjv\/dLbwfmCugq5FStWs19PmBuQK9C6A2OGuZCprubsFebH6aOZoyx\/3VQydIM3cXrxo2+bxWw39oTFuoaPsRSGjcaphV4yjfY5DsvN0Ck70tRPyEhV0PygLMq1ffXprM\/6TQGQAvMMfdlhOkRxxxhOzR8XBz4KjYerP88ssvhh5Bw+EUnZcLYSjipvGBVHDmdnCtsJu+09iUYpt1pjlzbhrCuMlVgfxGe8JjZZ999jHH+bltBzsMFB2LELBPZ1Ifc46zn7F+nCnK1jRd0LH\/dzVzu542UGlsbS8wV3pGQdqrPK\/2B9EKNoVh10M5c9ebJUgztwUKWviQnc+Unar7Bz2zBYnr6eKCuS2QAHMOKBEjJkgzV5pFy1Gay4szX1O9mayZ+al02rmabP7hK3MQy+savXTOLzvvhDRzzaC40CxBfslhHZ2\/jCKsh4K\/zzSYa22UZsEdkcud33jjDeOfzlVy0GHsHDlIBG+erqQ36+jFyspbu+W5YKf\/K4drA7G+GwTmrrAIKy9VYG5z5uW2KxejkuIB81GNj49x7V71ssHcNY66YM4O4chtpgt1US8YTo4SM8Yr2Zw55UCzAOb42Hu1gfyhkvRwE3niWnnsNbema0oF5psH84DucQPge10\/lZVRK0aFZgvMufNz9erVUrlyZSlfvnxWeswNfesH5q7Wp8bBoEoHgbn6Q5MPoGSfWIzaEUGauV7U4GcAdb1ZooI5Qbcmbtdcpv+60rea8YC5vUPQ+CsNNi+XDU3aeOZv11P99XkQQ7AX7UQ7ORGqddKgZsRpyUZKCsypsB38yuv\/bDRKy0zWm6Wwf3kN425ESqX3Sarzy3uzFIy+HfcHf3I08Q4dOpiLnKOeUk5m7rqhb1noT23azbjuuSkIOP3qEAXMeQagUdfBeNoTxJmTj5ZvCx4\/P3M9xh9UvnqzAJxBwscGc06CYgi1OXOpVteEyiWVL1NO6v82LRbmV10f\/WgWu36uC6UXzWKDOVo8l4lkSyun7kmDuWsM9XIRjGcSpfLZZL1Z\/CiWbHip0C9Rys17sxQFc+L2wJFr2IlMgLkdYEuj9S1uc7iMGlVwZZud\/HyYg9aCn7argKqAm24wt+vYosku8uPcn4oYdKOsaeqpgsGLVrLzUKOo6+nilmN71fi5PvrVzXWd9BK4SsvYF4LoBdNR2pzqZ5IGc7dCUaIZproRfvkl480SRLFkw0uFNkYpt7R7s+hcsDXzTIG5fY7i1GvulltmcklCa3MXJUnXRvPmzQvFTo9CQ7hzPAqYo10DYr\/v0CjuJRcmYLzK13C2cRf2fy9EoZfsvMPA3KZ71DWR367veRShGeZeSVmcI3h58EDjwpyNlBIwt28EiidqYrobnIw3i02xHNK8hnx43j\/3faa73iUp\/2xx5pkGczs2EON3xr\/vk4k79TGxuTUsqs7HY489Nmkw95ojgKEdDRBtssW6OcbvOt4UlWax800WzOOtI2Cu4Q283nXBnMNA8doQXLrFqxye0eiUhBtu3fGQeJuSkucTBnM79C3UCiegiIcR5cahlNQ8QibJgHneiyVCB0d4JNtg\/scffxjg5MQn\/uW4KeK+yOccKiJqIp\/jWuse+orQvNgjLnWDLenqZQWXC6vhPN1gzrafpBw5gAyY+3HmiewItMHp0Mz9BJTrJ6\/PAeZ856dpxwPmfry+bRuwBaVdVxvMs3lJRUJg7uWSyESNen1cPIskmWcTBXP3vs9sBM1Jpt259G6mwRygBkgJeTxixAgTVIvj\/EEpFSdAvRwBvt5QS0b+0SR22CzTYB5mAE0GzL36E4Pk8dULTkunKtk+9G6eYWBuv6unZ700c4014xUVMYrxFqG5zYolsqhsbXO70cfn75mq5seVT0JgTgm2Zg61ctZZZ5ng\/7mometNQ1FjqTw3Zbm5EYbUoOq2svCmwrEc8t4sm0zfuP3p9VmmY7PgjvjEE0+YSyyYjzfccIO5nQjATmdSv3Y7rj+7gfeqHmbuokRD71pvs7z+zDA5fe96Cd036nWc326Tq5mrUdHPOyRRMPfjtm3ja6r6OgjMObxUpll7+WBrg9DiwsC86q+zZUnjxKjUVVUaydBfGxqhzVhn63BhwmBu955725AGzgrt4TQ\/kKg3yxnvrZLxs383tdu3fgX5\/LIDCtU0ilcJL2Tjubw3S8FQ4WXFQSFupmratKkMHjxYWrRokbYZ5wfmRPxEMcCgronbfQq500WsVbxgTrZhhjsb6L1Oi5KHq51GMb5GbFLoYy6Yu3VsWWd7mbVstW8++rxphw9njg\/6iHW7mVuXEkl6yUixj2fuNj7K5RSJdFgi7yTizeLeKjTu7N2ka8vCIW+jeJVQ32w8V9q9Wdg1EUhL7\/PEjkOUzW+++cZo7M2aNUtkKoW+43fe4oSzLjZATtwOwt9WmT9JzmxbLiOaeVilXVC2hYX9rus14gfmUSiJsDq53xMSuHqZDbFAW7afeZiw4nt2K01X\/Ri7IUhD4drlkOe4P+qZW5eCkl+7qzdsKd9W3CP345kDzgSTOeSQcAstdzRiDK1ZsyD0ZbZTIpx5ni9P7ahlmjPHq+TMM8800RIvuugiadiwobltCHsON1CVKVMmtQ38v9wwgEKr2NcaYmBd37ST0cxxTzyn8Wp5Z1Bfqda5bwyc4qkMGiQR\/UhexrgaK+dL\/U3LPQ2eNhARg7vxtquLaO3Jgnk8bUn02UTA3I0I6ZYdRkfp866vOeNB6rxnK+lS7S955OlR5jJqUk6GwLVdvLhByCvpM3zHjUPFGczzR\/gTXWbe72UazNHMJ0+ebK6N4wIINPEKFYL9flPhzeK6JnKF4cU33C5dni+4fQYj+spXhpgfN2pi1B73A1t93+XM7Xxtv3EFJT\/NnPdsDTabmrkX8GrURL4L4\/31BqOgPg\/rV62D63sPmHPFXLeKi6XDN8+ax7gp6cHqfXITzKmgHobwunxZtV8i0Q0fPjx2\/2bUCZrO5xLRzLsN+1bG\/VhwcUD+irjkRyfTYK41JjbL\/\/73P7nllluMVn7QQQcJhnpcEd2UCm8We53wN3ajunt1ky4PT4kZxJY8eIn8Mf6VrIC5F7D7gTkgpeFreS8qmIcBa\/KzqYC\/jwfM2a3U\/G2+LGnU3nc3FAXMXe1d48S\/V\/8I4\/5JnYiaOLpyJ3M\/aTbiOEU2gMIJcquPDdh8xnVxQVfFpWIAE80j3tgsLsXC6b1T2tcucqt83psld71Z3LlCLHOMoK+++qpxWeQ8RLqoFrdsPdLPPDpq7gtGK8fg6N7wHja\/1XskDHT8NHPXaOh3EEbz59o0rqTTFBXM9Zi9y50jHBIx+Hr1i9sHYQKEOzqf39zBGDf96hDWr7oD4Do6HTu9wQkfd94nNgtAjlZeLLxZ9PJmeMHLL7\/cBPz30tbDJmemvk\/Em0VvFaKO8F6NtllV5DBJNrxUqE+UcvPeLEVnF5z5zJkzpVWrVml3UbRLt+OznFBprpS97xT5vk6H0MBXfoClWun3dTsUokG0TG7Z4ao59\/i+q1Xq6U4b1MlD7xhNFMzVNdEFWLRW2u0XcyUew6nejaoHicLAHAFzZ5kjjHEzHjB386XPMMZurVXXdLeCud5f+mmZZoYvz9YtQ9QpsmauE0a18Vw6tu8nHOL1Zhnx+QIZMHK2yU4HBZrA5Vyz4aVCnaKUW9q9Wey5ANWCZwthcDOljbtzUS+n+GqXz+Sm9+dK5+3mh4K5aoLqNqjAAugBmK5AUM334IMPNl4zetOO1sUNhatgrr9Vo6c8PZZun6qkTPvyZn3PBWEFczcUAJcoBwmxMLC3+xTfcoQZZdsXQvthAG2DZnmj5cm+YK6GZdtO4AXm9KN7t6pdLrFZuvfqEQvfkCmlVcuJG8w1SiIZ4PKVyqvhXH\/1oAiMUZ6NlzO347FcdvDOMqTXLpkejxJXXrY4czrSnSPauZwI5Vh\/9erVzYXPvXr1MrcPpSNhUP\/60duk7\/LXTPYAEMAWlmwwsf\/ecd4UqV1pS6E81KDJ4RXaJIu+M9kr2OLNsaZclSKXRdiaOe9NKVPXCAuSfRoSAMVHWz\/X9xS8tRyeQwi4Xh88F9TuoBuV3H5SYfNFhdZGSw7TzPme9odd2CHVdhT5Y0msOPdgFP83\/f0fF0fafOjcN2Xydq3lqK0\/xN5rcPOrUqlt4XMpYWOdqu\/jBnMKVkBnAagrVrIV0jw1TGlQGVGfjRfM7Xgs+SP8yY5owfu5AObc9cl9jZo2b94sP\/74oyxevFjef\/99IYoh3i\/cPJSOdN2t98uc6Z\/JF198kY7s83lmuQdQCM5f+4nULb9Vcjo2i59249V\/qaZe4ok77fVsPGD+wU+\/yWGPfmualU3eK8vzMuXF5wKYo3D4udVmImSzzsO77rorNE5Mygcgn2FaewABzb2y9Q4+Xd7oXsWAebZSQpp5piqbKjDX2Cx+9V70xybp8sg3gjcL6dBm1eS9s9tlqpklupxUgjl+48RZCbrTVcMx426IrWP27NnSoEEDc3ho1113Nb9ROjjQw3VymQTzZO6iTdck2bhxY9au1UtXmzKZrwrqhYffIT\/d1itrscxpc86Cue4I+vXrFxq8y+9Z7ejTTjtNTj\/9dN8xBsw7PbMw9v0LvevJfvWzE2A+kxMxE2XhScKRemgOOzBXImVzwpgAWkGgSGjbH374wVztR6At3BK5Ng6efNasWTJ9+nT55ZdfDKVyzDHHCPOLk6HJ1i2oPVF2iCgS835fH8umc7MavlkuXLjQ+M5r4pDe+IMKgjMAACAASURBVPHjzUlXO22\/\/fbSt29fYwB+7bXXjI899gH1tad\/Bg0aJJ06dZJTTjklkSEp9e\/YYD7u8sMkaNzS3Vk5CebxGFmDntWOBgAIieqXDntsqkycXxCXgcF494zW6e73UpM\/47N06VKjHScLmK+\/\/rpcc801gWBud6zf6eU1a9bIpEmT5JFHHjHA\/+9\/\/9soDMnWz29QJ775qpx28ZWe9QbENXaL\/T5U3+l71fP0jPjyyy9l3LhxBsA7d+5shBV9A3Az50l8BoDjpADYIwBwJ+7atasMGDBAypYtK6+88opxLUaw4aXmFyrYjTvjtlPXmX7unjtRDzj7vbBDhuywEN7XXXddSpwsvOaCfsZuzc+ZI6weeTAPgLJUATlFRNGIzPZk0EexGmXL4b+konsqaZao46nBtsqVKyffffedAN4tW7aUnXcuuEwZjRTtfs6cOUZrXbRokVx22WUJBb+KMm6PXXm+3PHqe0XA3AZyBW9+8\/nTXy02v\/1OIdsB7VauXClDhgwJBHPquWzZMvn444+Fm47oy9tvv13uvfdemTp1qvn\/xhtvjAUos9sVBOZQoQ888EDsMKHX+vV6P4hCJY\/bbrvN7KbZNaUiuWAez86fupK8wnvrnCzb7z756fajUlHVhPPIKc08Hi+ZKM9GWfx6sIMezBs+E55Hvi9mA8zRpgAqAA7XWb3MGToBYxUaK3w5gbig4AB9r2P+qegNALn1pcOl3oS7ioC5xgEiCBenRN2koXO9FAwXzInbvueeewptJ3FrEgelhg4dKmvXro3RMhycQktH47322mtNmAOEH4A8Y8YME\/7A9vwhLz8w99v5uJ97vW+PkRvLiXVLKAYM17rOTzzxRGPEJm+oIeoeD9AHXSOobcSOAmB7lXnzzTcbYedVV+w4A4e8LNf13jsVUybhPHIKzBl0TpVG8V+P8mwUMLd9y\/OxWBKeR8UCzDt27Gioif3339\/EToEvjxINNJleQVnofstLRcBcQ0eEKRBNbptkNHT71neb1oCuANigjQBoYriTDjvsMFEA4nnCbhDVkc\/eeustE+OdZ4guSYJvh9aAZz\/++OMLCTc\/MA8CZPudeDVznucAFOAdBObsSABSN3mdSlcw\/9e\/\/mUoKdet2t4NkK8KE83brpNdXhSMSWb+xPNuzoC5nwukujuqhoWE5pKBgQMHmi20nVzXyCixWWyK5fHjdjGhSqPeSJSLz\/nFjaGfotwMlOrnMn3TEPX30szRSJkzaOQktHY8Xi688MJ41kvcz\/qBuX3UXy989spctXf73AMeKBhxcXW84oorhJOuaNQYf6GM8OLhUBR2Cj6vUaNGIVDEfjRq1ChTHEZgBAAAeM8993jeg+oH5kGeQHZIYC\/O3M+N2aVYUq2ZK2Z4xZNi3tx6661GkDE\/bC3cj2opVmAej595ELDGvQpS8EJYbJYy29cTNB9N6sViX+xLrBP3ot8oMVLIMxvP5WOzeIM5QeKgUw499FBBIcAoO3bsWANgXPicrhQG5tArKBB+ScHcjY\/tdwmMe+OR5uuCIuA0ceJEA\/6A1qWXXmruS\/UKe5BKzdw98Oe226VREgVzW4AA3FxriTBH29e\/vTzl\/PqvRIB5uiZ5JvINi83y+S\/rTYhSEtvdGYMKLmK1Y7HkY7MU+N67fRL1MxbjqlWrjPGRnUEysWOgSM4444xQbxYvzRw+GT\/zd955x2iibKXDPCpSNUcrnv5IEZolqmZOPBfirXiBORwugunvv\/82NAp9bWvmdpx2GxTph7ffftsIMzxcMA4H2QyicuZ6kU27du0McOphLfd9PQuA26h7mCtMM7fjxidCs2id7HyUewcvoKBIrvE1DMzfuuVCaXPKZamaMgnlkzM0S0K1D3kpbAuU58vT0euF88yGAXTJkiXywQcfGAoBFzw1gKo3AlQUcwOqAc5VeeN09cbL746Xay4oKoQ0CJdf7Gu48gEv\/SDzft9QKD72tGnTTN2hRuC+oQb4vXXrVmPIxJBHXCO0bNwOEaIK5vTNPvvsY9wQ+d7uE1w1u3XrZsDdTlG9WXbYYQez06FeNo2RCs5cOW7XeybqmHkZa8nrpZdeMhfqkNTGgJBw3SLDOPM8mEcdiQSfCwJzN3Z5Nq55SrBZxeq1bIB5rnWQ3zxUCsXL8G67LbreLBjncKvs3r27VKpUKXZIinYDUPXq1TMHgUi1a9c24Exc94svvtjsTjACsjvBJZE8SESX5N3rr7++CK3oxXnbYO36meNZA0B26NDBODPcf\/\/9pgw7jlOQa6DtzcJ7qsnjHIEL6ddff52UN4vuBlTAYRimz\/r37x\/bKdicf9C9xmEKYybnYqhmXhI4c68Tgz+vXC\/Nb\/\/M9HWYR0EmB6SklZUH8+DzDro7tP3MjZY4dq7xYvFzW9R5gqY9evToIqc\/9fvWrVvLEUccYXzM9S5fNHVO5aLdA2iaMBLj4ZKuw1NR53Y6\/Myjlu31XBQ\/81wI1RAK5sl0QrbfDZKaNsVy3O51ZOTpu2a7uiWy\/EyDua3FRe3QPn36yODBg1Ny0tCrzI+fvFcG3vaAL9evvLj9btAJ0KjtKs7PhZ28zFTbwupRrDTzTHVaOsrxc020DwpRrrp95fJ1cNQzSv1Ku2uixmSBP46SCIULkBMC1\/VaivJ+lGeCjvPb79vxWbIZ4yNKm\/LPFPSAYsyHo56XJu33z2q3lArN\/Pnnnzen2nSx2pqQTbFkw5XQCJO5c80kSIVLZN41seAADHywG3hKVxo0Qps2bczRdS+vhlSvyFzS3lLdttKen45t3gCa5png55poHxS68KAG8kDvFqYmUa5ly\/XnknH9S8e72XBNxNhH8Czc9LyS7bLHUXeMfxyySdfVcnkwF3Owif7VPuZ\/5gaxcdzEroof112Sz9h5cV4gKHGoinHlNGu6wjRo+bk0tqVCM7eNEy7FkvdiSa9EzTRn7tUaAm3hopjKKw7j6bVcWvDx1NvvWUIJY0ANSxhUEZIk3CmJXoq3DMJ0ypQp8tRTT5kYOu4NT4A87pb4enPaWxOHnHC9JL6OGz\/GrkvQydSwOsf7fS6NbakDc9vwmfdiiXfqxv98tsAc1z0OGREW9qGHHjK+1\/hX33TTTebYuyb8sQm2lU4PjjDOfOUrQ6Ri2\/0L3R35x8evyLrpk2THC4cGdnqUGEWaQaJBqtwKUCaXfkBVeSUOMeHnTygB3ADRpnmHg2P0NcKVUMYcXNKEhv74449L3bp1BdfG77\/\/3oQjYHzOPvtskx8HowgDwPjh5ojbJcn1IXfB3HYzjH8GB79R7MD8vffeEw4E6OEKv0kR5MKT6k6Mkp9XR9sUS14rj9KLyT2TLTC3FzQgAZhzCpSgVJwiRRtkHhPTmxgn+BmnK4294zI597Exnt4sADk\/JL0MGCBf8tAl5rOgOyXh+zHckogxExZFMJVgrhEGAeZ3333X+KsTFgGsAITpc410iFaOrzl9DwVCaAW09BEjRpi462jcYAdaOrsnbpTCD\/6SSy4xp4c5wIMmT1hcBALATrsZR+53hVKxT5zmwTxgJmu8ArY83HmHtqNhKPFx1QhnuQrmem0cl1CM\/3lVwcKpuq1MuGifQtc8RfEW4d1cfq60e7PoNPYDcw2NSzwS4poDDMRrYeufrhSkvaF9L7rxuFjR1Tr3lT\/Gv2L+L1d7Z6nWpa\/vvZK63niWgy9eh3IILIU2C\/jdeeedBkg1jAGgy+lYPiNptEE1CmulOABkB50CDzAuN2vWzCh45M2uh4syeLZHjx7y3HPPGTBHe0dD33333Y1GTXRHTp\/27NlTnn32WRO1En93olgqpYLGzsEgBC4RHAFyAovZPvFo5Qhn8s9r5gUjFUiz0IFwjUhGJHGjRo1MaEgC1RQnMMeb5dPfq8gVH\/7D893dtbYM6r5bofWbSq8SMs5GfqXdm4WFb0cQbN68uTkwQ2wZNEeMotwso9ETGSfbIJoOQA\/biruAHgXI7YM17JrteNteQbUAe3vdosXbFA2gT5yW4cOHmy7gNiKv2Cl8x3uAKbc0rV692uwICCv8xBNPGLoKKot3AXO4bYzR9D91APgBbbRp8gH0iTFD0CuiWKKMwIlz2EnD+oI5xFonITwAedoIiCM4ohxs9IqSmIqxDhvbVJQRNY9AMGerQ4djEWYbV1zB\/LGX3pAB7\/wmi\/7cZPpFg2rZwaP4PO\/NklxQLfrQDciVaW8WuHLiVaOITJgwQY4++mij5aFZQqUADIcffnghYyggwsGhIKNa1AXl9VyUBW\/TLUbAdO4byJe7h1ns2CF+ccZtmkW1cqVC7GiGaLxXXnml0eS9qBvdqbMjJzQAQoKTo48++qhRYODDAXTNGyqG\/\/fdd1854IADjPbObgAAp9+hW9g9I0DI59xzz42BNyBMyALew0uJCzdoH2PIzorYMuQPHcNzpG+++cYIBKgdPFo2b95sdhKpuIfWHd8oY5vM3Inn3VAD6IIFC4wF+sgjj\/QFcyzbbJ2Q1Nwx6HW9UjyVStWz2tG3DH9DTn1jWSxbOzZ0qsrK5+PdA5nmzOFkJ0+eLBxzZ0cGcNicuU2zZGrMwha8zZHbdVIO3aueUCyApp1U+\/Tz5rDBHG3e5plV41bbQlA\/KZgTlOuCCy4wVQA8CVqFdo7BeeTIkQbMoXi4mo7dPWCKVg7lwviopwvvE9yKsAMHHnigcUv8+eefY8GuAGWMnyRis7hg7ho4aT+eMhreOJ1nCcLGNlNzjHJCwdweZD\/NPNfBfLuT7peZ6wukdt6DJZPTq2C3s3jx4lgI3GRKj7Jw0N7gjvFk4VIG4nTjGQGAoHHiGYEGpzSLfYFDMnULejeo3jaQw5Hj1aKcOXl6Abrf5cRKffKeFxjHo5lHAXPKgbaCcmF38+GHHxrNmntG0YTR2AFvgJW+ZwwQInjC6J2ljAMatl4+DeVFrHlscwgIvFmgbrglyk8z9wLzF198MXZjWR7MrdmJJGZCIrXZrrINYhuGdbo4GEC3DnzOBC0ijTunnXRtUTNd6zafr9MDmQZzLZ6tt3pJ6KUUeFywe1QjMc8CQoAMv9OVgsB8yYOXGPC2jZ02h+7lzeKleds0CYBqa916eTJrll223p8ZxJmHgTlcObt2\/MHBBDh7tOpevXoZ0OVzpVnQtAFoaC8EKwANFcY1d3iv2FEJEQJw7Ng3oEfIF2FAeRzuAoOiaOZ5MPeZzQw6nBSJiYEkhhtl+4pEzWVvlqOuGyYr2p0aa5kdh8X1Lc5lLxUaEKV+eW+WwpPYPXnoTnFAEL9ojGrZOgGqrokAtyYAff30zzw9Wfzii9vxuaEt4KAJG+tevYgNAUNnkDdLGJiz60FoEIKCPiTGDUZLPVGLGyE+5fiN42KIcAXEsWHQ39jgAHLscVBj7JagUtRrxrYJQJXBkdOm+fPnGxoXgaZ\/024MrrSHhNfMihUrDPCTPwoFggAKKMx9M16BHmW3GG+eiT4fSLMgUYlxgfECYwU0C4YKNBws3ywUtkVIUmIX56JrYsenF8imSgWn0Nw4LMXpOjjqH8U7prR7s+hC0BjcAAjAhO8yW34OpwAcKshVa41yiXiiiyyXFnyibbDf87taDbzAZx9DKIeFAHTcINHU69SpEztaD1AD0BhPAXS0bvzOL7roIkONkVwwZ+yGDh1qBAPUDgZU3Coxqi5cuNAAu1\/CGA5nD21TasEcH1y0bzSW3r17FzKAwkXiFsRiYOA04D0dGmQAdQ03Xjdpe2lPDJxu2\/het5XqI8tnrvsRi2j\/keti2dmHhIrbdXA0Ioq3TTriq1B2cbo2jvoCOGzTMdyzq2RLjwLC9l5pBt1p4gKXB\/NUwHzpyyOXBHWgZs6WiW0K1mQ\/P3Ms0mjmXFuFlISf9LuGy+X6+F99W\/0kpg3aNvBHOc3mgnneiyXziy1bnLlqj5wwRJsDwDlgokZCnW95zTzzc6IklVhswNzesgLmXidAWRQcymDBoL0fd9xxkblHL6u8PdB60QAUDsYWvYxVt2FhbmY2mOe9WLKzhDIN5mz12XZjgONkJy6zzE\/mJtt9dpRs55ViY2eHX3IUzdy9+MLdCdq7Tk406g5VFzzKiPK62RmNfKmp7gEoH+6RLTY3DRElDZ6RuAgkL62Yk3doQ0Q10wA4YR0XBuZwXSR4Ndcn1s+X1i4zD+ZhI5D+7zMN5hj3WFwY2QDUY4891uwu+fE7KQivHgbm7lx177C03d\/oVfvQDQv+iiuuMO52+VTyegBGAu+ZbKdIfuZRKokXBQsXl6Koyba+27Ef3Pf9bta2D02oxd7OxwbzgxpVkY\/P3zNq1fLPpagHUgnmGDHDtCDmCtQccT\/QvgF1PDqw6fB5Mpq52yW2V4lL17hGQgCdGCQ\/PHePHFZzs5Srs7M55Ylf+abliwQXRRIeLXxWnBNjjp0CCtYvGiUGS84CEOa2uCb1hY9i98tEG5MCcwAclyS2jni14GKE+1CUCHT21lNvy\/ZrsBeYu6E\/vUKB2mC+b\/0K8mLv9EXGy8RgFccy8BvmUFkqjlLjkYDXQ5QtrYLpQQcdZHhyfMkBDtvPmv5MhjO3wdx1F\/RyH9Q5\/9qFR5s1E9UVsbiNexQBzuXSGKOJy1JcUy7x5fRhIJgD1tOnTzcntEh63RaHMUhdunQx7kcsErhzjujiNcCCCbpPMR4gp5wwOoZnvE552WB+7r515N5ezYvrvCm29UYb5sAI4JVszHAWP7uxqGDOSUMCQBHND3tL27ZtjXsn8zNZA6jy5xqMytXEvWJo67zXKJ7FdlBDKh4E5qtH3yflWu4tc7ZWix0cWjdhtIndXuvcu4tVl0TZKWayQYFgzkIkUDx8OfQJhiIuv+U3idNeCuZEp0NrYjvJwvFbuFE8WNwOiArmrkHUBvPh\/VpL\/73zmnkmJxdlRdHSotYpHk2IuNpE2iPQEiGbEShQcOrHrGUmopnrfGRHqWFn4wFzDtMUZ3ohbLz8dmObxz4pW8Y9ZV6fd9gl8tmS9XJKm1qy+eXbzGfbHHaGlO02MCz7nPk+np1iJiodCuYK1mhW+rcL5kxOwlsS2IhDAxwQ8EquNhO1gS6Y20eX8RjQ\/zmgYMd0zoN51B5O33PZAnNaZFMdcOccDccji\/mqye8EqB464jnbQOoF5G5ZXv\/zmQojlJ699torfZ2e5Zz9dmMbZ02WJYP7mdrNrdFSPm\/QWU78viCAloYzqNyrIHBXcUjx7BQz0Z6EwZyoZhzI4CQXWg\/+5XgPqMeLV+W9Ir3xnLpxMdntmAqah5dm7h4a8vJIsMH81oOqyul5zTwTc6pQGVB1hDglIl6ybnlR3MA0BC6VUDoQ3pxEXBYMb0T21EuB+U30P\/uuSb9Ocj1Y7OdcWsXrlGRpcVEMGnNCFGx9\/OJCYL50YxljDLZtCBmfqAkUGGU+JpBtwq8kDOZoF1ApaOQc64crx5c3nQGL4m0lnb3zvT+a1+pNuEsqrJgVbxb553OsB8LcwPRyCqrN+JNUiGCgh0dXgz3fsXPAeM8RdJeCsZvut\/vTZ4JcE\/WZvItiQU+cUm+THNimWUwzH7eyrAyZXy7HZlq06oTNx2i5pOaphMGc4uHMOZxBoK1XX33VcJIYl+JxT0xNM\/xzYQHpok53Wfn8098DAHFUDd\/1KAHMOa3MQSICPRGmwu8iB7cl7oEh\/d7eEfodGrLzKu3zkQiRjb96qchE+aX7FcXSJTOe+Zju1REK5lzB1a5dO3MQiMtvcfPyMoDijkhALm4E4faQfMr3QLZ7gMNupJYtW8aqQghWaIDddtvN7Cz5m1OjSrtku84lufxEYreX5P5IddsCwRxNhqvjOB5N0uu1uHpLNXPbNZFbuDkBihE0nRfkproT8vnleyDfA+nvgURit6e\/ViWnhKQODQH277\/\/vjluj\/8uniXvvvuuMXbhEpZP+R7I90C+B+weiDd2e773ovdAJDDnODRbUbhGr8RWFR\/vXXfdNU+xRO\/7\/JP5Hsj3QL4HUtYDoWCOqxfhQ08++WTBxQsDkpdBkds9SI888kgkN6+UtcAnI9tPOFdiJ6S7zdnIX93ttGx8\/bnFRk9YBo2DbVSMEuwqG+1LVZklcT76GY9LypiH3b2Qa+0MBHOuXuJSCE7QcUWUVwhRnexcBMDCxrpL0Cui1GUr2REViR1Dp+MPHxTMK1t1Le7leh1b1zYFjYN98AuvKOYZxvOgi02Ka1+VxPmoghjHCHttlZQxD7t7IRfbGXptHAeCuMcPN64oF0LkwoKzXdIUNOxbinKhjiWlDn5XiNG+oHFwY+n4HRgrCf1U0uajaqzEc2fcbDAvqWPuHlzMxXaG0iz2YiLmAqfomjVrlrPeKn5H\/Uuq1pdNsHP72q5L2Dh4aT4lcQcV1g\/ZHL9Ey\/7uu+8Mlcpve8zC2lqcx9wGc5gHeyfptjtb7YwM5tyuXb58efPjl3iGcKdBERMTnUBR3\/PSxIO0x6j55p8r2gNelz1oaIawcXA18aiHd4rbOIT1Q3Frj11fF7TC2lqcx9y+ewHvPfdOYhtjstXOyGBOZbnbk3CiXnw43i6cAp08ebJw0IgGZyOFTahs1KmklukGTrP\/V+3FprdyYcJneixK8nwsLWDuhuwOG9NiAebPPfecuYCCeASA+uGHHy54L\/AZkouwo0OHDvWNmpiJhRS21ctEHUpzGcolcs9mLm5FMz02JXk++oG5Upq5Qj8kM+Zedy+EjWnO0ywsUmKwELoT18QPP\/xQxowZY2Kdly1b1twuRFyWoGBFyXRqPO\/aGmDeABpPzyX\/rN33QePg0iol3QDKheh2uOaSYJB3QUuN3n5tLW5jTvsuv\/zyQq62ukJycW6HuiZyqTIGz4ceesiAOVo5Hi4TJkwwtApxonFLPPDAA809i9miV\/y4vLxrYvIA7ZdDcXTfSl9v+OdcEl0Taa0XmOeiy14iYx5290IutjMQzAknOmzYMHN1HKCOYZOQoYA7hzygWvAzRfu99957DbATiCsXoiaWxEMaiUzKdL\/jHhpyD2jl2sGKdPeHX\/4lcT56gblq5+AGyZ0PxeWgWNjdC7nYzsgGUA4MTZo0yVwNR+JUKJfk4n9OwpOFy1n5jJtc9PNsLZ58ufkeyPdAvgdKUw8EgjkXOROXhVOduJxxbyGeLLgfrlq1yoQW5dZ1pVbQ3vFk0ZjRpakj823N90C+B\/I9kM0eCATzZcuWmRjl3OmJ5n300UcLHi1QKVwPh7cCN1Tzf6dOnUwEReiXfPjbbA5pvux8D+R7oDT2QCjNgv848czvv\/9+o43379\/fBNyCJyeW+fHHH29cEsePH29C315wwQUm6H9JTrQd4Ya9ICzZp2axJUA\/8UMeGI5zwb4Q1ob898WnBwh4x44a7zKv8yB8z6Ud7KqjKl1gAJFRo1zgwXPr1q0z81rL5zM\/TMCtee3atVK5cmWzLtwy3AtGyIs22HGWoIBZZxp2xGu0eIfT682bN5datWoVnwGNo6aBYG4bbew8W7duLVdeeaW8+eabQnwGBm7gwIGGhiEoVy7x5QzynXfeGbsdKUrfnHXWWdKjRw\/PR5lMXMCB\/YDf0ExeCeMQrpt4+OCyec011xh3TibVZZddJvfcc4+Z4Oxq8infA1F6wJ3LeJcNGDDAgLKuuY8++kheeuklM7+8FAU3Jg5KCTeIrV69OlYFvYRG5\/aXX35p1vv5558v++23n1n31EUTdxew7inPdT\/kngPsa5dcconxhHOT1ueQQw6RL774wtQbJkCTHQOFu4a58J0Lcx588EGjTCFo+JsbpIYMGSJVq1b17Mow75Qo\/Z\/rzwSCOQON5ETSAohINnxljznmGANUBNhhkPAzJ8Ii33NYRMOf5kLj1c+cNrRp0yawSjz7+uuvG7D1it7HBCbEL+2uUKGCucSaiH9eGhDaEZOYS66fffZZwa+YicidqdWrVzcUFTuZI488MlI32f7yxMVAeBIbwytFDSfrLuxIFYnjITwCEHp33HFH0i6r+ZAMYnZzzBu0V8AXGxXAO3jwYJk4caKhOVmHeJzpPan8xo6l5z\/cMdeQDCgZCAXeZa088cQTsXUMiLLzxsGhW7duMnXqVDnjjDMMcFIHwJ07VfFgQbnjN8CK9sy7I0eOlLvvvtt4vLGrt5PWh\/sQ2OFTf2xuejG8e48rWji7f9YOjhaUj5C5+OKLBYHgl0o9mDN5RowYYXzKAaCzzz7bcOac8OrevbuZJAA+Hfrwww8LEnzevHmxy3LjWPdpezQoGJRbqE7sfv36FQFzJhETcuzYsWZSMuEB8+OOO860l22iJp5lQiLcAP0FCxaYxYVg5DcTHlBHuNgXfmCTaNq0qWdf2KFm3Qhuaeu8JDNOJZgXl4idSXZZqLKhMUGwVZGYQ2jWCPeff\/5ZHn\/8cTnvvPNiO0ZAEQ8zQJW1gKaMHYwfXI2Zkzx\/1VVXGa07SMCz1qFcUezQ1GfPnm3clLkmUkHcBXPqSNlo09AbCAAUHU1gx2OPPWawRWkfzqxAryCg+CEdeuihph1gDK7S1J0Di9SXz1CqAPj27dub7+0yeJ9yWL\/nnntuoTth+Y4dMv0YlXZK5xgnk3eoNwuA1KpVKzNg0CgMBiAOgOFDymdwVXy+ZMkSI5EBslzpGBfMmSR449hJBxMKhUVhgzkTkTtPaTMa0X333SfQTCQ04yuuuML4199www3GCExeaPBoLwA+k\/i1114zII0GQfgDysfewAQksRvo2LGj+d4LzF0gK41gTj8FxU5PZhEUl3d1LiP033rrLTn11FON9vzxxx8bjRcgRztHe9aAeHxOTCU\/v2koCt4LAnPmM3OfHajSKIA2dcBmRgiPIDDX\/tX6o8n7Jb3chHKol16EA9jSXugUEgoT64qQIqoQ0VZuO0PgBZXhlu1eqFJc5oNbz1ADKC\/ASyHpkISAFWCElHW5ccCQbVouGfVcMGfbxslVcMwOLwAACW9JREFU3Yba28oddtghBuZMUoKGYfiF\/1Ye0TWesC384IMPzORCS7n00kula9euBuC5cef5558377IFHDdunOHO0STeeecd8z1CEFrn5ptv9r2hydVwo4K5alkIDbbiJHviesUUP+mkk2JzhAXKItdkH\/jgM\/d7GzAItIU2xTtKs7iLmWfsWNgu4GgERi2\/tGvn2n\/777+\/fPrpp2bOQGVokDt2ihylZ\/6RmJcoJzZl+Pbbbxs3Y70Nyp1L7pxgXkOlMoc52s6ahxKJAubz5883ShDJpXt0TBcvXmzqiALIzsBO2l4UKups40pU2oTdBPVm5wwnD+2J7Qomgb4qSSkSmGuDFdShDpD8SGy0z1yIx+I3KF5gzrMKUvbktcGcBYBGDs2Exsz1efjceyVoJ4w2PMt2Eb978mfiM5GYiPDnCDsmOAKFRQRdQ9+xOJhcXt4xXjRRPGCOgcy+DcYO5bly5UqzXWbbTNK\/0fbcMtyAQ\/o9C5C2unEs9Hnl72mHu+ux6+KGXfDa7sdDmZWkRapt0fZDJwDKaNwYPFFIGGM8WKZNmyYY8Elotti3FMxZr+wgsXexM4QiBKwZlyDNHOUNDZ61fsopp8Q08TDNHO83dp1QHFBACG7WBu2w28QuX8u3xw1Kh\/kLTak7WTRuPnc1c9doq\/nQP9ArKExQSwgX6tSwYUPDs5ckz7vIYE7nkegQDRIE2DExAKNciMnitYCTAfNkAAEffaz8aFBMZBYBffTuu+8aegbBwTNwf1AyTDKvWPFewO0VR9yuqx6h9tJe7P7Yc889YwAOsPsFFSJv1xDFZ3qcm4WOQHAvAOEdBCCaOW3GAGwbQ+26kB9eGGHX+3nVI5lxKk7v2kZwxguDIV5XTz31lFmDfIaG66eZz5gxw7gWs5aZixg0oQYB\/zDOHO2YHzj3qJq5ug\/ax\/51d6r9zpyDJmJHiNZsJ4BbDZtw+1Ca0KRRwZz1hUMG9j12MxhtAXaAHy8z5qJrkC1O88GtayQwp1OQbEwEOhgwx0+VzoXHIpIiEk4t0LnUIV5gjibDYJKYnAApg+xq5tqORx991HCRQQnuDuOK+45yfkrrcDF2nz59jHEUTZiEAQoN1isFgbmXNmPn4WfMUkCkHqqNUz+ba7QpFD+NWGkPNCsMum59bHoI2wEUk1di58Ids7aHjkux6HupNKrm0jyNUhcbzHke4cj44frKzg8wx6bjpZmzK+Q7hCu7TOYhnllQM7jY8j6avt+cUQoVWiQZMEeIYFtjhwAliTIDXQTYYqvSBD3CfEEQHHvsseZe4Q4dOsTcFsNoFnUhhtKk3ewS0PJZ5+AVLpDffPONoVFtV8go45Crz4SCucZcgWtlgTP48Ml0EglQZIHBHefigSEvMNcgQDooyiP7gTngByh7uSuiMbCdbdeuXSF+GQ2CfsHijxBEgLBoAD0mpXJ5LCys+X78XabA3HYntYNnMeZubHLtN134gEkYmLNoVEsP28XZ3LkL6nkwv9ooU4AVLn8umPtp5tAvACiKAyDOnGbuYtRkTQNuAKsfmBN0D6WHuUB8piicuZdmzme27YMysedgj7PtM9Ajo0ePNo4U7HABfhQCtHKSlwGUU+nqKszulPXGj3rpKJjzv2rt5IXThu1VlqtgHVav0AudcXtiG92zZ8\/Yokay0nFMDnzMMfYxIeCWsSbnUoqHZ\/VzTQza2mv+CAR7MiLc0FjReDEgsfBsH1rVPBTM\/SZTsmBuT2DGxY9m8TobYG+PEVgku43x0CwsxCg0ij13vIA7T7NcbfheABXlid0VO2TVzJXKoh\/1chCUL4QpO2eM0owhtBhjDvdOP7PGAU4FcwCWXSsHkzDS8z3cN1QFxlMoNTxLECwoIxwaAiuoD\/Yf9TOnHjpPAFawAnDmsBx1pi04CSAs0NChJHFBhL6lXPVeYQ3Zh\/9QjngWHFLPOWhDhA3KGmc7unTpYmxUCCyv51HeaCMYZnup5RJ+xVOXQDDH8EEHQiEg2ZHGTBCkK5ISUH\/66afNNu+zzz4zg4amkEtGhWyBOYOABxB++Uxu\/PThgznwgyGKxYEPLQsMIGci+3kBuQdm0mEA9eLMbeBMlQFUDab0j71VZtHZYO81bvGMZTyLoLg8q+0HnNkJqocG4Iwtix0PYA4NAo+uJ0E5eYkmC2hxF4FLrXHEHcUDg72COfkBzrgeA5Zo9AAq4M6ugDIw3uPBhSukrnn3BCh9yzxH8OAJxvtw2FAotIf3EQiAP\/MNoYNi2KhRI1MfnARcJcKdO7YXDLw+goA6o\/WjRPhp8rgYsx6hjtVxobjMBa96htIsxE1gewxnDKcJmMPvYkzBqIcHBxIRacoEwKCBtMyVpAsAfiwsJgPbTmgRtpM2peK6M9ptY\/JwUAoDjq21onlz+pEtLcKPic8pO\/oHzQGNA0AnsVBYnPStl0eLewtPmAFUXf7UW8V2TbTdAd0ttesa6J4kDXNNtOkZL9dEr3rb8a7d8BFerpG2x02uzLFM1cPmzAEwQBJQZfeFhwbH7xkzwBkNGGrQPRBjjzmUBbtrTnvq2Qm+Z45iIwMU2W2zdhgn5i38MrtMaFbmO2OGsMAzRoUBdVLNHIUG6oe6MnbsTvFLB9TR4nkX5YaxBmsAcQBctW2\/nVgYZ+6OSbzPZ2pMU1lOIJjTuUwGXIP06D4cJpL022+\/NVskpB8aAFsmtioYF4KO1aay8lHy0gXAwSYs90GJiYeBBCntgnk8nLlSKPByaDycaGOi058IPLZ87GY09gVUCzsgLPe6qOx6Jupfne7j+lH6P5XP5A8NFRznZ\/0BggAk8xvNmLnF2QXOUMycOTOm9QLsXsf5AVg0b9Ywhns9M8JunHmIYMbTA1BGc2XOsi7Q+PG+whMGxYPdOeCNQsL654Q03jQaJwV35lGjRhk7kZ44ZU7Ynlu2nzxauX0fQh7Mo6+gQDCHywKQ8GvFQq7BeOBPASK4MugWrpTDyAcowX3lUvAo9Y3nBFuY4Q0tmzbiR28\/CziTvGgQzR9ezv7eL1Icn\/OsVzyXoGFLBMhKEpgnKtCiL4Xcf9I958H\/\/MQzl5h\/KC3w0Qrg0C9Rkzt\/7TqwTpQTj6dOWjYCiXVnv+u39rQdUE1eLr1ue+J9Pmp\/5NJzoYG20CTxtMAlCdchJDoSGWkPxYJhlJOSaL4YVJDcbN3sWCW51ODiWhd3ix2lHSUJzPOBtqKMeP6Z0twD\/w981gvuEY74VAAAAABJRU5ErkJggg==","height":119,"width":297}}
%---
