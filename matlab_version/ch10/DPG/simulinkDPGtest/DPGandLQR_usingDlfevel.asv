% dpg_lqr_fast_autodiff.m
% 基于自动微分的高速DPG算法（向量化+预分配）
clear; clc; close all;

%% 1. 环境定义（对智能体是黑盒的）
A = [1.0000, 0.0099;
     -0.0099, 0.9704];
B = [4.9503e-05;
     0.0099];
Q = eye(2);         % 状态代价矩阵
R = 1;              % 控制代价矩阵

% 求解真实的LQR最优解（仅用于验证）
[~, ~, K_lqr] = dare(A, B, Q, R);
fprintf('--- LQR最优解 (Baseline) ---\n'); %[output:2e681697]
disp(K_lqr); %[output:3c35878d]

%% 2. Actor-Critic参数初始化
rng(42); % 固定随机种子

% Actor初始化（线性策略u = -Kx）
K_actor = [0.8, 0.6]; % 初始策略参数   注意不能让系统失稳

% Critic初始化（Q(x,u) = [x;u]'*H*[x;u]）
H_critic = dlarray(eye(3)); % 3x3对称矩阵（x:2维, u:1维），使用dlarray

% 学习参数设定
alpha_critic = 0.05;    % Critic学习率
alpha_actor = 0.005;    % Actor学习率
num_epochs = 1600;       % 训练轮数
batch_size = 500;       % 批次大小
gamma = 1.0;            % 折扣因子（LQR无折扣）
sigma_noise = 0.5;      % 探索噪声标准差

% 预分配内存
error_history = zeros(num_epochs, 1);
critic_loss_history = zeros(num_epochs, 1);
actor_loss_history = zeros(num_epochs, 1);

% 预分配批处理数据
X_batch = zeros(2, batch_size);
U_batch = zeros(1, batch_size);
C_batch = zeros(1, batch_size);
X_next_batch = zeros(2, batch_size);

fprintf('--- 开始Model-Free DPG训练 (高速自动微分版本) ---\n'); %[output:02f43ce1]

%% 3. Actor-Critic联合优化循环
tic;
for epoch = 1:num_epochs %[output:group:174df7de]
    %% Step 1: 收集交互数据（Experience Replay）
    % 生成随机状态（重用预分配数组）
    X_batch(:) = randn(2, batch_size);
    
    for i = 1:batch_size
        x = X_batch(:, i);
        % 执行带探索噪声的动作（行为策略）
        u_noise = -K_actor * x + sigma_noise * randn();
        
        % 与黑盒环境交互（观察代价和下一状态）
        c = x' * Q * x + u_noise' * R * u_noise; % 即时代价
        x_next = A * x + B * u_noise;            % 系统动态
        
        % 存储数据
        U_batch(i) = u_noise;
        C_batch(i) = c;
        X_next_batch(:, i) = x_next;
    end
    
    %% Step 2: 更新Critic（策略评估）- 使用自动微分
    % 将数据批量转换为dlarray（一次性转换，避免多次转换开销）
    X_batch_dl = dlarray(X_batch);
    U_batch_dl = dlarray(U_batch);
    C_batch_dl = dlarray(C_batch);
    X_next_batch_dl = dlarray(X_next_batch);
    
    for iter_c = 1:5 % 减少Critic迭代次数（从10减少到5）
        % 计算Critic损失和梯度
        [grad_H_dl, critic_loss] = dlfeval(@(H) critic_loss_wrapper_fast(H, X_batch_dl, U_batch_dl, ...
            C_batch_dl, X_next_batch_dl, K_actor, gamma), H_critic);
        
        % 梯度下降更新H
        H_critic = H_critic - alpha_critic * grad_H_dl;
        
        % 强制对称化（不提取数据，直接操作dlarray）
        H_critic = 0.5 * (H_critic + H_critic');
    end
    critic_loss_history(epoch) = extractdata(critic_loss);
    
    %% Step 3: 更新Actor（策略改进）- 使用自动微分
    % 将策略参数转换为dlarray
    K_dl = dlarray(K_actor);
    
    % 使用自动微分计算梯度
    [grad_K_dl, actor_loss] = dlfeval(@(K) actor_loss_wrapper_fast(K, X_batch_dl, H_critic), K_dl);
    
    % 提取梯度并更新Actor
    grad_K = extractdata(grad_K_dl);
    K_actor = K_actor - alpha_actor * grad_K;
    
    % 记录与最优解的误差
    error_history(epoch) = norm(K_actor - K_lqr, 'fro');
    actor_loss_history(epoch) = extractdata(actor_loss);
    
    % 每100轮显示进度
    if mod(epoch, 100) == 0
        fprintf('Epoch %d/%d: 误差=%.4f, Critic损失=%.4f, Actor损失=%.4f\n', ... %[output:927a93e7]
                epoch, num_epochs, error_history(epoch), critic_loss_history(epoch), actor_loss_history(epoch)); %[output:927a93e7]
    end
end %[output:group:174df7de]
training_time = toc;
fprintf('训练时间: %.2f秒\n', training_time); %[output:3a125ae7]

%% 4. 结果输出与可视化
fprintf('\n--- DPG (Model-Free) 收敛结果 (高速自动微分版本) ---\n'); %[output:031d64c5]
disp('最终学习得到的增益 K_actor:'); %[output:73554fa5]
disp(K_actor); %[output:8138bb8a]
fprintf('与理论最优 K_lqr 的最终误差: %e\n\n', error_history(end)); %[output:6bc57924]

% 绘制收敛曲线
figure('Name', 'Model-Free DPG 收敛曲线 (高速自动微分版本)', 'Color', 'w', 'Position', [100, 100, 1000, 400]); %[output:1e8bf8c3]

subplot(1, 3, 1); %[output:1e8bf8c3]
semilogy(1:num_epochs, error_history, 'LineWidth', 2, 'Color', [0.8500, 0.3250, 0.0980]); %[output:1e8bf8c3]
xlabel('迭代次数 (Epochs)', 'FontSize', 12); %[output:1e8bf8c3]
ylabel('对数误差 log||K_{Actor} - K_{LQR}||_F', 'FontSize', 12); %[output:1e8bf8c3]
title('DPG收敛过程', 'FontSize', 14); %[output:1e8bf8c3]
grid on; %[output:1e8bf8c3]

subplot(1, 3, 2); %[output:1e8bf8c3]
semilogy(1:num_epochs, critic_loss_history, 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]); %[output:1e8bf8c3]
xlabel('迭代次数 (Epochs)', 'FontSize', 12); %[output:1e8bf8c3]
ylabel('Critic损失 (对数)', 'FontSize', 12); %[output:1e8bf8c3]
title('Critic损失收敛', 'FontSize', 14); %[output:1e8bf8c3]
grid on; %[output:1e8bf8c3]

subplot(1, 3, 3); %[output:1e8bf8c3]
semilogy(1:num_epochs, actor_loss_history, 'LineWidth', 2, 'Color', [0.4660, 0.6740, 0.1880]); %[output:1e8bf8c3]
xlabel('迭代次数 (Epochs)', 'FontSize', 12); %[output:1e8bf8c3]
ylabel('Actor损失 (对数)', 'FontSize', 12); %[output:1e8bf8c3]
title('Actor损失收敛', 'FontSize', 14); %[output:1e8bf8c3]
grid on; %[output:1e8bf8c3]

% 保存收敛曲线图
saveas(gcf, 'dpg_convergence_fast_autodiff.png'); %[output:1e8bf8c3]

%% 辅助函数：高速Critic损失计算（向量化+避免循环）
function [grad_H, critic_loss] = critic_loss_wrapper_fast(H, X_batch_dl, U_batch_dl, C_batch_dl, X_next_batch_dl, K_actor, gamma)
    % 向量化计算：一次性处理整个批次
    
    % 构建当前状态-动作矩阵 Z: 3 x N
    Z = [X_batch_dl; U_batch_dl];
    
    % 计算当前Q值（向量化）：Q = z_i' * H * z_i
    Q_pred = sum(Z .* (H * Z), 1);  % 1 x N，每列是一个样本的Q值
    
    % 计算下一状态的动作（向量化）
    U_next = -K_actor * X_next_batch_dl;  % 1 x N
    U_next = dlarray(U_next);
    Z_next = [X_next_batch_dl; U_next];
    
    % 计算目标Q值（使用当前H的数值，不追踪梯度）- 向量化
    H_val = extractdata(H);
    H_val = dlarray(H_val);
    Q_target = C_batch_dl + gamma * sum(Z_next .* (H_val * Z_next), 1);
    
    % 计算TD误差（向量化）
    delta = Q_pred - Q_target;  % 1 x N
    
    % 计算均方TD误差（向量化）
    critic_loss = 0.5 * mean(delta.^2);
    
    % 计算梯度
    grad_H = dlgradient(critic_loss, H);
end

%% 辅助函数：高速Actor损失计算（向量化+避免循环）
function [grad_K, actor_loss] = actor_loss_wrapper_fast(K, X_batch_dl, H_critic)
    % 向量化计算：一次性处理整个批次
    
    % 移除K的维度标签以避免冲突
    K_unformatted = stripdims(K);
    
    % 计算策略动作（向量化）：U = -K * X
    U_policy = -K_unformatted * X_batch_dl;  % 1 x N
    
    % 构建状态-动作矩阵 Z: 3 x N
    Z = [X_batch_dl; U_policy];
    
    % 计算Q值（向量化）：Q = z_i' * H * z_i
    Q_vals = sum(Z .* (H_critic * Z), 1);  % 1 x N，每列是一个样本的Q值
    
    % 平均Q值作为损失
    actor_loss = mean(Q_vals);
    
    % 计算梯度
    grad_K = dlgradient(actor_loss, K);
end

%% 梯度验证（可选，仅在需要时运行）
function run_gradient_validation()
    fprintf('\n--- 自动微分梯度验证 ---\n');
    
    % 生成测试数据
    N = 20;
    X_test = dlarray(randn(2, N));
    U_test = dlarray(randn(1, N));
    C_test = dlarray(randn(1, N));
    H_test = dlarray(eye(3));
    K_test = dlarray([0.5, 0.5]);
    
    % 验证Critic梯度
    fprintf('1. Critic梯度验证:\n');
    [grad_H_auto, critic_loss_auto] = dlfeval(@(H) critic_loss_wrapper_fast(H, X_test, U_test, C_test, X_test, K_test, 1.0), H_test);
    grad_H_auto = extractdata(grad_H_auto);
    
    % 数值梯度（仅验证对角线元素）
    epsilon = 1e-6;
    grad_H_num = zeros(3, 3);
    H_test_data = extractdata(H_test);
    
    for i = 1:3
        H_plus = H_test_data;
        H_minus = H_test_data;
        H_plus(i,i) = H_plus(i,i) + epsilon;
        H_minus(i,i) = H_minus(i,i) - epsilon;
        
        H_plus_dl = dlarray(H_plus);
        H_minus_dl = dlarray(H_minus);
        
        [~, L_plus] = dlfeval(@(H) critic_loss_wrapper_fast(H, X_test, U_test, C_test, X_test, K_test, 1.0), H_plus_dl);
        [~, L_minus] = dlfeval(@(H) critic_loss_wrapper_fast(H, X_test, U_test, C_test, X_test, K_test, 1.0), H_minus_dl);
        
        grad_H_num(i,i) = (extractdata(L_plus) - extractdata(L_minus)) / (2 * epsilon);
    end
    
    max_diff_critic = max(abs(diag(grad_H_auto) - diag(grad_H_num)));
    fprintf('   Critic梯度最大绝对误差: %.6e\n', max_diff_critic);
    
    % 验证Actor梯度
    fprintf('\n2. Actor梯度验证:\n');
    [grad_K_auto, actor_loss_auto] = dlfeval(@(K) actor_loss_wrapper_fast(K, X_test, H_test), K_test);
    grad_K_auto = extractdata(grad_K_auto);
    
    % 数值梯度
    grad_K_num = zeros(1, 2);
    K_test_data = extractdata(K_test);
    
    for i = 1:2
        K_plus = K_test_data;
        K_minus = K_test_data;
        K_plus(i) = K_plus(i) + epsilon;
        K_minus(i) = K_minus(i) - epsilon;
        
        K_plus_dl = dlarray(K_plus);
        K_minus_dl = dlarray(K_minus);
        
        [~, L_plus] = dlfeval(@(K) actor_loss_wrapper_fast(K, X_test, H_test), K_plus_dl);
        [~, L_minus] = dlfeval(@(K) actor_loss_wrapper_fast(K, X_test, H_test), K_minus_dl);
        
        grad_K_num(i) = (extractdata(L_plus) - extractdata(L_minus)) / (2 * epsilon);
    end
    
    max_diff_actor = max(abs(grad_K_auto - grad_K_num));
    fprintf('   Actor梯度最大绝对误差: %.6e\n', max_diff_actor);
    
    if max_diff_critic < 1e-4 && max_diff_actor < 1e-4
        fprintf('\n✓ 自动微分梯度验证通过！\n');
    end
end

% 运行梯度验证（注释掉以节省时间，需要时取消注释）
run_gradient_validation; %[output:1a8a29c3]

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":34}
%---
%[output:2e681697]
%   data: {"dataType":"text","outputData":{"text":"--- LQR最优解 (Baseline) ---\n","truncated":false}}
%---
%[output:3c35878d]
%   data: {"dataType":"text","outputData":{"text":"    0.4179    0.2910\n\n","truncated":false}}
%---
%[output:02f43ce1]
%   data: {"dataType":"text","outputData":{"text":"--- 开始Model-Free DPG训练 (高速自动微分版本) ---\n","truncated":false}}
%---
%[output:927a93e7]
%   data: {"dataType":"text","outputData":{"text":"Epoch 100\/1600: 误差=0.4606, Critic损失=1.1236, Actor损失=96.0670\nEpoch 200\/1600: 误差=0.4371, Critic损失=0.5775, Actor损失=152.7641\nEpoch 300\/1600: 误差=0.3843, Critic损失=0.1731, Actor损失=151.2455\nEpoch 400\/1600: 误差=0.3153, Critic损失=0.0694, Actor损失=177.9980\nEpoch 500\/1600: 误差=0.2157, Critic损失=0.0166, Actor损失=216.1066\nEpoch 600\/1600: 误差=0.1005, Critic损失=0.0004, Actor损失=196.2683\nEpoch 700\/1600: 误差=0.0101, Critic损失=0.0028, Actor损失=213.0055\nEpoch 800\/1600: 误差=0.0341, Critic损失=0.0015, Actor损失=171.8101\nEpoch 900\/1600: 误差=0.0342, Critic损失=0.0005, Actor损失=191.6348\nEpoch 1000\/1600: 误差=0.0208, Critic损失=0.0001, Actor损失=197.3962\nEpoch 1100\/1600: 误差=0.0089, Critic损失=0.0000, Actor损失=173.0756\nEpoch 1200\/1600: 误差=0.0020, Critic损失=0.0000, Actor损失=202.9423\nEpoch 1300\/1600: 误差=0.0011, Critic损失=0.0000, Actor损失=204.2877\nEpoch 1400\/1600: 误差=0.0014, Critic损失=0.0000, Actor损失=185.3215\nEpoch 1500\/1600: 误差=0.0010, Critic损失=0.0000, Actor损失=201.5249\nEpoch 1600\/1600: 误差=0.0004, Critic损失=0.0000, Actor损失=175.9688\n","truncated":false}}
%---
%[output:3a125ae7]
%   data: {"dataType":"text","outputData":{"text":"训练时间: 6.18秒\n","truncated":false}}
%---
%[output:031d64c5]
%   data: {"dataType":"text","outputData":{"text":"\n--- DPG (Model-Free) 收敛结果 (高速自动微分版本) ---\n","truncated":false}}
%---
%[output:73554fa5]
%   data: {"dataType":"text","outputData":{"text":"最终学习得到的增益 K_actor:\n","truncated":false}}
%---
%[output:8138bb8a]
%   data: {"dataType":"text","outputData":{"text":"    0.4175    0.2909\n\n","truncated":false}}
%---
%[output:6bc57924]
%   data: {"dataType":"text","outputData":{"text":"与理论最优 K_lqr 的最终误差: 4.349102e-04\n\n","truncated":false}}
%---
%[output:1e8bf8c3]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAUMAAACBCAYAAABaZNgMAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnQu4XUWV5ysvEggESHgECBAQCBBoBMUHoEDbCiI6wiBKt4Ngo92K+jnazKc4I\/hAdGgf00P4fHWjtC9axumRqGkUCe9XQBCIhiQkkmCIQIA8MOR15\/vtw\/9k3UrtvWufs\/c+59679\/fd755zdu2q2qvW+tdaq1atGjUwMDDgmquhQEOBhgIjnAKjqgbDF154wa1Zs8btueeeuaR+8cUX3ZNPPul23313t8MOO+SWbwo0FGgo0FCgLArUAoZ\/\/OMf3YQJE9y0adNS+\/3UU0+5Z5991o0ePdrtvffeDRiWNcJNPQ0FGgpEUaByMKQXy5cvd2iIaHs+IKINcn\/z5s1DCgg3bNjgnn\/+eTdmzBg3efLkhNjPPPOMW716tZs0aZKbMmVK1AAwCVDXrrvuus0EILpQ14477hisD7px0Y+867nnnkv6zDigfWdd69evTzR6+jV27Ni8qiu5D8+sWLHCbbfddm7fffftug3efdOmTQktx48f76AHY9ZN\/atWrUp4d+edd07qoc+M6bhx4xJrKGZckIE\/\/elPybj4fAM\/Pf3008m9qVOndk2DtApQWP785z8n7e+yyy5dtSPZQAGC1qNGjXLLli1L+LzT+uFHaMG4QWsurEh+h0f1W17Hs+StFjBMA0SBJPdDQJn3Yr28D9PDQDD9\/vvvn3SF72vXrk0GJtYtIMCD0QFRXTAO93Dp7rXXXkFNGbBauXJlAlY8D\/NlXdK+Q7SGqdatW5cIM5\/lSp44cWLyjgBH2lWVNh+isX7bsmVLan8QjhDY\/+EPf3AAD8K+xx57JJYIQAOIaQyL8hR1bty4sW3NQCeBIQAOWNJO2kVfeReAevvtt98G9DVmgAoWU1UXvAYY7rbbbgm4WJlNaxOgR7kBoOwlGsAXug+d4Glbf5F30VhBIylUS5cuTcaTOmOUD6t4heStNjC0xGVgYWqYAILyMgx6iMGZVSjDy\/rob8HUJyzPweTUjUD7FwKPVkA\/JPj8B1h22mmnRJioQxf9k1YFU4YEteiAP\/HEEwkAcfGOCChtczELMvNJawlpGGg5aE4wMQy5zz77DNLiJEhZIObTXLQAAKE5AA1gwIyArQVcwDhPow8xcawQZIEhgsYYWboAPDwTAkPVRdsIAjxYFAylqdI2tGZsfDBkcoJPdEFHaASdoaXV7hBONCY7BvAq\/CXe4z7txgp8LG39cllgCE9a2QPU4An6GAJDKQV20i0qG9TBe9Mu4+zzEfKL\/MCvGs+8d8+Tt1rB0AIihNQsYWd7GEyMwIvCSPzxGSYHMHQJDGE4GNRezNYwGXWhWVkzk4FEcLh4ltmG52FOaUX0w2pbvlblC6oEhTYZPJ73L9rRu2lwaRdmQ4B4RwCIP8ymkDZm6Ub9ACKDTN8RJNrlOeoCKASulOXdKEcd0NTSGTClLgkm36Ut6d19kJGmJc2QiQS6SpulzarAMCSI4ocQGDKxMMHQV8aci3eFT+xvdsyggQUvjblt2wdDtBXoDN3pB\/RgwqMNxgOg04Tijx1WBXVzn3ahoywH+qWJijJ2fPJAIOZ+Fhj69AzRQW3w7gAZ7wY\/aLKSPIZklfehDcur\/lj6fIS8MPFIifCVBepE3nTFyFvtYGgBUUTOIq4ECi0OZgip8WlmkQhqzQ8RhXoBLd8\/YrUta5rkgWGMFqaFIQREM6vex\/4GGCK4AJcER8wE2HHfBzkYAzAUyKcJADTkD2aBWZnZAUY0TCYNQBiGRninT5+egGsaGNKGBQP6y3tZUyZGEP0yabQEABgzhA0gsxMndYhGPj9YAS3SHx9w8sCQuqGj\/LgCMNFbAor5DJ\/xHvRNWj3l7W+MJ3yPoAMiaGTUxdjAm936EK3ZaOmiCVdukxCA0W\/xjzWTNekUobM\/wYcwwgdDaZ9p7VgTPlbeegKG\/ssy6LxcmtpNeZkflkGzNAGe0eyhQYNAmB0wAUKlhQ+foGJ62gIgeD4PDKUlYRpIK4R5YSiex0QCeOkTAwtT++YH7XKfspShv\/IXauYOAbjtP+9mhdHekxkpDRS6A1yYu+oTzyKolEEIeB\/6ETKTqZtnKQvQlwWGWuihfoROYMB4QMM0l4reFdPKTnISHMbFuj+gN+\/Fe1qLQ\/UAOtYtkAeGjDX1Md7QFXoIwASM0vrgZ2np9FeWCyAPANI2dOX9NSkvWbIkAXz6e8ABBxTBm2BZ6mLyY7yt5ga9mVBom3dOu3hP6CZNW9YR9fG8tcZohzZCC4XUz9jYhboszZA2sYZoBx7Wc\/RXkwttQ1d4KVbeKgfDGIe3iC0fIv4YP85QhKasBYg0HxHlfOaFUDAhgyefTyxHZYEhgke9XHLO8znk\/JbvkQFEUOwl8NUqGZODwE8+lxBt\/HfgeQSTZ+2izOOPP57QBGY\/+OCDkz7DUNa09uuCVjCxhNmaajKzEVzGCzNPl2Z7vvvaonxO8hsLeH2\/cMhnmAX2tm2ZSNYS8J33nfgM\/UlbmjELZoyt6Mn4yh8NLaARf\/C4Fm8oyx\/3pbHL3WPpTF1yGekdrVIAeGqlnPvUx7jTjsYIcIH\/+R2rg35YP7NvJtMevui8C\/7gz5r8AD58ak3Xoj7DLDCk30zs0MqOqSY9axkUkreqg67ziBkCLX91ijJoLGh1CKYAIk8zhEkQROrD5BM4MZswWEWuNDBkwOVvpL4YzVXgov\/qB\/UgQHIJwJCAKzMcM6E0Z8ALQOE3qwXK2bxo0aKEOekbzyMA9EuaBcIFw0AbyiDI1C2tBMAB3NCcqcdqYsy4\/EZbgIlWUnmWWRgQRhgAI9pFoCwYWqGR5slv0kAsc4fAMGvRTHS0JhJ0ov+8n78aWyYY8r7y+ck\/KG2Lfuk3xgvaw4NMRvQPWvOfe9CEsYKO8usK3AAxLspAf8ZHAKAJif\/0g\/GwoATdqFOmOjTS2FOnD4axSowFHgAZHgiFgpUFhrwTdIVnuKwrC38tdISf7YJPrLxVrhnGAE6ez1B1+ARNA0NABEZXDBhMA4DmgWdWX9PA0M7ogAyDhC8HhtZKob\/aBUBIC0aLY0ABAUwJBECOZ5nKMBe+KBhhv\/32S7ppwVCLHmJM2lUb0jSklUhzkIDCODJnFKpDfZRHA5c\/UdqNTCL1gfcHnOkb7+xrgb6vJy28R34duwJpwZB+otkg6LxbyNyir9RPn7TKiYAC7qHLmlXW2e6XBdR5vzQzmbHQohT9p5wmFwBHPkMt6In29BdXCf2QycdYi4aAn6wHtDx+10Ii\/Kzxoi0md7swA6\/bsBPxftrCiwVD9Q9ayhT2aQJdFYdKXxRbKI3Wlue9KSvwSou9pC25rtI0Q9EZPhSNrd+bNvwV7lh5GxZgmAVizMCAEQOQBYby+dm6rGM3y0yW3wLmhmGZlWBMxZuhleqShguDADbM7IAJWgsgxn0NpvybMkGtCW776S9wiLEpA3MCdsyOMIXAUf4zG3bka12U5V0AKplbabRGyBSGYxdQfDDEVEf4\/dnbrxdwQOC0Os19xpB2bDwck4msBfrIO1iBiNEks\/jH8kCWzxBwps+aEBQBQZ+1Wkq\/pZ0JNKAZAC+Qg+bwkCInFGLFOCkMS7STHz0U1yqrCN5hwSaL9+kXPMS46FL4m+Jo5VdmbAE\/n+diNcksWmdZVZaP6JPogSaKIoE8UcaPoS0ib0MGDPVSVg1OC62B+SAYs6514op5QkHHCJXVIBAwu3rtgyGMibZmg661asmAM8spOt4PAFY\/YGoGUOE4\/A5DaLcFbco3wjtR3q4ii7HUNwQJJqHv0Im6FDrCd2l8AhYb1wiz8z7SbnmWZ+TIl\/Yhs1eLRDbgOBRG4\/\/mh6KEhIN+AAICYP7TH4Qe4EDwZE7b+jThWTBM0wyhhcxJykvAQv3J0wwtDR599NGkCmmDacIvDUdaEvzNZAegarJgrKlbk6QAlMkSHg6Fw6g938UQKhsCMPqDzNjJnBCrNA1f1kiWZigNnL7x3nbRw9InRjO0E63ddEA91oVm642VtyEDhho4XhKTAKIUNXu1upwVyCwi+ia5AIcBY0BlGvlmhw0toKyNK1PditBHyBACgFUzHWaCoukBHvyCcnan7ZJQ32Q2+FocDC5fi8wufpN2Ci1ph3vyfQGCvBt\/WhhBCBkH7doQAypkqCwwpH5tvaI9u0PEF+o8MEwDIwmIzFjKoamHAvR9gJFrwIa6iAbQirqlGWpxRP5R6kKI9buN92SM5OtlQhRQ2kBuTUp5YOhP1iEwlKuFdhRF4Mf+arLPA8M0Ois4Wv5QACtmN401663v1Vdk5BKi\/TS5jpW3IQOGAhnrNysKhtIu80JrIKwFQ2ZoBTZr0AEL6qM\/FqQsgPEcAuZfKgNYwhgIkHZy2FVuBEOLRmkmMnX7Jgv1ASYIhLQJQE1BxvwG6MIkCAIASHnN2PzO+6E5cg\/HtI2V899HpiS\/5\/kM08xkaIJGSH+0kyZtAUVmMrRnXDRB0mffTA4JKfShLegBHRA2frOrq6Hn\/Bg63lsTj8AQ3qJuASVjzJ+C4iknYNJCmDR2rcoy1lgTjAl+ZN5LGr4WggCFomayv93OvmPaAorA0G8rK\/bU1qtdH\/I9Ug\/vm7Utz48LVbQB8uCDodwIsuJC2\/Ji5a2nYKigTwWUMtCh7T3MSvhAIKL1NRUFQwYpL+gaosKAWrnToMl\/B2BoP6m\/N5n6GWzAhHdJ0wwBPuoHaBBoDahWHBEEQJJ+0N+sumAcGNNuyRNAAMTQSFqszBQxl2IMaUvOdi3i0CbhN\/wOgPEM4C5NVrFdjAk0AVTUFxhWq\/WxCygSNqs1hMDQJhSgb2j70BDgpo8KOrfxb1Y40cr0jgI\/3ol6ARzFAYZ2EMGD\/EE3AAt6+Gb\/woUL27t4rFbImNJfeEPgqN1PjJ00Qu3coB+8jxZj+KxFNwAVmhddQCkKhkqSQd8VuQBdBeLQ3o\/pFK3pLzSVj1RyZHlau638iUdKC7TAekLemKz8iZY26JciEdIms1h56ysw9HcV8LLy3SnMxCZA6AQMBYgCVzQRBRbb7XgwMsyusBP5zwQCIUGVr0smJuAAwwNKNnhXmggDrJAKMQT1wzgwLoJjQy2UuUR9gBlt2IvV3mBY3o13Aqioj\/YUUiFzmXIwEQxjY\/hk\/mvCon95ZrJdYADU5L\/MC62hDfpH32z6thCN6Qd9R6i0zQ16AbpandWYWSFDsOiHgsT97ZbWvGX8AAB\/L7y2K1qg9MGQsCb42PpnpbXIyW\/9iXyWmWrdG1r9l+sDXpR5zfjTN\/6gA31PC62BnpTP8i9Cp7T7lscYT9qRCwMahdwK9FMxl\/SfftqgdoUDaXEvtH0V2bEy40+qdqeYwtvSJrNYees7MAyZJzBfKE1Pp2AogVKiBhv4Kt8OABLSDnz\/EeURPu0oUCAoYCDTWquJCK20QGtu044WexBo1aU4MJmDgAZaCe2JGSgPQ\/KbNrfLfIPZ6I80FJnMElQrsPRHQAnz0A4gDoAo8UCWZqjYN2iq1VGltYoJuqY9BEZ+MwCP90WoBMwImcx\/+mtBywotwg89oAtCxaQiTZ\/nmFSk4Vh+A1wViK76eWe0k7TQG4EhtFKcpUJoNA7yCzLOfKZtxkqAJ4uA8dEf7Wu86AOgZq0O7uNXpF+xQdchzRD6KL6Uz\/AovMu7MI6K0ROPUdZORArupj+KM9REK1kIpQTD5aCJyS6QpS2uWDAECzROCgGDd7Un2vJFEXnrCzCE2bMSv4YAstvfYCAGQ2BmtwLF1C2tRaETMAB1MMtpxdfuP1X4DAMPOFAWJrFxezYg2Qo0\/ZEfRVuaJFA8o9kfrQQBkv\/RBztrwggoiVu0M7BdNVdsJvXlXTBgWUl5\/T2zCucQs9MXxWbaCctqjNAPgEIYbLAyk1xW7ju5SeANgRrmXNrWTYGhDboO0UogJ58oPKCthr5rCMFXoDi8ZPcgy8yUZpsVH5k3Zty3CUEYQyYRQFPuGfgzRDM7+Sgphd0EEBofvz9MPryntVKgRUgWBYaKkNCCjLW6oCfWGTIiV40sghh5G5FgqBVRmEwb4UN7U7OYSf4xmAVQY2BhGn8llzqYZf30V2l1U29aXTANQmqdxNbkQjC5tPpGX2Quw+CUhdlhZJmlfriOnwBWz8UIls3KE1M+qwy+PxhZ9OVdGDdoAJiF6Kz65LOlHHUgSGgcRY6SgFYKes5KhKs4RzQq+kc7gKjVMNGeAEH6xT0tlCDYAL8PeLwHfAnPhMxQ3odxzVr5jqU\/\/KYwJvqhiZE25NNOq4ux4J3ge8CGeuSqKKJcqA+8T9o72ThHxlHuH79vjBt94l1i+jBI3vphO17swJVVjoEGcJiF+IzAV5k4s6x+59UDyMGQdi+xNCQ0EYXIwDCYGtAAIdWulrz6m\/v5FFi8eHFbo5S\/lzEBUAFmQF7WBFoNoAeohBYO81trSpRJgZ5qhmW+iOrCj8jspHAXa3LJlJPDHgAcTmCImawdJgicEkhgsiCYgKVWjhFE5dpLy6RdxfgM9zrRFG3iDlbV4UkmIJnS0BsNEe1FkRJolt2m2x\/utK36\/YYVGGpLnb+tR+DIfUACrYgZGY0InwX3u80NV\/VAxdSPhovAEQbEhYNdPhTMD8w4NBSZQmgsCCNmTmhnS0ybTZkwBbS1DVrzmUkqtCCHVqj4uYaWvaXAsAFDZl9AUDFHaIbSCrVAA\/ABAICEQFBb4eSr8DPJ9HZ4slu3iy\/93M+0vg0VWndCZyZY+S7TFl\/qHLPhTOuy6NgXYAiAKdNyESd3iAhaRLBgiANbm8sxFdNWPWEYzJyYPG5lDUA39aAFotH6wqptVvgLyzhVrps+ZgHhfYuecN\/\/zdPuw6\/cehBWFW11W2canbPq1UTMGHR62FS3\/dbznfD1gBtwo1zrDKDfrPyxW\/jcze7sGVe6f1vwofb\/g3c50a1YN9+dduAlwa7+edNqt\/3YSc7WpeftA7cuv8q9btoHk586oXVZdGrA0FASUEbDfPDBB928efPc\/fffn9w95phjkv\/6rt\/877YMz2TdP+WUUxJzdc6cOYkG67eR9\/x5553njj322CTjsQ+G\/SSIaYwqWs\/4+nJ3yG+\/4fYb1TqTpuyLjEFHHXVUm85F6z\/66KPdBRdcEKRz1mSMj1CB+lmaJaviKAD4DKu6ROvPf\/7zidKRd73i75\/OK+LWPLPJ7bTbWOcGti26YsF6t9eM1kmNy+fu6qadtPV0QJ5bs2SSWzFvB3fIO550O03Z9hjaz73tnm14OrdDJRToCzDkPTBbEeJu4w1DmqFvJqet3Ilpbr\/9djd37lw3e\/bshMSnn3568l\/f9Zv\/3Zbhmaz7Z599dhKUDejy3n4bec9ffvnl7qSTThokpMqYUoQv0rYMFqkjr2woprMNht9Y7na57oNu9Nqn8qrp6D7hNUcccUSbzkUrYVKaNWtWNBh2MgYAYt5xDkX7bcuL1meeeWYuGL7vm618maELPTGAfZ11jYq2Hj45qI4RDYZpudA6CeS1YJjMTGZ1WQsoaYA71MFQ7x6rleAvRXup8hjKtJhOC4bjF811E2+b1ZlQ5TxVNxiqO7HaufIOKgFxFUQQrc8555wEDJWqDiWB69z\/vWcVzXZc54gGw46pFnjQB0MbWsNKclY8V8I0D81za356lfunJQNu9n\/8KmnhTW96U\/L\/hhtuaLfIb\/53Wybv\/hlnnJFE\/0sz9NvIez6kGcYIot3zqah+ZUkpcxxUV1pMpwXDsU8+4ibNubSK5pMA7To1w6wxYOJRXCHB7rhJ6gTDyy67LIkewPLhOv7445P\/h7+nFX3QL9eIB0NfO+xEK+x2MBMBfXiec\/\/zXUlVDz6\/xV21eLN7cnzLnwN46WLHiv\/dlsm7\/7KXvSxZ\/BAYageM6sx7HpPnoosuivYZEnStHRDEuRFKo7yBaCWdxLh1E9M50sAQWgNEaO1KZ6Vs0vBNtwuHWbwvWp9\/\/vmJZqi95kwU+x0z1r3mnIndik6pz49oMPRDYKAsgiYfotT5UikeqEyaobuiBYb+tWL9gPvPd25I\/CYnn3yyu+mmm9pF+M6l3\/Lun3baaQlTCgyLPv+Zz3wm0VjzFlDQhPFNKoMOAkCcZdZ2thg6dxvTacGQ9iZ\/5x0xzRYu0y+aIfxMPGFappfCL1bggTSf4bsu3zu4gFGg6mJFM\/yEtqIRDYah0Joyw21iR8yC4aqB8W7yqPBBQtT3w\/Evd\/805+521UXN3CrNZD+0RqftacVSmUo61UbKiOmE1nf\/bqn7yx88+RINt7rn9\/rhu2OHLLccGvCRRx6ZrOzbMz5yH3ypwMyZMwstoPCYjiYgyJo\/6Gz3kDM+hHBxYR1kZUeK7WdWucrAcNCKyhbyCA3qRvs2P3M78hrRYAiNbF44vocSp0bSsuNiFgxvWreju\/ieZ5K6MFlnbr\/BfeHAwefIoimeeWfr7OOiZm5VZrLMMe2LVYICbVrnvs5Voc\/2XOWihOsmptPXDFttS3UY5d668KtFuxMsz\/sfeOCBbsGCBducUx3TAJNcmgae9rzy9em+dp9oW6SyPSsprc0aFNOnomXSwDBr5VhtkLBozBjnvvX+x53K8zmkVf78i2vdaZ\/YcVD3fnvDanf3dc+1n20PccpLjBm1nbv0rbeN7NAaAaJd6ao7WNWC4a2b93T\/7ZbHkyGTo1mO55+8dju314StcQGvvenFbcrwjMqH6kDASJQgM9lvI+\/5NDNZOR6VwzCBmIGBhLn44x2VZw5zuZurfDAc3JsD\/u8F3XQveRYz+dBDD3UPPPBA6pGhWY284Q1vSPXNpj3HGKCFKmsOQfwEPss1EUpU3PWLZlQQAkNiBN\/1hb23eeqnX1rpjn37zm7NM5vdzd9pKQN51+kf38M99fgGt\/y+0W76XoclPL3vUWPcyRfsloCoLgumts7XnzfFzTiu5bd875HXRocx5fWr6P2+iTMs2vEqyotp3MUnua8tHeOuXfJCW6D4IMczn1+\/90T3pRmtg6y5Tr2\/FWRqndO2vHLo6Tf2RbM7RGDo3+d71vNpCygSRHbZAH5sP8RsQzgFkDpasgy\/Ie+ctfUxL6ZTsW+rzvuxGdKtzqVufYm98BnaMZArQudC6yWhPVq5zdxeBU9TZwgML\/jmfsEwvx998o9JQPWJ502JBkP1uxtan\/u1fdzD1+xZ2CVRJs36Agz70Wf4+d9tcj97MjupKbrhHSe3YrWsyRwzQN0wDvWnBQOHBNEeyqOU9H7y2Jg++2XKiOn0A4EHg2KrxW4AsSo6Z9HLHwMmNR3shGuCfcu4KxgLTGQl5+1kDGKe8cGQXR+YuaHLanIxddsyvaB10T5mle8pGPoZjf2O5sUFlkkIO4OiGV6xZIz7ydKWZqjVbJnw+o3v7z9grDt\/+pik3N\/+drSb\/0zLr6jEneqjXweLGTNmzGhrhv79vOdjwVD+K6Xxh6aEV5AsUyfgZWV+zqJx1zGdy5e70K6IbQBxy2Y3+ZrwCn8eD\/RCQC0YAngslmAaE9Kkc7xZRMF3aA\/yiklGmve+ofs+GPq+QjRBNMJur17Quts+2+d7CobqiDRDBBWG4TsxccyYdYXV+GD4o03T3f+6dUHSRd+fp99CPkT8h\/79UB1V+gwxiZWsFkHUljsrbKww6yQ5K6RlMldWXWlOfT1TlobYCwH1NUO0QDRAAaHeUcc84E\/0M453Mg46mwQ50nkylq+ZeNxOq9zp\/zB4x8l91z\/v7r\/++U6aHPRML2jddadNBX0Bhv7WJQVg68zWMl84RkDRDL+3Yqyb9ft1SXHfn6ffrE\/vzpfM5S8v3OSuW745eaYXPkP2eAOGylvon+pn31+p6RHEqrSSNHrngeHqUy91m6bOHPR4J3uY+11AGR+Zy93wOZo+kxvJZAnuBmA1IVpah8CwG9PY9rnfaZ1H374Aw77yGS5f7gDDGJ+hJa5dYZZ2mEX8oc44eYyVdz8PDHl+y\/hJ7rlz\/nlrVQPOTf5useDskUhnXCM6WTFPMywLCKUg9GLrYx6vxd5vwNBQyq4mFwVDqpF22IBhPvvFgCG1rJ\/5NvfCsf+lXeH4R290E+\/4en4DL5UYDmAYs+1Rq9bKbG4XZbbRDC\/aMwnpLMtXqMEY6rTuCzCEmL4z3v8ezf1dFLRgGLuAoubwbc49rvWNVegrHmsd4G7v81m\/VbGA0umunU6f64LUwXCPtPqefff33MDY1qo9eZ8mf+es6KbrFtBOaZn2XOy2R227JGQHH7ANm7JgeMgpG9whSUzfgLt\/9mp330+79xU2YBjNjnEF\/ZXluleS2+bELbOd+\/ZH3U9GH+KuuPGhpPP+vmH95u9NPs\/Nd4e455JnLnbHZe5drmpvsvyt+KHsWTD+KIQOyup0e17cCA8uFasZ6qn2gsqAc5PmXOLGrpwf1WzdYCg+wlTtdgyKbHsk0Qa+QiWSZSx1xKml9YkfGdXej1zWwkkDhlGs2HkhFgKyUvR3XnP6k3YHyuzxR7jL5tyXFI5N7rrrqBfdB9fekjxz1Y6vd\/96\/S\/bjfl1VJHc1X8zHQif9sbsQMk6E7gKGqvOomC47oQL3YsHndTuUmzsYS\/A0NKtjDHoZqdPe5JfvtyRz\/CNF2\/dJHzNh1eWOsS92Ade5gv0jZnMSym\/HjFZPUvh9ZJm2AkY8g6fWtcCwNsnHu7+YfZvegqGZTJK2XUVBUPab2uHo5zb6ReXunFPPpLbrV6DYW4HIwqUBYazrv2U2\/7wRe0W539334jW44v0Yh94fO\/yS\/YcDK1Zh2lMLBZhAd2m\/89\/9W1LWJ9hp2D43mduSPYtLxs7xZ39y62BrL3QDDuhQV3PdAKGq9\/8Obdpz0OTLnJMAKE2eddwBcPYoyysZvj9+z7qXtyudQYKiye\/\/mprO6mf+brT7\/jBDzvsMHfPPfckdYY2EmS118k+8LzxL3K\/p2AYchqXdRZKESKobBlgeNCCOe7nZU\/6AAAgAElEQVSd01o7UuyqcgOGg0ekEzAcpB1GbtMbbmAIDYocZWHB8F8eemd7ENhx8heHvjr57me+7vQ7GwnIEHT11VcnYBhKPpLV3rnnnus+8IEPjNxEDVYzxDTG8Yvjudea4SXjTmyn9S+S9v+hW37piDnkIr0Xe5a5\/DqqzGfYyURQ9zOdguFzZ13ltuy4e9JdG4SdmNCjnJt89eA4xOEIhkWOssgCw9GbWplibHKRbr6TfIQMQTfffHMChqHkI1n1c8AZx1mEEhbXwZ891Qz9F7TAyD2AsU5QlIB+8pOfdPPnz2+n9fdzFdK3tLT8Y1Y\/FQRDv46q8hnWwTRltNEpGL7w6ve69Ye9OekCiygtP+K2KZS1wDIcwLBbekPr3z\/2m+TMY10EWxc1Y\/PKYyaTSPeOO+7oyEwuehJht3Txn+8rMLSd6zReqxsCSUA56Im\/NHOBNrLyDSr4+v33b3RPvzjgTthttPvYwa3zYZXdpqq9yUXen61gxKNlnetbpL4iZTsFw41TZ7o1p346yarMAgrft7k4Z2Tl\/OSQqX4BQ\/zg\/JHV2m59hA5YQv6+5SK0zCvrg+HmTQPuXz64rLAZm2f2NmZy3kgMoftlaIZslhcYEnzNThZ9t6R49+PTKjsQyic5flgSXzBz24SuCl\/i904OhOpmaDsFQ9rcmsQh+1CNHX91udt9zeKenI4XOwYk0mBfMYdyTZkypRuSpj4Lre9dcIP7xWOfScrM\/seVbsWjxc3YPLO3MZNzho+BYLBjkljiZ4A5iH2rMwBYr1CWZqh9yvIXsrrMmSrXbpruPjCulQmH71\/YeGTpma7RPgBkVuY5X4MrDQy1u4Fy3aT\/70SCuwFD6zdU26PWP+8GJuy8TVfIll33fln\/DJq0MSBrzbJly5L8hoxBVdohtP7hXRe7hc\/e7AYGnLt\/ditLTZ7ZW\/R+YyZHgCHR+KQwyvL\/KTi1F\/GFPhh+7GMfS3yGaY5lymdlpfnvh411b5naWlHWddaD493azaPcqZPWuo8e1DKZr9x4qLvu3t+nOps7yXQdOrw8BIaK6cREJuM24Fnn1Q0YvvCq8936w09rd3fM6hVu5598pDXJDMqY7dy0O7\/ijt5tdHvSKfqOnfixYseAcSHTDLkm0aqquiwY0oZ2nuSZvUXvN2ZyxAjqXI7QgkivtoWFcr9JQInUZ8N7WrwVr5yVfPW0qaPd\/zhs3CDKnHRH6yt12gw3\/J4Wk5W1tzlNSGMEkRRfTFBoJL7pHDGcpRTpBgxZTUY71OWn9vIB8a0Lv9Z3YMhRDPAYF1ohuzequhIwvPtit3DVzUkTylSTZ\/YWvd+YyZEjGAJE\/UYVda4cp+V+K8tMxixWeA3vhrn85dHHJpTSooz8iD95YrO74tFNhQ+USjsQKg8M2SUgoOcz+e96cXUDhls1wNbxlKGteT4gcvyonVxi37kKzZAFFHJJMhnhq41xIcX2N1QOWn\/1xjPcmg1\/GgSGRc3gvPKNmVxglAR++Ea0itaLhAy2yzb3mwS0WzM5BIbnz98+aVam76yjx7ljdmmdM0twdmgWLttMxhwjzTzbHRkDJf8sMISlFS0HDNPPR2EfM\/uZdfUDGGISy6LAHQQQ1rE3HFp\/9hfHJ6RY9+R494NPL0w+FzWD88o3ZnJB8bBHWZZloikMhxU5rZbaDfJonaQ1YkWV2RiGBHz83G\/dCqglhV1BnvfsFvfhBzYOohTtzzlmffLbufducAvXtoKzY6+QxsKCCECn7Nb4AwnlYFFFB6hLCHWgfGx7ZZcrg9ZbdtrDjV7T0nZCl9UOd7jrn92E388p\/BpFNcNFi1p7f+EzLiZ7+WMZBy7GBBBEM6\/jsmBocxgWNYPzyjdmcgejKUAsI2uKFgLohsDVgiO\/MyMrxxtggZkCWPq538oQUNrzNcPz5m1wC9YMBjsY6wdHD7jJo1o5D2MSwlpSh4T00UcfzR0NFrKgUy9W623nyqJ11guvedOn3ca9j2wXic10k0fnrDZDYEgcob2YkAij6fbc6tzBfqmABcNH71hX+AjQ2Hb6JaYztr9+uZ4FXZcBiAwyoIbJByiKwWz6L2Zl2iJ0RCYJGiKH9Pi538oUUKsZ2m15GgAxzj9uN680MKQi32eINsK74ie1F1p0L7XDMmmdxfxWO6wDDENjwG9aOUYrRHPnYtGkDleFBcOl925yv\/xWK4FIng+w6P3GZ5gDwwyEEl3GIHYnoTUAIae9WTDk8HSF8ihJZt7WPn81Oaa\/aWWU9Zr7Wkm2ZZX77Quutcx847Pj3Od+N9iUzmp\/5syZwQO3sxZQoA\/mG\/RCKy7LTdEJneoCwyQmceLuyb7libfNcuMXzS3U3aJmstwU8BwWyP7775+0Z8Ob7AIK2qG2ahbqWIHCFgyfemCS+\/erHk6ezvMBFr3f+AwLDEpVRcsGw7lz5ybb8TSDM5tz6Tuf7Qyv77aMND5+u2zjkW7duFbohOpAM8THcvLin7r9Nj2T3LvCvcKt3MDR9Pn1w3j82U3tjz32WNIv\/IP2ZEE\/zlCuBQCRCQLTue6rLjBsbd+7tP16RbXDomCImQxdAcWsMag7tEYLKI\/csN7dcV3Lz5rnAyx6v\/EZ1i1FgfZCYKgs2SEzOa3LElDCXwDEW2+9NSn6ute9Lvmv7\/rN\/27LaHGE31hJPujYEwbVccoppyQLOg888ID7fzNb51BcNvGNg9rMqv8Tn\/iEI8uHBUPrM0T7IGQDJ30o6JqdPmjPvVpVrgsMEeglZ3y7NjBcsmRJMuEBiHaxKjQGMp0ZozyrpRsxs5ph2an+bb8an2E3o1TSsz4Ypi2g5C0a+GA4e\/bspIexaf8pq2d+dtKk9uIIPsNX\/NVbkrp036b9l0k9f\/v93d\/+vLUIQpsqG+oDqY58MKQckwAgi7msEBo0FH63ZrHMaZ5hcSmPNiUNVbuaOsFwymve7ubtdXrSdtWaIW0AhizaMdlw4XvjD7+tHYM6t+NJM9S+5LLHk\/oaMMyhalGfoa0u1n\/ogyF12NCa2FXrKsBQWWqykrv+8JiBZAWaS6vKnYKh6Ldu3boksBcaEl+IYPo+QrRDQJKFJRZU6rysf3bp0qXtpkNO+6ydOFk7gaiUd973iGPdjdPfm7QxZstGt8e15w\/aXdTJTp8YWgGK0BgtkX76E5LlU\/pZ1RhYzfBXVz7vlvy2ZYkUXSDJK98soMRwxRApUyYYAmach2LBDTKENMMdtmxwPz9hcELYbsGQttA8COtAKHk3VtStf5AAbO7JN1TnMInW+GYvueSSdtMhp7127VCo6H2lSrv+4I+222B73q+fmuDWHf8B57YMDDqY3q+\/jOzLaOiMRSiFF\/RnjKr021owXPh\/9nJz\/+PuhBZFF0jyyjcLKHVKUMVtSUDxF3IMKILKVSTTtX1Gn0N1+JmuFYqjMBzazHo+zUwuQiJWlhFGNMe6L9Ga3T533nlnu\/miO3GyEmZQKU59ElHcuMcZ7oVJ+yerytP\/\/e\/c0rd\/Y+sro5QPtExov\/2qsy9Df7T4KmMOLRj+\/Itr3ROPrUrevegCSV75ZgGlbimqsD0JaDeZrkkAwZWWCVv3\/UzXfg7EvOfPPPNMd9FFF+WmSNf2O59sACDxh7gYehFzWKfPkBRet6\/dza06+t1h7lFaxC0b3eRr\/npQmaKryX4DTDja+WTvYToDLvgWMZ8Je6oq7tOCIWefsAuliqvxGeZQtQ6fYVkDa023bjJdywTJMu\/8TNd2wQXTOiuTNvWHEjWwkqltYAgbe19DgogfEcEjNlN7w2325bLomVVP3WA4b948t+Kc77W69JImGOqfv8BSFAwXL148aAyYaNLGgHtMSFUnKbFgqIw1VYxxA4ZVULVHdfZSM3zjwVPdZ6c9m7w5pvLmSbu3z2CRpsl\/aZYhzTBNEBE23y8IEBKPWHX6qLSh7AUYrnzrV9sB2PQrOUPl\/B+3jlBpXwNul+suTI4i5SoKhkxIXEw4LMyg8QGGdgcU9\/EhEpjN\/6pjPRswjAOUnm3Hi+tevaV6qRmefsIr3afGPdQGwwNfcVw73Zc0Tf5L20zTDNEGWfUjCYU0Q18QEVJ2BQGSVaeP6icw5L21PW\/nH3\/QjVnXArxV7\/k350a1VvN1SUPsBAzTxoBxQHPXuTOMEWNQVbp\/vUsDhnE4UisYFjGZY8Nh4l4zrpR16neT6ZrWYh37mG8I6aF77uKuPvzPbTB8Yfykdrov1cd\/pfUKaYZoJWmCSBu8nxJVQN+qQjliqN0LzTAvn6Hdxzxh\/s\/cDvd8pyPNMG8MoA+ASMA7mjmfq7zaYDjg3DUfWVnaofF+qE0TWlNwFBX\/5\/tJ9DtCSkR+zFEBBZvOLW5j37rJdE1DMfFvM2bMaGdgpryCrzlI6orHxgxKRuozXkhjyQJD+iQwxDRDQ6ljX2y\/aYZZTLD61EvdJnPaHtphmZohWyUBZEJpGAvGoY4EuwLDLZtGud9\/f1pph8b7oTZNaE0uxGwtEEoiYB\/Hh0IZfCiAoz4LCAo01VHRXprJMNbF4x5Kdq0oM3bWAkzITGZvMqYYiyH4BNPMZMI5mGz4j5ncyzhDNFylv+9o0HIeKurUt9rhTnMuda\/ee3wwIUZas9ofzsIU+4+1gOK7Kngefiespuqgd4Ehq8i\/+OK60g6N90NtmtCaAhwsMxkihjL82tRbMIn2F9e1VayXZjI0+a\/TN7tTJrcy15x6\/4TCZnLaAgrOfP508ZmsOdAbbUWZVQoMZddF+9FM5qVaWW52a\/sQ\/+q3lxcCw6zVZDsGyhrEGDB5VXkolwVDQmuquopOPH4\/imrhZb9HrT5DaYZoeqGN6b5myGon+d7qBsOqtRUGMcQ4B+84yl1z7OCdKIo\/1LY+MUCIcdAGFVpDOeiGsNnf+F1JbSnPmFR5TOVQMpPV15Z22Ao+PPmxb7uvf+nS3HhOPevHdQKA0BmT2F78jkJAnCEaZJXHtTZgGAebtYIhXYrxGWJS+Png4l6nu1J1aStpYMjvdieKPVRKb6btfVmzKKEcJAVA2NjmhTCiafsZrvkd7ZsA76rO7B2KYGh9h6+ZtNp995zDo8FQ7wufsx9cSTAIieI7tLZnVBP0DhhqrLrj4PDTAsMVC9a72V9OPyah27YbzbADCoZWlW1SBs7yYB9nnVohr9FrMLTHBaAJKnnDt5Zscu87oHXWsjTELDC07ga0QwCPpA34EP1EoqwuV72aGWKRumjdiYBuPYq0pR0u+LtpHYGhdfMAhCRtoD91hzM1YBgHUrVrhnHd6k0pCeh73vOexKmvVPk609amzuc3\/zu9ts9k3ccsYkueQmvUxg2vbKWE13X3qi3uow9udH8xdZL7xmGt81KuXrrJzdv1qG18WTp\/w565wSSj74CeD3z4DJl0RsIOlLzQGkt3u5BSBAxjzqHxuRvaMwZVJWtowDAOTxowNHQSGD7yyCNJ0tUrr7wyufuhD30o+a\/v+s3\/bsvwTNb9j3\/84+6uu+5qg6Ha+JtHvjVo5GQWc\/8NS653U194Mrn\/wNlfcy9\/+csHaSxo1FzKVGOFC78h2rb9jXL4snqR\/r+fNUNo+OLBb3Drjv\/7hJ7\/+tbd3TuPP7R9Zk6WaGmHEP5uwBf3g07HU5Ygwmk0+SijEGZyVckaGjDsYzBU2nlMNF29PpODfkhA7777bnfbbbe56667LuneWWedlfzXd\/3mf7dleCbrPqmhWHmUZqg2\/vqJn7bNY+oTGOr+x5+5vgV4r3yLG31mOFGDbyZTHv8gznp7Vq\/Goa7zey1L9jsY2uMCLjtxV3fRm4+IAkO9Y2gMRG9MZbkr+I2yVW4yEBiu+oNzJHeVhpyXn7Do\/SboOg5026XSFlDKOC2vYFe2KS4B7UUKL6UJe9WyX7u3TB2T9M2uIOv+f3rqpvbh8+4Lc4O+LH8BRUAP7fEhKqwJTQXtsK7ze4cSGNJXmcqv2nu8u+0jryoEhiyKQF+AT2fooC0CfnavONo6ikCVbgqB4XOLJro\/3j65CbpOAYpazeQioTV1BVqHBLQXKbykKUx98Wn3jWPGJd2yR4zqPmZY+xjSFDC074TprGQA0NTu9AnFenY7ocQ+3++aIe\/x\/NuucJsnT3f77DTGLfnUcYXAkOfz+D2WVt2Wg9ZfufHt7pG7Vri7f9gEXafRs1YwLBJ0XVdsYQgMdToeCV65Tj755OS\/vus3\/7stwzNZ90877bRklVdmsm0DsFs7bkf3xhtap+b5fWiD4UU\/cgcc89pUIUXbDsVqylwjp2GVBxFlCfFQAMMXDzrJrTvhQteJZihtnJ0+aZsMugW52OelGVZ5gDx96WTl3r5DE3RtqGGDrnupGfbSTI7JlP3u\/ca4C1821rkMMNTEw4JJXoB7L2lddYB7twKKqXzGjB3ctee\/orBmCGuHfIexIFZWuQYM4yhZq2Yo5pDfxAppP\/kMe2kmx2bKvvPM\/Z274GupmuFQ0cL7HQyf\/Ztr3IWv2dN95R1HFQbDUDytL5axh57FiXO4VAOGcdSrHQzpVr+vJqOd1Z3puuhBR6FEDXbI85Ji+L7EOHYpr9RQMJN5216bbmVQvAHDOCr2BAzjulZ\/KQnoUNAMY85Aidn62KtFlAYM6+NvLaDcP3epu\/k7W\/3QZfegW5dEryeeBgwNR0hAh4NmqNfK2\/pYtkDE1jeSwLDXllCjGcZxZQOGATDk+Mq6M10XPSIzRjOMY4HelBopYNgPcbUNGMbxeOVgGONEtl2tw6Gs9sjkws4M9gkTGCsBPeecc5K9yWmR+jyfl8k67z7R+n6ma+q1bdq9tDGZruOGvJ5SgABjyfke0Jh92uPGjUuSFBBgPBLAMC\/OsIroCXYZscUPnrbWwWd\/cbxrQmuyeb9yMKxH9Iq3QiAyWUTYswvjADbD0UwuTpnunoCe0FW7L9jdwnGY0FhHkxJAPhLAsO4VfaUJI350n332GQSG37zlfe7mG+5xt3y3dYB8FVfjM6yCqjXUiebCVjS2Q7GIYMFwqJrJNraQdPME\/NoMNj5Zq9DCAUM0IrajaZsZGjhhVPjOAEkytIwEMKxbM4SmtCkaN5phMSAZFpohTIBZCwAo84d8NZCD3Swk2dRh3gipzrRFW\/HBsOrYN\/o0VGZRJSkFzGT6Y94BegJTzF5205B4A1qTqYXnuLgn4RxpYMj7l+EzFL11PINA1o6BdmxZGlswpI6q+Xqo8HQaRA55MNRKHS+oVFQWHPkdPwpgCAgisGgqlOEzDEUuwTpNtzrAMDRBWCaI2RmhQHhSUAkMrWASq8g9TDIAjwu\/IH8CQ4SUCYfDgnQEASbzSNAMRe9uVpOhMeDHRCQwDI2BNjA0YFhMG7SlhzQYIlAIGJoITICzHs3QCjrCCvOETieDySjra4YXXnhhpSe2MQAw9xFHHOEefvjhQUeCxg4l4D5r1qzUDMx5YJi2b1nt63RCvuuUQj7znM6w8bVGH2z5Dm2feOKJZNLBZOc7fkSBYdW0rprOsePVSTnRGlpxAYa+6e2PQRYYNrTOHoUhDYZ25kX7sGBImnXNllaAs8iB\/1DO\/06Yt+5ncJRzqpq9pEnE9MU\/uzr0jF3xFBhqYonRLtP6MZRoHaJzDH3LKqOkvRYMi4wBtL733nsTy6nfLxbbjjvuuJ50sxYwxJnOjEYSUWX9TdMqfP9IDFWYDcsAQ9qCcfgbCpdM0lBf8zTD2PerCgyHEq2z6JxFx7wxiJ1MugXDkUDrWH7OKlcLGNqTwljEwEdHaimdFsZ9Fj8Ay7LAUIfxZJnJZRBwuNcRAsMYM3m40yXm\/fLAMM9VoTZCYNiMQcwIFCtTKRgCeFwE3OrYRJ3SRpwfv6G6oxprJbgMMExbQOlFjsRiw9F\/pf3A4Cznff\/1vjc9KttVYcFQrgp8sJjNWsTqVV7K3lC4mlYrBUP8b9p5wODh9K8DDCGVDa2p8nyJaoals1qriDP0wdCGddhV5s56PLyfytMMY9\/eB8NmDGIpV6xcpWBI6ArgByASk5YFhsQoobkRBqNZr9irdF\/aj03sZrb1Ndy0uov+3v1b9kcNI4nWvTx6oUw6+1qpr3TYBbmhyNeVgqHEzvoMQ5qhdoOovI2pqkt0yzSt\/diwtLp5NwWL81nxkGm\/FzHz83Y\/1EXXUDvDjdZZtOzl0Qtl0pl37Ae+rpJvawNDtEP8g6z6slDi+wx7rRnGxibmDUYoNiytbuoKLfSk\/V4k92Devti896jy\/nCjdRqtrNuiF0cvlEVnaYTassp3\/JW94Osq+bI2MCT8xWp+hCuwa6GMBZQyCGRDfezgd2oqWz9PWt0wVygeMu33on2JDd0og35F6hiOtE7TgLMOhKp6fMqms7RDC4ZF+Lcsvi7Ca0XK1gKGmMbEGrLljcwa+BLZjYAfkeBRtMZuVpOLvHBa2bIZp9dgGJM6rYpEDTFjMdxonfbOvT56oWw6N2AYw90pZcirxoAAhGzYtwso3MNE5EIo2a\/KtrpOQmu66GL70TJNihDTFDGHyzCTy6BJVXWMJFprIcHf7aPfq4x0KJvOw52vK9UMGQy0PtRjYg5Dq8kwAwsHJFEAEDGne7GaXIWzWeZELxZQqgKyMuodabQOael1aOVl09kHw+HG15WCIcTDJCaoOi3oGj8YZfCtcLGQAjBiNtd9lRmb6MeGpdVd9PcYmqSZZ\/1weLz6P1xoHTMeWWVwIbEjq0ikQJE2y6SzD4Z8L8q\/ZfenCC3yylYOhiKYwNDfjqfUUJjShJmgRZLuqbk6o4C0ECYU0dbWlHe\/s1abp4pQwAJCHRpikb6N5LK1gGFsogaYBLDEXGa2bK7iFIg5Dzkv5KN4q80TeRQI5TSMyRqUV29zvzwKVA6GLJSwLU9JQNX1xx9\/vH1wDaYCCywkCSX+cMKECcE35N6yZcsSnyJaD+VsUCtASuJWTG1WrnUBxhz4xJ8u4h0x3wFd0tCnXbRJG8zgLPDg38SEV4p1+kG7LJAA4qyO9+riPckdCK0POuigdjdCtKaM8hTqsCnb74bW6aMYy9PwDFET8Ku9LAg2dO6VtGzbbuVgSJMIKCBFUk8AT5oJAAhwAUxaUaY8iy2AFLGI9kJzBIwsgFkwBOxoC2YVIAJSgC1mI4s1yunGc4pz1FEBacMis4Z2eQ9AkYUh8icCjLRJn3m3NCAP1Y22oF0nABPgjP80dMWYU0qYyvMhWtNf3oW6oBXvAj14H5\/WTGCAPGMkYC1Ka+jFM9AeQFAm8iz2Hyq0zuJpxtCmgdOpgAAfoWXwiGJGxdPQSCBZlM7QFf5ZunRpQmuC84cLT5cJlXm7sioHQx1diKDCDHIU68AgBk8xh3pxviPMNnEpYAMDwmjSCikP0OGPhAG0swVmkqZJ\/Qg1QEO9CCgAphAe6gAc7EUbMCwaI\/2n7wgzdQG0MDefpS3q\/A\/ftEdLTANHDYySdMqM4h2K7DSx\/dYkI0AF\/Ogv9PEvS3M\/vANaQx+Nl3y4ndCacVC0wHCiNbTBxw2tsuhs6a7JVGAontZYKC1XJ3SGp7EAGHv+lDdU7Q9Vnh5WYKj9jAAHA4JwKAU8QACwAFRoKpypy0zJrEoYDoCoS+dxhOK1ADurdQAKtAVz0R6XTBsYEuajTdr2mYZ7lOXZ6dOnJ\/1R2nV+F9DQXwCRLMiAjX8KHfeyNCE\/G0wZYMh7Qm\/6CMDxDjLxcRvwuzRDnf8C8EJTqxmK1pRhDKwW0wmtaRM6WzcFfR3qtIYfsC58OjPR2wkoTTMUnQFHeN7mKOyEzvC0JnZL66FO57IAseeaIcKJsCF0DD4MxIAx+IAFnzHJYATKwURK3y8iYFoAFjIlqIvnYUT+lDfRJxrgirlsL4AYv6OOCOUeZjqghiYoswUwkd9RWgBtARDSCukHv9FnmaWECPEbYJ4WLhEalCJgqLAd+q6D5pVOC+BGKNAw5dfU+0voBESU4XmBPb8zeUAbHZ4FXRgzBMrX4K3WwWfRBTry7qI1z3PRHu6F4UJr+VoZf3hUE6J4hPuMi6WvaC\/lAJoxLrI4+K9T73x6q17RXa4T2hFPUxf8xz3GYDjwtPVpawIRDUKLUGllrNzxWb5cyU7lZvKiRYvaAELjzJIMGP4q\/jNofNZAM6PxgnQQ4UGwmSXFPNyDYQAoe8FUEI3naQOB47M0Q5XFZAPwKIsZDiPTBmAG4wA0gIEEGhOY8jKx6StAKQanH5SnLeqDKSnLZ98Ppz74xxTwe1EwtCem+UdHylTS0Z064lMaM7SURqz3AtiVf1LahVwY\/LcX9UJfLvl3eR5a29VsaAXtoAPtKPh+qNNakxF0gjb8wQfQhHfUJM19eJiLScBfSBHYCcB8OnMfmuLTpX7qxZpikpcVwO9YO\/APbaNgYL4zRjw31Hma91SIGHRH9mVxheJmffm1kRNSyBgHuYZsHZWD4eLFi5OBk1lmhdAKGAPPPTEEn7VjRQsiABMMxyDDJAgZL4Y2Zp3SvoZoNRjqABDRPtEa5bDnM0wFIwnYBFKAMVoioAdTw+CUxbfDPc3sMvP5zmJK2hVK6xQKvbDP27Rm\/oCrnzJHoYnMeV\/AVKc0Px30HqI1YwKNeXfRmr5TNy4EJhLfnAvRGtpSjou+DXVaM+EhZAJCgZpobTVETSya7JkkmXSkLUMT+A1Qg+cBL\/E0z\/A8YJBGa8pYnob3BNZDnc5WQUDeZfFYn7qVJWiZlRgjbTOCJvDKwVBmMh1BU5BvENABVHSGsRiJ3zTz8XIwHcwFEEEMwNBmb5ETGmLpXNkQCMFsLMAg1DaUBkCDmegXAi7gVn94DsalfWk5GgwYEf8QfYLB+c\/snbffNBQLWFQzpJ\/2fa0JIM0ZWtoQI5uOHg3bX033aa2FLNEbWlMHAsqiioSWcn64iE9rvTPPDBdaWz+x9Q\/DC4CXfK5MZHbS8OmsiUY+Q7uwopC0EK3TeBoFZDjRGbr4NLRWlkCS39ImZ3GbtswAAAU2SURBVO6l+Qz7Agwxl9C0pBXyHwZCgP3QgKytZNKU0MZkvvmAKC3OhuVYnxYCDgj7K8JZGpvVINEWACFptFnbq+oCQxtiAWhry6Pi32SC+H3VO\/tgSPmFCxcmgoYmieZhTXTRHLrwu6U1YR\/ShoYLrQWGTJxyq2gnlSZV\/iPIAJfvRxaduW8XUGQCA4qaeOTqsFEQ0v59npY1NlzoLDBUshPLr1aJgG6hMuLLvgBDOfnplFbWEEwNJr8DRDoAm0FES7MrYllgSF2KmROIWmezGFRb1BTuwiyi1VUEPG1HgEJNZIrLjwgYSEuUNhBzxm4VYGh9I75mCG3on8KDoBHvgADyTtKI\/dk2DQy1Wm01ZGjLODMpIPgSYtFa2gqTxXChteVfNEP4FTqLboyJFguhrT\/JZ4EhPK3wM7RwAMHyNM8ywXPh8tFYwNMs5ugaLjydpRnqmGD4rq81Q4BEoSwIKQMHkyCgXPj\/MNcAQ0xdmIByfpxhFhjqHoKHTwahxAyXz0RB0cRiiWnoF20BCOoTz4XCYaRVIsg4sMWktEWfZa7wvMzw0K4OMWiWzzAmzjDNZyimgCF4L2iLMKJ9c2kRCi2GMaA8l10553uWZihQkw\/TpzVgyG9MCjLjMRnl1Ie+WuUe6rSGH7SQx7gr0Bq+g4fhMfgacOKepYmlc0gzZAyYqOWfpXyIp7WZAL6Wea0FPXhzuPA075\/nM9TGhTQZ6rlmyEsAfACKVtw0SIpx0xnKCC2\/acHCmrxZYAjjAXjUz5Y4hF6MAxCjBWlRAwamvAVdzGtr3oT2RtNHmJr\/MCiCDIjyWbsRaIP7CrNJW7yoajUZutIv0VqxloyB4iGtz1WmKxqLwmwAai6F0tg4Q35fsmRJO\/QGrUOr6jwvWlMOwAWMobUCuOnfgQceOCxorYlFYAiteT94h3e2\/m3KQGvRBbohmAqS9uMMKUcdTHrQVEAn\/zQaKDytxRlobRUJTXCMNXLANZR5WpNqJ6vJdtEEGuhYDrvuUJvPUMkX5ExWeAeDpz8Gi\/toiFz8DpCgycmHl5d+SvGBgB3Pw0zMyjAEjIlGh3DaUBntMlH4jdqgD4qNhPEAaVagATqFy9BfwJD6IDh14xOSyU7dgExoB0pWnKE1vdq2zksftDCjwbMmmOI0mUignd7Vmlb0xU4wmKsImOLaaIZ6oJV27PhgqEUY6tXqv+ISeZ62eR7Gs7TmHvTxx3Mo0trGCEJnaAjgafJTmJdoDX3hRxu0z3tDP4Uc+T5D7hOWRr2SE+tukqavWF2Ff6lOJnh4eKjztL8o6scQhhYr\/TLWiukpGAIOgAmCiDCgXflBqFbotZiizNgCkzwwpA61BTNqccPuMJH2p9g6+uNrbwqgVUAr32mbeiA8AMgFMKKBweA29guQUOB1aEFG7+rvQPGBL+t7WmYaAJv+AkTQjb5oBw1CJZ+hJh\/RRos+VdGa+kOhT0OV1v1KZyYaQFVBxOKhoUrnIjJRRtlKQ2vkwwCZmakAnqy0\/ggvjEZ5Zlx7yZTmnr+rxC+HholG42\/\/EogpFCSNgIqJBJD5zIwc0vC02hdqB+Dxt\/rZ9vy9yUUGMytNl9UEi9RZFa3RDpmE0nbkDFVa9xud4Wl4Sok\/\/AiBoUrnTnm4k+cqBUOBjzWHe3XGSSfEqfoZNM405u1EM6y6v0Xr7wbwi7aVV76hdR6Fyrk\/lOlcORj6JEZAmKXSsrmUMyRNLQ0FGgo0FChGgdrAEF+GVpOzuojpyZ\/2dBZ7naZ0Q4GGAg0FOqPA\/wclSFiJNJP3jwAAAABJRU5ErkJggg==","height":129,"width":323}}
%---
%[output:1a8a29c3]
%   data: {"dataType":"text","outputData":{"text":"\n--- 自动微分梯度验证 ---\n1. Critic梯度验证:\n   Critic梯度最大绝对误差: 2.946257e-01\n\n2. Actor梯度验证:\n   Actor梯度最大绝对误差: 2.873561e-10\n","truncated":false}}
%---
