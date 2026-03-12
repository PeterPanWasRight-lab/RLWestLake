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
        fprintf('Epoch %d/%d: 误差=%.4f, Critic损失=%.4f, Actor损失=%.4f\n', ... %[output:6b768607]
                epoch, num_epochs, error_history(epoch), critic_loss_history(epoch), actor_loss_history(epoch)); %[output:6b768607]
    end
end %[output:group:174df7de]
training_time = toc;
fprintf('训练时间: %.2f秒\n', training_time); %[output:381047b1]

%% 4. 结果输出与可视化
fprintf('\n--- DPG (Model-Free) 收敛结果 (高速自动微分版本) ---\n'); %[output:7364fa34]
disp('最终学习得到的增益 K_actor:'); %[output:29d833e7]
disp(K_actor); %[output:91b2df5f]
fprintf('与理论最优 K_lqr 的最终误差: %e\n\n', error_history(end)); %[output:3833c950]

% 绘制收敛曲线
figure('Name', 'Model-Free DPG 收敛曲线 (高速自动微分版本)', 'Color', 'w', 'Position', [100, 100, 1000, 400]); %[output:931c534e]

subplot(1, 3, 1); %[output:931c534e]
semilogy(1:num_epochs, error_history, 'LineWidth', 2, 'Color', [0.8500, 0.3250, 0.0980]); %[output:931c534e]
xlabel('迭代次数 (Epochs)', 'FontSize', 12); %[output:931c534e]
ylabel('对数误差 log||K_{Actor} - K_{LQR}||_F', 'FontSize', 12); %[output:931c534e]
title('DPG收敛过程', 'FontSize', 14); %[output:931c534e]
grid on; %[output:931c534e]

subplot(1, 3, 2); %[output:931c534e]
semilogy(1:num_epochs, critic_loss_history, 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]); %[output:931c534e]
xlabel('迭代次数 (Epochs)', 'FontSize', 12); %[output:931c534e]
ylabel('Critic损失 (对数)', 'FontSize', 12); %[output:931c534e]
title('Critic损失收敛', 'FontSize', 14); %[output:931c534e]
grid on; %[output:931c534e]

subplot(1, 3, 3); %[output:931c534e]
semilogy(1:num_epochs, actor_loss_history, 'LineWidth', 2, 'Color', [0.4660, 0.6740, 0.1880]); %[output:931c534e]
xlabel('迭代次数 (Epochs)', 'FontSize', 12); %[output:931c534e]
ylabel('Actor损失 (对数)', 'FontSize', 12); %[output:931c534e]
title('Actor损失收敛', 'FontSize', 14); %[output:931c534e]
grid on; %[output:931c534e]

% 保存收敛曲线图
saveas(gcf, 'dpg_convergence_fast_autodiff.png'); %[output:931c534e]

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
    % H_val = H;  % 必须要去除梯度的传播，否则结果错误  半梯度法的来源（和赵的书上写的P127-129)
    % 本质上又是一种快慢系统辨识；critia-actor构成另外一对快慢控制
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
    % K -> U_policy -> Z -> Q_vals -> actor_loss
    % 向量化计算：一次性处理整个批次
    % H_critic = extractdata(H_critic);  % 去除标签可以防止一些意外的bug，但是这里无所谓

    % 移除K的维度标签以避免冲突 这个例子里也无所谓
    K_unformatted = stripdims(K);
    
    % 计算策略动作（向量化）：U = -K * X
    U_policy = -K_unformatted * X_batch_dl;  % 1 x N
    
    % 构建状态-动作矩阵 Z: 3 x N
    Z = [X_batch_dl; U_policy];
    
    %% 关键：用神经网络前向传播
    %% Q_vals = forward(net, Z);  % 1 x N
    % % % 梯度能正常反向传播，因为：
    % % % K是 dlarray，支持自动微分
    % % % net是 dlnetwork，其内部计算会记录计算图
    % % % 从 actor_loss到 K的路径是完整的
    % % % 计算Q值（向量化）：Q = z_i' * H * z_i
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
run_gradient_validation; %[output:1c602238]
%%
% 本示例展示如何计算神经网络输出相对于其输入的梯度（自动微分）
% 1. 创建一个简单的示例网络
% 2. 将输入数据包装为 dlarray
% 3. 使用 dlfeval 和 dlgradient 计算梯度
% 4. 运行测试

%% 清理工作区与命令窗口
clear; clc;

%% 1. 创建一个简单的全连接神经网络作为示例
% 这里使用深度学习工具箱创建一个简单的网络
% 如果您的实际网络结构不同，可以替换此部分
layers = [
    imageInputLayer([10 1 1], 'Name', 'input', 'Normalization', 'none') % 假设输入大小为 10x1x1
    fullyConnectedLayer(5, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(1, 'Name', 'output') % 单输出，作为 loss
];
% 转换为 dlnetwork 对象以支持自动微分
net = dlnetwork(layers);
fprintf('示例网络创建完成。\n'); %[output:2b3c4a33]

%% 2. 定义一个函数来计算网络输出相对于输入的梯度
% 这个函数将被 dlfeval 调用
function [loss, gradient] = modelGradients(net, K)
    % 前向传播：计算网络输出，即 loss = net(K)
    loss = forward(net, K);
    % 计算梯度：loss 相对于输入 K 的梯度
    gradient = dlgradient(loss, K);
end

%% 3. 准备测试数据并计算梯度
% 创建一个 dlarray 作为输入 K，并启用梯度跟踪
K = dlarray(randn(10, 1, 1, 'single'), 'SSCB'); % 'SSCB' 对应空间-空间-通道-批次维度
% 将 K 设置为需要梯度
K = dlarray(K, 'SSCB'); % dlarray 自动跟踪用于 dlgradient 的变量

% 使用 dlfeval 调用梯度计算函数
% dlfeval 会自动设置梯度计算所需的环境
[loss_value, dL_dK] = dlfeval(@modelGradients, net, K);

%% 4. 提取并显示结果
% 从 dlarray 中提取数据
loss_value_extracted = extractdata(loss_value);
dL_dK_extracted = extractdata(dL_dK);

fprintf('网络输出 (loss) 的值: %.4f\n', loss_value_extracted); %[output:267886a1]
fprintf('梯度 d(loss)/dK 的大小: %s\n', mat2str(size(dL_dK_extracted))); %[output:49a6034d]
fprintf('梯度 d(loss)/dK 的前几个元素:\n'); %[output:951ca7ed]
disp(dL_dK_extracted(1:min(5, numel(dL_dK_extracted)))); %[output:8087e997]

%% 关键点说明
% 1. **dlarray**: 必须将输入数据包装为 dlarray 并指定维度顺序（如 'SSCB'），
%    这样才能启用自动微分跟踪。
% 2. **dlgradient**: 用于计算标量输出（此处为 loss）相对于一个或多个输入（此处为 K）的梯度。
%    要求输出必须是标量。如果您的 net 输出多维，需先汇总为标量（例如使用 sum 或 mean）。
% 3. **dlfeval**: 推荐使用 dlfeval 来调用包含 dlgradient 的函数，它会正确处理计算图。
% 4. **适用性**: 此方法适用于任何由 dlnetwork 表示的网络，无论其层结构多复杂。
% 5. **扩展**: 若要计算 loss 相对于网络内部参数的梯度，只需在 dlgradient 中列出这些参数即可，
%    例如: gradient = dlgradient(loss, [net.Learnables; K]);


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
%[output:6b768607]
%   data: {"dataType":"text","outputData":{"text":"Epoch 100\/1600: 误差=0.4606, Critic损失=1.1236, Actor损失=96.0670\nEpoch 200\/1600: 误差=0.4371, Critic损失=0.5775, Actor损失=152.7641\nEpoch 300\/1600: 误差=0.3843, Critic损失=0.1731, Actor损失=151.2455\nEpoch 400\/1600: 误差=0.3153, Critic损失=0.0694, Actor损失=177.9980\nEpoch 500\/1600: 误差=0.2157, Critic损失=0.0166, Actor损失=216.1066\nEpoch 600\/1600: 误差=0.1005, Critic损失=0.0004, Actor损失=196.2683\nEpoch 700\/1600: 误差=0.0101, Critic损失=0.0028, Actor损失=213.0055\nEpoch 800\/1600: 误差=0.0341, Critic损失=0.0015, Actor损失=171.8101\nEpoch 900\/1600: 误差=0.0342, Critic损失=0.0005, Actor损失=191.6348\nEpoch 1000\/1600: 误差=0.0208, Critic损失=0.0001, Actor损失=197.3962\nEpoch 1100\/1600: 误差=0.0089, Critic损失=0.0000, Actor损失=173.0756\nEpoch 1200\/1600: 误差=0.0020, Critic损失=0.0000, Actor损失=202.9423\nEpoch 1300\/1600: 误差=0.0011, Critic损失=0.0000, Actor损失=204.2877\nEpoch 1400\/1600: 误差=0.0014, Critic损失=0.0000, Actor损失=185.3215\nEpoch 1500\/1600: 误差=0.0010, Critic损失=0.0000, Actor损失=201.5249\nEpoch 1600\/1600: 误差=0.0004, Critic损失=0.0000, Actor损失=175.9688\n","truncated":false}}
%---
%[output:381047b1]
%   data: {"dataType":"text","outputData":{"text":"训练时间: 5.53秒\n","truncated":false}}
%---
%[output:7364fa34]
%   data: {"dataType":"text","outputData":{"text":"\n--- DPG (Model-Free) 收敛结果 (高速自动微分版本) ---\n","truncated":false}}
%---
%[output:29d833e7]
%   data: {"dataType":"text","outputData":{"text":"最终学习得到的增益 K_actor:\n","truncated":false}}
%---
%[output:91b2df5f]
%   data: {"dataType":"text","outputData":{"text":"    0.4175    0.2909\n\n","truncated":false}}
%---
%[output:3833c950]
%   data: {"dataType":"text","outputData":{"text":"与理论最优 K_lqr 的最终误差: 4.349102e-04\n\n","truncated":false}}
%---
%[output:931c534e]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbQAAACvCAYAAACVUiuqAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnQ20ZlV53\/edYZjhYwYYRAaBEcSvCNQwmlZBI9SIZom1JVZFE5RUTZDYNjaupmlTMKHSrMQmNZKYmGRFVyzWSNJUrHRMgga\/YnFAAygf8jGADAIzI8MgM3eY6fqdO\/97n3ffvc\/Z5+M9737vnLPWXffe991nn3328+zn\/3ztZ8\/s27dvn1ti17Zt29zWrVvdcccd5w499NAl9nbD6wwzMMzAMAPDDIRmYGYAtIExhhkYZmCYgWEGlsIMDIC2FKg4vMMwA8MMDDMwzIAbAG0KmODJJ590jz32WDHSNWvWuFWrVhV\/8xnu1YMPPtgdddRR85\/XeaXvf\/\/7Dq8z99OPf\/GMRx55pPjumGOOcStXrqzTfVJb3u\/hhx92e\/fudUcccYQ78sgjk+5j3I8\/\/rg75JBD3EEHHZR0T66Ntm\/f7n7wgx8U7\/H0pz\/drVixovOhQstly5a5ww8\/fL5vPZfPoa94q+nDlzqvNp0X3Tc7O+tYc3v27KnF63We+9RTT7kdO3YU68Ku1y1btrhdu3YVn61bt65Ol8G2rL0nnnjCzczMFO8i+UG4h+\/4f+3atUG5UvZweOjRRx91q1evLuRd6Lr\/\/vsdc8n3Rx999HyTAdBak3X8HQBagArX0572tAJ8uAABvuOCQWPEj40QZvze975XAAl9ItD8S89A4B1\/\/PHFIoldjBGhCSOG+mozDt3LguQZjH337t0FGB922GHFvPAuLOaqa\/ny5cW7hAC86t5xfa95ZmwnnHDCvCDiPR988MHiPasuhEps3pmX++67r5gz3vvEE090PMvS9xnPeEbrmPO08GrVXI7re\/gXYQw9\/DUH\/0LvqquKf1kfgBcXPHHssccWf997773zgPbMZz6z6jGV38d4h\/cA0FJkRughuh+lLrRO4WHmUEqB3o++pg7QrBAum3Ems4sFWknVjhugdfCD1iPwiAkJEV4MXle7him0gHgWjOEL+QceeMDt3Lmz+Pykk06Kvi3MRVsWLGPHykoFtSpgfeihh4oxIAR8wQ6d0TjRBFnEqYBmQaNjEjbqrgzQpHTQMe9rL5QRXTGlhO+tkMNCY21wtQG0aeXVRgTq6KYyQNN6hMdZ0\/7F5\/qujH9ZL1j7XKxpQK0toDFu1hb8JzkT45177rmnUJxYl3WBE8sSBU68jvUlBV7zgSyQkmcV\/KkEtBBfSeDHtFgELK4cNHmELP\/7V1UfmmCIxCSXZU8i3OkPLYW\/Na6UcYQEWwzQpHHBYOvXr6+15HAL8Cw7F2hEZIaKYRFYaPW8A0Iw5AoE6OTu80HNZ7bYAKsADfeDLFQWuuZTz2Xx2PegDQDnu0dZ5CwY3wrSuDQO\/rfKEP0zV3K7qr0EELyAgGlzVQEa7xyaz6q505iskLPWfBtA64NXcT8xzwJu5gFat+VVzQv9w1sx\/m5D09C9KYAWAwLxW4x\/eV7MEm8LaHq2HVuId5hPFFvGYRWnlHlEfiBvkDuiM7+RO9YKs3yHPLfhhqwsNCYD9LU+0ZSJsIu1qr2\/ENQ+BdBs3wgXhKZ\/sTjoS9oUz5MQhsiyICACRAJk7ZUqJKwgq1qMWF9W4AO00nAALxiG50oDA9QYl9XqQ\/NKn76QhYZYFDAn7807AiT8X3ZxnwSVrBDF05hLBAFj5ffmzZsLAUd7vmOOuUfWYaGp7ffrE4\/SFQONKkCTxulbQG0BjTkBLLkEtvAFi5S54weaMJ9tAM0KUV\/7txaelLYQnQBtWXVl8xlSvtrwqjwEPJOx4s6GV7rgVbmu6CtV+aqSL7HvGTPj51nwIbTnXWQ9sUZ5V+aqDaBZOcYasOueZ2vNhCxArRvkr6+8pgIaMVnFw5GPvnzT\/FgPlD4TrfmOMfzwhz8s8ACeZa7Ef4Ae3wVlUk5p+xCTlyIWZBG5jIlsHImJwOfqW0\/0i0XC5MhKQqO27jUxAt+HXJVMKm2wFiREff+uZSaYknfw3YCMAQCGoRHEPMtaEqmAZt+7bH581yvPl2uO75gHxmg\/Zw4ANQAN8LNC0ApA7kcQ+MyPFcQ7asFKY6srDEIuNALq0JJnI\/xFRwltxsdiRcvjYlHJ9VkFaLHxyRIuc+nVfTfaVylR8DHjl8tRwO0\/S8pAVRy0yRh1T8gK7YNXEaTwn5Qc1go8Jx5uw6uWZ8YNaOKhEA20RuFrZBX\/h5KC4Gnxd8zlWPacFPqHlFTuSwU0hUGqnuVbmSjYyBuej\/xX0or9HHCEx\/lMhoH4QmCdlYUmQINovBRCNXahXfFiMKWuqrgZwhAh4k+aFS4xQNMz5KqTMJd\/V8F7JpaJB+zKtDW5USzxaJ8qJMS4IX+7iEt\/MI42mAM0zIG+9xexQE3BVrQgFpF9HwlhgWEs65FnCFiU1VXG5GUWmvz3jAULWLE05k4aLuMRcEIT7oEmgB5KA6A97YBWJSRCgGZdQMwFwGTjcHyvucfSj2nu0FlzrXGMm1cBLbmXGTfj5EJR5B34vg2v5gpoVXSOuRyRTVK44XdfmWb9a12ibMYum0mtNimABn+xBpUDYK1D3xOghA9+A4LIHi74DNlp3YgCNQwQ+ABvBvSXJ4n7JA+zAjQGBkFEFGtm2snHrEWwye3EJMjdVJUIIsL4GTRVFpqeb4HLChCboJGScACRsH4AC+vKSRESEBsLiPf3wdMqBQCW3Ld+Mo3vjuD9BPTSAlkgYjIFd+XebRLwjS2gqjiQDZbLZWIXi1LOBXY8x4+n2vZ8x6JlEfGu3I8lhACAjlzQBJ6iT78vWStlLkeEL0BOH9b9DCiEXOpNwbZK+MmNwxisteqDUpUyGHrOuHnVusgYn+jAb9YANIzxKjIE\/rW0g764oK1y47+XXdPcyzsqFq62th97P0IVRRsaw1uKBbFGFWMti6FV0bJKIbSu\/lCVpDZZjimAhgLMepG733rZ5IpkTm0M11p0ipv5iU\/IGsAMtyx0FWBKIZfSVuBBTi5HEQwE1v4oG1jkhQECNA25Dpk0mEgTWQVoSjDgfjuxdQANIcFYxPyWSesGQn0mrRISCEMAQNmJLGxSsOWisO9hNRiew7wyTygKmPe+1iR3g\/z51qxXyr4EZJUV6r8Xz+X5CFU\/a8kCGsKA8TGnMC7vxVj52wo4H9AQVNC2KsuRd5YixNzxQ9\/wEZqvBTT6Yj7hL+6Ryw9Aw40ZAzQEqZJYFBdU\/FRg6nsfYoBGX1XxR821BIlc6fTJ\/XyOEIaGvgU2zqSQprwKPVjncr\/ZBAHRnc\/on\/VmXdqab23pEPiJ12WxIwQRsophiraK2zLn8ALtFGJg7uQRUnzY7umbtxL2J51BB94FfpZwjwGa3b+XAmzWYmZ8xJU0VtaXjR2rv3ECGrIUXpKr309UkyLsZ2Qzz4rn850sNTsHUp79XAltSVDWY6GU5whoDMy6xxAgxGnkulKQEKEAA6eCEf3GrIHUPqyQUEqskidi\/ucUBvU1ZutWsIF2FpDiWrrHpuaK6D7Qqa2EQ9mY6IPFjqYkiwgBzsLUXia7v4W+7BYA9S1G5JkKeCvmYQWBFBgF+kNjk6ZtgaxuTMvSHgGGImBdG7Esx1gMLQRoNikGvgVENGYlDPE\/NLNunxig1YmJhKws1hH8g\/APBejHCWh1eRX6s8YV8BfviQ+lrFhNnjZSUPh7PpYyM1OAt1yS8IrcYShs0L7M5SjLQYqvaAUA4uqEzn4M3NIK2RRy68UArSqe6q8JP6bJnMBfjA9wCbmOxwlozClySbxuQx2MXYkcoYzsFJlkXefwA3RGvrCVSEYKazlbQGMSbKKCCKosMMssqWBEH1UMFYuhcR9CHgHBArKbU+s8vwrcyiw0uccUN2MuGJe1CpWRV5ZKrs21jFsbobGc6AvmQPAhVKTVwqiynNCm+ByAs8kgZYDGO1tBL0HAXLMIlclEv7wT3+u34mc2Q1TgRhsAg0UCKNp4amieEW56loSabdcFoIl+oU2hNqWauSvLwNQ7olzwXize2KZ26Ml7IUSshyJlTqAz\/SMk4CPf3ePPo7UMuuJV6MfY4TMu1hbvD720\/1G8ymcWtGzWsNYDffAu4mHxqsIZynSG\/xTLtvFkhRV4Fvf62cx2K4FVqgQYZXs2ffnDc0UDWfVlsUzFwez6Zg6QTWXeCeaT+aUN82QVyphMsrG0Mpej5JG8GKJdyIPl872ebb0Q8DzjRCmAVngZsPzgf96TsUNL6Mf61xYp5j1rQJMgRIjyQgwYX7Rf5qgOoFQBmh8vCRHbtzLaaLl+\/1UuRxYXICThwyIQQMAUyv6y8TM9g3dXNQ0EH0KQuZP7UPOIcJEbU9oPggwlgufJbRATsCHml9WtTZNyx0g5kF88tiFbsUHRR2CrsTKulCoLmouQ0OkC0KSJ1t2bJrortsn70QcLG7rZ\/hCAcoNB+xj\/h5SMKoWq6ns7jq54VSAKf8HfvLvciLwnvChe5X0BcMXUoL+sA9ryw3dyq5fxasxCE8+X3SvXu7U4BGhlYQcrf2RB8huFEWFtFerQxnU9w9LB9llFv9TvfW9TVQwNujF+pdsjo6T0q3yeHz\/TWLDKBVQoEKwheEAhE3\/bjBRD3hs5B8grdpk9oPHSNqMR5uVFLajVATRrulrtqsrkly+e5yuBQASxCyAUwyurbuK7iaqEBARksfL+zIvcKLwLhIZ5bGafz8ASuLQHpABAMR6aEVqcXZAIT57BhTCBiao2cscAjT5sEFiaMsxbVYLLVhDw3Y\/0A118C01pzkqD5j7FTJhDP47VBaDVSfFnLPCdNHQ\/cQWhxdxYQAspZLkCWiqvwk+0VezPbqSHZ7RJHl61m\/gVemDNMZfwkGJnUpjKeDUGaFqDZYlPamM9NWW0Z11iMSvOZdelrGPlBkiGhGRBroDGmFmH\/GgdQR8AiveCx+1c2fdXyEZhEuhMH\/AE61oxNitbFVqhTxR51glu\/KwAjQmIbZhjwDC6BCqagAQAE6Y9DFVJIQogIvytFVMHFH2QgFGZYMbju+Jo2wbQ0N54b8W0\/FR7iC8Q41kQN+RO05i1SJSpJbDC7y5T338HNjIrZZp+qmJXZYAm1y3CWm7jqixHnsl7ag4QAPADWi0AzN\/wg+8a8hd\/DLA0N20BrW4GW2gvITzEe\/BuLHD\/HeoAWpU27j+\/bgzYF7hd8WqduCFzxZrg0nhsTK2MV7sANBvrjgGarbhjacI6VdWhkPxJBbQqOvO9P6ex1P9YX\/6aVpFl5GkodivFGYBTtSR\/i5KeZecHUELWIFNRrOWB8JULyVz1IRd\/VoBWJvhTiCafallZqrZZjqFx2PiQnywRG3cMQKV5KBAOMwi4QwLHbiPgWdznJxzYMQAAKlyK\/xlzH60WZoNJQrEfO2dl1p+eUwZoofmoAjTFLOz+OgErGhr3+9YWi+Tuu+8uXFC0Ofnkk+cVC+4ts6T97+skhdSx0JhvhBbjU4aXL2h8QJPFjNCWsG6ijNkCr\/CM9gzazepVa24cvGp5wSYLKEGEMelz5gClSHs+mQfmU4AmV3ysxmkXgGYFbRntUQrhX7lHLf14pxANbYKXCgiELLQqOlnhby2aOhnZcrHCJ9DCxrxCgOYDTpXckMxQ1X6FUuib9e8rJjYezfsLLLMDtNTK4paI8p2nAJoN3KrieIyhqhjFfl9VIdrvK8TAFhhtey3mmAYtK417qmI30oZkoaJh8Vy5MENMjhBV7C0lXb8K0HimLCu5iOQe9GN\/fn1IGyex8TZfGbJCT3EpO6dYobHtA20ATdZsTFiIT\/zvq7IcRVebUStLugmg2ZJSCAMEs\/bcpdC4a161biq590maUSEA+EK8ipBTzJW2tIOWEqKifWiPJvMvpQ+eDyWFtI2hpViFrL9QyMPG0GTl2HhvXUDz6YTixzwx36kWuS2VF5JLsT2MNoYbs87Un3XhQmtwQNZdTAHVFi67yTorQKsDILZt6oJmwSo7xme61D5iY7SWUopAqAI0mA0hg9BGqDNuLt\/laAU+31dpQrRRkg3uGgSrjcWhISEQ7DlGljH9dNzQfFQBmg8+2hwbSmenf\/oDVBHquJflcsRdo7JWmn8Bnk0eob0SD5RMMi5AU4C7bZajKrD4Akxza+lcl3dtOSFrkdkN7FWVeqyg7IJXxROywgU6CFPAy\/KqYmWy1rTvCouL+dLnvJvdYGzTu7XRuW2Wo83aS7HOq5LSBGjwqZR7CwZ1AM2vpiQlyuYR8DwUAr8KjF3XKmWntYN8oz1rEt4LAZoPpKEKIPYZ0B+LlL5Z18gZG8vkfn6khDImW0lG73ZAABqLgviLajmGJreuUAgJcptYEqsmgKClHamoSsO37i+YWHEUBcntPjQLaNqU6Mcdq5hHY0dQMDdo\/Sp\/xHeWQbUnjc\/l+oxt0mV+YUIlXvCucsHahAf6wpLRhmWNhzYqb4Og0GZxmBf6IQy0qLhHGY62Jid9MmY7J7JumsbQZM34VlXdfWiaS7kVrSssZqFBF\/EJmrUy4RgL7wUfIfAURypTCpkbAFfZoH62ri+E6B9h5x8ppGd0yatSquQulJUllzHvC0ApscJuLxB9mV8JWM2ZeBVeVyWNEED4Cm7ZPjTxl79JuA2g2ULVzC\/PYMw8g\/i2FFpVOqlS\/m3NWNr6MsFPgkuJi+u0Ca3lWHZ3qCwhY0hR9GmHTOJHR0bpXUVnn0+V2coaWFKAlpJyD1FgEN+v3gWgMfGAA0LDppXL74z2KVeJgANGqjpdIARoNqtLzMKCl8CDgdFOrTDi2XyPsPfBRFqtAAUgYhFpsyTMqNIzzLMv3G11FwkhngHzK4sLQWIr\/1uAkV\/eAh\/3SXNkDrhsDEBjBviYY+ZczxTgMedtAU0uXQu4ZZVC\/NJsjE\/76GJunhig8R7ME98ri1OFr7VHTQIrVAxbc4R3gnFpjTBfCEd\/8y\/CQha8lBiUi0JYBI5d8gVrU15lHSi2Cx\/DJ7y3jSHzLAGV3osxqeKLlBil9GsN8r3+9gW73HrcQzveVV6RskohIcumDaDpfaCzqviIVxi7LeNVBj5SWqS80y+yDpnnKya2oo3WFrKorM6jpXcI0CyQwmuaS1UACWWpo7DyA+9Z+ahnQRvGjvyAb1X2Tx4deI61UcjZXCuFVGkg9vuqlHvaIlQQjjBDaGF2BWhiTPpDyKsUjD5n0UAcxpJ6wrQVEjAcDM5nEk4WXGxaPO\/J+0p79+eJ+5kXBD59aLOlr93F+gfkEKJye8pdJCvSB\/dUmtpkAN0jEOV9VG2bsTMXVuhJW2OeGY8EIG1t2n7oNISYBafqEAhMgRoVCspqOdIXCoHStHk2zwQIVbHDboilHT+y3ljEAJBcbpoHK5zok8WNMNCc2ZMq6A+a+5VlQu9uaVOmYcMLjN\/fC6r76\/Cqqsf4iijPQPDZOpoWyAQm2vZgsxrFq\/YYJ42N71RdSJ\/B6yigWqeW1xkX32lDspRQZSf6SnEZoNEHPMFYRS8Bk7K0bdknX\/HxvR+ylsRL8mLYbQFlHhK9f2iNSgG1MiG0di2gMa8CJrn9dY6ZLSOmOWQdSFnVPjMfwHg3+FlrF1owXvpnnSB7GKMy1ws+WAqAlioop7WdFRIQWRUVYHqYwq\/bZmMkMD4uTX7D7AhAGAQtDGbwwd1mJ\/Ed7XSUg0DZCiLFL6Tt+ZUAYD4Y3wpm+g2BFp\/DlBJw1q0kQAPQbVV9aynVoW8skF2njy7a+mc7qXiAtmqIPgAI7x6KdfgKiFw7cnGqD945xSOg95JVZ928gCGHycastTq8agWZnsH7w3O8k1zXGg\/vhXCUpcr\/AoE6vNoF3er24Z8tCD+j+PEOcoXSJ++hWqG+RSWAlVCnrfZ\/2biirC2rPJWNl3lmjQK4Wnv8ht\/KXNkW0GzpK\/jMP2kausGPopctC8fnyCY+U4KSHS9jURydz1kL\/lmSiqkNgFaXM732NsGhywr09jG+GwcGxFooY1iACcanTSxtOfbqihHE+ofBFLC2x+RgUSCM0J5il6+NaRGzOG1aPotCmjDMrtMVrMtRGm5KiSd\/PFIGYvGhlmyRfDtCgfHLTSshRwcsYi6UhKpSRdyPUEPY67gcWc58x1yVBf7LBixvA\/RRqbFY+1RepS\/VRPTdir6mLm+GQNTWGmXNAYZ1eTWZQB01VIxQ9UpZJ0pogv4ohNAYOsWsXw1FdW7pQ8qsPmOtAyZ11zx9M0Z5BVTrtOz1LaCh9CJvoGVZnI91zHqmfdV72mfLFY5MKDZQB9zfxcb0wUJrx7ESOjAnAgXh46eDt3vC0robBYAfnbXm+8xZSMylL8BtXc86+2eW1uwt3beBHxCI8IbioAgtLBhbRUQzoLMNbT3PpTs7w5ulzsDUARrMjkbDpU14qS+b0k5ptZi+0qCkLSs7zO71wm9OW0AMcFOcJOVZQ5u5yiaKISjTEkEWqrii\/TDM27hPGB5oM\/4ZwHJUsVwsA9Y1a1rZm\/BFGa21569q7+X432R4Qi4zMHWAVpV63WZiy\/aIENNQNqL+BsQGQKs344AW1hb7ihS4lztMSQwAmspY+Zaa3E0q75WSeVdvhEPrvmaAdYSLUnsfATV4AaBjncMD0Fcp3CiOfCZXlY7G0R61vsY9PCffGRgAbT9t5PtXyi+LTBYalhdgp5OoATFlow2AVo+5lQKPaxE\/P0IJN62y\/LTJU0Fgm8igShHMvT26pt4Ihta5zABxI+3HBKSw1okLqYqFYoG2qgn8gHUutzWxmNA5b7m84zCOfmdgALT98y03JoKWWJgFNFUfeOYzn1m0tv8PMbR6DIvLGCEGkOn4DIANhQGry1ZwkIuXJ\/Cd9pmF3JH1RjG0zmUGlLlHLFVA5rsZWWMoPNqbZveXVSWp5PKewzj6mYEB0Lx5lkvTBzRZZDRngeH7B+BsluPgy09nWlXUliLhl+ySEAO8VB2C3quOrkkfwdBykjOgau0qcgtIaS8Yaw1+UPo3ChD8gsVGO50KETsscpLvNTx7sjMwAFpLQEshn91cndJ+2tuoZJX\/HikHESLEiJUgwLDmEGIIN\/ZroUSkls+Z1BwOtE6b+VRegPaqAgEvwAOqFAGvYNE3SVFPG2W81UDntjM4nvunDtDGMw0LvcYsNFqEXI5V44Hxr776ardhw4aqpkvm+02bNrmf+qmfmq\/FqBcr08pVgUP70SQw2HOCECOBQBUh+KzpnqpxTnJxqvovX++OX73c\/e1bjhvno7LpO0brOgPESsMKIymES3sESfbA\/a8KLVKUaA+IsVaJuQF0fV5drOk\/+Yc3uRm3zF10+lXF0Hc\/tdN97u7L3aM\/vMutO+wFxWdbdt5a\/P7Z0\/+nu2PbF9319\/\/e\/Gvymb3oz372Z7de5HY\/9cT8\/dx75\/br3UWnXeVo+8bnfdh96rZfGLlH\/fG9vWbcjDvliYvc297884vWdJ\/znvKsAdC8WQoBmnUx0twmhVRNMv1x\/+WXX15sRtYlgEMg2M\/s\/3zeZTvbF6WbXvjCFxabW1mg\/jh4dpPPXv\/617tzzz23OH8sZqlpAyeCiQw2G0sjJkLyBwJOoKZSNzqmns3QUi6q5r\/P76H1Yb\/6teKRP\/GtK\/p8dOmzROtrr7120anebQZ5xhlnuHe84x2ltC7rXxUq4AG2u+hcPuZRxxtpO4yNp0J\/lBrATtUy+rTSYms6dS5f9POPlDbd8eget\/rogxbazDh3+7UHu+e+evf8Z+wr3rePw+HQABaartlxhrv1W3e6E86a2\/LAtflv1rqjnrXHrT75sUXP3XH3Gnf7\/z24+PxFb3nKuTVzNVND16\/\/s68PgJZK5FzahQCNDMhY2n7VuMX8559\/\/gignXfeecWt11xzzXwXfGb\/54su29m+AIfTTjut2NUPcPjjCI0t5bOLL77YXXjhhVEhZ+tJ6hgYYiGMASCTS1FZjgI7baZGmaBt2SGmVTQZ1\/cCtIO23OLWXHvZuB5Tu1\/R+oYbbugU0FCQrrzyysaARqUJFBfADDpjiQFgJH+g\/OmcK8XQ2DbD5mvt+9QBmFWV4mtPWMUNsTWd8pwNrzvCveh1R8w1BYgApJTLA66UW2ybh767yx17ysq6t420HwCt1fRN5uYQoDESeyZYneSPGPNjxXBt3Lhx\/kX5zP7PF122s31JyCEM0Iz9cYTGlvJZFaDhVkJAqQq39h4xDuJkWG0ILuZNJXL4DKGHoFO5ndSTwfvkImuhrf3Tf9nno0uflSugwXf8oKyIrgInPAdYbvAC9IcX\/DJugBv3+UUQxj3xdQHtnX+4vhjSJ\/\/D99ybr2jnHq2DgbXnIQaa+x86AFrtGV16N8SYXzXYKOGjSyf02lnosp3tS0IOEEFY+ONgDE0+O\/vss90VV1yRpLXb6hAAnCrDY32pioQ9boYxMZ8AInuR+nQz6dm4uYjlhep2WkA78tPvdssefzgLhs4V0MomBxBjnstqbdpKI31OdGxN61BWFU9mTAKzPsc3rmcNgDaumZ2ifmPMf9ZZZxVv8eUvf3n+bfjM\/s8XXbazfUnIITDQkv1xhMaW8hnuRqy0shjaFJFvZKhVew4toK2+9jK3YsstWbzqtACaDrnE1di3slKHUKlr+qjjV7g3XLp0koMGQKvDJTXbxmou1uxm7M0L5v83L3Nu25biWS+9blfxu0vLK7W\/3Cw0xq3yRTp4dewEMQ\/oum6nBbRDbvqUO+SmP+\/zdaLPmhZAU3w116QfTXAM0JR5q72Vr3j70e65Zx6WBQ80HoTxcQ6A1ngWq2+EqQgck9qbc3X7gvkvfPaiF9q6b6XbtPdo96t\/d\/cBbaFNSoiNo26nBbTDvvz7buUdf1vNyD20GACt20lOjaH903c+zZ3yY4d29vCmeSH+fU37GQCtM1KGO\/LT6cf8uEbdxwDN7+zBJ\/e5999zqPvmg3N7cXT5Wh+f85m0wDrtbF8ScnfccUeR+Wb7iz2TZ1W1K4uh2ROaNW7cnWQ3soE25eh30rSJq1WRb5eVAAAgAElEQVSdD1ZGrHHV7bSARvba2o\/lkRiSI6ClbKxOWXBkR\/ZdzzEF0MYWO0vJClnmnNsbmb0AmpH+HzhebFEHA6ClcGTDNspGtKfp2q5yOZE4FdDs2C+9ddb9ww\/2OUAuFGjmMxt45t6UdraNhNzNN99cJIXY\/mJ98ZyqdmWp3F0JMZ32q9MP6rLQuOp2jgBaMagF6TPJrMelDGjMct\/n41UB2nn\/7unuuOetqsuW8fZlANXdU\/b3FLffBkDrfLKnr0MLaJd\/e4\/77Jan3HGrZtzb\/vEp7rSZ7e6UZQsbIP23u\/SWWbfzOS8tPu4ieWTSSSF+pRDeSzX8UEBiG7HtvKh9FyWwQls0\/E3zdep2LgY0Rj4qII676qd7Z2Is39NPP73YKI\/y0tV16qmnttqH5o+jrvtZqf7E3Ng83tfVO6A5527\/ys75eFzISPvouzZHMyr3PoUFts\/93ce3uh9\/+9F2H3atKRsArdZ0Lc3GIUDjTa1bD4A7\/YgZ9\/4XrFg0CcTa3nLjzCJX3zS6HEMUbirEQmnzdTmoLqBV9R8GtNG7jv7h\/e7M+z9d1VWn3wP+z3rWs9xtt91WZLR2dbGvsaoqTJ1nlfGCilmzvUMeBAFaF7xQZ5xVgJbibgSgnvOSw9wM1pdz7i8v3+Ie2TxKG\/rZtXOv+\/gv3l+0eedH1xf6EYD22KN7CjdhUS1k\/x63WBIKYKcrNLYRgNzf6Rf\/9FFHf1yHH3xMUSprGjKXp770ld3wLKLV2fhch5GbtLWAdv5XdxduRK6YW2\/f7C73xVcs3tGv7EjdO40uxzqAhgCjOgvVI2xVkC6FWAzQGGeTup3095Vb73Gv+uSWQAWIBUtt5fdvds\/48u80YadG96A8Pf\/5z3c33XRTp5VCXvnKV7r3ve99nQm6EKDBB1QEEb+ziZqYGRf7JIm\/YoGyL7GvqwzQKFlVtnlaQPHgbXOW8jUfXNiHWmf8eoY2a\/Pbdy0LvCyg+cDGd2uPP9htfWCxoqP7B0CrQ5kWbXXECMVJATEuCSn2sehsrRaPaH2rBbRLbpx1m7bPRWur9pdd+iMHudesWz7yfIFa0\/1qk3Y5pgAa8TEEGOn8\/M0PQuy44+b282CZsl+JxauTi5sSKQRoXdTtpMzZLa\/+UOmwlu14yB159S80HXqt+3KMoaXwAsoLm+\/hAVzSWJpkNE96j1oZoFVZZwKX4567siiB1RTQQvPXNZ31Lt\/62NHud37r9zpTXGoxb83GU2uhhYSR3p2FgOCzQFdzXjprHrPQUrMXv3rOqLUGqE1rlmOKEGMrBmBGNiNKCuWOyqpFtCFUiIe6rtu59e2xvWhzFlsfySJdCzrNedtajj7tfAsNMGMtY4FhpcMTOVwxQBup0xgZaMha6uqdpoXOXb1vqJ8B0MY5u\/stRu1Dsy7HOo+1oIbLkn7aXrkwvy\/EEBZ8Rgaj1cQBOdL121pldt5iSlHXdTt55t7Dj3Hb37Bw\/EefGZC50DrGs6oQwrYOXMzQmP2lXFhnFsgeeeSRwmrDalehgLZroe79MUD757+yzh1z0lzl+tA1TjDjebnTue48N2k\/tYDGy06zyzElzZ53pB166XVnLpD3N+5a7v7XvXNnHelK6W\/Saft2vKoQon1lWGN+3Uba2xgK4+d4mS5Brcmiid1TlSwQs9ao+Ujtx3FduQs6xUUFYLgXVYyaJBCyXwE26mgSM+Ni\/xkeGHim7ytGZ7nobJLF9u\/NuiOfscKNG8wGQJvjgqkGNIEaDGavXJNC6sTQ9D6Ke71629fdq47dnxJlSmj57crS+3OKoVkhpndAaCG8oB8auD2dGODjMyy3aQU03jMGauN0PU4LoMkSgw\/sBb35jm0ffIcLkpjqpFyQIUAbiZ2ZnRoka3C+WR9X7nTuYw6mHtD6mKQ2z4glhaTG0Gw763okuQSA1JXS36Qrhdh5tNmKCCyy2BBYISHF98RQSArI+aqy0DT2Xc8+2+182SWLXuWwv\/uQW3nX9Z2\/Yu6CTrwgRRRe4GgY3I\/8bQGONrgaxxVXTZl8n85lmY0DoKXMaHdtphbQYKpYLUdiMBwxghan7MfupqxeTzFAq9fLQmsLajaVv25\/kxZyvhDT+HFFQju7LQEXFK6nPlOz684n7VMBLWypjS9JZNK0rprLGC\/gbuZcNPbO2Vga\/AAvwBOTuHw6k7F43i8dGxxKH65Gq9RyaG9uB7n2SaMlCWjTkOXYlMh\/8dKDi0ojXDds2+vec9OClVanz0kLuZgQsydaE0fBalN5M5JE2IM0Se28bI7bAdpCz127HydN6yq+DPECIEbMjN9Y6CinKDkkhcAP\/naOqmfU+d7nQe1JVB+pgNYnmDG23OlchwZN204doIViL6GX77t6QIwAsbT9pgQDzAA1XU2ttEkzf0iIkb2GwCJ2RpyEoD+Ciw20uKAUY0O4TXovUoh+dQBN94dial0fDjppWlfxus8LZD0+8MADBYABZvCB6M137FPkNGvc0+Oo40j5M+K17Hmzf9cBtL7BbAC0OepMHaBZppq242Oapu37AsG6Hf\/l13a7+384GkSvEiA5ML8vxBBQuJfQvkNCClckwAbY+UIu5X37aNME0BjXAqgpm4D9aW\/obMjTBmi4GrHOuCyY2QkB1FCAuACeshT+uufe2ec0ATSSQIid9X3lTuc+5mNqAY3JEaNijVEVRP8j8Hw3QR+TGdXa95+H1hWg\/cTTl7tfP\/WgVlbapJk\/ZKGhlaN1I8RCF6CHEoOlNg7NvC2PtAc0qsjMZbIedv2H3crvfrHtkIr7J03rqpcAmCiEoI30tMfViHJTZonDL7Qrs9ibnHun8cKj0LSuy\/HvP73NfWtjvOh41Xw0\/T53Ojd9rzr3TTWgoT1xWYbLDdRiLseUfWO8W6jdSUescldtWLDKcDum9JfTPjRiY\/ywn0jjSmFchB9CbFKbasvG2BTQ6HPbWz\/u9q0Y3VPVVSxtKQs6f+O1pU\/Tc+\/oo+ysRZ\/O\/nEx1\/zWQ+7B2+dOpu\/zWsp0Tp3HqQW0acxytBbaOeecU9Douuuum6cVn9n\/+SLW7j1Pfnk+OYR+n\/\/Ssyv7s32J+UmFB1j8cYTGlvLZhRde6C6++OKpqPuWukhS27UBNDez3G192ydHHrX6ut90K+79eurjo+1yFnQAktzIbV+U+Jrdo9j03DssMxTjWC1Yn85+\/cYB0NpSsvn9Uw1o+NlZrNT7s9c0ZDly7AbXxo0b54fOZ\/Z\/voi1m73xr90HTls4bubSFa+o7M\/2JSFH\/AGLxx9HaGwpnwFmgFrZURMIBCrpU96ozd4ylcmiekgOVytAi2y67sJKyxnQpJhipTfdXwb\/suaJvZ1yyimLWCFU4ix27h1ZtLLO1BFrxMqYAdByWG3hMUwtoPE601b6ylpocpmR6KCLz+z\/fB5rt\/yxh0eyHX\/+u0cWR9OU9Wf7kpC77777CkHg38ezm3x29tlnuyuuuKIU0NB+iYXxmxgJ1T+IiaVexNIARMaNIOzzcMeyMbYFNPqeSxBZKJ500JZb3JprL0udmmC7nAGNAZPZinuQC35gfxlZrlUXma+4oOEjYq+xvYp1AC0l9i46X3DBBQUfX\/i7o3vQPv6eh6qGPpbvp+Ug17G8\/P5OpxrQeIdQGn+upa+6dDniIrTZjr\/pXuS2uZWlLszcXI4SZIqDQDd+EGq2Rh8VIwAvgIzfqhyBoKaCSC5XF4A2u+5Ut+M1CwA2s2e3O+rP3trqFXMHNF5OVpbS8QEo4qu4ENl3yGZqkkRoh2tRVWXgBax8NlrHEkhigMZzm557hzKNV+Ouwz7mDl6zUNrq4ZvWuIe\/eUQrejW9eVoOcm36fin3TT2gpbzkJNvEkkLauhxZTHaT9ecPerb70szxpS7MXFyOlh4IKKwtBJlfwy9EN+09wqrLbYN1F4C2YKXx15yl1tbtOA2AJlqn8kMdPggBWhfn3r33ve91p130wAib3rLxSXfL5\/tPCGEQ0HkaDnIdpzxecoAmiw2NLrfz0Owm6LYuR9yBLzximfvIhrk42tZ9K91Ft66aCpdjjKHZa8YP2jeCDaEFyAFcFC3GpZLiihrnginruytA45gZjpvRdcimq9wh3\/qLxq81TYBmXxI+wJ2IhS4rHssdNzP8kHqFAK2Lc+8u+eUL3IveOndgry7cjbZsG2Ntcro8\/U1b5nIqPcbZbskAGkFeMU4uVUIgnLXQLKBVnVgtope186uG\/NLuF7tpqbY\/TqaeVN9dAdqolTb3Nm2stGkFtK7oGAI0+m577t0vf\/Bn3HNftXvOkN5\/3fqxE0vXIM26WPsh+SA6owCiEPqygHuafDZNmctTDWhiVNX6w+dOqm2dfU1dLZpYPzFA68JC45k2jkZiyDc3LwSk\/SSTXJJCxj3nk+q\/S0DzrbQ25bAOdEDrmh9E5\/\/+v9\/pVp\/82Hz3X7tqp9u8aU+xSVwXc2\/\/53N76kXbdrYv0fmOO+4olHt\/HDyryWcpiV5dz3HT\/qYS0GwiiECMjCd+cgW087+ztsiISrG8LDGrtDkLaP9l9nR3zZduGHlGTCMbtLmmSyZ+X5eANpIcste51Rsvcyu23FI8fOdZl7hdz5nbd6hrxQOb3OrPXxEc3ABo3dJadP7Ubb\/gHt\/9cNE5R7j93ccedbd\/Ze4A0klcA52nsJZjbEM1IJctoN1\/vzv\/\/PNHAK0rC80mhnz4+0e4T9wS3wYwWGjjFTNdAhoxNKw0XXI5bnvrx9y+FYcGXmTGzcw+4Y76xIWLvhsEXbd0nwe07\/yCe3x2DtC4\/vL9j7odj+wZYmjdTnet3qbSQrPHO0yFhRYAtCrLS1SsavfGE5a7X3zOXF3H\/7f3aPevv7hQFPW8l73Y\/eTObxYVRR744T73fw9\/odu2b2XhRx8stFrrJKlxl4DGA20lfu1Hi514rQEue\/wRd+SnLx4Z7wBoSeRLbhSy0LiZ+BlXWRyb76vWdOra9\/sa1vQUWmiW62zhUT7PNoYWALSu\/Oix42ROXTPj\/uhFC8fMaN4uuvUQ952Hts8D2oHsb0+WYIkNuwa0x37yMrfn2FPnn77q1s+6J1\/w2srRrP3Ym5zbt5B9NwBa5ZTVaiA6\/8k\/vGn+Pirsf+6\/zrkbhxharenstPFUWmihGbAZSzkBW9dCzn\/3GKDZ2Jp\/D9mWg5DrdB0VnXVN613PPtvtfNklgYHqmJn4O9isyKVKa9L5SQhrUz6tCReEAO2LfzrZ+BnvsVTpXIdGSwbQ9NJyR+oYEipPTPKKCbmUPSaMO6WdBS+qkfynHznIbThy7hgSymFdcuPsSJksPn\/NplWO49pvvvnmovqGv3eGNk0+27Bhg7vyyisbFydmvxGCios9R9DRXtqnRh3I3K6uAW0hjsYc8DO652nlnV9wu075cedm5mhtr2WPP+zIjJxmQacsZt4htKeUvZhYQwjyPk9fYFyf+cZvuxu\/\/+n5KQfQ7v3GXMWQYR\/a5FZmVoCGYN25c2dR26\/OxT2UUaKA6KQBzB93TMh16Ue3gIb15f+vMdnPAboPLvuxYtNyTntWqoQYBzsixNDKcylKrPntGtDo16\/tOPesfe6oT7zNzczOAb+f4q\/xHHLjJ90h37x6KjR3QICq+2yY1qbpKl645557Ct7t+2y8GKAdM\/OjxdQPMbQ60rvbtlkBGozCoX0cf55aoy+3yiCpgNZVDI3n2UxHrLErz5irogBoYbHp4pnXbnhy\/n82Yk8yhgYwIchQQlScuEyIIfAopowQy62OI5M6DkDb\/oYr3d7Dnz7CVivuv9Gt\/usPLJIEixJG9u11xNOmwRWlkIGtbF\/GC\/AOyg0WPIoshYn7uhjXtTf9gfv6gx8fsdAeumUuOWuIofVFicXPyRLQ8IsDapxEG7sQaqrYThu0OtwSOW2qHpeQ8+fEApr97mtb97pf\/ObsSHPATu5Ivjj7K6MukrasWMflWFeIUfMRS5wYKQpPbmWwxgFoO171K272+DNGyLLyto3usK9+dBGptr\/h993ew5828jmxtKUGaCg2KL54dPAwnHjiiQVP9HVB56u+9ivujm0LJ4p\/9F2b+3p89DnTQOdxT1JWgMbLIrT4IZaCwAKk\/IsjIxBsAB8aGuCHD92Pt4x78lL67yOGRszstesWL2gfrAT2XzhzYeQXf\/tg9+1tuxvFy0LxvXEBGgoMAEil9ZxOU7A8MA5AW0gMIU42F0Mrqxqy9W2fcq6IOy4cQXPyX76jiJfecMMNi+oKpvBwrE0dWlc9p45yQ+yMWDnrHcVG3o6qZ3T1\/QBoXc1k9\/1kB2i8IiY7TAuoWf84\/2OVkRgAM3O0BAzdd5aTJYPdExeqIdlHDO0tJy5373n2nLvDXqHKIXx\/xpavubeuX2gfqgFJu3HXfasjxNDIiZVSdBrLfZI0jy3DcQCav8GaZ1fVddz+xj90ew89an6Yh84+5l55z58sCUBTIogvG7oXjfEeofOv\/Z+z5ms4XvfHj7o7\/35nUgIXvaYkeqW2s33JQsst0atP2mQJaEwAoAXz4l5AI8dHzv9o6LLeEGyTtsooioyFiO\/f\/i0ixoRclzG0Z6yacVe\/dPGeMzIZY7XbbILI625c5R7ZPlp\/ToqF3iM2Xr9dnbpvKYBGpiMnkwNm0J3sRv+E8j4XTNmzxgFoPG8hNsZRMm+sfN1Quv\/r7vidqQY0PDBUAmKOuSaZFFQA2ufOmqeDUva7TPSi85T+bJthY3XmG6sRYlu2bClATRdWGQKty\/iJNmgDTFZY2gr+\/jHsIalSB9AqpVKNBsTElAii2\/yEEL+7S09b5V5zDPuZ5i57EkCNRy9qWscNVQVoZLtiAUuJQalJTRZq8w5N7x0XoDGex15zWa2Tq\/0EESy0W7\/6t1PpciS0oONjUGBZ++vWrWtKptb3+YD2jc\/8wG36zNz2gZAiOBQnbj3lyR1ka6HpDQj84mbkbCQCwBQfBtS6umy1EQtaZE\/CiMTwAFb9HdsWQHsY3T\/CfZxCzs6Bv5F60\/a9xf6z2HXSEavcVRv6BzSdbcW4tm7dWswZAgphgDXG91hkOkGBdrgZOZG4z0y2JvzVF61TxrbzzIvdruf+05Gmx13109kAml\/lJ+WdWP8onJN2N\/uAds1vPeQevH0yh3raeRuSQjK30EQsm9GI0MPV2AWoKf6FwEQDhCFkoWFtcQFQ1oLjuSSk0F4A6J9+a5msDyHnVwvh+Tds2+vec1Mc0HjXK390hTtl2Y5iuFUAmCJwaFNmodUVYswxVnOOp1OH5qMPWqfSIRR7m3ZAY52y5uruU02ds9R2PqB98j98z1H6atLXAGiZAZqtEuEzBxo72Y8IxVWrVhVM7cfPQpUlyphMrgBS\/rEKBGgSvARcsQj9\/22fWGZ8T7tJCbkQoP3KzbPuuodHK0uEtLnfOnjhuJku3I5lgIZiQnKHXMjQmx9ZYXzu05T\/0ciJoXShxIxT6OQEaIUF\/PY\/H3nd1X\/\/x27Ft6\/tbArquJerHuq7n\/HM8KNsZns\/6x\/PyaT4YbDQqqg5ue+zcjlqI6V1N6VODdpbqDxOyv16rg9oNqaGxaYakeozZHH4sba+hJzvcrz823vcZ7csxB79eZA2ZwHtX31j1t36WBwEU+ayjpAri6EBbrKceS5KB5Y5wizXqy9a13l\/C2q2HFadPmJt69C66nlV8VQlheD+R8lB2aRSzCRAzQc07UHrMnuR+Urpb8hyHOWs7ACNeJm09qpFYL+HyRF4TUpfNQW0lPFJyF1wwQUj56Gl3Funjd1bxn0XbJopKoXELuJWp59+ujvs3hvdf3zmXAklLvautblOPfXU5FqOVUKMwD8xNVzDXAgvlJZcQS13QGMOq1L+69C+LaAxXyT8wItY7vxfVSkEjwj7UJERk8p0jAFaSlYi89tluyHLMWNAq7OYumwbA7QUl2PVOCTkNm7c6PjRRYCbCzec\/cz+z+ep7aylxX2\/P\/s8d9+Koxf1r2ciDJ71rGe5u+66y71\/dqHiAXvXdq5YHRxbaLz+Z+eee67j5+STT56vyWfnCDevBBfupBQhxhYOsl0RYoAZmnmflSGqaKzvcwS0kZOvUQq23FIrW7Ls3dsCmupyys1s49I8V+uSv633JYeN1TZtXxbakOWYulLG1y4rC218r1nesw9otLYuxlhaf8p4JeTe+973ultvvXX+lpe\/\/OXF39dff\/3IZ\/Z\/vkht9z\/O2OfWzixkWr3328vc0S9+5aL+9UwW3\/Of\/\/xCy33Htr+Zv\/eyu1e5FaeFxxYar\/\/Zu971ruJ07higEat8\/PHHi3dWKnaVVk5bJfDgjiZx50Cotp\/CXylt\/FhaV1ZaW0BDSUFZsR4Zm8kYA7QcSl+FAC2FFuNsMySFTFFSSBNGSE0SCQFa3bT92PhiWjtWDJe12vjM\/s\/3qe0+e\/aaEUAjwcPvz\/Yl5gdM1s8+4n5m943zr3DpilcExxYar\/\/ZxRdf7C688MIooJHYwwZZLFFp5sTKSMknNT8mxHiOXJS4HknCkfXahDfGcU+OFhrvufVt\/3PkiJlcAE00wGrHjUgSCDyhwgnwKCEILj8+PunixAI0shvJcszhGgAtM0BrkxTiM1SdJJEQoNFf3Y3VIaaOCTmd34T7RBef2f\/5PLXdH79ktXvBIQvuSwDN78\/2Jeanev1Rbpf7yCnb58fx8989soi\/+WOjQdVnqZVCADTAjSC\/koAAKCwvPcMXYlh2CDjaA36TTt\/26Z0roEHru\/\/FH80P95Cb\/twdctOnWsvgthZaaP5wQwNsXNpYHwI0uz77ru1pY2gDoLVmo047yMrliJBD82qSFOLPCpoeC3nSWnxMyHUZGObdP\/SK492PLXukmAZVCeEZsXqMfpkcG4OjtiPXuGs58gySAhBiuJ5QQhBi\/B8TYgAwiSJY31RZz+nKGdAefN3vuCcPOnx+urqw0roGNA0O+lNIAR7FRR3jBRQi5AU802eBYgtot39lp6P0VQ7XYKFlZqHlwBRdj6EvC+24Y5\/u\/uIFc\/UYtUk61UJDI77oqK3unz9jrmL\/l5443P3m3csrrbGQ9ZhqofnzDKDpbDS5mUJZq8TScFlSAiu3bMcYrVPSr5mPLtv56dwPnfff3BMr1iwCtNgzaVh1Yvm4AK3OGlQcts49bdvGAK1L+qXyw5C2P0rNrCy0toyW4\/19WWhYYxuWPeretPwu97IvzLke61hod33jK8VBobqaVuAnfkYcLZYUkiONuhpTn7RmzGUnI\/vp3Cte8Q53+9qXzL\/qYV+60q288wsFj9y07lx33+oXzKf0x7wH\/jNzoDVz3mSrThuaxwCta69LSn9D2v4AaG14ufa9fVloKbG2WAwNC235Yw+PANrrbzmiVwstdWJxUeJmwr0yiU21ZeOM0TqHdO5nnXGm+5uTfnZ++MTQiKXNZUDuPztt4Qg1xxlqsZMa1ElTazyV1lXttCeNTN3QuYlV9zf93gLa9R\/f6r7zpbms3RzozLl3kzyFvumcdnVfVhYajNJ0Y7U\/IW02Wnc1ufSTs9YO8xNjJHaJtm9Pvr5i9nT3v7+0UBarD63dFi72acC+M9wrpHrjcsTdSAxt0scH2XHmHEOD1p95zr8dmdaVt33e7Xreq0rZvSzW1tbliCIFLe0FPTkDkdiZPWXDtsEiI3nIJgiRrdvXVQDatWc5t885Vdrv69llzxliaJnF0A4kQEuxqGDeLtuVWWhkFnJIKIeFchFHe9\/fLwS7Y+Ogrc18rNLaiZURB7OJPySDIKD8o4J8IaY0boQeVUS6PEKoC4GUO6Bhodk4Wso7r772Mrdiyy3Bpm0BzR6OqwfAC+wzJDEEazy0gR6LDLBjvidRLcRaaAOgpXBRf22ystD6e+3+njRNFppf5NgWK+7KQqsSYgAddRv9C62cFH8SFdDGBbD9UbL6SbkD2te\/s9k99Lr\/5r2I8TMGXnH5D77njvjLfzNWQMN1DJCR2cglQONz\/zgmvufYITIcVQpNiRHVFOqmhQW0XCrt82aDhZaZhdYNu+XVS85xFd\/f7gPa+V\/dPV8PsqsTqwVoCCHchrgZEWRVQkzljlT6Krf4GVyXO6DdcMMN7sEL\/qwWoOFWm9m9wx111UL8TR10ZaGxqR76s3EeN2QVLwBo\/JCu36erUe89AFpeMtaOZrDQxkybnFO5AbSbb765ECJK0bZV+y2gpaZ3Vwk5AVqZEFu\/fn0BdLrYc8Y84oZCiBFjyfGaBkDb9dQ+t\/WnPzEyfba+o18mqwgUuRl35Kff7ajWb68qWlfRKIUXADd74CsuSOgPH0xKqbGAlsvhnoOFNsdtWQMajOOfXly2SHBbIHhh9r5TeWPjmiaXI+9gE0N+9ZZZ99ffnztOpmuXYxmg+UIMNySghpsxt71nlu5TAWi7drkdr\/xlN3vii+aH7id+LAa1uaZ+u74ATYli9ry8SR4nZAGNTdVsrs7hGlyOmQMalgOuJrLwKHek+ApaGszNjw6J5DOVUFKGYw6a\/DS5HFmUP3vScvfOkw8q1qc9xbprl2MZoLFpmpRs6AmNcUlC05zoGhJg0wJojB3QYh8a+9FC167nnuN2nvnuRV9ZUOsD0PwYGklF8AYJI5M6E20AtBzgOzyGrC00hgygsR8Gi4u9JjZNGzDjCAqYHIGLBk+wGP86zI7ratLXNAk55qosMSRlLquEXIqbyRdiKDTQGTdkzmeiTRutq+g5e9ypbserLxtpdsimT7pDvnV18VkVrav6bxpPRdHFcwOose5RgPq8YjG0oVJIn1SYQkDTsS2xbCe9EkWEYW4qsMNU1PvjXs7NwlU1yWvahFxZYkjKPFYJOQmxuplt0JekAcANy7vPjbQp702baaN1ynttf+MfuL2Hrh1pKiutitZV\/VdlvMbS9lFucT1jqbGPknXeZ83WGKClVPZgTrpsN1QKGeWyrC00xdDYb1SmhclKU3V2\/9otrJoAACAASURBVP+qhTXO76ctKYS5CCWGdJ0U4u9DU2Zb2WZa+ADa5mqlLUVAgx\/8mJqqjLQFtDYbq6nzKcW175MXYoA2VAoZpyRN6zt7QCMgnGqhqZgtmjzuKTS3SSeHTFtSCGxjz1ZTpmNXSSFo3f6p3DyThB5oXQZoKCw60sceCprG6uNvtVQB7bHXXOb2rDt1kZXWFtBshyg4rFlCCvKq8BlHBsEzofR8VY1B0WHt93UNaft9zXT952QNaLzO5s2bC0aXrzwUQyPGhguC2Ata3wMPPFAsDEojhTbp1p+m5ndMW1IIb2ozHT94+x736Qeeitapo72t+VdVKcTOpI4L0gGfJPbwGTFQhBkxUb9ShOr3IcSoFpLTtVQBbfeJL3aPv\/Lfj0w1afwvfu6J7sorr+ykELW8MaxbhQ4UcoAX\/LPxGAwb7VnvfZ+LNwBaTqtuilyOYlrVbIPZEXB+liMCEQ0NSw7rDC1\/EsHiEJmnUci9+5SD3M+snyuB9dktT7nLv71nVJitmHHbZ9mftPiqo7WHhBgZjbiSyHDEzeyXt0KAobHnkMHqv\/000jpVNPluRzIkX\/7EV8cKaIxNp5TnZJEPgJbKNf23y95CY0rQ2tHM0cjsBcDhUiT+QlAYRiPGwmdo9zkUrp1GIRfKdOSzK89YUWRB2suWx+LzKkCzBYihK5ulVb9P1hjJAtAapSRUr1GFivtfLuVPnEZa15lDH9R+4ltXNAY0WV8xF3PquOAFFJ8+FRwLaB991+bUoY693bAPLfN9aCEOYCGwCHIVaktBaw8B2n9\/4Qr3j9cuCy5KC2plgNaVEENRIRFgEmWPyqTSUge07W\/4Pbf38GPmp+Ccu\/7IfeQ3LmvkckSZITQgQNMeUtyLvpvZbqhG+dGle\/rOep0HtH3OffTnBkAbO1LXeMBUWGi8D26m7du3F1YYTA9jY4mtWbNm4nGypSbkQqn79vDPB5\/cF7XUygANdyL7CqGlLlyIABQWtixq6IvA4zef+0KMfqA9sZacrqUOaLuefbbb+bJL5qf8+NXL3d++5bhGgBZS\/NhbZmNoanPPPfcU23D8bEa8NljzffPCYKHltOpGxzIVgKZEAJvqrddgAajqRI7TPK1CzqbuWwDjbzIfuWwb\/sdSq3I5WhqFYmj6nmQgwM6PnUxKiKXw1rTSOuXd1MZ3O972cyeMHdAII6i4glViUI44Tw1XG2GHvq4B0Pqa6frPyR7QtPmSV0MT4\/wjHUpJnAXm4srRBcW4pnEfGnvOrtqw2AoTaInNaPeFMxeY7kN37nG3HfPC5LhKGaAhrLDISeEmW3VeoG7dWqRyI8S076c+24\/njgMB0GbXneZ2vObS\/RO4z932cyd2AmihtH1RCcWGtc76n3ShBK3pX\/vcWcXwhhjaeNZS016zBjR852S84WIi8It70b8QbmRBkp6P4AsdCNh0crq4LybkzjnnnKL76667bv4xfGb\/54su29m+FEBGQWB+\/XH80r5vuLUzu0am4PEVh7tXbVw49JP+XjLzfffP9t053+73T3yTu\/jii1sLOdyS\/ORcjNjnjwMB0Hjnwkrbu5fNg+5dZ6x2H37Tj3bm9leWKxnL7CPNbT0vANqZbsejTznOQ8vlGpJCMk8KQUAQOEYzg7ljF23Q4rSxOhcGE\/OTenz++ecXwKvrvPPOK\/685pprRj6z\/\/NFl+1sX2J+skERIv44XrvrFveje0YX630HHe3e+PmFz9Tff9z5+fl3eOx5P+7WvO3XGgEaNKQWJ0AGLbHEiZ0otoJbCQDO9TqgAM0QYfd\/fXlngCarXbVbc6S1XI47Ht0zAFpmBMraQhNz1y19ldMcx4TcueeeWwxz48aN88PlM\/s\/X3TZzvYlQCNGRQKGP45jZ37o3r376yNTeemKVyxqR4Nbr\/+8+\/OXHLzQ9gNfqA1oWNrESpTBigKDdSYeIKNtqLbfjLO71tx3vPo\/u9njTp8fTJeApsLDWGah06qbzUC3d8UAbShO3O08N+kta0BT1Q+YO+ZOlFuS3xKCTSZiXPfEAA3LiItYkS4+s\/\/zeZftbF8Scrh0dUyPHQeZjh85ZfvItLz+liMWjVfvMJIgcvHvuZN\/\/LW1tHZKWgGsKC\/EQ\/3DG3WKAlZbzoIuZI2Pi7dS++0a0GbXnep2vGahCn+XgMY7ETtlfyLrnpgZP7GKP3zX937TGKB1WXSYeUjpbyhOPLoKsgY0hoqAICDMXhPiaFbQIQABABjMTx5IXezjbjetMbSj3C73PveN+ekhfvaB2X+0KNZGA+JvG45cVmy81nXyVfcmAxpzhDsWsLIJID5tAF9onqNrmbEeKC5H3tVmO3YJaNYir1qbbOcIlcSquq\/t9wK0B2970l3zwQWFdChO3HZm29+fPaDZs4\/IhALQ+FHSANpZrtXXy4Rcl65EnpPSXx2XI32+f\/aL8xz23b2r3cdXbgi6HOWutFZaXUAjTlYVN8npFIXQ0hsArb1AYr3rcNeq3lRhpu\/EoRigVY133N93bYlrvHW24oz7Hav6zx7QeAGVvtKmar2UX\/qq6mUn8X1MyE2DNvcjy3e4337hCsfes4tunTtXzhYi9t\/BFjWuA2iqIIKAwkLz3Y08N3fX8oFmobHBmo3WxfrsMClkEmu07jMHQKs7Y\/21nwpAs9NBRh4\/uKdCgq+\/qUt70rTuQ+PtABpdqeehyUqrA2g8Q0VoVYfTHthoT6zuuypEGpXnWh1IFhrvK7fjOAGN2p+EHEgKIqZGlmvfFpnPAwOg1VkV\/badOkDrd3raPy0m5FICvjy9y3Z9BJCbAhquJuJoKCu6sNhUr4\/PSAwgfjZpgRbjigMO0C68yrllB43FQqMCCIkhtkya5QsSnEL7Utuv2OoeBkCrnqNJtcgK0JQcECpxVTVBSume9IGeIW0ulPnWZfYiz0zpr06WI336GZgpn5EYsuGMDe7kD\/xVclKI5gxLjGxG+MBWYUczh66ce2Uttyqe6Pv7Aw3QnvFPXutu\/pG3dw5oKpagQsXQXkdGiTcmuYVjALS+V1b687IDNJIDrFae+iqTyniqGt+BZqFdeOGFnVQKqZrXHL8\/0ACNZIHf+O3fdac\/75TayksZ\/bSFI1ajddJbOAZAy3H1zY0pK0DLd5qaj2yak0LKEkCYkVBiS50Tq5vPap53HoiA1tWJ1aKovDS4l9evXx8lNFs4iPGy97Tv+o4DoOW5\/gZA64EuB1pSSEqKr\/YaMf3sI+JKtcxztcR5hwHQ2i8o8UbO1YEGQGtP53H1MFho45rZ\/f0OLsfFE6y9RnyjYz9y33uUwiYDoKXMUnkbshqpzYqFVlYRBrckCUSDhbYwn8M+tMHl2H4FVvQwuBzHPsXZPGAAtG5IwVl4uBNjMTSSlXCHT6oM2mChdUPncfQyWGjjmFXT5yDkqieY7EYEFLEQSpzFLrLfaEe2Y46p+wOtq2md0sJmOeJiBrjYBwnIwSuTLlQ9AFoKFSfTZgC0Mc\/7IOSqJ1hxk6qTh4fSV9VzGWoxja6oHTt2FCWwQvvQKKiAq5o42ySuAdAmMetpzxwALW2eGrcakkIWTx2xD9xGVlgRV0MTRyMPXexJQjtnP9JQnLgeO04joOkNscqIq6k6EJa5qtbUm4XuWg+A1t1cdt3TAGgdzSjaJIzuB7KHpJDwBDNfW7durXX0B6BGdQgArY8L2jHOdevWFUJUliQuL\/5PpXUfYy17xjQBmhKGsL6IocUujpjhh2IB4yqmQOIJ\/MaZgfYaAG3SHB1\/fnaAhtaOJu7XapQwQUM74YQTFr0RzMc1ibOyNLZQZtaQFBJnPjRvWV6UOaJOn\/a2he7qs37ntm3bitJLWIzwGwBGxRcu3F1sM0DgWmE3uJfbC7pU9zPzT9txHB+jYtkoLpzNNwDaBtf1fsP2nBLuITtAQ5BQCYDAPwIDoQdQ4HKAiQE0tDJlOUkAtgE0MTDamNLImS76VIFemNp+Z6eTdirNM2jt9Zk\/NSmk7iKAl7ACfRdlCl3hQWI1WAGy0KzGDrgBcpYnBkCrSyFXrGvS9OV+tlWCYu5nnkJpNOQCgFaVIBRb3yqITX+26LUKCkD7mIX2\/27b6C7\/9V93m754T\/2XHtMd02SJj2kK8qsUYgENgQGIQSi0dwHa2rVr5\/+WtdYU0MTsLBALWriaYGwWDNW+9bfv3pCrEcanoOoAaPUBbRzMLQuLmn9Wi0+lK2PyXY4DoI1SKmUTfQptRas6NVyhK+sVC6rsiq1vK2e4X0q0tcbKXI4Hwsnkmteu6JzCC23bZGOhwVBoXACLmGvcgGbdSiwmm2VnAdJqeGjtuKJoD\/Mj9OwxK74lN2jti1l03JVCpHnDP1gAFtBS6YrlNQBauXgZh6BLdTmmCL6y9Q2PsG7lTpaXxYYzBkCbm+Vx0DmFfk3aZAFoOrwRBsN9gBDC5ThuQJNrASCVJYggE4DxfBjc\/z800SyewUJLY\/5xVwpBIWI\/G\/SwlnVduvqANsTQxmOh2V5RFBVmYF36F7xT5WLUPbH1zfe+Ryfk4RkAbQC0JqBa3AOjss8IoYM7AVfCuAFNg\/W1wpDPPaTB2ZetArRLLrmkOO8rl4u5veiii9xf\/dVfdTou4lW5BJCtexFXcV26lmU5hg4alTU+0Lo9l6OMoJgAcJS3gl+REcTb+Iz4epW7Mba+BWjEvWWRoaygWNuQQRWgDXRuT+eue8jCQrMvhRACHLCUqgANxgb8uGBGNLcmWY5dAFqMMFibV199dWG2HyjXpk2b3Jvf\/Obk12WOqJ6OOxfhhaDp4moLaHXHMNC67oyF29uYGrwAT7C2ATT2L8rFH8pADPUYcmP6CmoI0IY1vTADddd0N5xQv5fsAK1OUgivK7cEwqSqoGlsemKAVsflWDb1jM2exFyfTNN1B3QIuYuq5h+rR9X3u3jjGKB1RdfQGAdat6Ocwg\/MI0ptaC+aSmPBYyeeeGKlAhQDNEYqBbhuUtlA53Z0HtfdWQIaggiNjNp++MHHmeXIxFZpcLG033ER5UDrF82bGCbaeBMLOzZfPqDRzmrmA13z4zTWIm5F1n5ov6lGjEVFrA3rrWpjdWh9+xZZVUghv5kaRhSagSwBTVmEGjCuKBic9HmArsu0\/Rig1UnvHlir\/Qyw5we6A2rQmp+Ylcd3cjWXPTkEaANd29NqnD0IfKqs9Tobq0OAlpK2P873HPoezwxkCWgEg0l\/J0YG49n9KbiL+A5fuq0aUtdlYKczliqcsgF3PGQ5sHq1JaWq3rzOAZ8hQJOVlrJhvmosw\/fdz4DiqUoGCWU0shFfGadYcVSQKbti6zu2sbr7txp67GsGsgI03IsIIZjZz3JkQtDiBW5o6KRmq65fG0Dra7KH54RnQGn8tkpEbK4ANGIrqanbw5xP3wxs2bKl2HKBZwbllbqO\/E01Earwo+TyN8UMqOIyXMMMaAayATRcCDArF0IrlrYvxkb4IdRIIuCzHAGtbwuvbokf5tpaRyHrZ9Bi04TFQOu0eUppBVghD1B0Ypdd+yl9dtVmoHNXMzmefrIBNPZokeFEfAQNrGxjNbUc2bNGUBgXJIFhAslcXSYVtJnyvmM1TUv8WEXAVwr6jDOoILU\/52jquJRytsgGWrdZKfF78djwo72peGdY7ySJlRWxHs9oXOE9SimH19Xzp31NdzUPdfrJBtA06JS0ffzmMDcghoUBk+uwPxJGcrhiJZZiBY7bjLlpiR8\/tuAX800pD9Rm3NxLIgiuZKxsPwVb8RQ0dgAN91JVvKTteJrcP9C6yay1uweeAFwIO\/R1PtpA53Y06+Pu7AENyw1fuS1OrHRemBpQQ5NBYzv22GP7mLPKZ9QtsVTZYUWDpiV+rPJArML\/P6U8UJux41ZiTxGX3UCrPrHasMRxPUk7xxoH\/HK5Blr3SwnfahvH8TGhNxro3C+dmz4tO0BDW7fHx+jFZE3456GpHA7uShJEJnUsuyVA3RJLTYnn31e1n472dv+Nb5FxPwoEVi4A16aaQtU76YBPgAyrtcyFZMui5aS48I4Draso3f57FBtkAsqPzXhmWwdrvg939EDn9nTso4fsAC320sqEg3l9tx3CEabHQstBex+Yv5x1bTUIBBJuo6pLYAsAplSHqOqvq+8HWnc1k4v7wRpDwWXt232HOgg2hW+6Gt1A565mcrz9TA2gjXcauu29b\/eEb8XGjsGhnXUjTsrlKHBCOcGFmHpR7xHFxT+wM\/X+cbQbaN3trAJeWGMkfPnbOLDITjrppKRN9d2OasESH2fZtNCY65btmtSa7nq+m\/Y3lYAG05P9RmxN7ga0fiqJ8FkO1yRKLNUt8eO7GENJIbYCeVflgTRO3MN14p7E09ja0VfcJJWPBlqnzlS4nQooYJHZmqcAGNYYmc\/Qvmmt1najW7h7oHNXMzm+fqYG0NhsyUWmm6+F8HlTrX9cU9t3iq\/mwJ7rxmdVqfeTSNtvSitltabU7xsXXUP9DrRuPtviBRQnLkIGgBgxXGW0SgGaNKANdG5O577uzAbQ7J4L+\/I6d6rMVUZ7AA8Nj4SGcaTGNyFI35swm5T4mcTGasXQVN4oJRUfV6POwiKGVqeafxPa1b1noHXdGZtrb\/kPix13uV9sOBdAY7wDnZvRua+7sgM0FSLGf457SVW3ywBNx0loL1MOiSF9EXBan4MLiYA\/gf2UOBpghkuZ9l0eMTOt87dUxo2iwl5Ebd\/gvahYw7oH3KB3ToC2VOZ9qb5HdoAWC7rGAA3tHusMAEzNmFuqxJym91KBWfYSQnMs69CWC5SarVu3FunxKCqAWR9p2tM0l0tlrHhYcJGrMgjvRXYj\/MFnrPVcKgEtlTlfau8x1YCmY9khCuWwJlEOZ6kxRJ\/vQ3IP+96UCIAL0p5WjWtS6dqAGQkkxFeGa2nPAEoOwIbV5h+Mi9JDfG1QapY2DzR9u6kCNB35gZDTBku0NhJFBgZvygKTvQ86Emwna9VP0+Y76AuIERdNOQNtsm8zPL3rGUDpEbj5m6oHvuh6tqe\/v+wBDYFGEgAuRzQ3fOrEUvhbF4yu+FlKggH3AY46GZtj3rkfbZBz1hCsOqKEOA9X6Cj4GPmVwIAA5r7Ue9FIuYeFyvsyHlxsfEYcAQuUcbJPBwBgLnJJgGm7FLDGoKmy3Xhn4ijMCbTiXfkOi44YC3G3gdZtZ73\/+0Vfa4kzihQ60wZwUyk0\/3SIYU33T8\/cnpgtoCG0CRYj2NavX1\/sQ+HCh3733XcXwh6XI2AGg8Pc1HisOo5dBFD\/ZMspLqNMS\/rkM57BcxkDYIJb014sLsDHF6yAjd1mkLI3TiW8eBbP5rkCNGJIgDiuFsagGoj8vxQAjflCkQCsfWtbx9cw\/7ibeHfmCFqnFqUdaJ2H2OmKztq3xprgVA6t+YHOedB5kqPIEtBIEABcADQ0OtyJMDGghcDbvHlz8b\/SfMmAox1avAoXl00qlgBVJwAMe0igAE0WGpYVYCJg5X8BCAIYQGNMPthh+fGdrMbUFHMdbMiYGAvjYy54vo7JoU\/en+8ZS6r1588HAAo4AMbMWZ1To7WVoivGxaLF8lLgX6Bmt3IQP2MefUCzlmxImWlCa94L\/kJ50OnoA63bU7sNnXl6Ga2b0BnFCNnCWuD8RdbbQOf2dG7bQ9WaLus\/O0CTS8IfNMyHQAPEBDBqw3eKv9g4S0zwCoxweyDMbfyNyWThweAIMy4dz0LfZFJildEGANIhhAADlhW\/ARw+p\/\/YFgIEJgDMc7hozw9Zfbwjrk4WKWOQy42+AFCEOve3KQGlo2HUhwBNCkQZ0zCvXVuGAlhlPeJSJOvNp3VsXAB+KAOuKa3lxmYc0H2gdVsxNXd\/WzpDF0IQvvLSlM7QlnVFv\/A0CuywpruhddNelgSg8fLUb7OBXz5DmAMofA6gIXCV+WSTQwA0gZn88zC9v2cJNwXWAO1DLrsQoDEO3BkAn59lp\/O8GJ89ZdtPcIgRF2sEi8xu2Iy1BcwYM2OJXSluSIACl6a1TgVo9Dup0lLESBBM0I25lBXNmHh3zanNhtRn0AbXtL3a0Jp+4Ts7RwOtm4qo0fua0JkeoEkI0NrQGb5BibLrYaBzN3Ru2suSAbTYBABigJS0MDQqhB4\/MB8MyfcAHgJIlpXfnz0\/DfCzVpjayrqiD74HZAFVLDJ+07\/v5mOBym0htyFjQeNDEPsXsQSdxKzK8dzPguXHaom8K8+TFYkFByDRr1yZALwsK9yfVW5I3pH3scCVA6AxTyFaK16mMdoYWoz5u6A184oAtVb8QOumYmr0vrp05u4QrbugM2taHhHReqBzN3Ru0suSAzSEOy4kP00bYY7WjKtOJ1MjnBF0ABoaelmigKpNMMn0TUCZTDqEOz88F9Aqs67KKlVYd4o0eyU1WJeYQNM\/24vFifUIiHLJEmHxYxkCsAAmY1V\/ijPRPsUFyRyGCr02ATT7bNyk9M1YGTcAzHh9i5bn6Lgf2sXa0heWKP1JM5dFDpijCDAHYn5oiVBibvjeWvqW1igLzK\/GGVI4tAhRGOgPOmFJ22ugdTe01ppWLAueEu1UGk3HxPi0lsIhOeGvaZRGVe0vozP9Q2t4w6f1QOdu6Ky1w3ximIjOimMSI1dyXWhN0w4ZD30UplGfyAmUfORnNjE0DU4JFbyAdTNhqcjlaAULggmhw4VQ9RMwVJGfuBOLR5eSOZgI380pMKENz+UH4cmEA4Ax0FT2IRMvcBGgcR9C2AIHRFR1DMbBuyO8Wdy8Fz8kJkhQ05ZFKqDjGVwAtc4Jq0pAkQWJFWezNtsAmrZQMD9YzVIMmD\/FHBmnSpTpHbUxXlmq0Jwf+mDOtWWCd+NHAkwJQbSBpgUj7+cXbeuwbmnRWv35WqPc1cwdP6I17fAA8ByOLbHXQOt6tGZebXkrbU9B4eFz3IZc0Bl6QT\/owv\/QlN+iNd\/zQx\/iCe7lM\/g6tqZtnJ31yDrTmpYy6dN6oHM9OrN2kMcoEtAM8JHHSOuf9Us75p92tLc5CACalQnasoOMgoa2Xq+NndIuK0CzyRfS3nmBMu3KChllQgqg5FaUYKIvEJ7Fo4lhkjTBTCqTy4Lgb1+IlZnPEAaw8DMnBWhKUBGgWItNbhOICGgBAlhxSttn8TFmhCvjY9EBdjAL78Hn9A9gVl2xDMnULEcl0qgckU7AtlmYjEHvKSBXQWI0M20\/oB3\/K2tVY+ddQvHUqnfDSoNXmA+BlHiCRYAGzv8IUZ4hWisBBIC35bdUnstPwBlovbLYB6iEpxRah4qPI9CgF7xX54IHlYEMnaUY0Rc8QH9K7NCWGda0lD3uSaH1QOf6dBYdWYNYTsw5blzkqfIEfM+UZKTi\/2rnyxQZOzZeLmtOinM2gCb0hlltvb677rqrEDwIICZC1UI0cXaTtbQtuzhkoSE0WYAINiwamDoUQxOoSnCnbN5lDPwQM+M9EOJa7HIvaoM4whHBClHs\/jQWoDaOq1gr38u1qjJR3M\/nMAuCmUXMM1OSQSSgeXc\/8SM1y5H3QhjAqFZIKblFc48wYJ6VGao9fZoHv8SVDu\/kfoEl90o753PmWJYX98vNybwx54yLeWXOuVdxSeYIWmPdMgeKj6bQWklC2iIx0Pqpwv1qebcure+5555CIYOHoC90sLQWnaGbbScLDTDDu6E1jWzQflEAjR\/RDVrDK6ypqjVtac3YhjXdjs5K6mLtsO6RB9ovbL1c0B+wsierKCcBWuqSjNJ2KOhp6cwazwbQEEownU2A4EUsoPlZbLw06C5XQlkZLICARWP3moUATZMm\/33KRm2BlqwrEcAW4IWQ1sXoa6S4w9AiubhPWZ2yTuU\/5jfvIbCgPW0RMlVH0lvB429MbuNyZAyhjc5WQGiPDyAcqq4vLY2+ECbwgy1GG6M1ggyaMreq5cl86F1l4dt9RnJt8hwBZOyMNRalFhfCcaB1e1qzpqEvgok5Zd2L1mVrWpq7aK01zbrBSoSuUuzkihKt5bVBkw8VwYbvLK0BT+sxGdb03Aw0WdPQBFpbOpdZ5D6dfUCzSWE+nbMBtNgLlgGakjyYLAQ9gg0AQCOIFSquAjSBkFx6sYxJjRdrC+1ClohdLIAU32kx2RRw\/31lTsfmgf7tRmoWn5JHUt2joSxBn1n4PzVtX3PJ\/If2gNktECo4G9sbWAVoMVorzhqyOGWJy0KDltALugBkSjLhtzaY+\/sGrTsEHhto3Z7WZYBWtqZFN5\/WIUCDjy2t+V\/0RvEDFGO0xtpnrQxr+pmLxFGTNd0E0BRusQZFTH6NrOl9oYyIMvjs+TsJel8Qkg2IC0nCCsuOAK6Cy3I9+MOtAjRpaoBF1dlbKlelKiX+njcJQwQhbWRuxwopW4tOmiSaIu\/qu+pU5ofxph5qOgkLTdYT76ZSYXUttDJaS0v3y57B\/CFAEz8AsPAP80gfWHLwkJ\/NKK0dBUlZnCG+GGh977ylXEXrGKDB42VrWt6KVECztMayhubil5ByKSXRAt+wpleOiFBrPVXR2SqDdS20OoBm6ZylhSaXEUAAgyMIBWhK++Q3jKeyUwhOBDY\/dlH4FlYKoAlEcUmpCC7E43OEnlLRRbAQUCk+g6AEXAFaVfyIHVCp4LqsTJ6DUObZ9KF4mrRPxlMH0FJiaE0stJDLU+BJf8whz8Y9GCoHpoQRZasi2JS5CA\/wuWjNnKhkF32poLMfaPYBDQEmZceeeG0tYwWbLa1xk7C4lGU50Hqxe7surW0MDa+DzjOUFaU1rQ3TUjjkPvTdw76FBo+olqpoLTorLkcbYq52TSu7ThmzfhWhYU3PxcXrrGnWDjRmrcot7IcnoDNrWgkkPCMF0JAbPp2zBDSZlggYXhKGVfUInZ+lDEYBFpOAcEPwI5D4m3tUukgongJoNkECYgAkssbkv6X\/GKD46amMAaIpcE1\/NtipsfEMLAZlXsqd5sedbP8SAqmHm1ZlgM0qcAAABE1JREFUOTYBNO2Ts5YXdMKikSISy3LkecqI4n0RXtwDoAngLK0F+rKS7rzzzsK6IshMjJW2XD6gMeeMh4VilQPaKQ6LqwnXqWhNP3JgqCSbbw0PtHaFB6EOraE3tGZ9AypYRspqVv1M5t7Smv+Za9ohEO3+Rh\/QoJESEkRrrWn4imdrL6ti0TyX50mRGug8t\/e17ZpGOZVRIIXBZjn7dEbZj22s9l2OgKRP56wBTZtfbRAfRlTZKmUgAhT411kgIgBaYyj4mwJo1u3IpDHJCFcYnmcAHghBhKOeqaw9FjaBSsYuH7ySNbSnxVqWIQ8u76LUc75HWOPz51m2fxiOBQ6hNc6qc+EYG\/2HLBrGl1LLUZmO0EFp+xqn3SDrjym0D00b2u1WC\/qSJq34Fv1CO35sv2j70MHX2kMuR4EnvwFNlCXAi7Y8DxrKmoTWNmmEewZa3z9\/vI\/2EWnPYSqttR1D7kPt8dRWGujCuvZpTTtt7\/ALCIRiaLKmLK21Pw1eYz0xBq1pZAVrA34b6NwNnaWIyrNk1z\/KY4jOzH0qoMGDPp2zBDTFjWBG7SdicpTNCCMyGTCm9YvD9FootEU4+\/UcUwHN1oezfcr3znOx0PhfwMKCsBufWTQ285B7AA3tkWG8aJCAC9+x4Lhfm0VZZPzNmGmnuI+sGIQv38lqtfHEEFBKy2UR+xtIU\/eh0YfSn\/kbQOOylUIkFOzuf43H7qezIGb3iwFy8IC0dvsuvCNAI0uMtspkhA5y58ZiaCp4LE2cfgRcGoP4TpVCcDPzjIHWzWmtdSyPgjwporNobEP6Pq21Zcev2xlLCgnRWv37a5r\/SS5ASGKtDWu6+ZqWEhqqFFSnUkhsa5HNcoRvRuicY1KI3EBKnQcUQHcGrpiJLU8ln7ef38L\/TKp176UCGhOlo2O02GD6UAYgz0GwK+sQQYgWGbKWaCsXjQAIMMD1oXIw3IerUxk+CG3tTeN+BKuqhDA2mASQ4n7t0YkBmkCIe8q2EZTdr+9s6as655P5fWvurNtU7leVLqoazzhobZMQRMuB1uG0\/Sr6iO+gZ250JhFIHhHtgR3o3JzOKbwwrjbZWGhYIlgNgJOsBYDBpoPzOSamYkxyTWHSWvBQCaPQpPEcuQTx5cb2pOheCVasPdqWFT6W+8KvNRYaB++gqhmynPgMQApt5g4lpdh+WYC4Equ2GUijwU1blcVZxXRdARrvhsVUtudPm6T9MY2L1ko4CWXDMdaB1qPZb1W8wvc50pk1jcxBJvi0HuicfohuCv37aJMNoGn\/CVoxwgRhiYCz9b5klfE5gl+bsRHmuAlie8\/6mMhpe4Z\/HlqT8XcFaE2ePe57VHqsjeU57jGm9j\/QunymlgqtBzq7fCqF4EqU1YIm5x\/0iTWBNuVbVMRklPabmumXKgiWcjuV0iJmENtGUPX+SxXQFGgOVZKpmpMcvx9oHafKUqL1QGfn\/j+OeOJ5G7lOLQAAAABJRU5ErkJggg==","height":130,"width":324}}
%---
%[output:1c602238]
%   data: {"dataType":"text","outputData":{"text":"\n--- 自动微分梯度验证 ---\n1. Critic梯度验证:\n   Critic梯度最大绝对误差: 2.946257e-01\n\n2. Actor梯度验证:\n   Actor梯度最大绝对误差: 2.873561e-10\n","truncated":false}}
%---
%[output:2b3c4a33]
%   data: {"dataType":"text","outputData":{"text":"示例网络创建完成。\n","truncated":false}}
%---
%[output:267886a1]
%   data: {"dataType":"text","outputData":{"text":"网络输出 (loss) 的值: 1.0737\n","truncated":false}}
%---
%[output:49a6034d]
%   data: {"dataType":"text","outputData":{"text":"梯度 d(loss)\/dK 的大小: [10 1]\n","truncated":false}}
%---
%[output:951ca7ed]
%   data: {"dataType":"text","outputData":{"text":"梯度 d(loss)\/dK 的前几个元素:\n","truncated":false}}
%---
%[output:8087e997]
%   data: {"dataType":"text","outputData":{"text":"    0.5192\n   -0.3357\n    0.3419\n    0.0981\n    0.3685\n\n","truncated":false}}
%---
