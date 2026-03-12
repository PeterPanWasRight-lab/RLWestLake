% dpg_lqr_fast_autodiff.m
% 基于自动微分的高速DPG算法（向量化+预分配）
%% todo  收敛不了
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
fprintf('--- LQR最优解 (Baseline) ---\n');
disp(K_lqr);

%% 2. Actor-Critic参数初始化
rng(42); % 固定随机种子

% Actor网络初始化（线性，无偏置）
actor_layers = [
    featureInputLayer(2, 'Name', 'state_in')
    fullyConnectedLayer(1, 'Name', 'action_out')
];
actorNet = dlnetwork(actor_layers);
% 强制偏置为0以保持纯线性
is_bias = strcmp(actorNet.Learnables.Parameter, "Bias");
actorNet.Learnables.Value{is_bias} = dlarray(0);

% Critic网络初始化（简单MLP）
critic_layers = [
    featureInputLayer(3, 'Name', 'state_action_in')  % 输入: [x1, x2, u]
    fullyConnectedLayer(16, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(1, 'Name', 'q_value_out')
];
criticNet = dlnetwork(critic_layers);

% 学习参数设定
alpha_critic = 0.001;   % Critic学习率
alpha_actor = 0.0001;   % Actor学习率
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

fprintf('--- 开始Model-Free DPG训练 (高速自动微分版本) ---\n');

%% 3. Actor-Critic联合优化循环
tic;
for epoch = 1:num_epochs
    %% Step 1: 收集交互数据（Experience Replay）
    % 生成随机状态（重用预分配数组）
    X_batch(:) = randn(2, batch_size);
    
    for i = 1:batch_size
        x = X_batch(:, i);
        % 执行带探索噪声的动作（行为策略）
        dlx = dlarray(x, 'CB');  % 转换为dlarray，维度为通道×批次
        u_policy = extractdata(predict(actorNet, dlx));
        % 调试输出
        if epoch == 1 && i == 1
            fprintf('调试: x = [%.4f, %.4f], u_policy = %.4f\n', x(1), x(2), u_policy);
            % 检查网络权重
            is_weights = strcmp(actorNet.Learnables.Parameter, "Weights");
            weights = extractdata(actorNet.Learnables.Value{is_weights});
            fprintf('调试: actor网络权重 = [%.4f, %.4f]\n', weights(1), weights(2));
        end
        u_noise = u_policy + sigma_noise * randn();
        
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
    X_batch_dl = dlarray(X_batch, 'CB');  % 通道×批次
    U_batch_dl = dlarray(U_batch, 'CB');  % 通道×批次
    C_batch_dl = dlarray(C_batch, 'CB');  % 通道×批次
    X_next_batch_dl = dlarray(X_next_batch, 'CB');  % 通道×批次
    
    for iter_c = 1:5 % Critic迭代次数
        % 计算Critic损失和梯度
        [grad_critic, critic_loss] = dlfeval(@critic_loss_wrapper_nn, criticNet, ...
            X_batch_dl, U_batch_dl, C_batch_dl, X_next_batch_dl, actorNet, gamma);

        % 梯度下降更新Critic网络参数
        % 调试：检查梯度格式（仅第一个epoch显示）
        if epoch == 1
            fprintf('调试: grad_critic类型 = %s, 大小 = %s\n', ...
                class(grad_critic), mat2str(size(grad_critic)));
        end
        if istable(grad_critic)
            if epoch == 1
                fprintf('调试: grad_critic是表格，有%d行\n', height(grad_critic));
                % 计算梯度范数
                grad_norm = 0;
                for idx = 1:height(grad_critic)
                    grad_norm = grad_norm + sum(extractdata(grad_critic.Value{idx}).^2, 'all');
                end
                grad_norm = sqrt(grad_norm);
                fprintf('调试: grad_critic范数 = %.6e\n', grad_norm);
            end
            % 遍历表格并更新
            for idx = 1:height(grad_critic)
                criticNet.Learnables.Value{idx} = criticNet.Learnables.Value{idx} - alpha_critic * grad_critic.Value{idx};
            end
        else
            % 直接更新（如果是dlarray或其他类型）
            criticNet.Learnables = criticNet.Learnables - alpha_critic * grad_critic;
        end
    end
    critic_loss_history(epoch) = extractdata(critic_loss);
    
    %% Step 3: 更新Actor（策略改进）- 使用自动微分
    % 使用自动微分计算梯度
    [grad_actor, actor_loss] = dlfeval(@actor_loss_wrapper_nn, actorNet, criticNet, X_batch_dl);

    % 梯度下降更新Actor网络参数
    % 调试：检查梯度格式（仅第一个epoch显示）
    if epoch == 1
        fprintf('调试: grad_actor类型 = %s, 大小 = %s\n', ...
            class(grad_actor), mat2str(size(grad_actor)));
    end
    if istable(grad_actor)
        if epoch == 1
            fprintf('调试: grad_actor是表格，有%d行\n', height(grad_actor));
            % 计算梯度范数
            grad_norm = 0;
            for idx = 1:height(grad_actor)
                grad_norm = grad_norm + sum(extractdata(grad_actor.Value{idx}).^2, 'all');
            end
            grad_norm = sqrt(grad_norm);
            fprintf('调试: grad_actor范数 = %.6e\n', grad_norm);
        end
        % 遍历表格并更新
        for idx = 1:height(grad_actor)
            actorNet.Learnables.Value{idx} = actorNet.Learnables.Value{idx} - alpha_actor * grad_actor.Value{idx};
        end
    else
        % 直接更新（如果是dlarray或其他类型）
        actorNet.Learnables = actorNet.Learnables - alpha_actor * grad_actor;
    end
    
    % 记录与最优解的误差（从神经网络提取权重）
    % 提取actor网络的权重（忽略偏置，因为已强制为0）
    is_weights = strcmp(actorNet.Learnables.Parameter, "Weights");
    K_actor_nn = extractdata(actorNet.Learnables.Value{is_weights});
    error_history(epoch) = norm(K_actor_nn - K_lqr, 'fro');
    actor_loss_history(epoch) = extractdata(actor_loss);
    
    % 每100轮显示进度
    if mod(epoch, 100) == 0
        fprintf('Epoch %d/%d: 误差=%.4f, Critic损失=%.4f, Actor损失=%.4f\n', ...
                epoch, num_epochs, error_history(epoch), critic_loss_history(epoch), actor_loss_history(epoch));
    end
end
training_time = toc;
fprintf('训练时间: %.2f秒\n', training_time);

%% 4. 结果输出与可视化
fprintf('\n--- DPG (Model-Free) 收敛结果 (高速自动微分版本) ---\n');
disp('最终学习得到的增益 K_actor (从神经网络提取):');
is_weights = strcmp(actorNet.Learnables.Parameter, "Weights");
K_actor_nn = extractdata(actorNet.Learnables.Value{is_weights});
disp(K_actor_nn);
fprintf('与理论最优 K_lqr 的最终误差: %e\n\n', error_history(end));

% 绘制收敛曲线
figure('Name', 'Model-Free DPG 收敛曲线 (高速自动微分版本)', 'Color', 'w', 'Position', [100, 100, 1000, 400]);

subplot(1, 3, 1);
semilogy(1:num_epochs, error_history, 'LineWidth', 2, 'Color', [0.8500, 0.3250, 0.0980]);
xlabel('迭代次数 (Epochs)', 'FontSize', 12);
ylabel('对数误差 log||K_{Actor} - K_{LQR}||_F', 'FontSize', 12);
title('DPG收敛过程', 'FontSize', 14);
grid on;

subplot(1, 3, 2);
semilogy(1:num_epochs, critic_loss_history, 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]);
xlabel('迭代次数 (Epochs)', 'FontSize', 12);
ylabel('Critic损失 (对数)', 'FontSize', 12);
title('Critic损失收敛', 'FontSize', 14);
grid on;

subplot(1, 3, 3);
semilogy(1:num_epochs, actor_loss_history, 'LineWidth', 2, 'Color', [0.4660, 0.6740, 0.1880]);
xlabel('迭代次数 (Epochs)', 'FontSize', 12);
ylabel('Actor损失 (对数)', 'FontSize', 12);
title('Actor损失收敛', 'FontSize', 14);
grid on;

% 保存收敛曲线图
saveas(gcf, 'dpg_convergence_fast_autodiff.png');

%% 辅助函数：高速Critic损失计算（向量化+避免循环）
function [gradients, critic_loss] = critic_loss_wrapper_nn(criticNet, X_batch_dl, U_batch_dl, ...
    C_batch_dl, X_next_batch_dl, actorNet, gamma)

    % 调试：检查输入维度（仅第一个epoch显示）
    % fprintf('调试: X_batch_dl大小 = %s, U_batch_dl大小 = %s\n', ...
    %     mat2str(size(X_batch_dl)), mat2str(size(U_batch_dl)));
    % fprintf('调试: C_batch_dl大小 = %s, X_next_batch_dl大小 = %s\n', ...
    %     mat2str(size(C_batch_dl)), mat2str(size(X_next_batch_dl)));

    % 构建当前状态-动作输入
    Z = [X_batch_dl; U_batch_dl];
    dlZ = dlarray(Z, 'CB');

    % 神经网络预测当前Q值
    Q_pred = forward(criticNet, dlZ);

    % 计算下一状态的动作（使用Actor网络，不追踪梯度）
    dlX_next = dlarray(X_next_batch_dl, 'CB');
    U_next_val = extractdata(forward(actorNet, dlX_next));  % 不追踪梯度
    U_next = dlarray(U_next_val, 'CB');  % 重新包装为dlarray但不追踪梯度
    Z_next = [X_next_batch_dl; U_next];
    dlZ_next = dlarray(Z_next, 'CB');

    % 计算目标Q值（使用当前Critic网络，但不追踪梯度）
    Q_next_val = extractdata(forward(criticNet, dlZ_next));  % 不追踪梯度
    Q_next = dlarray(Q_next_val, 'CB');  % 重新包装为dlarray但不追踪梯度
    Q_target = C_batch_dl + gamma * Q_next;

    % 计算TD误差损失
    delta = Q_pred - Q_target;
    critic_loss = 0.5 * mean(delta.^2);

    % 计算梯度
    gradients = dlgradient(critic_loss, criticNet.Learnables);
end

%% 辅助函数：神经网络Actor损失计算
function [gradients, actor_loss] = actor_loss_wrapper_nn(actorNet, criticNet, X_batch_dl)

    % Actor网络预测动作
    U_policy = forward(actorNet, X_batch_dl);

    % 构建状态-动作输入给Critic
    Z = [X_batch_dl; U_policy];
    dlZ = dlarray(Z, 'CB');

    % Critic网络评估Q值
    Q_vals = forward(criticNet, dlZ);

    % 平均Q值作为损失（最小化代价）
    actor_loss = mean(Q_vals);

    % 计算梯度
    gradients = dlgradient(actor_loss, actorNet.Learnables);
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
run_gradient_validation;  % 注释掉，因为网络架构已更改

% 本示例展示如何计算神经网络输出相对于其输入的梯度（自动微分）
% 1. 创建一个简单的示例网络
% 2. 将输入数据包装为 dlarray
% 3. 使用 dlfeval 和 dlgradient 计算梯度
% 4. 运行测试

% %% 清理工作区与命令窗口
% clear; clc;
% 
% %% 1. 创建一个简单的全连接神经网络作为示例
% % 这里使用深度学习工具箱创建一个简单的网络
% % 如果您的实际网络结构不同，可以替换此部分
% layers = [
%     imageInputLayer([10 1 1], 'Name', 'input', 'Normalization', 'none') % 假设输入大小为 10x1x1
%     fullyConnectedLayer(5, 'Name', 'fc1')
%     reluLayer('Name', 'relu1')
%     fullyConnectedLayer(1, 'Name', 'output') % 单输出，作为 loss
% ];
% % 转换为 dlnetwork 对象以支持自动微分
% net = dlnetwork(layers);
% fprintf('示例网络创建完成。\n');
% 
% %% 2. 定义一个函数来计算网络输出相对于输入的梯度
% % 这个函数将被 dlfeval 调用
% function [loss, gradient] = modelGradients(net, K)
%     % 前向传播：计算网络输出，即 loss = net(K)
%     loss = forward(net, K);
%     % 计算梯度：loss 相对于输入 K 的梯度
%     gradient = dlgradient(loss, K);
% end
% 
% %% 3. 准备测试数据并计算梯度
% % 创建一个 dlarray 作为输入 K，并启用梯度跟踪
% K = dlarray(randn(10, 1, 1, 'single'), 'SSCB'); % 'SSCB' 对应空间-空间-通道-批次维度
% % 将 K 设置为需要梯度
% K = dlarray(K, 'SSCB'); % dlarray 自动跟踪用于 dlgradient 的变量
% 
% % 使用 dlfeval 调用梯度计算函数
% % dlfeval 会自动设置梯度计算所需的环境
% [loss_value, dL_dK] = dlfeval(@modelGradients, net, K);
% 
% %% 4. 提取并显示结果
% % 从 dlarray 中提取数据
% loss_value_extracted = extractdata(loss_value);
% dL_dK_extracted = extractdata(dL_dK);
% 
% fprintf('网络输出 (loss) 的值: %.4f\n', loss_value_extracted);
% fprintf('梯度 d(loss)/dK 的大小: %s\n', mat2str(size(dL_dK_extracted)));
% fprintf('梯度 d(loss)/dK 的前几个元素:\n');
% disp(dL_dK_extracted(1:min(5, numel(dL_dK_extracted))));

%% 关键点说明
% 1. **dlarray**: 必须将输入数据包装为 dlarray 并指定维度顺序（如 'SSCB'），
%    这样才能启用自动微分跟踪。
% 2. **dlgradient**: 用于计算标量输出（此处为 loss）相对于一个或多个输入（此处为 K）的梯度。
%    要求输出必须是标量。如果您的 net 输出多维，需先汇总为标量（例如使用 sum 或 mean）。
% 3. **dlfeval**: 推荐使用 dlfeval 来调用包含 dlgradient 的函数，它会正确处理计算图。
% 4. **适用性**: 此方法适用于任何由 dlnetwork 表示的网络，无论其层结构多复杂。
% 5. **扩展**: 若要计算 loss 相对于网络内部参数的梯度，只需在 dlgradient 中列出这些参数即可，
%    例如: gradient = dlgradient(loss, [net.Learnables; K]);

