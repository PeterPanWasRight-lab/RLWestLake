% 清除工作区和图形
clear; close all; clc;

% 设置随机种子以确保可重复性
rng(42);

% 并行化选项设置
useGPU = true;      % 是否使用GPU加速（如果可用）
useParallel = false; % 是否使用parfor并行处理批次（仅CPU模式）

% 检查GPU可用性
if useGPU 
    executionEnvironment = "gpu";
    fprintf('GPU可用，使用GPU进行训练。\n');
    % GPU模式下禁用parfor（避免数据转移开销）
    useParallel = false;
else
    executionEnvironment = "cpu";
    fprintf('使用CPU进行训练。\n');
    % 在CPU模式下，可以选择是否使用parfor
    % useParallel = true; % 取消注释以启用CPU并行模式
end

% 如果使用parfor并行处理，检查并行池状态
if useParallel
    try
        pool = gcp('nocreate');
        if isempty(pool)
            parpool;
            fprintf('已启动并行池，工作进程数：%d\n', pool.NumWorkers);
        else
            fprintf('并行池已存在，工作进程数：%d\n', pool.NumWorkers);
        end
    catch
        fprintf('无法启动并行池，将使用串行训练。\n');
        useParallel = false;
    end
end

% 生成训练数据：三次多项式 y = x^3 - 2*x^2 + x + 1 加上噪声
x = linspace(-3, 3, 500)'; % 输入数据，500个点从-3到3
y_true = x.^3 - 2*x.^2 + x + 1; % 真实三次多项式
noise = 0.1 * randn(size(x)); % 高斯噪声
y = y_true + noise; % 带噪声的训练目标

% 将数据转换为dlarray以用于深度学习
% 注意：需要将列向量转置为行向量以符合'CB'格式（通道×批次）
x_dl = dlarray(x', 'CB'); % 转置为行向量：1×500（1个通道，500个批次）
y_dl = dlarray(y', 'CB'); % 转置为行向量：1×500（1个通道，500个批次）

% 如果使用GPU，将数据转换为gpuArray
if executionEnvironment == "gpu"
    x_dl = gpuArray(x_dl);
    y_dl = gpuArray(y_dl);
end

% 定义神经网络层
layers = [
    featureInputLayer(1) % 输入层，特征维度为1
    fullyConnectedLayer(10) % 全连接层，10个神经元
    reluLayer % ReLU激活函数
    fullyConnectedLayer(1) % 输出层，1个神经元
];
net = dlnetwork(layers); % 创建dlnetwork

% 训练参数
numEpochs = 1000; % 训练轮数
learningRate = 0.01; % 学习率
batchSize = 10; % 批大小
lambda = 0.001; % L2正则化系数

% 定义模型损失函数（用于dlfeval）
function [loss, gradients] = modelLoss(net, X, Y, lambda)
    % 前向传播
    Y_pred = forward(net, X);

    % 计算均方误差损失
    mse_loss = mean((Y - Y_pred).^2);

    % 计算L2正则化项
    l2_loss = 0;
    learnables = net.Learnables;
    for i = 1:height(learnables)
        l2_loss = l2_loss + sum(learnables.Value{i}(:).^2);
    end

    % 总损失
    loss = mse_loss + lambda * l2_loss;

    % 计算梯度
    gradients = dlgradient(loss, net.Learnables);
end

% 训练循环
numSamples = numel(x);
lossHistory = zeros(numEpochs, 1); % 记录损失历史

% 将数据转换为普通数组以便打乱（保持行向量形状以简化后续处理）
x_data = x'; % 转置为行向量：1×500
y_data = y'; % 转置为行向量：1×500

% 如果使用GPU，将数据转换为gpuArray
if executionEnvironment == "gpu"
    x_data = gpuArray(x_data);
    y_data = gpuArray(y_data);
end

% 开始计时
tic;

for epoch = 1:numEpochs
    % 随机打乱数据（使用普通数组）
    idx = randperm(numSamples);
    x_shuffled = x_data(idx); % 普通数组打乱
    y_shuffled = y_data(idx); % 普通数组打乱

    epochLoss = 0;
    numBatches = 0;

    % 计算批次索引
    batchIndices = 1:batchSize:numSamples;
    numBatchesTotal = length(batchIndices);

    % 预分配梯度存储（仅当使用parfor时需要）
    if useParallel
        % 获取可学习参数的结构信息
        learnables = net.Learnables;
        numLearnables = height(learnables);
        % 创建单元格数组存储每个批次的梯度
        allGradients = cell(numBatchesTotal, 1);
        batchLosses = zeros(numBatchesTotal, 1);
    end

    % 分批次训练
    if useParallel
        % 使用parfor并行处理批次
        parfor batchIdx = 1:numBatchesTotal
            batchStart = batchIndices(batchIdx);
            batchEnd = min(batchStart + batchSize - 1, numSamples);

            % 提取当前批次数据并转换为dlarray
            x_batch_data = x_shuffled(batchStart:batchEnd);
            y_batch_data = y_shuffled(batchStart:batchEnd);

            % 转换为dlarray，保持'CB'格式
            x_batch = dlarray(x_batch_data, 'CB');
            y_batch = dlarray(y_batch_data, 'CB');

            % 使用dlfeval计算损失和梯度
            [loss, gradients] = dlfeval(@modelLoss, net, x_batch, y_batch, lambda);
            batchLosses(batchIdx) = extractdata(loss);
            allGradients{batchIdx} = gradients;
        end

        % 聚合损失和梯度
        epochLoss = sum(batchLosses);
        numBatches = numBatchesTotal;

        % 平均梯度并更新网络参数
        if numBatches > 0
            % 初始化平均梯度
            avgGradients = net.Learnables;
            for i = 1:height(avgGradients)
                avgGradients.Value{i} = 0 * avgGradients.Value{i};
            end

            % 累加所有批次的梯度
            for batchIdx = 1:numBatches
                gradients = allGradients{batchIdx};
                for i = 1:height(gradients)
                    avgGradients.Value{i} = avgGradients.Value{i} + gradients.Value{i};
                end
            end

            % 计算平均梯度
            for i = 1:height(avgGradients)
                avgGradients.Value{i} = avgGradients.Value{i} / numBatches;
            end

            % 使用平均梯度更新网络参数
            for i = 1:numel(net.Learnables.Value)
                net.Learnables.Value{i} = net.Learnables.Value{i} - learningRate * avgGradients.Value{i};
            end
        end
    else
        % 串行训练（原逻辑）
        for batchStart = 1:batchSize:numSamples
            batchEnd = min(batchStart + batchSize - 1, numSamples);

            % 提取当前批次数据并转换为dlarray
            x_batch_data = x_shuffled(batchStart:batchEnd);
            y_batch_data = y_shuffled(batchStart:batchEnd);

            % 转换为dlarray，保持'CB'格式（x_batch_data和y_batch_data已经是行向量）
            x_batch = dlarray(x_batch_data, 'CB'); % 形状为[1, batchSize]
            y_batch = dlarray(y_batch_data, 'CB'); % 形状为[1, batchSize]

            % 使用dlfeval计算损失和梯度
            [loss, gradients] = dlfeval(@modelLoss, net, x_batch, y_batch, lambda);
            epochLoss = epochLoss + extractdata(loss);
            numBatches = numBatches + 1;

            % 手动更新参数（梯度下降）
            for i = 1:numel(net.Learnables.Value)
                net.Learnables.Value{i} = net.Learnables.Value{i} - learningRate * gradients.Value{i};
            end
        end
    end

    % 记录平均损失
    avgLoss = epochLoss / numBatches;
    lossHistory(epoch) = avgLoss;

    % 可选：每100轮打印损失
    if mod(epoch, 100) == 0
        fprintf('Epoch %d, Loss: %.6f\n', epoch, avgLoss);
    end
end

% 结束计时并显示训练时间
trainingTime = toc;
fprintf('训练完成，耗时: %.2f 秒\n', trainingTime);
fprintf('执行环境: %s, 并行模式: %s\n', executionEnvironment, string(useParallel));

% 测试拟合效果
y_pred_test = forward(net, x_dl); % 使用训练好的网络预测
y_pred_test_np = extractdata(y_pred_test)'; % 转换为普通数组并转置为列向量以匹配y的形状

% 计算均方误差
mse_test = mean((y - y_pred_test_np).^2);
fprintf('最终测试MSE: %.6f\n', mse_test);

% 绘制结果
figure('Position', [100, 100, 1200, 400]);

% 子图1：拟合结果对比
subplot(1, 2, 1);
plot(x, y, 'b.', 'MarkerSize', 10); % 原始数据点
hold on;
plot(x, y_true, 'g-', 'LineWidth', 2); % 真实三次多项式曲线
plot(x, y_pred_test_np, 'r-', 'LineWidth', 2); % 神经网络拟合曲线
xlabel('x');
ylabel('y');
legend('带噪声数据', '真实多项式', '神经网络拟合', 'Location', 'best');
title(sprintf('神经网络拟合三次多项式 (%s训练)', executionEnvironment));
grid on;

% 子图2：训练损失曲线
subplot(1, 2, 2);
semilogy(1:numEpochs, lossHistory, 'b-', 'LineWidth', 1.5);
xlabel('训练轮数 (Epoch)');
ylabel('损失 (Loss)');
title('训练损失变化曲线');
grid on;
xlim([1, numEpochs]);

% 显示网络结构信息
fprintf('\n网络结构信息：\n');
disp(net.Layers);
fprintf('可学习参数数量：%d\n', height(net.Learnables));

% 保存训练好的网络（可选）
% save('trained_network_parallel.mat', 'net');