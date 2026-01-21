% 清理环境
clear; clc; close all;

% 数据生成
x = (-5:0.1:5)';
% 尝试不同的函数
% y = x + 1;  % 线性
y = x.^3 + 2*x.^2 + 1;  % 非线性
% y = sin(x);  % 周期函数
% y = 1 ./ (1 + exp(-x));  % sigmoid形状

n = length(x);

% 网络结构配置
layer_sizes = [1, 5, 5, 1];  % 更大的网络
learning_rate = 0.001;
epochs = 20000;

% 初始化参数（使用Xavier初始化）
L = length(layer_sizes);
W = cell(L-1, 1);
B = cell(L-1, 1);

for i = 1:L-1
    W{i} = randn(layer_sizes(i+1), layer_sizes(i)) * sqrt(2/layer_sizes(i));
    B{i} = zeros(layer_sizes(i+1), 1);
end

% 激活函数
activation = @(x) max(0, x);  % ReLU
% activation = @(x) x;  % x
activation_deriv = @(x) double(x > 0);

% 训练循环
loss_history = zeros(epochs, 1);
figure(1); clf; %[output:7a62d76a]

for epoch = 1:epochs %[output:group:96b7ad65]
    % 前向传播
    Z = cell(L-1, 1);
    A = cell(L, 1);
    A{1} = x';  % 输入
    
    for i = 1:L-2
        Z{i} = W{i} * A{i} + B{i};
        A{i+1} = activation(Z{i});
    end
    
    % 输出层（线性激活）
    Z{L-1} = W{L-1} * A{L-1} + B{L-1};
    A{L} = Z{L-1};
    
    % 计算损失
    y_pred = A{L}';
    loss = mean((y_pred - y).^2);
    loss_history(epoch) = loss;
    
    % 反向传播
    dW = cell(L-1, 1);
    dB = cell(L-1, 1);
    
    % 输出层梯度
    dZ = (A{L} - y') / n;
    dW{L-1} = dZ * A{L-1}';
    dB{L-1} = sum(dZ, 2);
    
    % 隐藏层梯度
    for i = L-2:-1:1
        dZ = (W{i+1}' * dZ) .* activation_deriv(Z{i});
        dW{i} = dZ * A{i}';
        dB{i} = sum(dZ, 2);
    end
    
    % 更新参数
    for i = 1:L-1
        W{i} = W{i} - learning_rate * dW{i};
        B{i} = B{i} - learning_rate * dB{i};
    end
    
    % 动态显示
    if mod(epoch, 500) == 0
        fprintf('Epoch %d, Loss: %.6f\n', epoch, loss); %[output:0f8b854c]
        
        % 实时绘图
        subplot(2,2,1); %[output:7a62d76a]
        plot(x, y, 'b-', 'LineWidth', 2); hold on; %[output:7a62d76a]
        plot(x, y_pred, 'r--', 'LineWidth', 2); hold off;
        xlabel('x'); ylabel('y');
        title(sprintf('拟合结果 (Epoch %d)', epoch));
        legend('真实值', '预测值');
        grid on;
        
        subplot(2,2,2);
        semilogy(loss_history(1:epoch)); %[output:7a62d76a]
        xlabel('Epoch'); ylabel('Loss (log)');
        title('损失曲线'); grid on;
        
        subplot(2,2,3);
        error = y_pred - y;
        plot(x, error); %[output:7a62d76a]
        xlabel('x'); ylabel('误差');
        title('预测误差'); grid on;
        
        subplot(2,2,4);
        bar([W{1}(:); W{2}(:); W{3}(:); B{1}(:); B{2}(:); B{3}(:)]); %[output:7a62d76a]
        xlabel('参数索引'); ylabel('值');
        title('参数分布'); grid on;
        
        drawnow;
    end
end %[output:group:96b7ad65]

% 最终评估
y_pred_final = y_pred;
fprintf('\n训练完成！\n'); %[output:5ac59a31]
fprintf('最终损失: %.6f\n', loss); %[output:91f98de3]
fprintf('最大绝对误差: %.6f\n', max(abs(y_pred_final - y))); %[output:472f67e9]
fprintf('平均绝对误差: %.6f\n', mean(abs(y_pred_final - y))); %[output:23b136cc]

%%
% 清理环境
clear; clc; close all;

% 数据生成
x = (-5:0.1:5)';
y = x.^3 + 2*x.^2 + 1;  % 与原始代码相同的非线性函数

% 准备数据 - MATLAB工具箱要求特定的数据格式
% 输入数据应为 numFeatures × numObservations
% 输出数据应为 numResponses × numObservations
inputData = x';  % 1×101 (特征×观测值)
targetData = y'; % 1×101 (响应×观测值)

% 将数据分为训练集和验证集（70%训练，30%验证）
rng(42);  % 设置随机种子以保证可重复性
n = length(x);
indices = randperm(n);
trainRatio = 0.7;
trainIndices = indices(1:round(trainRatio*n));
valIndices = indices(round(trainRatio*n)+1:end);

XTrain = inputData(:, trainIndices);  % 1×71
YTrain = targetData(:, trainIndices); % 1×71
XVal = inputData(:, valIndices);      % 1×30
YVal = targetData(:, valIndices);     % 1×30

% 显示数据维度以供调试
fprintf('数据维度检查:\n'); %[output:9945c17f]
fprintf('XTrain: %d×%d\n', size(XTrain,1), size(XTrain,2)); %[output:291985bb]
fprintf('YTrain: %d×%d\n', size(YTrain,1), size(YTrain,2)); %[output:30aad269]
fprintf('XVal: %d×%d\n', size(XVal,1), size(XVal,2)); %[output:891ddc5c]
fprintf('YVal: %d×%d\n', size(YVal,1), size(YVal,2)); %[output:7512053a]

% 方法1: 使用dlnetwork和自定义训练循环（更灵活）
% 创建网络层
layers = [
    featureInputLayer(1, 'Name', 'input')  % 输入层，1个特征
    
    fullyConnectedLayer(5, 'Name', 'fc1')  % 第一个隐藏层，5个神经元
    reluLayer('Name', 'relu1')            % ReLU激活
    
    fullyConnectedLayer(5, 'Name', 'fc2')  % 第二个隐藏层，5个神经元
    reluLayer('Name', 'relu2')            % ReLU激活
    
    fullyConnectedLayer(1, 'Name', 'fc3')  % 输出层，1个神经元（线性）
];

% 将层转换为层图
lgraph = layerGraph(layers);

% 分析网络结构
analyzeNetwork(lgraph);

% 创建dlnetwork（用于自定义训练循环）
dlnet = dlnetwork(lgraph);

% 训练参数
numEpochs = 20000;
learningRate = 0.001;

% 准备训练数据为dlarray格式
XTrain_dl = dlarray(XTrain, 'CB');  % C=通道维度, B=批次维度
YTrain_dl = dlarray(YTrain, 'CB');
XVal_dl = dlarray(XVal, 'CB');
YVal_dl = dlarray(YVal, 'CB');

% 记录损失历史
trainLossHistory = zeros(numEpochs, 1);
valLossHistory = zeros(numEpochs, 1);

% 训练循环
figure('Position', [100, 100, 1200, 800]);

for epoch = 1:numEpochs %[output:group:7a52cde1]
    % 前向传播
    [YPred, state] = forward(dlnet, XTrain_dl);
    
    % 计算损失（均方误差）
    loss = mean((YPred - YTrain_dl).^2, 'all');
    
    % 记录训练损失
    trainLossHistory(epoch) = extractdata(loss);
    
    % 计算梯度
    gradients = dlgradient(loss, dlnet.Learnables); %[output:14cee5dd]
    
    % 更新网络参数
    dlnet = dlupdate(@(w, g) w - learningRate * g, dlnet.Learnables, gradients);
    
    % 验证
    YPred_val = forward(dlnet, XVal_dl);
    val_loss = mean((YPred_val - YVal_dl).^2, 'all');
    valLossHistory(epoch) = extractdata(val_loss);
    
    % 每500轮显示一次进度
    if mod(epoch, 500) == 0
        fprintf('Epoch %d, 训练损失: %.6f, 验证损失: %.6f\n', ...
            epoch, trainLossHistory(epoch), valLossHistory(epoch));
        
        % 实时绘图
        % 在所有数据上进行预测
        XAll_dl = dlarray(inputData, 'CB');
        YPred_all = extractdata(forward(dlnet, XAll_dl));
        
        % 子图1: 拟合结果
        subplot(2, 3, 1);
        plot(x, y, 'b-', 'LineWidth', 2); hold on;
        plot(x, YPred_all', 'r--', 'LineWidth', 2); hold off;
        xlabel('x'); ylabel('y');
        title(sprintf('拟合结果 (Epoch %d)', epoch));
        legend('真实值', '预测值', 'Location', 'best');
        grid on;
        
        % 子图2: 损失曲线
        subplot(2, 3, 2);
        semilogy(1:epoch, trainLossHistory(1:epoch), 'b-', 'LineWidth', 1.5); hold on;
        semilogy(1:epoch, valLossHistory(1:epoch), 'r--', 'LineWidth', 1.5); hold off;
        xlabel('Epoch'); ylabel('Loss (log)');
        title('训练和验证损失曲线');
        legend('训练损失', '验证损失');
        grid on;
        
        % 子图3: 预测误差
        subplot(2, 3, 3);
        error = YPred_all' - y;
        plot(x, error, 'g-', 'LineWidth', 1.5);
        xlabel('x'); ylabel('误差');
        title('预测误差');
        grid on;
        
        % 子图4: 训练集vs验证集
        subplot(2, 3, 4);
        plot(x(trainIndices), y(trainIndices), 'bo', 'DisplayName', '训练数据'); hold on;
        plot(x(valIndices), y(valIndices), 'rs', 'DisplayName', '验证数据');
        plot(x, YPred_all', 'r--', 'LineWidth', 1.5, 'DisplayName', '预测值');
        xlabel('x'); ylabel('y');
        title('数据划分和拟合');
        legend('Location', 'best');
        grid on;
        
        % 子图5: 参数可视化
        subplot(2, 3, 5);
        % 获取网络参数
        params = dlnet.Learnables.Value;
        param_values = [];
        for i = 1:length(params)
            param_values = [param_values; extractdata(params{i}(:))];
        end
        bar(1:length(param_values), param_values);
        xlabel('参数索引'); ylabel('值');
        title('网络参数分布');
        grid on;
        
        % 子图6: 网络结构
        subplot(2, 3, 6);
        plot(lgraph);
        title('网络结构');
        
        drawnow;
    end
end %[output:group:7a52cde1]

% 最终预测
XAll_dl = dlarray(inputData, 'CB');
y_pred_toolbox = extractdata(forward(dlnet, XAll_dl))';

% 最终评估
fprintf('\n===== MATLAB深度学习工具箱网络结果 =====\n');
fprintf('最终训练损失: %.6f\n', trainLossHistory(end));
fprintf('最终验证损失: %.6f\n', valLossHistory(end));
fprintf('均方误差(MSE): %.6f\n', mean((y_pred_toolbox - y).^2));
fprintf('最大绝对误差: %.6f\n', max(abs(y_pred_toolbox - y)));
fprintf('平均绝对误差: %.6f\n', mean(abs(y_pred_toolbox - y)));
fprintf('R²决定系数: %.6f\n', 1 - sum((y - y_pred_toolbox).^2) / sum((y - mean(y)).^2));
fprintf('训练数据点数: %d\n', length(trainIndices));
fprintf('验证数据点数: %d\n', length(valIndices));

% 显示网络结构信息
fprintf('\n===== 网络详细信息 =====\n');
fprintf('网络层数: %d\n', length(dlnet.Layers));
for i = 1:length(dlnet.Layers)
    fprintf('\n层 %d:\n', i);
    fprintf('  名称: %s\n', dlnet.Layers(i).Name);
    fprintf('  类型: %s\n', class(dlnet.Layers(i)));
    
    if isa(dlnet.Layers(i), 'nnet.cnn.layer.FullyConnectedLayer')
        fprintf('  输出大小: %d\n', dlnet.Layers(i).OutputSize);
        fprintf('  输入大小: %d\n', dlnet.Layers(i).InputSize);
    elseif isa(dlnet.Layers(i), 'nnet.cnn.layer.FeatureInputLayer')
        fprintf('  输入大小: %d\n', dlnet.Layers(i).InputSize);
    end
end

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":34}
%---
%[output:0f8b854c]
%   data: {"dataType":"text","outputData":{"text":"Epoch 500, Loss: 58.314673\nEpoch 1000, Loss: 35.194764\nEpoch 1500, Loss: 19.471350\nEpoch 2000, Loss: 14.197554\nEpoch 2500, Loss: 11.336804\nEpoch 3000, Loss: 7.537453\nEpoch 3500, Loss: 6.324567\nEpoch 4000, Loss: 4.256906\nEpoch 4500, Loss: 9.567758\nEpoch 5000, Loss: 4.078873\nEpoch 5500, Loss: 4.208449\nEpoch 6000, Loss: 3.689853\nEpoch 6500, Loss: 4.121325\nEpoch 7000, Loss: 2.390338\nEpoch 7500, Loss: 3.428038\nEpoch 8000, Loss: 2.269690\nEpoch 8500, Loss: 1.711168\nEpoch 9000, Loss: 2.993207\nEpoch 9500, Loss: 2.524851\nEpoch 10000, Loss: 1.433164\nEpoch 10500, Loss: 1.657512\nEpoch 11000, Loss: 1.793776\nEpoch 11500, Loss: 2.073732\nEpoch 12000, Loss: 1.839940\nEpoch 12500, Loss: 1.737634\nEpoch 13000, Loss: 1.820852\nEpoch 13500, Loss: 1.805472\nEpoch 14000, Loss: 1.294739\nEpoch 14500, Loss: 1.122665\nEpoch 15000, Loss: 2.157650\nEpoch 15500, Loss: 1.820087\nEpoch 16000, Loss: 1.792435\nEpoch 16500, Loss: 1.659708\nEpoch 17000, Loss: 1.735969\nEpoch 17500, Loss: 1.728679\nEpoch 18000, Loss: 1.699832\nEpoch 18500, Loss: 1.653573\nEpoch 19000, Loss: 1.559495\nEpoch 19500, Loss: 1.036474\nEpoch 20000, Loss: 0.765226\n","truncated":false}}
%---
%[output:7a62d76a]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAccAAAESCAYAAACIDx4uAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnQn8V1P+\/w+FakKSPflGGRn7logiZA2NqGZ+FEN2xpKdsk2W7FtIWWYyMimNPRTJkGQbQvSlFAoh2wj\/\/\/PkfJzv7S7nfj73fj6f+7nv83h8H9\/v9\/M599xz3ue8z+u817PML7\/88ouSIhQQCggFhAJCAaFAgQLLCDjKahAKCAWEAkIBoUBDCgg4yooQCggFhAJCAaGAhwICjrIkhAJCAaGAUEAoIOAoa0AoIBQQCggFhALhFBDJUVaIUEAoIBQQCggFRHKUNSAUEAoIBYQCQgGRHGUNCAWEAkIBoYBQIBYFRK0ai1xSWSggFBAKCAXyQIFcgeNXX32l3nvvPbXxxhurFVZYoeT5pb2ffvpJrbLKKqFtLV68WH377bdqpZVWalCPz\/\/73\/+qJk2aqN\/\/\/vcl90caEAoIBZQyfNWyZUu17rrrxibJDz\/8oJZbbjm17LLLKnj8zTffVO3atVOtWrVyaov3f\/3112rFFVdUjRs3VosWLdLPNW\/ePPB5s0fwzDLLLFOo9\/bbb+u\/eT+lUaNGTn2QSqVTIFfg+O6776pLL71UDR06VME4t9xyi6bg0UcfXaDkd999p84880y1ww47qEMOOSSQwiz4Cy64QDPiJZdcErrw6+vr1V\/+8hfd7m677bbUu9Zee211xhlnLPWujz\/+WI0ZM0bBrGFl5ZVXVgcffHBoH0pfKtKCUCAbFPDyMPxz+eWXq5dffnmpAXTv3r0B\/1PhhhtuUC+++KL+\/cknn6j+\/furK6+8Um2\/\/fZOBGCfGThwoH5n+\/bt1WWXXaaf8+Nx0yC8fswxx6jjjjuuwR7BsxzAV1ttNTVhwgR17rnnqk033dSpH1KpNArkGhxZeN98840+qW233XZqn332US7gyGnyoosuUv\/+97\/1afKKK65QHTt2bHDiM9NCdj6Y7Pnnn1c33nijatasmXrttdc04P3vf\/9Tt912m27DADEnxz\/84Q\/6xAmTnXTSSeqwww5T66yzTqE+zLHzzjvrVzzzzDNq6tSpavjw4Rrww8o\/\/\/lPBVDDpPx91llnBVb\/xz\/+4bwZFLsE6cOUKVPUkCFDVNOmTZ2agSZsVnPnztX199tvv6WeZ16HDRumv998882Xok3U9\/\/5z39U3759C\/3x0sLbh7\/97W+F+bNp7DQgqZQ4Bbw8bP7faqut9Hqh8BmH2vXWW68BaH3xxRcaoDp16qSOP\/54NXPmzLKAI+tm3Lhxeo+wNVEGWE8\/\/XQNjnPmzFGHHnqo3h+kpEuBXIAjIPTggw9qUJo0aZLitNi7d281evRoTV3UrPx97bXXanXKKaecorp16+YrOX700Ufq\/PPPV7Nnz9b1ASbAEcDp1avXUupaNlKkRr7v0qWLVtMgvbLIf\/75Zw1Wyy+\/vGrdurXuCwzMqRO1L8+edtpp+n1t27YtMDRAbJh8\/PjxauzYsZHg6JWaiwGmpJdi3D4YULJP8WweAKUBWNq89957C\/SI+733HQAlczBixAgtBXz++efqiCOO0OuHA423vtmI+\/Tpk\/rhIun5qIX2Hn30Uc3nAAnAt+WWW6q99tpLXXPNNZq3osCR\/eHII4\/UGqGw4j10of6cP3++foQ94tZbb1VHHXWUPtSyJim2JmqjjTYqqGl57oQTTtCgxx4B8MH\/aKfQctHeZpttpt544w312WefaVDfcMMNa2G6qnoMuQBHo1ZBSgGU2rRpoy6++GL1+OOP68lBMkOlcfLJJ2uVBZsfYGarUWgDgEVV0qFDB4W0wMJHMnzyySe1FLb++utr1SmMg70CBgMIAURUsDfffLN655131FVXXaXtj2YjDVKrGmBF2sQuacCUZ439Y8GCBfrvKMkRkKirq2sg4cSV2pJeyXHB0a++rcJaddVVl5o7A2bMJxuKd27t75lvPxWY\/ZlfH7yfAaijRo2KJREnTdu8toep5JFHHtGHTvjE8OT9998fqVbFTnjqqadqeyGaJAqS5AMPPKAP1PC7KV5TBmuEd3DI\/f777\/X74Tf4lj2Hwncc1JFG77zzTr2\/sH\/cdddd6qWXXtJ7Egdu6rA38F4K2iQO6+w7\/ACUXv+FvM53muPOBTgaAnqlJ7PpIamh+uSEduyxx2qgHDRokNp22221+pPT5NVXX604HbIB9+jRYyn7HswA2HCy4znaRLrkN04BMMmnn36q29liiy0KDMApcI011tAnRwpSpDHce20Xfipfr6Tkt1i87VDHBZgMcDBewNeoMr1qRltNCdAbKcv0xavCNWpI0weYnzYpfmrSMAawx0Y929ZjnjMHAyOVG1uQ93vG6WdvtsHuuuuu04\/Z9iPvuhLpMc0tK7ptL58ANsYpxvs0JhWAjt+A1E033aTX+iabbKKr+mkr\/HpgH6DCbI7ewxiaJzRLXbt21c0i9QLw7DPsBQD95MmT9UHrd7\/7nQZTDt1ouKSkSwEBx183uunTp2sVKbYGjN6o7lCjsRjvu+8+DZDYINgov\/zyS99Z2XfffTUTsqABO06MnP4wpsNwqFX4HFWJsYl5GxowYEBh401KcnSRdvwGZBgZtY8BPFvNyEkYMKHYas3rr7++UJ932\/\/bm80HH3ygJW4DllGStF8f7cMBUjmblFeKNhsXKquw71Gn+WkNGDPPcYACWL3OWn6HDxcnjHRZO7+te8HRa0O2KWOrR5EYWZPY\/I3HqOFB9gMOvUGlWHBkzbKG2WcwkXC4O+igg\/T7AXV8Cs455xy15557qhkzZqhXXnlFq2uRcKWkS4HcgCOLj1MZm3GLFi0UIATTeKWAp556Sqs\/sRlEhWi4TI3xaqXu4MGDtVTIZyx8v2JLjsYl3Fy5aZwIbJsjbZjTL6rcqFOt+T7MIcdIb7zPtq+ZZ81G0LNnz6UkNXtjCpLE7D7Y9kE+d5FozfNm0zOSrAGxagDHOONwWUdSx50CZg1ijsDRa4MNNtAqT9YF\/IcTHtIiHqnmsAQPYds3KlDzNkwiaIBoA+2Pt\/AMdk3aQZ3K4XfWrFla88QPvgIcDil8x8H6xBNP1IBnzDbwOaYWPGPZI+g\/du5nn31WP4cNEl7bcccdNXCvtdZazg5s7lSTml4K5AIckXxYkC+88IKW4lBj7L\/\/\/lq6MeAIYLH4UWmwKJESqGtKmDu4TVSkj7333lt\/xDO0g7PPgQceqG0ZGNr5+4knnvBdjfQLOwmOBW+99VaDOj\/++GPB0QAnIm8xz3o\/95NiXDZvrwrIBjVUyDAsjg5eMDIem0GSmLcd21vVpV88b4DR9hStJnAM6otsQelRgJAH\/ALuuOMOHT8MMHFAg5\/4jHVqaxfsvzmUoiGytULY+FFpvvrqq3ovgL+8XtWG52zTgssIzYGOd7A\/oK7v16+fmjdvnrYpYq\/Exsm+xEEe8wufX3jhhWr11VfXWiyJeXShdPF1cgGOgBQqUySw22+\/Xdv9CHtgQfPZrrvuqk9sqAiRGtnYMIwbzzbIa06j2AdNGIVNdoAP1QsqD+OVRjt4oMFQ2BlRy2D3AuQADvs0yqmTBAXYPThResERJoHxsWlSMMzTb1tarBZwNGCcFjj6ASM08VNv8nm5bY4GvP1UuMWzqjzpQgFAh0Pdc889pw444ADNi\/ZBJQgc\/cKg8EGAn9dcc03tRINTHTznV4pVqxqbI23C90iHu+yySyF5AXyP9zwmAXwWOOADpACklHQpkAtwNCT0c8gZOXKkVrEAjKhhsCnhqQaDAXbGOSYq\/tHr4s87Ud+QGQdvUjvrhTc8wGzsQcHGBAijvkECRh0L0BLQjBGfEySG+rBSquRowhbMO5JUq3o9ZqMkR68q1R63n6RbCW9V+hQ1jnTZOt+t+9kczUGF2GTCPNDokPkm6ABjzCGvv\/66NrHgR4BqlsOzXxhFseDIwZ2DNeBsPNyRgE0GLw7vhIqh0cI+ir8CkrCU9CmQW3AErAiwZ2GysLFDonpFukPtCegQ6L\/11lvrWSgGHM1z2BJgMpx+ADjUrqhJbK9OP684O3wEaZUTpHEIwSkEj0nUMOedd56WZoMCg\/1CC1w276Qccmy7ok1H6BMHHF08Bysd5+g9QIRlRUmfvfP5Bj9vVQ6qmCmYD4L84Xu8yDkIw\/u2BobDKGEVmFcIyu\/cubNWbyI5wsfsFxxQ7QNvseDIDBEuwt4AABNeRgIK+oktEocc9iEO2aiFUbXSP+Ik7ffnc6bTHXWuwBGVCqCEWpWFd\/fdd6vDDz9c6+4xnhOqwSkRSQ\/AQaJkkQI6huGQ2PxyLJr4JWwCRq2Kp6rxKmNRw4hIe8QoAb6AsHH6MfFUZMxB9coplb9hSvqAigh1rB1qYIMnfeI71C\/eAsjRj7PPPlt74BrJJixDDra83XffXTvk2KEcfqEapYZyuNocw+w6dniJXa\/cGXKgbZCtNl1WltbNYRT6w+fwwh\/\/+EdtZ8Tm+NBDD+kDJnHNqF3hVfgB0GNdc4g1fIckx0HU9lCFRwFNvErhY\/YO1hdmk2Jtjsb0AlBjxjF2zYkTJ2qPdvYVfCQASWyoJDQgRpL4aYl1THfNVy04Rm249vfeuDvbE5NNnryjMAcMQ25TVBPE1plC5gk2aAAEzzMKqhccdLABcEIrxuaIVMcpDwa0kw7DENjjgsAR2wN1OOni1m36GiS9shngZUugcJCHrbG7heWL9QNVv9CGdJdk9luXJACVm0N4GfMEBS9RDrhoiDhw8jkqUSNxocXBoQw1JgdHVKbY9QE9TAl+5gocaABW1LIk4OA39n\/4C0\/XKP4yvgnsRfA53rBItdg1Tb+wdXJY32OPPbSWiX4A8CadHankyActkmO666wqwdGr8vOqyoIM7Jz2goLCSSVlXLj9Qh6QJMPyFaL7Z2ETfOuXBxSmwdMNQI3KE8ppkL6YrP1MsQnbgBHwnPMrUX0IWypee6vLshIJyIVKDetIAoD4NEvyCfgQbQv+A\/BzVNA8fIdGBp7lN4DjkreUdlF18h6ecbl5g3GafSKMz4PoYfYNAD8obCtJWua9raoER++khGWc8G5GXmAtRmKq1UVhQixc7WACjvFXQlwax3+DPCEUEAqUgwKZA0eTlcVkKfGqGr2emZKppBzLSN4hFBAKCAVqiwKZAEcAzty8APlxPLFvPbClQ6+kGHSS92bCqK1pldGUmwLmVpVyvzet9wl\/pEXZ\/LabNR6penD05ub0s+nEBUcYH6M2AbVShAJJUICUfsSjZW0D8Bv7caPe0rzx+YMXJkEaaUMooCmQNR6panD0AiMEDrrI1KhZXdSqJpCczcy+hkbWcEMKkIQZj1k8fF2cFPJKP4CEjEq429cKOI5+ZJK6ef\/VhD9CFrXwhzvHZ5FHqhYcw65isiVFP4ccc9s9U+fnkGPAsVY2M\/clGq8mKe0IRyEGS8AxmHa1tp4ue2yWuuKBl9Qr53aqCbCPt+rdawt\/uNMqizxSleDol17NnoZiQjlM8DvtZHGi3JdhcjWF+d1oWWvrScDRbd6FP9zolNU9tyrBMSjbRFAWlKgkAN7A3FrbzNyXaLyawvxu9Kq19STg6Dbvwh9udBJwdKdTxWvW2maWFkGF+d0oW2vrScDRbd6FP9zoJODoTqeK18zKZlZpd3qyh2Bz5HLVvNscgxxt6uuVIlsZa+rdd1sXZaMLu3javq+yXIwj4OhGaQFHNzoJOLrTqeI1swCOEm5S8WXSoANBbugjRy4BR8rs2XOcwdFkH+Ii3TAANMDpl\/A9LQoJOLpRVsDRjU4Cju50qnjNLICjhJtUfJkUOhDkho7U2Lbtb\/10BUeAkcu1uQ3C75Jdv5GbZwgZSbv4gSNaBIqtQfDLR5yneoDj7Nmz9W0Zhi5B4\/ejXZ4+y8Ke6+WrqnTISZv5szBRWehj2vPElVwkendNsszGREJonnEt1Ocn7B1Bc4HEiORoiis4uvatUvX8wHHWrFm6OwCBKXxm\/8\/neaoHOE6dOlV16tSpAI5B4\/ejXZ4+Gzt2rL4uLEvhcwKOrVtXag8KfW\/ewNHcVgBQmcK9ee+\/\/7464YQTCuDFDQh+txJwYwHXAHH1T9h1PiQ14Goi7uXjPjzu6Dv55JN1xqRNNtnEd0785qJ+Yr2q22UJUCyjflHNm\/9LvfVWR2e1alUuul875QeOAAGFW2dM4TP7fz7PUz3GSkx1u3btCuAYNH4\/2uXpM+6n5CowAcdq5vyMxDnmDRwBqfPPP1\/fiWcKf7P52pdL4xjDJbUUb33u1+N+vLq6ugKYmvrccwnwcv8eQDp58mSFvc9b\/Gx7S82F8cKZOFE\/PqdxY7XDWp3UlCn\/iA2Otu0xjG0GDBigL70uRxGboxuVxeboRidqZXE\/E8lRJEf3FV6Gmo8++qhOWYck+eSTT6qNN964kMKMS2X33HNP3QtA5a9\/\/as68cQT9QXP5kJo7tXjYulGjRppFd91112nrr76am3be\/vtt7V0iQOMSRs4ffp0hdMLNsAVVljBTXIcNEipwYMLdf9z9NGq72OPFX0q9ssGZUCTS3d79Oihk+0D3OUASAFHt4Uu4OhGJwFHdzpVvGYWTjFZ6GMaE0kCCC6C7d69e4PmH3vsMX1BtAEHJM3LL79c39g+cuRIrWrdf\/\/91ZAhQ9SVV16pU94BjrfeeqsaOHCgtkPiAINdkjrc+I4UaUulSJ7cth6aNALHFNsLp65O\/WfUKNW3b9+iwDHszky\/TFAPPPBAGmRv0KaAoxuJBRzd6CTg6E6nitfMAvAE9dF2AKk4IUvoQL9+\/g8DjjNnztQSo13efPNNbduxJSfy6uItCBBi0yAeE2\/OoUOHLuUFescdd2hbY5cuXdRFF12kwZHNbb\/99iu8BhsnatxAcLzrLtX6vPOU+lWdqurqlBoxQv2nSRMBxxLWQlYfFXB0n7ks7Lne0YhaNWNq1WWWcV+Q1VwT50ewxVsARxipTZs2Db768MMP1fbbb98AHJEecazB0xT7I7ckoEZFcsTZxi44TrzzzjtaujvnnHM0OE6YMKEBCAPA3EASCI7duqnWt99eaLa+3wWqbsSgku0pUWpV+kOdKVOmaPBPu4jk6EZhAUc3Oonk6E6nitfMwikmqI95AEcXyRH16Ndff63VpO+9956WKpEen3vuOQ2gyy+\/vF5neLdijyQO7d1331V33nln8eA4e7Zq\/Wu8n0b2X8MbklhPpg2bOUzOYL+r29JkIgFHN+oKOLrRScDRnU4Vr5nEZpb2IPIGjniQ4kxz\/fXXa29TW90JrVF5Iv0R1oHDzUcffaQGDRqkvVMpbFSAJCpWW2rkb+oRj+cFx9hqVQOOv6pTVdeu+t1ZWE9x1uuoqR8rLjx+7ZT2sb1v47wn63UFHN1nMIs8ImrVjKlV3ZdjtmriNWpCNPBU5cdkHWETQnVqpEFUqFtuuWVhgCQLGDFihHr++efVwQcfrNWPf\/7zn9Xvf\/\/7BsH9XnCM7ZDzKzgadarpQBYZP2x1GHA8o3udatOyqeqz7ZrZWkxl6q2Aozuhs8gjAo4Cju4rPOWaqEhfeuklddtttynu3ySjholNxEZowhpMGAZSI0yHo81qq62mwZWwDoCWNvAEpY0dd9xRg6QBRz5DEqV4HXL4DIBt2rRpYbQFxgYcWS+\/qlOTBEe\/5OOVSDrOmCbPXKh63DRd1TVZpDpt2k7d2KdDyjOfzeYFHN3nTcDRnVYVrZmFicpCH5OcxPnz56uzzjpLO+L06tVrKakP+yIxkI8\/\/rgOzUBNinMKqtTDDjtMbbvttg2kRED1tddeU4SAHHvssVoVa8Bx8803V9jzgsqRRx6p9t5776XBsVEj1fq225RRpyYFji4OOUnSOqot1t7e932nqyE1Cjj6U0zAMWol\/fZ9FvczkRxFcnRf4TmtWWDsI45Qrc85ZykqlML4rnGOrgnKk5giA47LfrtAS47jj\/tNhZ1E+7XShoCj+0yWwiPub0m2ZtWDI0QdNWqUDty2VV24\/A8bNkxTw3j1GdLYKio\/1VQWJioLfUx2KVZva1FzEfV92MiqFRz3HTlb\/dysldpxgxYCjgETKODozrOl8Ij7W5KtWdXgaAiKXcgGRz4HHIcPH65j18zfnK5RnaF2I3sKxfyNDSspNViyU+DfWhYXUznoUol3RM1F1PdRfa5GtaqAY9SsLfGQ5jJwPKTzfhl4FLVK5ZGo9tP4vmrBEcDDaaJr1646ns0GR76jEM9GlhTyTvbp00cHiZtgaVOfuoQG2IHdWZioLPQxjQVZ7jZLubIqycNWNTnksPYOuHKC+l+bHUVyDFmQAo7u3JrF\/axqwRFi+oGdAcMddthBA573fxs4mTrv\/3yWhYnKQh\/dWSO6JldTjRs3rlCRXKmkg5sxY8ZSD3fu3FmRhByV+RZbbKFYC+YePSoT9P+HP\/xBn+YJ1yBPq9EcJHVlld2pWpsrAcfo9UoNAUc3OmVlz\/WOpmrB0XTUKwl6JUUDgEY69EqKPE\/wuJ2T02xmTz31VNUGORPSUGwya\/clWz01SfptgPCFF17Qc4aX6Weffab4jgTi3AfHfY3rrbeeVmWRUu6KK67QN2wQF7nzzjtrLQPzSw5VPGD5MeBY6pVVQeul1uYK+vW88J\/q+416iOQokmMim0QWD5C5BsdDDz1UhwFUY+GuwazdnF0KHYlx5CDDtVGEbJx99tlq9dVXV\/fdd58GSgAIwEMq7NmzpwZJCqD44osv6ryqJANAYiRnKqEdXnAs9cqqv\/\/974X32mMlJd1dd93lfCuH6x2OvIOwE2zr5fZWPfjEwWpR54ECjgKOpbB14VkBx0TI2LCRIMkxCbUqG9o222yTQq9Lb5LFlLWbs0sdNapRkoJvtNFG6tRTT9Vghz35wAMP1KCIVMjtGpdeeqm+r5GMOAcccIB6\/fXXNTgChsznpEmTlgJHVLSlXllFjKVOAuApqIOJ0czSLedhc8XaO+jMG9W3Wx2uq71ybifVpmWTUqe35p4Xtar7lAo4utPKuaYXHL1qVD+HHFuNWnMOOcVmHv\/lF2ea64rles+vvTKXHJNflb87deqkbc7PPvusOvnkk9U999yjU8KRBQepErAkM84nn3yiDjroIK1+DQNH2in5yqpnnvEFxywyfhQ49v7LieqrPZY4vgGOa6\/UWP9te2Ui7Xu9NPksL\/UAR65MIyGFoUPQ+P1okqfPssgjmVOrsqByHcpRLtAq13t+3aVReQJuHIY23XRTnd0GWyEq0hNPPFFfRcVvrqgCHHGwwYbIZjRt2rRIyZHLjku+siohcEStOnjwYC3JuqpLzTPluLLKbGQLDxiuZ+fBY7dU6zT6Qv8NEJiCpG\/\/z+fGMSoP9QDHqVOn6oOcAceg8fvRLk+fjR07NnNmokyCo5Eec5kEoFygVa73eEQYJP3NNttMH4CwBwMGxx13nL6tAwnSBkfUrqhTcdjhBB8mOQKmJV9ZlRA4MmTb7uhNYmGTxIR4rL322jq5uh2vG08V4F7bgGOL3teq+u+bKxKQn9RlLd0Al0GbAjjY\/\/M5n+WlHmNFS8V1aXaSfL\/x5\/0zrpPLmpmo6sHRnaXda2ZBxA\/s48iR7gO1a\/brF++5cr3HBxxxvOHWDWyMF110kQ7Zuf\/++zU4Im0ZyZGwjKuvvlrbJFGrzp07V+2xxx46n6qfQ07JV1YlCI72sO1sT95JqkTycbP27hn7uCIZgORX9WcdsTm6bylZ2HO9oxFwlNyq7iu8DDUBCsAReyOeqAsXLtQSEwVnm4033ljfvMFJHYeqL7\/8Uh111FEaGEeOHKnDX3C+weuV5\/H0NKEcJV9ZlRI4loGssV5hb2THjJuvn5X8qkuTUMDRfVkJOHpoZVRHqLvKpRJyma4sTFQW+uhC6zh1sCEiKe26664aHJEcb7rpJn3zBt6ghExQCHExnqynnXaaWn\/99XU9Yl25BJnEACQKoNhJAEq+siqH4Pj3\/\/6oLnusXn1+1S5xpjIXdQUc3ac5i\/tZWSRHW23kzZPqTt7kamZhorLQx+RmRGm7IRLgjz\/+qJ1vWrVqpbMbISmiNiXpPIDIPY047Gy44YbqkUce0WkD7733Xi1N9ujRQ3377bc6UT1qVcJBAE5CP7jvseQrq3IIjs\/Oa6yOG\/WWtjue0f03Z5wk5z6rbQk4us9cFvezsoCjIaE3+LkS9hT6koWJykIf3Vkj2zWj5iLq+6yN3jue\/W6crmZ\/scTR5obeHVTndi2yNqRU+ivg6E7WLPJIWcHRkNLEJprb2MudBSQLE5WFPrqzRrZrRs1F1PdZG713PEiOo6Z+rIchV1j9NpsCju4rO4s8UjZw9AKiLTWiPsOhohwxXFmTHHHz98vK4r4spWapFJgzZ05ontssMn4YTbzj+fDz79UWFz9feEQy5iwhhYCjO2dlkUdSB0fb3hgkIZqg\/gceeMCd2iXUzMJEsSGffvrpiiTcUipPgY4dO2pbpl8pdT3ZFx5jSz3iiCMUuXXLrVExY\/MbD+C47ipN1HPvLdTS4419OuQ+pZyAoztflsoj7m9Krmaq4Bh2y3lyQ4jfUlYmCoDkp1IF5ieUAueYrFzmOniwUhMnLqHY08rysBw0SKkuXYomJdJ7kARf6nqyr1WzLz5+4oknlrpRpugBxHgwaDxIkKhYAUjAkfjHPBcBR\/fZL5VH3N+UXM1UwTG5bibbUhYnKlkKuLWWNeYH\/wBHU35Ryyz5kzjJuEkQ3Eika5WynuwDJJIiidbJhsMVa0ajUolbOYKuSwMge9w0XUuReZces8YfMZZ04lVL4ZHEO+PYoIBjlSYBcJy\/VKtlhfmRFPv3V6q+viE5NCbWD1IK1EyxlML4firV3r1766xA1QiOkNE46OQ9c05W+CPFpe\/cdCk84vyShCsKOAo4Bi6pamd+wBBJ0Zvprq5uibDYtWvC3BLQXCmMbxzVuIKNS5xJamASZtjq1vKMZMlbosZjpEd+51m9Wu38Uc41E\/WuqDUV9XxkNEz1AAAgAElEQVQlvhdwFHDMHDgCik8+qdRf\/rJ01488Uqmzz1YKgCxXKZXxSU7Qv39\/7bE9YMAArVI1HtxDhgzRCRDKWVzGIwAp3qpx1qTLmorTXjnqCjgKOGYHHP8\/Kg4aWdfArmg6DxhecEGqpsVAOmWR8cM2lzjjaXnK07qpPGbQEcnRHaLirCn3VtOtKeAo4Fj94IioiO508GC1jGp4aTOgePrpSh17bLqMkhSY+LWThVCOoPGTHOAfL84rhHjkKUG5gKM7zwk4utOqojWzOFGVIFjFmb++Xn336CTV9JiG120ZgMSmiG2xnCpUv3kodT1lJZQjDCBx0qHkKYNOxfmjEptCke8slUeKfG1Jj4nkKJJjdUmOSIkTJ6r6OyepuolL311Zr+rUxLrDVNenB1UcFA3hSmH8rIVy+C0W7I8Ufh9\/71v6N2rWHTdYJTAPK2DapmWTTCczF3B0x55SeMT9LcnWFHAUcKwsOJr4C9Smkyb9FsHv6RWgeGvdJeqoETuouq5l9LZx4LdSGD+LoRxRJLFVrUG2SJOOzuvxaoPm5JkLNYDyU41FwNF9VkrhEfe3JFuzJsGRLCPc\/0fxu\/kjixOV7LS7tZYW84OH\/HA944iRvwbqh3WpXz9Vf9gFVQeKSUiOWQnlWLx4sR6unSmJz7yZk+x6gKQBO2\/qORIJGImTeEkAsNeWq6met76uEwxguwRAO67XXIeLxHmvmZeo\/pVaD\/6YPXu2atu2baF\/QXTyo12ePsvinltz4Ihb\/MCBA9Xll1+u1775u3379oXtN4sT5QZnydYqFRwBQOyBJg7RCIZ2sH4hi4236zx42GGpB\/AnQbFS11MWQjlmzZqlSQUQmMJn9v987q0HQN7xzCw17aMlqtewcmvPNdXFExc2AMfVmirF53Hfa\/oS1b9S6sEfU6dOVZ06dSqAYxCd\/GiXp8\/Gjh2rLyl\/JuBO1Ki1UYnvaw4ckRqnTJmiTHwYzg7cEE\/GkSRO+pWYJAMyZILh7351v\/6hlELdyIdgSdjfI+u76jq63q+ZZGxHFu\/n2Ps4BX+xcKFapUWLJczvrWQ1pKXBSfVqcP0S5xmT3zSKXgYc6fvMum6q3WGddTxGpZ1sovptf18qOMZ5Vznq+o0HIKA0afKbipPP7P\/53q\/e+58uUqOnz1eoScnLGlSu7dVODX1ySS5hbv4gTAQHn9FHdCjqva79K7Yez9XX16t27doVwDGITn60y9NnEydOVIcffriAYzkYOOgd3qwifllGDPM\/9dRTVXEd1Jw5jTVg8bPTTj+piy9uVPjfmxKNcQdKWyGE94ZARM1RMe+gzTjvad16serZU6kTNp2o6vqVKZ1N1MCL+P6ll14KvdKqiCYr+khaYG\/UrEGDQ31qvF4\/v2oXDY6oWwHKaiylalaqcUxp9SmtNZVWf\/Ve9ssvvzQMHEvzbWVo2yspIklyuiPriClmog499FB1GKq7MpfGc+Yoflb8179UhxfuVYBjnFIMcMUBrWIBOAwcAcJ11lmstt\/+O9Wx4\/dq++2j1WxxaFLJunfeeae66667SjoV23ZyMxY\/e3k5xpnmRua9G9IeD847lz22RK3hBUfsj+aqrHLQwOUdAo4uVFpSJ8015d6LeDVzDY5saNtss008ihVbe+RIVT9yktrohXsbtBAXtHh4lmobW+3YVi2xGVH8pFHvsHiHS\/GqP4\/YdZb6+eclt0MBiKhjawkI\/Wgybtw47QBWrD3FvqaqZcuWS8Dh88\/1vY4mCbnLXCRVJ+2NDKAzzjhB4Ii0aDxazd\/VluhcwNF9xaW9ptx74l6zJsGR4RtJMUytWuxm5kreiYMmqpUnPai2nHh14CM2OBqgIbh9q62UWnHFJY\/ZCbTTtMXZtkj+hvlXWGGe+umndTXIGZNjmn1wpW011SuF8cPuPKVd1m81XVmVBN0ve2yWr\/0RydDYJB88dkt9NRbFgCMqVj4HWDu3a5FEV0pqQ8DRnXyl8Ij7W5KtWXPg6FWjltshZ+LIevXJnY+oQyZG5DP7FWF+Pu8CddeySxxQynWLhOsSEuZ3o1QpjJ9HcDRUxQZpCinoKH7gaNsiTf1quA1E+MONP6hVCo+4vyXZmjUHjpUI5dAS16+5P+uU51JBe77q6lT9BSOWxOtlQPwS5ndjtlIZP29qVT+q2jGRJjGAcc7xA0ekyBt6d1hKgjTq2nIkDhD+cOMPAUd3OqVes1xJAADFF15QqnfvJUPydZQBBAcOVKpDh+oTDSNmQpjfbamWCo68JS8OOUEUNeCIXdFIlACcF+xIEMDnttSJOnb2F785eNnPnNG9raLNNIrwhztVk+AR97clU7PmJEcXspQ6UdYlEQ1eVwBHALFvX6V23z1zgGgPSJjfZTVlU2UUNrJS+cONag1rEQM5auo8nWvVeKb23W4thX2SAsDhyQowAoY2gAKYFACS7\/psu5aWKGmTXK8UEw6y343TdT3a412lFOEPd+pVYk25986\/poBjjNyqQaBoSPt0v5Gqa5f\/7w7ar+EtEqVOUqWeF+Z3o3xajF+rDjkuVEX6Awj98qsCmEuSm7d1yrtKvCSqWYAT4OVvbJy0z9+8wzj\/2LeKAMAffv5dYAJ14Q+XmVxSJy0ece9B\/JoCjo7gCDDussvSYRDGkYZwyWpzqIm\/HBo+IczvRsG0GD\/P4OhGebdaxnaJRAkI4vGK9MjnSJQ4AZm\/DXgiWZokBOaWEcDYSJ68+coD6tQGv\/teffDDiuqFD74OvYXErae1WystHkmTYgKOEeAYJi0iIHL7fAZ8a4paQwKObmRLi\/EFHN3oH1XL2DNJMsDfRsVqJEqkQ\/O5AVKj3l0iWX6hX8HfJvYSVe7qzZQ6t2sLdeyDC9QLHyzS13SVqqqNGktWv0+LR9Kkh4BjCDiSH7R\/f39pkUt2a01S9C40AUc31kuL8dMCRxM+8uqrr6oBAwY0yB6VVRVY1EwBhBQ7kQBSoLFN8h03gQCSgKFXykSVi8QJONqS57QT6grgaFLd2Q5B5fKe5T3VEv\/pNxdp8UjUvJfyvYCjDzjq+3ZH1quug3dRdmYZJEQkxRoxKUauGwHHSBLpCmkxflrgiGcspUePHuqSSy7RKRRr\/dYa7IrGYceEeQCEgB5SICCJ1Ec9PkNKRAWLlGnup+R7Wy1715TZ6qYerdTW19drqRGHIWPPRFXr\/RsJFQADeE1JKpmBkY5tNXE1ZRRKi0fcOLS4WgKOHnDUwLjLINWvfnCBomSx6dZNqdtvr10Vqt\/yEXB0Y6q0GN8VHM2dkH369FHbb799odNBIU033HCD6t69uwZE6qy33noNnktrPG7ULF8tgAo1KqBpVK0GHOmF+dzkgwUA+R4Jk88AStSqR435WEuTgCrPGGcic4clYOz6twlfMaBp2mrTsmnBOei5977Q\/cCb1wA77wacGQd\/G29eA5ZIw0s8fVcphL3gfGTeh6fwEsl6CXAbiZd+JCH9ZnFNCTja4Fhfr27f5e\/qL\/XnNuTQ7t2VevTR8nFtlbxJwNFtIuIyvq3WjHrD5ptvHpo+zgDj+PHj1T\/+8Y8CyIUlwxBwDKe6SW8H+JgYSZMP1rYrDrj7dbXKcovVrS8uLIArzxoQsgHH\/I30abIA0TYgR7ElSAOIBqSMpGsA2c5La8DNpN4DDM0F07yHz017vNs8C4jjscsY+e2VYG0QXdKP73Q\/AWmKAVvv\/3y3xMv3e31gMO3G5ZEovijH9wKOv4LjyEFL1KgNMtxk6MLdNBaLgKMbVSvF+OaS5K233lp9+OGH2nZoJMewe00FHN3m1a5lbJGoSk255KF39f2TdvhH\/JbjPWEkRPspGzxNEgVAHOAyDkY2UMV7o1ttwNCAPvQw0iu3q1AqxSNuvfevlXtwXLy4tVr78ZFq+QH9G1IIYMyD103I6hFwdGOtSjH+Rx99pDvYtGlTfYOHDY5h95rm0eboNpPxar0zd6F67f156oDt2xcuO47XQvzaOBEFXRgNMPKdHXpSqTsxTeyoAeVK8Uh8Cv\/2RL7B8a67VOsJE5Qa\/Jt9UZMGYHz66XwZGH1WkYCjG2tVmvH9kpeH3Wtqq3X97ow046mWy8DdZqH8tSrBHwCfSYJA+Ihdxhy1qRo9fb5Wa57arbV2MErCXlgsZbcZMlV1XK+5dkzK4oXguQXHgQcfrJ5p1GjpOA3cUQcNKnY91NRzlWD+LBIwa+AYRWMznkpdBh7Vv2r5\/ocfflDz589Xa665ZtkkR3vs0z76Xr0053s17+vFavxbi9StPZfkkMVBaNBurdR+HZpXlFSDJizQ\/SLcJYkLwcs9mNyCY+uddlKtFy\/+jd55i9NwWGkCjg5EqgJ7SpDkSO\/D7jUNGp0Bx7JeBu5G6qqqhTPUJ598olq3bl0RcDTE8EqHD7y+UK21YmO1fV1lwREp9qTRM9V\/TttSTX\/2sZIuBK\/ExOcWHLfv1KlA7zmNG4sa1Wf1LV68WI0cOVLtueeeegOQ4k+BOXPmqL59+6q0L88Oor8fOLrcaxoFjni\/yrwHr3rhj+gdYbOr3lVn7bSi2mmtxRXlkeieLl0jl+DIZrZ43Z20Z+oZjU9Ro9aaqho3nlMM\/eQZoYCmQMeOHdWoUaMqQg0\/cHS51zSos\/DH6aefrl7gPjYpQoESKLCo80D1c7NVdQt1Tb5RvfbqkpkUe7kERyaKDYAfKUKBJCiAhFUpKcsPHBlT1L2mYeMW\/khiVUgbUODnZq30jzfBQbVTJ7fgWO0TI\/0TCggFhAJCgcpRQMCxcrSXNwsFhAJCAaFAlVJAwLFKJ0a6JRQQCggFhAKVo4CAY+VoL28WCggFhAJCgSqlQC7B0U7WbObF7167Kp2zsnWLLCvDhg3T77OTWpetAxl4kV8Scb+sMxkYiu6izHm8mQq6ESVeK7Vd2+QAnjt3rh5oVvbaXIIjG9qpp56qzj777Ab32NX2Eo03Ovu6pHfeeUdvmsOHD1ctW7aM11CN14bxL730UjV06NDM00bmPN5iDboRJV4rtV3b60lt\/u\/du7c65JBDqnrwuQTHWtrQ0lpdduJqOR0HUxlAIb5xyJAhOgF4lovMufvshd2I4t5KPmt6k+JXKxVyCY6ul8hW66Sl3S8DhjvssIM+3Xn\/T\/v9WWrfezVUlvpu91XmPN7Mhd2IEq+l\/NUWcKziObeDo+lm1IWyVTyUVLrmJyl6b3lI5cUZbNS20dH9\/fbbL5NSpMx5cYsvKAFDca3V\/lNG4r7yyisLd49W66hzKTmyoWEcNqow7\/\/VOlnl6pdslG6UNnRae+21dYJv7\/9urVRHLZnz4uZBwNGdboZWXMhtEuK7P13+mjUPjraUGCQh2nko27dvX\/5ZqLI3ioqt+AnJqspe5ry4ORdwdKNb1oCRUdU8OLpMnTjoLE0lW40qDjkuq2hJnSw76Micu8+zqSngGE2zLHmo2qPJHTgGnZCNaix6qvNRQ9z6o+c5y27qfqOTOY+ec28NAcdwmmXZ1JA7cGQqvUkAsupEEZ+V4z0hAeHR9PImAchKgHPQyGTOo+fcriHgGE4vbwIAUzsLe24uwTHe8pfaQgGhgFBAKJA3Cgg45m3GZbxCAaGAUEAoEEkBAcdIEkkFoYBQQCggFMgbBQQc8zbjMl6hgFBAKCAUiKSAgGMkiaSCUEAoIBQQChRDATygJ02alImgf+\/4BByLmXF5RiggFMgsBbzpI70DKcf1bHlIPJLFwH97LQg4ZpbFpeNCAaFAMRSohmTxWQPHoEQg9kHDe48pYUGrr766+uabb9Txxx9fzFRV9BkBx4qSX14uFBAKlJsCAo7xKB50b6UN8LQ4cOBAdfnll+s7co06tWfPnuqxxx4TcIxHcqldrRQwgbsE6poEwSz2vn37qnKonKqVLtKv2qCACzgalWCPHj30Jd\/mFnvv+jd8YSjjlZ68KlzzvQEW2r\/ooov042TpGjFiRFVdwB52b6WXjnb6Qe9tNVncN0RyrA1+T3wUNhhuuOGG6ogjjlBZuL07cUJIgzVHgTjgOH\/+\/AJgwROnnXZa4X\/auf766wv\/GyA54YQT9D2oQd9zXdOqq66q+vfvr7beeuvC7UDUv\/feezUYt2zZsix0p8+2ZOfNMx12b6X3Xka\/exq97ZdlUAm9RMAxIULWYjMs9hkzZuihrbTSSpm8p7AW50XGVBoFwhxyTFozVIl+B0IDACeeeKI688wzlbkQ3PTIAO\/555+vLrzwwqW+N\/X87jWslB2SMY0fP16P98EHH\/QFZ780ed47Xhl7fX19A89UAcfS1qo8XaUUMAxhn56rtKvSLaGAMwXiSI6YFbh\/0At+2NdwMvF+bySvc845R51++ulLfW+Do22j4\/NKgSPvjqJJseDoPClVWFEkxyqclGrpkp00OIs2g2qho\/SjuigQBQT0NiihuHk2CBzNzSZXXHFFZsCxFMkRWhm\/BD+1anXNfLzeCDjGo1duahsPNQa8xRZbBKpbckMQGWjNUCAOOHrt7EmqVatBcoyyOZpJ9zsseNWoXjVr1heMgGPWZzCl\/tvOAU2bNtX2FbnzMiViS7NlpUAccCzVIcd2sLHvkt1qq60ahD5UWq0aNQF+4BgWyhHVXha+F3DMwiyVuY9+YRteT7wyd0leJxRIjAJRGXIIt9h99921g4odyuEXalFqKIeJC8wiONLnsCQAiU1YhRoScKwQ4eW1QgGhQPVSQC4xrt65KVfPBBzLRWl5j1BAKJAZCgg4ZmaqUuuogGNqpJWGhQJCgaxSQMAxqzOXXL8FHJOjpbQkFBAKCAWEAjVCAQHHGplIGYZQQCggFBAKJEcBAcfkaCktCQWEAkIBoUCNUEDAsUYm0mUYv\/zyi1q8eLFabrnlIqtT79tvv1XNmzdXyy67bGR9lwq0+dVXXy2VVJnPeBd3vyX1Lpf+SJ18UIBYxa+\/\/lq1bdtWLbPMMpGD\/uGHH9Sbb76pNthgA73+eYYf4hR\/+ukn\/VlYYZ03bty4QZWff\/650E5UB+bNm6d5lOTkLv2Nak++L44CAo7F0a1iT02fPl0nNLbLxhtvrLg5Y+zYsb79at26tX5m5syZOvD42GOP1fkiSTbMRmDKyiuvrA4++GDN\/N7s\/C+88IK65557dDKAddZZRz9HjNbLL7+81DtJUj5o0CC9GZnC5nDfffepu+66S910002qrq5OfwVg33DDDeqZZ57RNxysueaaFaOtvLj2KPDOO+\/o9GZrrLGGIhk4\/PPoo4\/q1G5t2rRpMGBiFuEhbtU499xz1VlnnaWefPJJfaA75ZRT1FVXXaVB79RTTw0klFnPU6dO1fVbtWql3nvvPUWictrr3LlzJLByhRVAzO8VVlih9iYlIyMScMzIRJluvv\/++2rcuHH63\/\/973\/q8ccfV4AjgPfZZ5\/pzwEamJrrdVZccUXNYJtttpk+jU6cOFFdcMEFao899lCvvfaaOvzww3UdrqYBLEmmPGzYMDVgwAD9e+jQoVrSY4MA8GbPnl1geoCSTWfnnXcuUJF27rzzTnXttdcW7qV7++23lTm9A4Sko9trr730Mx9\/\/LEiDyWg3LFjR735MB4AVopQoFgKAC7\/+te\/9DqEDw444AC9\/pHKrrnmGs0HBPt369atIJ0hXZ599tnqwAMPVHfffbfq06ePGjVqlLr44otVixYt9MFwzz33VPvss09gt7iV4qijjlJ\/+tOf1KGHHqrbRpK89NJL1Zw5czQ\/wW+mwMOLFi0q\/M+9kSeddJI68sgjNY\/6leWXXz5Sei2WbvLcbxQQcMzgakDl06xZM8WpGEbnh9u3YW4Y8ZZbbtF\/c9pt1KiRZiQYyhSYEZBD8kOSRKJEhQRIGVD0giPP8hx19t57bw22flf2+N0sQH+4M84UpEg2EfqEVGsXP6kzg1MkXa4gBb755hstLT788MOhveAghoTIJd7wC5IlvNWkSRP14Ycf6rUJePEbfgAkObihOTFl\/\/33V+uvv77+F95DWkRq9GpBDGgCrMcdd1xB7erNsONCNvpikn271Jc6xVFAwLE4ulXsKZh15MiR6sYbb9TSFcDIhakmbyPSn7eYGzVQhXJ6xq5n1KYw2r\/\/\/W\/Vq1cv3VYYOJoNgE3FvA+1KqojU77\/\/nttP7z99tsLkiNg+OWXX2oVKoW\/Ae59991Xn9Lt4gXyihFaXpxpCrC+OUByAEONj8TIJcPGXshaBMTgiR133FH9+OOPWpPC2uVOwwceeECDHhoZ0sah+bjkkkt0OjkK2puddtpJf0892rv\/\/vs1wMKbu+6661L0A6zR2tAO6emQKm3JccGCBRr0OHzCFwC2OdzajYnkWJ6lKeBYHjon\/pbXX39dMyJqTdRDnHaR5FAFbb755prZYTzUMzDc73\/\/e3XyySdrFStqJhgTVY8LOHK6ve222\/QYjP0S1dB\/\/\/tfzcDe4lWNmoDqV199NZIOcjVWJImkQggFALzJkycXaixcuFDbEVHl8+NX1l13XQ16ANyIESPU3\/\/+d80nANwTTzyhbYW0C7jxPWufw93gwYO1rZ\/nMGVgV+zXr5+2WXodcszBEomSwy0aGAOQfEcb2OOxy996663alCHJ\/iu71AUcK0v\/2G\/nWhhsgd4CyGGvABw\/+OADNWXKFK0ytS9k5fSLzY\/T8Z\/\/\/OeCpBglOaKmok02ADaJ4cOHa+ZHBYUqya+sttpqGpC9hfqMAbWufZKPTQh5QCjgQwEbHJHExowZo0Fxu+2207UNWCL1YUqgAI44ypx33nn60MjhkQMn9kfWK2ufNcsh7+qrr1arrLKKPlgikbLO33jjDQ2ISHmffPKJgl+CClIoalmcgTjcGvsjgI669Y9\/\/KMGZQ62HEjRyhxyyCGF5oz\/gDjqpL\/8BRzTp3Gib8DuB+OgzsTZACbmNAoDoeoJA0c6goqTn1mzZjlLjpxiKfY1VqhGcdDBUYf+YLdEqjSONN27d1dHH320EptKotMvjTlSAEcvwIf1ibmANYq3NV7an376qVZ9Ir3hBGbCJfgcj+3nnntOe5hygOS6tkceeUSh9SC0gjp4YaOCxQPc2PI5eFJ4DkCFVwBnNDemDodLbI8ALSpfA3D4DhxzzDGaJ6MKWiEOp4Yno+rL98VTQMCxeNpV9Elz4zgOBUiSbAQwexQ4wrjYBbHBuKpV\/cDRZk6kQNRM2FJQM9mFDWLGjBn6I1S9qJSQXJF0UQX7lSCps6IEl5dnggIc\/CZNmqSBC0kOiXDatGla6sMzm0MkdnfsgzikAV6HHXaYluCM4xhepRTjLIbnac+ePbWmBG0MBTDbb7\/9fGkCEMOPHBRRyRrwveOOOzTIDhkyRIMuqlRCpJBUkSbx9EYCpY\/Gpm\/foWofTgUc01+OAo7p0ziVNxhwJBaK0zEnStRAgCOnU9RLAJatVqUjJk4S+wg2Dhebox84YhsxHqiALe8kdtEAntfrlI0A93ney0WvQfYfNrNtt902FZpJo7VPAdSTf\/3rXzVwATSYF5D0UKECkpgVkACxF6IuBcRQY1IXqQ8AQguC7ZB1jVYGL1Wc3tDaAGSEU2EXNLG6Xqpy8OQuSNSkdtgHEiOmDwOOPAeQI1H+3\/\/9n+537969BRyrZJkKOFbJRMTthgFHo2IxQfkY+bGzYNPgxAvzc2LecsstCwH3nFD5DOeCHXbYQeHcA6jC8MQbAnyclLGpmDhH+mdOrthdkBbNifihhx7S3+Flx6mXz22nHNthAS88Av05GX\/xxRfaKxAVrHGPF3CMuxKkvk0B1hp2ReISASLMD08\/\/bSO+91oo420pIi9EW9WJEiTNYe\/0XJwmERixHlt9OjR+jN4CBUo4IgkaMDRDukwfTCONfAgGh2T1ILPMYFQbGnSPGec1gQcq2c9CzhWz1zE6okXHHkYBsS5BkcXTrjPP\/+8ltQMg2KHQR0E43fp0qUQnIyDASdrgo6Nl503Q44BR1SxqKAAV36MvYT2OAWzkZAUwAAnzgmAMCdw1FecjlEpUXhH\/\/79dX+xBUkRCiRJAdYmBz4OgPyQnhCVKaBF8gw0LUiYtmcp3tfYJjkYArAcPjEVcPhE6kQyBSgBRqRPO5WcfQikbQ6aNh\/gfYq06ZdhJwgc8UbnQEmCD2yfHErphyTJSHKl+Lcl4Jg+jRN9AzZDGBgVEU4FRnLkMzLT8ANA7bLLLlrqw7kAx50OHTpo6Y5QCTYHNgpOxqhfCfVA5YNzwkEHHaQZ2guObA7EaMGg2DnJcPPss89qOyPSHp\/xHbaev\/zlLzqwGtuhUXMBpCY\/pSGIn+TId\/SVTCRShAKlUADtBKCDmpTDJIAIOCKdIUF6E1AgGaKGvfnmm7XDC4dGTBYcNOEZNCwAE4VDJqYBAI91jkMOvAWokiEHCRSbJ2CGkw9ATTsAoze2l\/bgBfgWvkIDY9scUc0CxKiDybrTqVMnyblaysJwfFbA0ZFQ1VINJsKWAbOjCoWhjAMAUhogZXJGAmgwOu7lfA44YkeBeXEOwOMOwIO5kSqRCmF6wJSNZLfddmuQ7mrChAm6bYKeqYvN0dh3OE2bkzM2lXbt2mngxCuPd1JQ1+LlGlUEHKMoJN9HUeCpp57S3tIcJlGddu3aVccVYps3mgu7DaMyxRsVYCMxAIc5eAqvbNYwPGS0MKhWcegh1Rv8gEcsyQMAP2ImjTQKn2GzpHCIREvCwTSq+DnkRD0j3ydLAQHHZOmZemsAEFIitwNwIjW3WARl\/ac+P97bLtg0+Mx8btdjo8Cz1G7fOzC+N7ZF73e0Tf8kFiv15SAvCKCA0bBgAvALyPd7zO82Dep5ecXv2TB+KGaSDJ\/Dn1G3gBTTvjwTTYGaA0dvNhZsCrZ3WDRJpIZQIB8U8MuDm4+RyyiFAtEUqClwNKoIjO9BsULRJJEaQoHap4DhFWzXOEyRuF6KUEAo8BsFagoc\/SYWOxtGdq\/0aAJ9i10MPzdrpZb9dkGxj8tzNV8wjJoAACAASURBVEYBr3NHtQ\/PeDvTT2xnXnAslT+qffzSv\/JTIGs8kktw\/PDz79UZx\/XX2SmKKYtb\/V4t6jxQrfT4GQKQxRCwBp\/B05cMRVnYADA94P2IQxdeyl5wBBi5DLhY\/qjB6ZUhJUCBLPEIw61pcPTGDpn5PW7UW2r0I5PUH3\/3qg5Aj1vGfbCCGvdBE3X0JovVdi2DkwzHbbfa6pMsgKwhqKldnRqqbQzl6A9eu\/yQ6SQL4Ig2hUKmItKhecHR5MMlaN0v0L0cNK3UO8iaQ7pDLkGuxjWPGhxv8m+3Olwt++1nqsmMcTq5QFLzlNb4Tb+zwiM1DY7GpsIgvSpVwPGfk95QY\/+8ltpmm21i8+HQJ2eroU\/OUad2a61O7bZu7Oez8gA0JAyEDb8aN4pqoSMJ38l6kgXGxwmHWNhzzjlHx\/yFgSMxdSRuyFMh\/ImYRUI2qnHNk5uVtJBf7XGZBsfmky\/XV2yZEJNS5yqt8bPmSFeZBR4xNKxJyTEMGBn4ZY\/NUpc9Vq9eO6V9USd9wHXU1I9Vn23XVDf26VDqeqza502Ccq70qcaNoloIZyStLDA+alSyGZGRKMhb1YyHzayYw2O1zEsx\/aj2AyFzQ0INGxyJSU5KY5HW+LN0gKxZcHQJngXYALhiwXG\/G6er595bqHbcoIUaf9yWxfBgJp4RcHSbpqyAY9il0\/Yl01kZj9vsxKtV7WvezI0NjkkeytIafxbXVM1Jjn6Z773sYcDxlXM7qTYt\/a9NCmMpA47U+fyqXeJxX4Zqp8UoGSKBU1ezyPgMLEpyTHLTdSJkFVSq9jUv4Fi+RVJT4Bh0MvZeEDp55kLV46bpqlhw3OLi5\/UM4fVabBvlm+Li31TtG0XxI0v2SQHHZOlZydbSWvN2aEwpKlABx\/KtjpoCR1eyTXltptp35Gz14LFbqs7tWrg+VqjX8pSnta0R1Sy\/sT3WYklro6g1WmUVHIPmodbGE2e9pbHmvaExpYQ0CDjGmc3S6uYSHFlge9\/3XVHAhrSI5AiwHn\/vWxoYz+jetrRZqNKn09goqnSoJXWr1sAkaDxJST8QO8m2Spo8z8NprHlDTzv8wlZZG1q4SJQCjknOdnhbAo4xpT4bHPF6xWZZqx6raWwU5Vva5XtTHsAxSeknybaSnuU01nwYoNm08EqUfqAp4Jj0jAe3l1twRK16+oHbxJb6bHsl4EgRcCzfgq3GN2UJHF0S8\/uNJ0r6iTMvSbYV570udcsNjoYW\/2uzo1r+w+cKcYA2aK694ebq3tuv1+EaAo4us5hMnVyD4yFdNokNbDY4jpo6T8c74pRTiyWNjaIW6ZQVcHRNzB8GjkmEEKS9wZeyxlzXfFKqUPug0OzlOwrgGASaadPOdfxxaZwVHrHHlVtw7HHTK6rXXl1ig6MJAyGEw\/477mLJQv20GCULY4\/TxywyvhmfX2L+YsAxKbCIQ\/ek6tr2z1atWql58+apsMQXYapQvz6FAVoUOGKn9APNJA4pfn1Ni+ezyCO5BccDrpyg\/vjHP8YGR1SpRlosNV4yKeZOq520GCWt\/laq3SwyfpLgWG12szjrwGv\/3HrrrXV6tm233TYwK1SQVBf0XgFHVVAHZyl2NtfguO0u+8TOcAM4ololM45xzqnVWEcBR7dtNqvgGJSY34znqaeeKqQle+mll1Tfvn0bpC0z35vvjN3MfP7xxx8XbvfAbnbPLVfr9sLacqN4crVMX2xPUpM2Lyhlov0MUp1NJ7+eudDOSIhemgZ9bkuOUe+PQ620eN7QQMAxzmxUoC7M3\/PCf6rVdzg4tr2Q7DgUkzbOxDymFesIEOMRW0wmn1JJmxajlNqvans+i+AYln\/YjMdOPB6W8Np8ZzZykwjbfG5A0\/u5S\/JsAJaSVGJt79rxG9fQoUPVJptsEig5Bo03aF2WQrsgmrrQrhg+kcTjv1Ett5LjQWfeqK99iZv+DXC0wzeIeSTOMQ1wNGpbpqsS0mk1giPSOo5QHBrwEq7EocG76WQNHKMS85vx2InH+Swo4bX5zmzkJhF21OdRybNReXLbCdcdoe7kaiaXWMAgUAgKjfCO64477tAJ14Mkx6BxhalVy0W7OLZfv\/5K4nEBR1UsOAKGduA\/\/5OAPI1wDgPEgGQlMvFUGzgCjGQlmv3F93oFr7tKk9hq8WJO01HPZAkcXRLzx3XIScupJK5tL2yeguyifvZAJFxsjkZqpV0blIPGW2mbY1xHIb\/+psXzWeIRQ5fcSo4HnzhYLeo8MLZEhhr1jO51hfhINmtK0uBoJxtAUkrjHVGbflqMEvXeoO+RFrH5IqmT9i9tlbZrP7PE+C6J+asNHL0em67zYteLExoBOK611lpaan3hhRd0M3aAfrWCYxKHibR4Pks8kntw7P2XE7VzQZz8qjZgmZysgCOfJ311lYmntENGyq1ahVGmvT1bdfxD26q4z9Fr7\/WquIvZNJN4JiuM75qYv5bB0SU0AnD86aefFDZXv5RvaYIjV4chpSIF4gDl0l\/j5BK3X2lKjt70gFnhEZsmuZUcWXgLDxgeS13pB45pxTraISNMWDmlJGPX40JoCna9tOyqruBkaG9L7dUSSpNFxg+juz0eU89s1n7xdWmrVf0kx7i5WaP66HVwMeAYZ7xJqFXx6p37zqsFgMwiOPqlBzzooIO057J4q7rueBWqZxglLjj6XXWV1gZdKSnJgDJTc+CmLdQfVlXq7S8bq6FPzinYVivhBGNo75X0y3loiNr8ssT4LuCIFHPttdcWVIs8Ewcs4gCSH+2Cni8mN2ucvtiSY5zxRq0Pl7Zow3j38ncWwdGm9bLffqaazBinrrjiCgHHCuFdrNfaRvg4+VX9pMRS74YM6rjX8SctCdV+P8CItGikM9v+MPerxdoZhpK0Ctll8jgs4IjjTdWXpkOUS7+oU6uSo9nQbNWiywYfpebzc4KJA45+m2\/UwaQS4Gik2zhStw2IWQdHe60IOLruJlY9vOd+\/PFHteKKK6pllllGffHFF+r888\/XbuNbbrlloeYtt9yi\/z766KOLeEvDRwyjtOh9req0aTtnZxqvqtO0mrT0Eqa+Tcvu6AVGxuY1zpt+4a2btANS1KR6HaFMfQCbg0PckJyo98X5vtbB0d7kqgkc46RQKzc4eqVbV6lbwDEO56Vbt+I2R5Pb8eSTT1bPPfec6tKlizrxxBPVGWecoV555ZUCIOJlR+HzUothlE2PuVX93GxVZ0koyPnGK+WV2r8gFWJaMZWAC+DovZvSz3PNSLBxHJlKpYffYcG0mZZaO06fswaO8ByemBRiBw855JAGwzXjMad9AcfPVPPJly+VFDzMi9ZPunU5WAg4xuG8dOtWDBy\/\/vprhdT49NNPqylTpmhJ8ZprrlHnnnuuOu200zQITpo0qQCIaYDjPufcpZ6d18g5S47XDmhLL\/ydlDRlp6izpz+NsJEwaTDIrTuIDsUsVSON0w\/b2cZuK0p1XYzkbmiJo1GpNtQsgeO7776rBg4cqC6\/\/HJNYvN3+\/btCyTPMjgGOeqUW3L0Ux0LOIpDjtMeOX36dHXMMcfoeKI2bdpUBByPGXqf+tuzXzur5AAFQjjYUL2g9dx7C51BNopAQeCTtJRkB9X7XbsVBI5JSY+2B+oSD9mPfUNrgtTZho5xJXdjv+SdSaiIswSO3ls4OHTW1dU1kB4rDY5eW52rUwrrAY\/IOLGJfiCWhENOHsExzMYqNseoXd\/6nhx+Dz74oD7B7rDDDoHgCMFR+8DUxP8kqVY14OhqxwvahJN0lvELWbDJWoyUFDQtSE+AOipSP+kpLCA4CUcYV4\/cKEk1jkRt6IuU\/+Hn32kHJFd7JfP8jxfnFZIQGLpmCRy9Ghg\/jYwXHEmWQUG1aP8dFJPn+rlfWzbAmXg\/A47edr3P8ywhWt9vtL\/uL16SxfTlqquuUlxdRZxjnPHafGaccOLSLkitGjV2v9hI80yMbVktXrxYX9mF0BKUPi+oPftgQh177AKOcWbh17rmJBukVh02bFih1QEDBjiBo6tN5W8jHlTHjJvvlCXH3lS9eVSTlOiC7I2GCEkFvrtIf2HgWOqBIMzpyGvPjJIMoyRLe1ka+6qRlF0PG0bKpi2jJTAHiqyBoy0pwiv19fUN+MqMpwh2ztwjPzdrpZb9dkHm+l2ODi9u9XvVeMHbib4qyqs40ZeV2FjFbI6m3zDnpZdeqrbffnu1YMECdfHFF6uzzz67aJtjHJvKPWMfV\/uOnO2UJSfMKYSxuG6yUfMVtdEnAcSuXqdRqaRKGbORWv1CM2zHoCi6Q884QO2VQk0\/bujdQavMg4o9L95x1yo4ctpfZ511liIJakviH+0QD1PXfGdi9byf21Id3yFhBbVlv\/ijjz5S999\/v1aZEih\/5aAljnlBfXFt135H2LiC1oV3vEbys+P7bBra46AuaelOOumk0G2hmDH69csbP2n6aM+jy5zSWbst75yGrRkBxygEsL4HHK+\/\/nrVrFkzNXPmTLXCCivo1E1kxsd7lYIq1dUhJ45NhYna7Kp3nbLkRDmFJOVJGqVCLBWIo+yM9tRFgWMcdabdLn3ocdP0pbxjqeMNzYiiO8+41KFeENAaG2QYQNrz4pXe0wLHRYsWqfvuu0917dpVrb\/++qFcRR9GjRqlhgwZopo2bRoM8h6v7zC1atBGFhanGCdwn\/66xjwyIACPH1SIJhG4ed5W4dFvP7Vm1MYcpy+GwN7x8nlUiIkZB3XtsQRNWlC\/SomfDPJAjus0ZNS5BuiZU7+SFo\/EgJrYVSsmOf7yyy\/qv\/\/9r7Y5rrLKKtrmiNRIKMfdd9+tfv75Z9W8eXPVs2dPdcIJJziDYxybCpeEHnDPPNVry1ZLOdl4KTl6+gK9cX96+U6+RN5myFTVcb3mkR6rBNOzSW9f13ypdvgOkB1z1KahUsxJo9\/Vqr2Xztw29oQPfXK2uvv5OSpKUqLhKHA0NKEfcTw+\/1O\/SIOj3zi9bfL\/JQ+9G+nstPrAZ9W1vdqpPtuuFUiTsPceOOx1tfZKjX3nz8zLqd1a63Xi7WNaF7maXKgcDhs1aqQmT54cODYkEuKFo8DRq0YNc8hJEhxtcLNBoRhAsongly2HDTro8zCGCQLasGuyigHHuExbzeCIXdOUMKAXcIwx62+88Ybq37+\/NgDvvPPOSznkrLvuujqUY7fddtPhHbfeeqtuPcohx8vsYTYVDO73q25q63WaqEG7tQrt\/bAXFqp\/z1ikxh\/W2rfeoAkL1Pi3FqlpJ9SFtnPUmI\/VtI++1+\/br0NDgIx6h2mY52nn1p5r6r67FvpHP4\/aroUa0DFYhWjac7n4dOvr633HEtanqHHabdJfwImxhpX97pyj9t2oeei4wt5raOM3f370tvt45513Ku4+jJJMXOfJ1LPB8YMPPlATJkwoNLHyyiurL7\/8UvF71qxZqm3btk7gGMfskDQ4+o2\/VHAMAt2wz4PmoRRANU5DtB0lOcZdB9UMjq5rXsAxxqwT48h9aS+++GJgnCMqVmIet9hiC7XGGmuolVZaKTJDThxwZEO7e9bKutfX9votzstvGEgWeG+NPqKD7yhHT5+vTho9U\/3nNH\/PTx6aPPMLdfAdM7SUhZRpv1NLk1cuUTUOPbBhqIjfC3sNf0sfLB4YsKkT1eO2T6MuF5\/SDySuKPrZnYyiJXQw9Imqa9qlHnQN6wcSN0AbNIdrn\/28lj57bblaA5oyt+SWZW5NoY9oHE7ttq4aN26cDqp33SicJkwp5QVH+7n11ltPAZj85hCJxsVFrUobrg5rWQFHV3q61MO0w8Edb1Ucl6IuV8675Oi65gUcXVafp46xEZIWDrA66qij1CmnnKIlRJx03nnnHR0Pue+++2r1apR7cRy1KhNLnKNLjKKLl2iUg4rJsNN3u7W0itYOIXHxHrVJF+XVatc1Nj4+84tnDJq2KLUqz8VxhqG+i4ONSWVHiEWUp6rpu8vVYVFzGPS9nx3YrpsG40+dOlWnUXz\/\/ffVOeeco1XcSImmlAKOUWwaNZ5ibI5B70xCcowaT5zvXda83Z6A4zORBwjoFbWm4sxRuepWzOZoBsgmwAm4R48eavnll29wWgYcKWPGjFG33Xabuvnmm\/VpLqzEtak8O6+xBqqoWLeg3J52X8I2ci8o2BtuVGxj0HiDknF7gZHxkbQ7KJ6xFHDk2ahDgR+oh8WW2gcFbJMuqepsQA0aT9QcBrXhN682GKfB+IDivffeq6XSb7\/9Vn3zzTc61Vu1g6P3NvogBw0zDgHH6K1e1KrRNEqjRsXB0TsoHHGwpfzud7\/TYEnBaw8V32qrNVR3+REkrk3FgGPYZu0i7dCXMO9Nr0el+d92ZIkj1fE+068dN2jhmx\/WlhjjAiPtu56iXaU72owKVTFzaoCMIH0XcIwKcXGZQ782guJbbYk5DXCEDmmpVaM2kqjxRDmu+HmVBr1TwDFqNn6Tury2zGrwVhW1avT8JV4DexrerMTMoP\/HOYQTNJ6tUSWuTSVK8nFVYYapGP2kPNp97r0vNMgVm+PT9A2AJOsLYLskFds8nf2F\/4sBxjjgGCekwyVUhXcDuOYAECXVUy8qnMN8H9aWAUL7oBQEqjaQzn3nFZ2ZxXWjiFq\/5vs0HHJMm6+++qp+zX777beUh2sUOBbjuBIFjt5QjChbnysN49ZzPRCadkWtKmrVuGvMqf7bb7+t5s+fX6iLHeX1119X06ZNU8cff7waNGiQ6tOnj7Y9vvXWW9oGQ+gHnnmlFC\/zu4Jj1CYdZjeMI13FHZvZwHnOgCO\/vbdsxG3XdaOIY3eMUm2aPpp4R8bhKlGHzWMciZVDhsmCFAaq5n1tf6pPHRyTCOVA+3LmmWfqdI2kZDT\/r7322r4ZcsLAPm6sXtDaSxJo465vv\/qua948a\/ffxPyJt+rSlI06cCUxd0m3UVG1Ks4zdno4Y1Phlo4LLrhADR48WIPjNttso7PofPbZZ5FxXC4E8k5UVJ5Q143VSDxGijN9cVHpufQ7qg4gRb7QHTdYIl2HZXyJaovv42wUAEWU+jMOHQzgBqmM\/fofdgCJI7Ha8xc292mCI4fGBx54QK2++uqqc+fOWnsSVljTrt6qdjvepBl8V+6NLCmgdVnTUXXirHkbIBkDsabkF01aChabY9SspfN9xcERBxucccyJlmHa4NirVy\/tsUcsGQ45G264YcmU8DJ\/lFrQxRPSK\/HYUmac50seXIINxNkoXCRjF9Wm3X3AJ86tGWGHnChPVXv++NtcPxY2d+Z9\/9f2y8Qlx\/fee097qb788suaP8grvOqqq4bOLpeFc2l4lEe3KziSJKNS6s0El3GspuKseW\/DACQhPeZWkK233lofWEotJsmELZEyN4TCoc73fs6cmWfs+MugWMygbDlm\/sPacl0jaSXKKJW2Yc9XJTiOHTtWZ8QntIO4I+yM5GHcfPPNE6GFHziGhXO4bqx0zmufdM1jmsjAEm4kzkYRdcCga3EkcFMfKdhVAg7rgwt4m3faN3WESZwGOM\/Z+ofEwZG+kEWKUKahQ4c2SAAQNM2oR0eMGKHsuxnDloSxP\/bu3dv3yip48LDDDkt4VVV3cy6JL8JGAGB98sknugqx2WuuGZ68woUa2IcJb7NBjGu1eI\/f57zTPFMKOPIO2mJMmLNo0\/xv2jV1osaRVqKMqPeW8n1VgiMqVFLHcXUKBdUqyQBcnHFciOEFxzCbWTFhFrbzTVxAcOl\/uerEAUcXu6OrarPY8QVJeWE3qnjf5fVYDQNVM7e37LZMKuBo+oYHN4H+F154oVazYn6IyrUaRUNjb6SeN+Wc4Q8Op5g08lRcEl+Umx7MB+k1bXB8\/PHHdYo8v8+RHM0zpYAj7zCaAwCSH8LukI5Nu3adMLqklSgjzbmoSnC01apchEwWHVRG1113nd4cSi1B4OgXzhHHTmb6ZZ4xzjG2g0epfS\/n88WAY1hIjKv0VuwYg+IU48yh7bFKP+hzkC3VAOnDBzdNFRwNPbBpPfTQQ2qfffZRSInwRFghrIkUjXPnztXVyIOJc1sYMFKv3DbHYuc7jefirPk03u\/XZqVtjnafvN65rh7aWVxTFQdH1KaogcgbecQRR+h58DrkcEo+8cQTdYosMulEbQpRi9ZvooI8HVGTHn\/vkmw2cQrPEXi\/7ipNnNWCcdovR924G0WYt2gcgCp2bEHSa1SYh\/d9Zhw45lQTOBZLF\/u5IA9Vvw3QdeNLol\/V0kbcNV+Ofgs4loPKS7+jouD46KOP6hANU\/DKw\/nGC46cdlErXXPNNdopp1Q9vh84BjlzZFktWuqSirtRhDnEuMaKltLnIBD0XnAc9Q4j4WLvJENPkDRs3jds92XUGccdnnicY1Q\/i\/keD3EkybDbO7J4yi+GFn7PxF3zSb03rJ00wNH2qA1yyPE7HInkWI4Zj3iHSQKwcOFChUdUv3791IwZM\/TddmH31bl03Y\/5g+xVadvJXPpbqTpxN4owh5i4AFXMmIOk07gHHAPyqMXpd5DWwLwPh5wbzzu26sHRmwDA0BhHt+HDh6uWLVvqjwQc5yluBYrj9VvMenV9JmlwtO9gpA8Cjv4zUVHJkS6RW\/XTTz9Ve+21l04TByjahaw4xDtuvPHGTonHXRacH\/P7qeSKccZxeX9W6sQFxzCnnLgAVSyN\/FS7cQ84BuSjwJE+8j5COcZffUrVg6MrTQUcaxsczR2MhJxce+21Ao4BjFFRcESlSiacbt266Zs3yIgzfvx4364Sv3XjjTfqgOhSix\/zx0kdVur7s\/J8seDop4aMC1DF0gipj1R8JsMN7cQJxaE+4EhoDzZH1sX44367qsrbL97X9qcP1Kt3ni3gWOykVdFzcdd8ObqehORopMWOHTsWYi\/92o3K7iNq1TLMOLcMcC3VBhtsoCVCbjEnEcBWW22l8z3aBSeChx9+WG233XZqs802S8Uhh\/d5N9ZySTtlIHdRryhmowhyyknbU9UM0M\/uGffddnYepEeTEMCPiLTdeMHb6vMHLxRwLGqVVddDxaz5tEeQBDgaaRGQNOEZAo7hM1dRyRE1apMmTbRu35v3Mc0FF6Q2su2OSAz8H7U5ptnPSrddzEbhB0Tl8FQ1tPLaPePEOJo2DDjy\/xnd67QkGlR4H+qpLIKjfYONnThA1KrZUqv6patzkfCKAce4V5IZvsnimqooONobDtlAvv76a7XccsuV7HATBSpBE2V7VNKG612CUe\/L6vfFgKOfU045PFVtcLRVocUAs+kvbUbFqDLe0Y9MUs0nX54pydEcRkny782qk8WNLCkeK2bNJ\/XuoHaCJEfqk8vVpKvzU5maYP0wz9M4Djm8M86VZAKOaa+OhNsPY35jGyNGkRI3vjHhrla0uWI2CmOvs+kWN6dqKYP2qsKLBWbUw5SoZOq874oHXlIrPX5GpsARHiCsg0JqMJEcl6y6YtZ8KevV5dkgcDR2RMCK4qcyTQMcXfrsrZPFA1fVSI7FELzYZ8ImygT9m43Rvoy42Pdl9bliNgo\/j9Vy2m6974+bAMDMlSs4mve1GHtEZsCRkA5z4w0AKeD4G4cWs+bT5m+zX8W57aNYtWqcd8QZt4BjHGpVsG7URKGKyzMomqkpZqPwA6Ny3krizY1abHwltlPWgesdnkiOkx8dk4lbLLimioLz28CBAwPB0fXGhQqycuKvLmbNJ94JT4PF3PbhvUnDby79bvswr07qRhHTntzKkfYqSaj9KHBM6DWZb6bYjcLrsRo3lKIUwnnBuVip1U4eH9Yf876sgCNOONyQwHVYbLph4Ci3cjQuZSkm+mzc2z68t3L43Z7hd9vHSSedpJo1a6Y6dOig1llnncTGILdyJEbKdBsScHSjb7Hg6A2niBtK4dY7\/1peB5xipVZAlbbCwjjogXkfDjlTxtxWVZKjX+Jx0jCSo5iUjFHeqnIrR\/WAY1yeYI\/jxo6w2zNMHdsh58EHH9TRA9gvk8wQJLdyxJ3BCtUXcHQjfLHgaHusFuMt6ta74Fq25FqO5AO8r9nLd6j\/3PO3qgJHL4WC0sdRz9zYwd955o9i13ypazbp54u1OXIFVaNGjRJPn5fFNSUOOa1bJ70ua6a9YjcK22O1GsAx7VjVrICjd2FGSY5yK0e2Jce+ffsWJEfXUA4Bx9+4pObA0Xs6JttO0GWueWT+OMhdLDjaHqPFeovG6ae3rq3GLYdKl3d8OuU+9eKtp1a15CjgGL2qil3z0S2Xt4ZIjqXTu6bA0ZtlJ+juuiyK+KVPdfwWit0obI\/RUVPnhd5sEb9X0U\/YNs+wOyajW3Krwfs+fuVJ9dI1\/TIFjkGjyzN\/FLvm3VZK+Wq5ZLLxi58UybGGJUfv8sNtnfshbekxz8wfhz1L2SgMKH34+XdlB0dj8yTtW9hlxXFoEVYXu+bUpx8ScEyKoBVsp5Q1X8Fu+746KpONgGP4jNWU5Og31DBwzGMcVxwGLmWjWH3gs+raXu3UfS8v0NeQhd1sEadPLnVPGv2uev\/TRWrY\/22qwXHMUZuqzu1auDxaVB3eRwq5ly\/dVyTHoihYPQ+VsuarZxRuPRFwzDE4Gvtj79691SGHHFKghFkUeYzjcmObJbV++OEHNX\/+fLXmmmvGduve7845aq0VG6tpH32vBu3WSu3XoXmcV5dUd9gLC9W\/ZyzS7z1qzMdq\/GGt1dorpedcMWjCAvXIs9MyA45m\/UNk70XHfJZnzYqAo3irms2nZiVHY29koEEOOXmM44qDOtDwk08+KSrmCWmKOxGJlbrmoPZq+7rygePo6fPVSaNnasmV33Mv7RRn2LHrDn1ytrp6\/Cs6D6+5Dih2I2V6wMQ+XnnllTrWUcwODQkv4CjgWBPg6BfkDMOHAWPeT8Zx9uBSNgqC6C97rF5fGFxOlSrjMw5BXDfF32knjzfve+2U9lUPjoBhfX29vks1qIjkWF1XWW8IjwAADX9JREFUVsXh2Th1Ra2aM7VqkIeqTYY8M38c5ikFHA1gRN1qEac\/rnVN+Ajxjeuu0iR1cM4KOLremZpn\/ihlzbuuz2qpJ+CYM3DkloG5c+cupUoVcIzPkqVuFJVK4G4SDzDicoBzJWI548+mKmhUunfvrm677TZFbs0wm2MeHdZKXfPFzEulnvFLPP7www+nkiFHEo9XapZ\/fW9QeizvBpDnk3GcKcryRkEoCZJj2ipV6Jk1cPzwww\/V8OHDVcuWLfWdjt7DZJ4d1kpxQovDW9VQ1y\/x+B133KHBsRgnvLAxSeLxaphxhz4IODoQqUovfnXruVLYPNu0bKr6bLum6yNF15vy2ky178jZZZFS43TSa5MfMWKEGjNmjNphhx0K3tt+KeQMf+TRYa0UJ7Q4c1MNdSXxeM7Uqi6LTsDRhUrVeSu6W8\/LW4v1tPd93+kbPMoBxqWMDkmxrq6uATheeumlaujQoVqSpOSZP7KsLYm7LsTmKOC4FAXyzPxxGChPG0UcunjrZgkc6SsAaatVGY\/tvZpn\/sjTmhdwFHAUcCxy58\/TRlEkiQqSFmrV0w\/cRpGyrtqLnQRAEvM3nK08rXkBRwHHQHDkDrtqD9qu5EZL2reRI0eqPffcU+gUMhHksAQcD+mySeTlyJWcT9d3m00zj\/yRpzXPuuVaq0WdB+qlwYXd2JknTJiQOM+bd2XpJqSazZATthHYGetdNwypJxQIowAbTK+9utQEOAp\/yFpPgwIdO3ZUo0aNSqPpVNrMJThCSZOxPhWqSqO5owAaiFrSQgh\/5G4Jpz7grPFIbsEx9ZUgLxAKCAWEAkKBzFJAwDGzUycdFwoIBYQCQoG0KCDgmBZlpV2hgFBAKCAUyCwFBBwzO3XScaGAUEAoIBRIiwK5BEdzO8H48eMLdB0wYEDoNT5pTUA1t0uw+LBhw3QXcevnOjApDSngl8\/3b3\/7W4PLtbNGs7zMu3fuvDGf9vd+CdqzNq9B\/fVLIZiXsYfNYS7BkYk\/9dRT1dlnn63at29fK2s80XHYmVTeeeedBllVEn1RxhtjY\/GmX8vykPIy797ru\/yuuuOQQCF7kP13lufX23cz7mnTpily75r9MA9jj5rHXIJjrW1oUZNczPc2cxgG6tOnj0iPHmICJsRuDRkyRDVt2rQYUlfVM3medy6CnjJlip5L1vwRRxyhgRGNSa3uGeYwxCK8\/PLLNTgaqbHWxx7FeLkER29+ySgi5e37oFO1fZtD3mgSNF57Q806OOZ93u25JM5z4MCBBcDwUz1mnQcAwcGDBysOvRyKDDh6x1qLY3eZu1yCI0xw1llnFehTy\/YEl0UQpGqxJUXvbQ7FtFuLz9j2Ocbnl6s0K+P20xDkZd6NtNS7d29tL\/ZKirVoimEfpGy11VZLHQRsU0Etjt2FJ3MJjt4LXv0ufHUhXq3WyfMmGWdOvXYqP7tVnPYqXTev827GDf2NerzWwZHxcQHxOeeco7OFeaVkAUelah4cbSkxSELMq9ogaDPOu3qtFJDKsso+j\/PuB4zMf62rFhEIunTpUrCnhqmQ87o\/1jw4umx0tWpsdxl7UB1bnSYOOe6UzLqDTp7mPUzS96oSa2mP8As\/MiuckK0NN9ywgTd\/LY3dnZNzIDkG2dOMc0nWVWFxJjtO3by49Mehibeu16vPa7cqpe1KPZuneY8yp+QlnMFPMszL2MP4LJeSozcJQJadKNLcRPMSDF4KDb2n8FpIJpGHeQ+SnmzTS14C4SUJgP8OkEtwLGUzlGeFAkIBoYBQoPYpIOBY+3MsIxQKCAWEAkKBmBQQcIxJMKkuFBAKCAWEArVPAQHH2p9jGaFQINcUmD9\/vvr6669V27Zt1TLLLBNJix9++EG9+eabaoMNNlDNmzfXz\/CDr8JPP\/2kP3MpCxYsULNmzVKbbbaZWmGFFVweCazzyy+\/6DEst9xyNZGmsCRilOlhAccyEVpeIxQQCpSfAiTNJ0foGmusoc4\/\/3w1ffp09eijj6rTTz9dtWnTpkGH8NQdO3asOuGEE9S5556rs2g9+eST6quvvlKnnHKKuuqqq1Tjxo11mINdDHAtXry4wecvv\/yyuvbaa9V1112nVl555QbfkWrQpBu85ZZb1GOPPbYUcUzmGoDVG4Nafkrm740CjvmbcxmxUKDmKQCY\/Otf\/9LgdNppp6kDDjhAS13z5s1T11xzjZo4caLiarFu3boVpEkkM27qOfDAA9Xdd9+tc46SVP7iiy9WLVq0UGeeeabac8891T777NOAfn5X4EUR2L7WDLB+6623GjyC5NqsWbNCxh4BxyiKJv+9gGPyNJUWhQJCgQpS4JtvvtHS4sMPPxzaC6RAJMS+fftqlSWSJaDUpEkT9eGHH6rWrVur\/\/3vf\/o3ITqA5MYbb6zWWWedQrv777+\/Wn\/99X3fU0q2JG9CewHH8i8oAcfy01zeKBQQCqRMAWL3kLyWX355ddNNN2mJ8corryzYC1GFTp06VWFf3HHHHdWPP\/6oXnvtNfXtt9+qBx98UD3wwAMa9I499li19tprK+yWl1xyierRo4fu+bhx49ROO+2kv8ceOWjQIK1+tQv\/k7e0Xbt2uh92sVWm1AOUbbXsM888oz755BORHFNeJ2HNCzhWkPjV+mo2lv79++sbJjiBUzgFc8ImvRT320kRClQjBQC8yZMnF7q2cOFCbUfcYost9I9fWXfddTXoAZhc+Pv3v\/9dO9Dsuuuu6oknntC2R9pFEuX7FVdcUdsgue6JVGvwy0knnaQOO+ywglT53nvvafXtn\/70p6X4xQt82EEPP\/xwDcI2iIrNsbIrTMCxsvSv2rfbYMgGwMWv5jqfqu20dCz3FLDBEW\/RMWPGaFDcbrvtNG0MWCL14Y1KARw7d+6szjvvPC0FYqPEJoj9EWnugw8+0PcdkjHn6quvVqussori1gok0tVWW61BknIADgAFkLnxYuedd9Yq2iuuuELttttu2p553333FS5VxinnoYceUiNHjlS33nqrbtuviFq1\/EtbwLH8NM\/MG9kQZsyYofu70kor1cxt95mZAOlo0RT4+OOPtQ0RlSUSHLbDe+65R0txn376qbrxxhs1YHXs2LHgkMPneJU+99xzCsnvz3\/+s\/YofeSRR9Srr76qVl11Vf0sISGoYC+88MIGkh4HSmyT8AqAC6B9\/\/336osvvtAergDy6quvrpZddtkG47rhhhu0WhUwtS\/MBqjpD\/UFHIteCkU\/KOBYNOlq\/0GTWxJ7C6fh9u3b1\/6gZYSZpsDPP\/+sJk2apIGrUaNGWiKcNm2alvpQXXKRMV6r999\/v775\/sgjj9TqUFSlJqQCOyEFRxwKqtGePXtqAONqJ8oxxxyjzQ52AYjffvttbbccPny4BlhsknjK4iR08sknq1122UWbLHAGomDzRELFxukt9B3gBGwFHMu\/LAUcy0\/zzLzR2B7nzp0rtsbMzFq+O4q98a9\/\/asGLuIVp0yZoiU9VKiAJGCDBIi9EHUp0iW2ReoCZniJzpw5U\/Xr109Lfddff732Ut16663VokWLtMr1\/fff1yrQurq6ArEB38cff1zbK3keCXWjjTbSkiWA\/cYbb2j1KkkBeBc\/ACTfffnll9reaRfTjrl8WcCx\/OtawLH8NM\/EG+1LYLHZ4MHHabhly5aZ6L90Mp8UAGSwKxKXyKGOQPynn35aB\/MDVkiKqDc5+CFBmqw5\/I2N8vjjj9cSI1Le6NGj9WcAKA46gCPOOQYc7ZAOnGouuugiHeoB+NIefSDOEulxjz320CpTgBD1KTGUXsnTnjFMGqhjjepWwLH861nAsfw0z8QbOUHfe++9GhBhagKgcTYw3quZGIR0MtcUIDsOEh53t\/KDvQ+V6bBhw7SUhzoTgDIqTohFvCO2SZxtADfWPw5pqD+ROpFMAUqAEenTTiWH8w58M2HCBA2UABoqVNS3s2fPVrfddpsGRvqCRBmUyg4VLPyGZGqy8Qg4ln8pCziWn+ZV\/0a\/sA2jYkUdhN1GilCg2ikAoKDuRM3JmgYQAUe8rpEgjU3RjAPJEDXszTffrDUkeI5iR0SVCqC+\/vrrGiAp2BzRqABirVq1UgDx0KFDVX19vY6HRA3Le\/HyJu0c\/xtb6B\/+8AedfGDNNdcskJB3I1VyEKUt3omUarLxCDiWf7UJOJaf5vJGoYBQIGUKPPXUU+roo4\/WoRioTrt27ap23313xWXGtkeoDYyAEd6oqFZJDIBkhxRHgD\/2RxvQUK0iERLfSLIAABUJFEkR6ROvVcAUIMTRx6hgUZUileIla6eu86agQ\/VL0gJCRSgCjikvGJ\/mBRzLT3N5o1BAKJAyBQBFQAovVFttGvZanvGry+eEU3hDMILaMonIucHDhGK4DBfpkZATVK7emz\/kVg4XCiZbR8AxWXpKa0IBoYBQQChQAxQQcKyBSZQhCAWEAkIBoUCyFBBwTJae0ppQQCggFBAK1AAFBBxrYBJlCEIBoYBQQCiQLAUEHJOlp7QmFBAKCAWEAjVAAQHHGphEGYJQQCggFBAKJEsBAcdk6SmtCQWEAkIBoUANUEDAsQYmUYYgFBAKCAWEAslSQMAxWXpKa0IBoYBQQChQAxT4fyhLYwEQkC4MAAAAAElFTkSuQmCC","height":219,"width":364}}
%---
%[output:5ac59a31]
%   data: {"dataType":"text","outputData":{"text":"\n训练完成！\n","truncated":false}}
%---
%[output:91f98de3]
%   data: {"dataType":"text","outputData":{"text":"最终损失: 0.765226\n","truncated":false}}
%---
%[output:472f67e9]
%   data: {"dataType":"text","outputData":{"text":"最大绝对误差: 2.734103\n","truncated":false}}
%---
%[output:23b136cc]
%   data: {"dataType":"text","outputData":{"text":"平均绝对误差: 0.654802\n","truncated":false}}
%---
%[output:9945c17f]
%   data: {"dataType":"text","outputData":{"text":"数据维度检查:\n","truncated":false}}
%---
%[output:291985bb]
%   data: {"dataType":"text","outputData":{"text":"XTrain: 1×71\n","truncated":false}}
%---
%[output:30aad269]
%   data: {"dataType":"text","outputData":{"text":"YTrain: 1×71\n","truncated":false}}
%---
%[output:891ddc5c]
%   data: {"dataType":"text","outputData":{"text":"XVal: 1×30\n","truncated":false}}
%---
%[output:7512053a]
%   data: {"dataType":"text","outputData":{"text":"YVal: 1×30\n","truncated":false}}
%---
%[output:14cee5dd]
%   data: {"dataType":"error","outputData":{"errorType":"runtime","text":"错误使用 <a href=\"matlab:matlab.lang.internal.introspective.errorDocCallback('dlarray\/dlgradient', 'D:\\matlab2025b\\toolbox\\nnet\\deep\\deep\\@dlarray\\dlgradient.m', 105)\" style=\"font-weight:bold\">dlarray\/dlgradient<\/a> (<a href=\"matlab: opentoline('D:\\matlab2025b\\toolbox\\nnet\\deep\\deep\\@dlarray\\dlgradient.m',105,0)\">第 105 行<\/a>)\n未跟踪要微分的值。它必须为跟踪的实数 dlarray 标量。请在 dlfeval 调用的函数中使用 dlgradient 来跟踪变量。"}}
%---
