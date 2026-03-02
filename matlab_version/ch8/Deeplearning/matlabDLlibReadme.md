# MATLAB 深度学习库示例：神经网络拟合三次多项式

## 概述

本示例代码展示了如何使用 MATLAB 深度学习工具箱构建一个简单的神经网络，用于拟合带噪声的三次多项式数据。代码实现了从数据生成、网络构建、训练到结果可视化的完整流程，是学习 MATLAB 深度学习基础的优秀示例。

### 主要功能
1. **数据生成**：生成带高斯噪声的三次多项式数据
2. **网络构建**：创建包含全连接层和 ReLU 激活函数的浅层神经网络
3. **模型训练**：使用手动梯度下降法训练网络，包含 L2 正则化
4. **结果评估**：计算测试均方误差并可视化拟合效果
5. **损失曲线**：绘制训练过程中的损失变化曲线

## 数学原理

### 1. 目标函数
代码拟合的目标函数为三次多项式：

$$
y = x^3 - 2x^2 + x + 1
$$

### 2. 训练数据生成
在区间 $[-2, 2]$ 上均匀采样 100 个点，并添加高斯噪声：

$$
x \sim \text{Uniform}(-2, 2), \quad \text{共 100 个点}
$$

$$
y_{\text{true}} = x^3 - 2x^2 + x + 1
$$

$$
\epsilon \sim \mathcal{N}(0, 0.1^2)
$$

$$
y = y_{\text{true}} + \epsilon
$$

### 3. 神经网络模型
网络结构为单输入单输出的浅层网络：

$$
\text{输入层} \rightarrow \text{全连接层}(10) \rightarrow \text{ReLU} \rightarrow \text{全连接层}(1) \rightarrow \text{输出}
$$

前向传播公式：

$$
z_1 = W_1 x + b_1
$$

$$
a_1 = \text{ReLU}(z_1) = \max(0, z_1)
$$

$$
\hat{y} = W_2 a_1 + b_2
$$

其中：
- $W_1 \in \mathbb{R}^{10 \times 1}$, $b_1 \in \mathbb{R}^{10}$：第一层权重和偏置
- $W_2 \in \mathbb{R}^{1 \times 10}$, $b_2 \in \mathbb{R}^{1}$：第二层权重和偏置

### 4. 损失函数
损失函数由均方误差（MSE）和 L2 正则化项组成：

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

$$
\mathcal{L}_{\text{L2}} = \lambda \left( \sum_{j} W_{1,j}^2 + \sum_{k} W_{2,k}^2 \right)
$$

总损失：

$$
\mathcal{L} = \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{L2}}
$$

其中 $\lambda = 0.001$ 是正则化系数。

### 5. 梯度下降更新
使用手动梯度下降更新网络参数：

$$
\theta^{(t+1)} = \theta^{(t)} - \eta \nabla_{\theta} \mathcal{L}
$$

其中 $\eta = 0.01$ 是学习率，$\theta$ 表示所有可学习参数。

## 网络架构

代码构建的神经网络包含以下层：

| 层索引 | 层类型 | 参数 | 输入尺寸 | 输出尺寸 |
|--------|--------|------|----------|----------|
| 1 | 特征输入层 | 1个特征 | 1 | 1 |
| 2 | 全连接层 | 10个神经元 | 1 | 10 |
| 3 | ReLU激活层 | - | 10 | 10 |
| 4 | 全连接层 | 1个神经元 | 10 | 1 |

**总可学习参数数量**：4组（每层权重和偏置各一组）

## 训练过程

### 训练参数
- **训练轮数（Epochs）**：1000
- **学习率（Learning Rate）**：0.01
- **批大小（Batch Size）**：10
- **L2正则化系数**：0.001
- **数据量**：100个样本

### 训练循环步骤
1. **数据准备**：将数据转换为 `dlarray` 格式，使用 `'CB'`（通道×批次）数据布局
2. **数据打乱**：每轮训练前随机打乱数据顺序
3. **分批处理**：将数据分成大小为10的批次
4. **前向传播**：计算当前批次的预测值
5. **损失计算**：计算 MSE 损失和 L2 正则化项
6. **梯度计算**：使用 `dlgradient` 自动计算梯度
7. **参数更新**：手动应用梯度下降更新网络参数
8. **损失记录**：记录每轮的平均损失

### 关键 MATLAB 函数
- `dlarray`：深度学习数组，支持自动微分
- `dlnetwork`：深度学习网络对象
- `dlfeval`：在深度学习数组上评估函数并计算梯度
- `dlgradient`：计算梯度
- `forward`：网络前向传播

## 代码详细说明

### 1. 数据生成与预处理（第7-16行）
```matlab
x = linspace(-2, 2, 100)';  % 生成100个点的输入数据
y_true = x.^3 - 2*x.^2 + x + 1;  % 真实三次多项式
noise = 0.1 * randn(size(x));  % 高斯噪声
y = y_true + noise;  % 带噪声的训练目标

% 转换为dlarray格式，注意转置为行向量以符合'CB'格式
x_dl = dlarray(x', 'CB');  % 形状：1×100
y_dl = dlarray(y', 'CB');  % 形状：1×100
```

### 2. 网络构建（第18-25行）
```matlab
layers = [
    featureInputLayer(1)  % 输入层，1个特征
    fullyConnectedLayer(10)  % 全连接层，10个神经元
    reluLayer  % ReLU激活函数
    fullyConnectedLayer(1)  % 输出层，1个神经元
];
net = dlnetwork(layers);  % 创建dlnetwork对象
```

### 3. 损失函数定义（第33-53行）
```matlab
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
```

### 4. 训练循环（第54-103行）
```matlab
% 训练循环
numSamples = numel(x);
lossHistory = zeros(numEpochs, 1);

% 转换为普通数组以便打乱
x_data = x';  % 转置为行向量
y_data = y';  % 转置为行向量

for epoch = 1:numEpochs
    % 随机打乱数据
    idx = randperm(numSamples);
    x_shuffled = x_data(idx);
    y_shuffled = y_data(idx);

    epochLoss = 0;
    numBatches = 0;

    % 分批次训练
    for batchStart = 1:batchSize:numSamples
        batchEnd = min(batchStart + batchSize - 1, numSamples);

        % 提取批次数据并转换为dlarray
        x_batch_data = x_shuffled(batchStart:batchEnd);
        y_batch_data = y_shuffled(batchStart:batchEnd);
        x_batch = dlarray(x_batch_data, 'CB');  % 形状：[1, batchSize]
        y_batch = dlarray(y_batch_data, 'CB');  % 形状：[1, batchSize]

        % 计算损失和梯度
        [loss, gradients] = dlfeval(@modelLoss, net, x_batch, y_batch, lambda);
        epochLoss = epochLoss + extractdata(loss);
        numBatches = numBatches + 1;

        % 手动更新参数
        for i = 1:numel(net.Learnables.Value)
            net.Learnables.Value{i} = net.Learnables.Value{i} - learningRate * gradients.Value{i};
        end
    end

    % 记录损失
    avgLoss = epochLoss / numBatches;
    lossHistory(epoch) = avgLoss;

    % 每100轮打印损失
    if mod(epoch, 100) == 0
        fprintf('Epoch %d, Loss: %.6f\n', epoch, avgLoss);
    end
end
```

### 5. 测试与评估（第105-111行）
```matlab
% 测试拟合效果
y_pred_test = forward(net, x_dl);  % 使用训练好的网络预测
y_pred_test_np = extractdata(y_pred_test)';  % 转换为普通数组

% 计算均方误差
mse_test = mean((y - y_pred_test_np).^2);
fprintf('训练完成，最终测试MSE: %.6f\n', mse_test);
```

### 6. 结果可视化（第113-135行）
```matlab
% 绘制结果
figure('Position', [100, 100, 1200, 400]);

% 子图1：拟合结果对比
subplot(1, 2, 1);
plot(x, y, 'b.', 'MarkerSize', 10);  % 原始数据点
hold on;
plot(x, y_true, 'g-', 'LineWidth', 2);  % 真实三次多项式曲线
plot(x, y_pred_test_np, 'r-', 'LineWidth', 2);  % 神经网络拟合曲线
xlabel('x'); ylabel('y');
legend('带噪声数据', '真实多项式', '神经网络拟合', 'Location', 'best');
title('神经网络拟合三次多项式');
grid on;

% 子图2：训练损失曲线
subplot(1, 2, 2);
semilogy(1:numEpochs, lossHistory, 'b-', 'LineWidth', 1.5);
xlabel('训练轮数 (Epoch)'); ylabel('损失 (Loss)');
title('训练损失变化曲线');
grid on;
xlim([1, numEpochs]);
```

## 运行结果

### 预期输出
1. **训练过程输出**：每100轮打印当前损失值
   ```
   Epoch 100, Loss: 0.349490
   Epoch 200, Loss: 0.331943
   ...
   Epoch 1000, Loss: 0.252959
   ```

2. **最终测试结果**：显示测试均方误差
   ```
   训练完成，最终测试MSE: 0.188404
   ```

3. **网络结构信息**：显示网络层信息和参数数量
   ```
   网络结构信息：
     具有以下层的 4×1 Layer 数组:
       1   'input'   特征输入   1 个特征
       2   'fc_1'    全连接     10 全连接层
       3   'relu'    ReLU     ReLU
       4   'fc_2'    全连接     1 全连接层
   可学习参数数量：4
   ```

### 可视化结果
代码生成包含两个子图的图形窗口：

1. **左图：拟合效果对比**
   - 蓝色点：带噪声的训练数据
   - 绿色线：真实的三次多项式曲线
   - 红色线：神经网络拟合曲线
   - 展示神经网络对三次多项式的拟合能力

2. **右图：训练损失曲线**
   - 纵坐标采用对数刻度
   - 展示训练过程中损失值的下降趋势
   - 验证训练过程的有效性和收敛性

## 使用说明

### 运行要求
- **MATLAB版本**：R2020a 或更高版本（需要深度学习工具箱）
- **必要工具箱**：Deep Learning Toolbox

### 运行步骤
1. 确保 MATLAB 已安装 Deep Learning Toolbox
2. 打开 `matlabDLlib.m` 文件
3. 直接运行脚本（点击运行按钮或按 F5）
4. 查看命令行输出和生成的图形窗口

### 参数调整建议
1. **网络结构**：
   - 修改 `fullyConnectedLayer(10)` 中的神经元数量
   - 添加更多隐藏层或使用不同激活函数

2. **训练参数**：
   - `numEpochs`：增加训练轮数以提高精度
   - `learningRate`：调整学习率以优化收敛速度
   - `batchSize`：调整批大小以平衡内存使用和训练效果
   - `lambda`：调整正则化系数以控制过拟合

3. **数据设置**：
   - 修改 `linspace(-2, 2, 100)` 调整数据范围和数量
   - 调整噪声水平 `0.1 * randn(size(x))`

### 常见问题与解决方案

#### 1. 运行错误："Invalid input data. 通道维度大小无效"
**问题原因**：`dlarray` 数据形状不正确。`'CB'` 格式要求输入为行向量形状 `[1, N]`。

**解决方案**：确保数据转置为行向量：
```matlab
x_dl = dlarray(x', 'CB');  % 注意转置符号 '
```

#### 2. 训练损失不下降或发散
**可能原因**：
- 学习率过大或过小
- 网络结构过于简单或复杂
- 数据噪声过大

**解决方案**：
- 调整 `learningRate` 参数（如 0.001, 0.01, 0.1）
- 修改网络结构（增加层数或神经元数量）
- 减少噪声水平或增加数据量

#### 3. 过拟合现象
**识别方法**：训练损失持续下降但测试误差增大

**解决方案**：
- 增加 L2 正则化系数 `lambda`
- 使用更简单的网络结构
- 增加训练数据量
- 添加 Dropout 层

## 扩展应用

### 1. 拟合其他函数
修改数据生成部分即可拟合其他函数：
```matlab
% 拟合正弦函数
y_true = sin(2*pi*x);

% 拟合指数函数
y_true = exp(x);

% 拟合分段函数
y_true = piecewise(x);
```

### 2. 多变量回归
扩展网络以处理多变量输入：
```matlab
% 修改输入层
featureInputLayer(3)  % 3个输入特征

% 生成多变量数据
x1 = randn(100, 1);
x2 = randn(100, 1);
x3 = randn(100, 1);
y = 2*x1 + 3*x2 - 1.5*x3 + 0.5*randn(100, 1);
```

### 3. 分类任务
修改网络结构和损失函数用于分类：
```matlab
% 修改输出层和损失函数
layers = [
    featureInputLayer(2)
    fullyConnectedLayer(10)
    reluLayer
    fullyConnectedLayer(2)  % 二分类输出
    softmaxLayer
];

% 使用交叉熵损失
loss = crossentropy(Y_pred, Y);
```

## 总结

本示例展示了 MATLAB 深度学习工具箱的基本使用方法，通过一个简单的回归任务演示了：

1. **数据准备**：如何生成和预处理训练数据
2. **网络构建**：如何使用 `dlnetwork` 构建自定义网络
3. **训练实现**：如何手动实现梯度下降训练循环
4. **正则化应用**：如何添加 L2 正则化防止过拟合
5. **结果评估**：如何评估模型性能并可视化结果

这个示例代码结构清晰、注释完整，是学习 MATLAB 深度学习的优秀起点。通过修改网络结构、调整训练参数和扩展应用场景，可以进一步探索深度学习在不同领域的应用。

---

**作者**：MATLAB 深度学习示例
**最后更新**：2026年3月
**文件位置**：`d:\RL\RLWestLake\matlab_version\ch8\Deeplearning\matlabDLlib.m`