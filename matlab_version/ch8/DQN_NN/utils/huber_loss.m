function loss = huber_loss(pred, target, delta)
    % huber_loss: 计算Huber损失，对异常值比MSE更鲁棒
    % 输入:
    %   pred   - 预测值 (dlarray或数值数组)
    %   target - 目标值 (与pred相同维度)
    %   delta  - Huber损失阈值 (默认1.0)
    % 输出:
    %   loss   - Huber损失值

    if nargin < 3
        delta = 1.0;  % 默认阈值
    end

    error = pred - target;
    abs_error = abs(error);

    % Huber损失定义
    quadratic = min(abs_error, delta);
    linear = abs_error - quadratic;

    loss = 0.5 * quadratic.^2 + delta * linear;
end