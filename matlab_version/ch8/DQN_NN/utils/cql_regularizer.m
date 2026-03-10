function penalty = cql_regularizer(q_values, actions, alpha)
    % cql_regularizer: 计算保守Q学习(CQL)正则化项 (dlarray兼容版)
    % 输入:
    %   q_values - Q值矩阵 [num_actions, batch_size] (dlarray或数值数组)
    %   actions  - 动作索引向量 [1, batch_size] (整数1~num_actions)
    %   alpha    - 正则化系数 (标量)
    % 输出:
    %   penalty  - CQL正则化项 (标量dlarray)

    [num_actions, batch_size] = size(q_values);

    % 数值稳定的log-sum-exp计算 (支持dlarray)
    % 减去每列的最大值防止指数溢出
    max_q = max(q_values, [], 1);  % [1, batch_size]
    q_shifted = q_values - max_q;  % 每列减去该列最大值

    exp_q = exp(q_shifted);
    sum_exp_q = sum(exp_q, 1);     % [1, batch_size]
    log_sum_exp = log(sum_exp_q) + max_q;  % 恢复原始尺度

    % 获取每个批次中对应动作的Q值 (dlarray兼容索引)
    % 使用与DQN_NN_Agent.m相同的sub2ind方法
    batch_indices = 1:batch_size;
    q_selected = q_values(sub2ind([num_actions, batch_size], actions, batch_indices));

    % 计算正则化项: log_sum_exp - q_selected 的均值
    cql_diff = log_sum_exp - q_selected;  % [1, batch_size]
    penalty = alpha * mean(cql_diff);
end