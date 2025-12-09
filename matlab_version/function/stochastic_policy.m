function action = stochastic_policy(state, action_space, policy, x_length, y_length)
    % 模仿一步随机过程。在state下，从动作空间action_space中按照policy获取一个具体的action。
    % Extract the action space and policy for a specific state
    state_1d = x_length * (state(2)-1) + state(1);   % 转到s_i编号的状态
    actions = action_space{state_1d};                % 第i个status可以做的动作
    policy_i = policy(state_1d, :);

    % Ensure the sum of policy probabilities is 1
    assert(sum(policy_i) == 1, 'The sum of policy probabilities must be 1.');
    
    % Generate a random index based on policy probabilities
    action_index = randsrc(1, 1, [1:length(actions); policy_i]);
    
    % Select an action
    action = actions{action_index};
end