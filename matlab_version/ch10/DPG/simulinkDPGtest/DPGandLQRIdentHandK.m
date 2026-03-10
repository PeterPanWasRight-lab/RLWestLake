% dpg_lqr_modelfree.m
% 在未知环境模型 A, B 的情况下，仅依靠奖励(代价)函数 r 学习最优控制, 不使用神经网络
clear; clc; close all;

%% 1. 环境定义 (对智能体是黑盒的)
% A = [1.1, 0.5; 
%      0.0, 0.9];
% B = [0.1; 
%      1.0];

A =[1.0000    0.0099
   -0.0099    0.9704];
B =  [4.9503e-05
    0.0099];

C = eye(size(A));
Q = eye(2);         
R = 1;              
% 求解真实的 LQR 用于验证误差
[~, ~, K_lqr] = dare(A, B, Q, R);
fprintf('--- LQR 最优解 (Baseline) ---\n'); %[output:9ee3eb4b]
disp(K_lqr); %[output:4a6a6c64]

%% 2. Actor-Critic DPG 参数初始化
rng(42);
% Actor 初始化：在无模型RL中，通常需要一个能让系统基本不发散的初始策略(温启动)
K_actor = [1.0, 1.0]; 
% K_actor = K_lqr;
% Critic 初始化：Q(x,u) = z' * H * z，由于 x 是2维，u 是1维，z 是3维
% H 是 3x3 的对称矩阵
H_critic = eye(3); 

% 学习参数设定
alpha_critic = 0.05;    % Critic 的学习率 (通常比 Actor 大，需要先验估值准确)
alpha_actor = 0.005;    % Actor 的学习率
num_epochs = 1200;       % 训练轮数
batch_size = 200;       % 经验回放批次大小
gamma = 1.0;            % LQR 是无折扣问题
sigma_noise = 0.8;      % 探索噪声标准差 (极其关键！) 越大频谱带宽越大，越容易收敛

error_history = zeros(num_epochs, 1);
fprintf('--- 开始 Model-Free DPG 训练 ---\n'); %[output:1a005d9c]

%% 3. Actor-Critic 联合优化循环
for epoch = 1:num_epochs
    
    % --- Step 1: 收集交互数据 (Experience Replay 思想) ---
    X_batch = randn(2, batch_size); 
    U_batch = zeros(1, batch_size);
    C_batch = zeros(1, batch_size);
    X_next_batch = zeros(2, batch_size);
    
    for i = 1:batch_size
        x = X_batch(:, i);
        % 执行带探索噪声的动作 (Behavior Policy)
        u_noise = -K_actor * x + sigma_noise * randn();
        
        % 与黑盒环境交互，观察代价值和下一状态
        c = x' * Q * x + u_noise' * R * u_noise; 
        x_next = A * x + B * u_noise;
        
        % 存入 Buffer
        U_batch(1, i) = u_noise;
        C_batch(1, i) = c;
        X_next_batch(:, i) = x_next;
    end
    
    % --- Step 2: 更新 Critic (Policy Evaluation, TD-Learning) ---
    % Critic 为了拟合得更准，在当前 batch 上多迭代几次
    for iter_c = 1:10 
        grad_H_accum = zeros(3, 3);
        for i = 1:batch_size
            x = X_batch(:, i);
            u = U_batch(1, i);
            c = C_batch(1, i);
            x_next = X_next_batch(:, i);
            
            z = [x; u];
            
            % 目标策略 (Target Policy): 评估当前 Actor 的纯策略 (无噪声)
            u_next = -K_actor * x_next; 
            z_next = [x_next; u_next];
            
            % 计算 TD 误差: 预测值 vs 目标值 (半梯度法，不对目标求导)
            Q_pred = z' * H_critic * z;
            Q_target = c + gamma * (z_next' * H_critic * z_next);
            delta = Q_pred - Q_target;
            
            % 累加 Critic 梯度: \nabla_H Loss = delta * z * z'
            grad_H_accum = grad_H_accum + delta * (z * z');
        end
        % 梯度下降更新 H，并强制对称化
        H_critic = H_critic - alpha_critic * (grad_H_accum / batch_size);
        H_critic = (H_critic + H_critic') / 2; 
    end
    
    % --- Step 3: 更新 Actor (Policy Improvement, DPG) ---
    grad_K_accum = zeros(1, 2);
    % 从学习到的 H 矩阵中提取分块
    H_uu = H_critic(3, 3);
    H_ux = H_critic(3, 1:2);
    
    for i = 1:batch_size
        x = X_batch(:, i);
        % DPG 计算梯度是在 当前 Actor 策略上进行的 (不带噪声)
        u_policy = -K_actor * x;
        
        % 从学到的 Critic 中计算动作梯度: \nabla_u Q(x, u)
        grad_u_Q = 2 * H_uu * u_policy + 2 * H_ux * x;
        
        % 链式法则: \nabla_K J = - \nabla_u Q * x'
        grad_K_sample = -grad_u_Q * x';
        grad_K_accum = grad_K_accum + grad_K_sample;
    end
    
    % 梯度下降更新 Actor (最小化代价值)
    K_actor = K_actor - alpha_actor * (grad_K_accum / batch_size);
    
    % 记录误差
    error_history(epoch) = norm(K_actor - K_lqr, 'fro');
end

%% 4. 输出结果与可视化
fprintf('--- DPG (Model-Free) 收敛结果 ---\n'); %[output:9ea4820e]
disp('最终学习得到的增益 K_actor:'); %[output:1ab434c5]
disp(K_actor); %[output:68f5cea7]
fprintf('与理论最优 K_lqr 的最终误差: %e\n\n', error_history(end)); %[output:62ec0753]

figure('Name', 'Model-Free DPG 收敛曲线', 'Color', 'w'); %[output:6e13add1]
plot(1:num_epochs, error_history, 'LineWidth', 2, 'Color', [0.8500 0.3250 0.0980]); %[output:6e13add1]
xlabel('迭代次数 (Epochs)', 'FontSize', 12); %[output:6e13add1]
ylabel('误差 || K_{Actor} - K_{LQR} ||_F', 'FontSize', 12); %[output:6e13add1]
title('无模型 DPG (Actor-Critic) 收敛过程', 'FontSize', 14); %[output:6e13add1]
grid on; %[output:6e13add1]
%%
%% LQR 算出Klqr  只辨识H矩阵的simulink测试
Herror = [1 1 1;1  1 1;1 1 1]*20 %[output:558656d0]

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":34}
%---
%[output:9ee3eb4b]
%   data: {"dataType":"text","outputData":{"text":"--- LQR 最优解 (Baseline) ---\n","truncated":false}}
%---
%[output:4a6a6c64]
%   data: {"dataType":"text","outputData":{"text":"    0.4179    0.2910\n\n","truncated":false}}
%---
%[output:1a005d9c]
%   data: {"dataType":"text","outputData":{"text":"--- 开始 Model-Free DPG 训练 ---\n","truncated":false}}
%---
%[output:9ea4820e]
%   data: {"dataType":"text","outputData":{"text":"--- DPG (Model-Free) 收敛结果 ---\n","truncated":false}}
%---
%[output:1ab434c5]
%   data: {"dataType":"text","outputData":{"text":"最终学习得到的增益 K_actor:\n","truncated":false}}
%---
%[output:68f5cea7]
%   data: {"dataType":"text","outputData":{"text":"    0.4179    0.2910\n\n","truncated":false}}
%---
%[output:62ec0753]
%   data: {"dataType":"text","outputData":{"text":"与理论最优 K_lqr 的最终误差: 2.821137e-06\n\n","truncated":false}}
%---
%[output:6e13add1]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYcAAADrCAYAAACCX1rrAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnWvMXldW37evcSaxx5fEsRPnYheKB+KBCdNWQYMARUrVDnzJAC18qBrqLzTTQgfaSkPUSm2YIg1FQyUXpGZUKlVVP5CpVJFCp6IgUJsKBY8LZaZuoXEmjm+xE8eOx45f22\/1O57\/m\/Vun8s+z3POefY57zrSI79+nn32Za2913+vy1573fLy8nLwxyngFHAKOAWcAoYC6xwcfD44BZwCTgGnQEyByYDD0tJSOHfuXLhx40b48Ic\/HLZv375qrNeuXQtvvfVW8d39998ftmzZUvx95syZ8P7775e+09V0URsf+tCHirb9cQo4BWanwM2bN8PFixcD\/951113F2uW5fv16scZv3boVtm3btvJ9m5aQIRhTduzYETZv3nzHq5cuXQrnz58vfmMt037Xj5VlbWXGe++9FzZu3Lgi3+bp22DgwID5dPHcfffdYd26dauqQsCfPHmymDAwNhbC3\/jGN8KpU6eKdx588MEA0Xlef\/31AhzK3uF3Jtu7776b1G0mKR\/KM8GYoICQ2qDNffv2JdXlhZwCToFyCti1btcUa\/z06dOFDGAdPvDAA61IKBkBuNTJg3feeSesX78+PPTQQwFZVPUAJMgP+rh3797kvlSNr6wCNsNXrlwJtMUGGLkDOCDj3n777cCYmh5k6a5du+7YUA8GDhAJos77wBQr3FVfGTjQHijf1q1iJ1ybfjOh7rnnngKEaPO+++4rJpmDw7xc9\/fXKgVYR1evXi2Gz24dwVclPNEmZB3Q2mtDNzaXEqYIfcAl1h7efPPNQhjz\/WOPPVZbvcpqo5gKEE3gwOaTsbLZBsjsQ1v0DbmDFpEKDmX0cnAoYa8FB2k8EBmw2bBhQ6GV8C8qKKrs1q1bi53Kpk2bCobNCw52cpTNPu0OmMA7d+4sVSGb6qBedgyMAxMc9dQ9TEg+jFcTMqUfqYuThYQQYEFCz7LnjTfeKMoIeNntdPXIJCFzQVf1zlsPu0F2gIybHTEP42eusRGpMn\/UtSshGGuytMVm6t577121i2zSrpvGqLnIXENzRsCnPtrN281WlfA8e\/ZsMUdpB0Es60BKW9AYYLEWCWhMPTJBs7aZg+zWYxqpDQGY\/m8BAl7t3r27sTtN4ACfkDH0wz70F8DSGMRn+lTWLnNem+eFgkMdRZpMQk3vsmj4wFwIJmHNJJFNcFazktqWFoLmwgSn3rJFUzaZ22oOKYLd0gTGxsK9bR1MdjSy+GGXBF2ZSDyMn0nIA81ZMExGPoDMLD4V0RawqzK7WZMBbVP24YcfblxoqQVOnDhRjLHKnJBaT5flMJFgLtBih+7Q39Kd\/9cBall\/ysCBscssGwuKecGBPjCHEMDMkTbmnjbgoH4CPghDaFP2QEfNYX5nhw2tASCAgD7SX+hMHQAEQAwv8B9WPfApph3yCIBgPfIwdoBe\/y+ri36UrTfrfwAgkEG0KX7y\/0cffXSlyqrvVaCMtrY\/g2kOdYsmxdZX9b5VBeMydmc0r0N6UeBQJqyYWExUVEuZzJh0csxBhxTAhe4sAsoyyRDs1pFvdygsEn63bdAOkxgNChChTFt1XkIJbaTMXCieyrzHLkg7prrybYV0FwKwbZt15dkZIrQEutDV7mrhDYIK+s2yU47b1nyBD215mDJueMaum\/rb7OpTwcHKkKb+2DXFvIWOAgI2JwCE\/R6602fWHDxhzUFzHmvW0fwvC4YBfAAkQIu\/68Chqv9VPss1AQ5WdWxisH5nEcmuBoO1gGAUxIQhbXwdduK0mXD0hwnDwkKQzeNzSBHstGcnMEKTnbQmbWodCBkmF5M8nnxf\/\/rXCyeXHFxSr+t2o3E\/mvioHSXaXpnmove1s2fhwReEIg7\/PXv2NDWR9HtO4KCNCB3HdFZl8rO8m1eT6hscGIsAvk1gRio4xL5BqzWwFvVhfbDBYaNz+fLlYmMjuRGDotaXIiDZ8aMpo0XgjOYp2zSWTTja0NpU9GLVxGzSHOiPNAvqkOPZmo+Qe5jZoF+sUajdUWgOUtfK1LKklR1CQQRFKlgBr0ljzSFxnZYZYwIHTQzGqJ29di2p4EAdAgE7ido691SeOtgd1QGJ6C+tgckeaz6WRyxiJjoP5RSdAWgBiNZEEPP2woULhR3a2mfpm\/pYFbQQCzAWoJyAMqOxGBEoCAv7ADSMDSFEXxEozDHKpUSrCaiqFrVtC0HDbhZwYEzaDCnckrEq2k4mKP7PetEYq4IutBbqgLOJvrav0kSbtET7Tgo4sKtHK5EWheC2808bCwug8cZPPLVt8x0bEPqLYGcO8Fi+yM+RwqtUWdbkc0gJtIF31JO1Q9rGJFcRB6bKLIHtWwhbR0wmNmX1WIKxQNiFwjBrjqhyBll\/RZXNWc5QO8GG8Dk02cDZzWDTZALb8L024KDFYye4VNUUAZw66eNyAhT4XSfkZWKRVsKYEYoI3NgUpjYAA95TeB9zgnakzksbQnAzd+Rol13f2nfVPnVTBx8JfL6LfT6aF9Ji6Tfl0Y6a7O0IesZWF07ZRG+reVAPY9VOGkEKcFhwgA98tBsVDXReqGyep9I33iRoHaVqfSngAK+sM9luNOA\/66MsvBWQhBbwhQ1qHPmjzSprXnVAexvGKoez1Saa+MPvahsZFs+JrsCBOZcKDgsJZW3rGE0hLGVioWkXML\/LwdomFLWsXr6r8omULRr5NnSQhok3j0O6CRzon+pnIT7yyCMFCVPBwWpcohkLSTuxtpM+lX+Uq4qasXUgsFiATHRLC425ypyCgJUzF9OTIqBsfU1ATz8wOyiuHfOOIqTgL5oqwtw6LS0\/+BthXBV9VUYru8mh3wjRto+to8yEU0b3OrNS2TxvS1+NwW7WrBm0aoxN4AAPmRtWM7TBFdaJXKadwsf4zFTcF7QDwBM+U1abEviqdRKfqyiTO4rWYhNm64zNWU3gYPsnrRpg61J7oY3eHdJajArDK5sE1ldgf6+KNqCM7Ib8bRe8fZ\/JwG8KQa1apBAWIkug24gbG20Q295T7dR9g0OZQ6oJHBgXY4Y2\/G2jXuxuKwWc2govylsAqmtDgo5FaR3Q1lwYO6arAEX9xBTCx4YqlvHShi6W7XTlZKWcFUiqKyUWPqadxjWPk9lqDmXO5XnBYRb6apwS1qnjawIH5oV2\/GxkdAYB0xL0B9wR7FYwxzRnPJSFbvQPOcFGgDXEPKFetFW1w4YA+UM5Ngh8H2uwTeBg5Uq8uWgDDgJpjUlaLGuYsaQ+Cmm3VpvewaGpc5YQUr95p03EhMwTitxhBw0zqQ\/G8ntVrK8ElcJg40gGEB5GUjcTxgJHFThQlgUoE4bArypcNKZRk2CPyzeBQxMPBIoyu80TPdbUln6XxtIUHSMzhNWKqMOa02LBLQHEomP3bc2PVf0r46XddVbt4mVztuY31ZXKb9snCw4KmU6lqcpZcEBIxvH+84LDLPRV38S3OpOgHW8TOGhsEtAIa8Aa+YGWV+ZvUP2sMywOrE82GAALtJO2LDpasydggSmKXTrgwN\/8XndauoqnNqCA+qij7pBfPA+qzjsosq3NAeAy8Fw4OIgBMJfFJBtZG3CQABHxQHXqoU7FJ6cuMAsOQnc5F2OfBe0ywRBCioKQQ1iTSCorvyNgYudlWb\/agoPsnlatTDHn0W9AU2k\/4gVcZ\/eWACzrf4q2UXXuxNYnQcLi1WK3v4vvcYRUagSJrasMHGw9sZMzFsTwWQIiVaMso50Eieprc5Ar7pM9k2PbmhccZqGvFch1aW5imqSAg04es8ZkXmYtMA8VjVQ2f2hLc4iyvMNOXPMJ4EdLsCCvOcm7aJ6ss3jjEo+hDvAVrQe\/ZfZK1RysaQ+whd+KymJNU482zPSJ9awURozRmtN0NsQGdywcHKzjkwEpIiAVHGRzE3GUW4nBYo9EwMdOGd5R0q44N0rs6Jajsir6pq0gTwGptnWW2d\/b1mH7FZ8ELQsxnRccqsxFZbvoJprFZzRm2X03gUPVLr5sHHXgUHUuRzs35qp2fPP6HPoCh1no2yU4KGcRazj2qVh5gKBj\/daFYmssimBjo8UD7QEH2ohNRoru05iaNkN19KJ+ZB5aiGRMCjhI89Zped4B3BgzgMYmlHVrAaAt31qBgzqEeQWCzPvgsVeIIuiMoGZh8KSCg90FQaCqxHvq6yz2dCaYkunFY7b+CphbBSIx6NTRro1g7yJaqawvVbvyqn636TN1pICDBaAq\/5NMdta5PMvOtgkcmjQHRbDYRItlQqMJHBhPWUh2Fd0V9YIARIgxz5rGPybNAVOP6IHggz52N1zmcI+Fd11klKLDoC88Vkoc3oG2tCn\/hXhgrQIpZ4DaCuUUcLA8pF86z4AWRXuY1GJZ1LYfc4GDBgFipcRv2wluY9zlnKK+JnBQCB51WZsbCwMEtuBQt7utE87xhGt7GK6s7qbdhX2njaC1jlmbRqFNHWX9teccqsJFZ+2zJnNZllzVqR1glUlJ5WRGsAkZtemoMs2In3LCs8vqw+fQhueWlmXnTqrmq8ZvhVTf4DALfWfVHGQyteMXOMDfMnBg7gAoPNKeqjZt0pI1zwAjHfxERpX5jdiQyVeREs3XJJRpTwEx9Jm+0I84PF00UASdTm7r3EocrdRG\/pXN1YWAA6YemKckakpI1RRlAXFwMPE+74CO2N1gMAJMscgaqHwCdmLFkVH2KLzKMeGsKUXhqXEcNOXtATrFh5ct5LI7JubdhVutIZ4Y84IDfWtzQrpte9YxWaYlyp5aF2VCH63TWHxv6ot2fgjUujxZbaKVrJCax+fAmKzwrTshbXMCWSHWNzjMQl\/NdRuIkLLpsP5E1hfmF8aqtV4GDlZ4physlRMbiwh122gs1q2CUZSh1Wp\/KVFXNiFgmXmyKtpSWRbiBJNsghVFhZyqSp8Ryz+bMDM+S2ajP8WrwcEBQkAspeEF0VHb6GwTONjwR8UVAxQMWhFCdWYlu5gQpjr0k2rCKhPmTQulTkOZFRxok509glFx2qiTNiqni37FuZWgU5w3hjFAVyYs5XlSdsx1oawpPg\/RztZjHdNVcfh2TNbcoB1qvFOsO+dgtRZ7lmJecGBs6k9VbiUb3x6bNmYBB9qs6nebcw5V9BW\/BHwpQpV3mOeMFQEp53yd2YVdus05Rh3aYTdFrTHvmMu6G8Gm4lG4tM4nqF7+tTLMrmnkC\/NQ64LfGAOBLXEqcJsGiHbhqUJqYznBnKdemY6aEuzp\/SYNJm5nUHCAyQoZpSM2fIv\/N4GD3SnHAqhJGDJhMFkBJLIPI9BQ31IF2iLAoQlcqpJ9NdGjqV79XjbBpSHZ6ActFtRsQKTsFq24TQnXWDVva9JSLLt1TDed4I3nngUT+p56QlrCOyXEOZXmlAP0bfI9CQz4bfP4I0QwJ9oouFnBQbt02qBeNgJ8ZjkhHdNXYxevZjkDojqqwMFmsaX\/yAg596EfQtkmjoSOzDXkigJPLI\/ggeYC77FbVzJE6I2fi+8oF28qbN4z6mTNKCSe\/yP45YSuS2EuPyprDZChndhENmpwiHeWEAfiMqktYeR8ghBluU6kFtkDW2UTRsAB03QPAUSWANMkidPpwkBdL0i\/ALKUW+DihH91QoBF05RquikMVREKCDBU4TJB3BU4aCzwUAvJOgSVswp+Ikja5OoXCPCOdfpJSJU5A8toa31C8YlphAM7QZ2gpb9KWWAjOQSCCAq+j4WXcgjZFOUsUrujVd+60BxUFxsqNjEWEKC\/dpaAUnzCd1ZwYP0hvBXuKIFXN55U+mo8MlUyV1LuNmjakLEGWO+sVZv+Q\/cwQDv5MSXEoZkOvdmzAHZdMXZpGtBF553ojwUCm5lBMg1+6OS0PUMUp79vs1mgbFUqmxgcdLjVrlPerwtl5ffY9N1ac4AYZbZ3O1DrHIwPakA4mMnuMn4QsqjTVg0rI2CZEygWhggJ5WhXHQhmAMkiL0JD8cxaZNr1sMj4dPl0fcS9y74NXZdMQvIZlZmshu6Tt9cfBWz45Tzp1u1aZy0zjwRoZTe4scEDPLTplAkQOcPat76MGGhtRCW\/sdu3mYARwMgsxmaBAAGNNgp4KYOrKBsncUyleJXzOwYHmz4ltW7KxdaYVuDQpiFbll0nzNFOq8nkAGGVMC1ukzrK8s2U7ZRhDmhdd2Oa6oe57A6ZZExc2tEtcLOOu+w9gDMlY2mXbeZclw4BsYiaNKqcx+F9a6aAbN6znBy3tcdmJTYVaDzIhbJNJ+\/qljtp2829\/aCEfKRoHGWHWAEFbUQtEAAsrHVdOFbWZoqMsZeWldURg0Pb1BmqMw63HwQc2jDCy64tCiikuU0a57VFoWmMVnxGkLa57Gcaox\/nKFbAoS5cs+3QdLdA3ztkbKEKdWzbRy+fDwWkButUZz498550RQFpiNi1u7qgqau+eT3lFFgBhy4OeqkJ63Poi\/BtM5321Q+vtxsKyG5rQ0K7qdlrWTQFMAPhq8RmT5x\/m6CFRfd9Lbc\/OrOSTcTFZEM7aXs6ey0z3MfuFHAKOAVSKDA6cFBGR3aY\/D1L6o4UwngZp4BTwCmwlikwOnAQs+bJ67SWGe5jdwo4BZwCKRRYAYe6zKMpFdkymHtwPDWFrLat15ZPBQfFP8\/Tlr\/rFHAKOAX6pIC9R6HPdtrUPVqHdAo4AAwvvfRSeOKJJ9rQxMs6BZwCToFBKXD06NHwqU99qjiQl8uzAg7EHysZXhed4+BZfNqwi3rbmJVwXuOXeOGFF1ZS+HbZhyHq+tjHPhYOHz486jFAJx\/HELMlvQ3nRzqt+i4pXuzfvz9PcOibAF3Xn6I5CByeeeaZ0YLDY489Fp599tnwuc99buVO6q5pOUR9Po4hqJzehvMjnVZ9l8SyceTIkeDg0BGl1wo44Lt5\/PHHw6uvvjpqcPBxdDTxO6rG+dERITuoxsGhAyLaKhwcOiZoz9W5MOqZwC2rd360JFiPxR0ceiRuVdVTMCv5Il7AxKlp0vnh\/OiaAg4OXVM0oT4HhwQiDVTEhepAhE5sxvmRSKgBijk4DEDkuAkHhwUQvaJJF0b58IKeOD\/y4YeDwwJ44eCwAKI7OORDdDePjYIXawocuByDG5N0IfiiOOTgsCjK39mu71Tz4YVrDnnxYjLgEN9vbK+WU8ZUSD\/PNYBdsc7BoStKzl+Pg8P8NOyyBudHl9Scr67JgAP3KPA8+uijxf3K3IfKzU5cx8n\/h7jLIZUVDg6plOq\/nAuj\/mncpgXnRxtq9Vt2EuAgzYCJxX2q0iK4vIO\/SZ8NaOTyODjkwgl3gObDids9cXDIhyOTAQcu0t65c2fAnMSjG9mseSkXsjs45MIJF0b5cMLBITdeTBocZGbKjegODvlwxHeq+fDCNYe8eDFpcNiwYUOWV3U6OOSzCBwc8uGFg0NevHBwWAA\/HBwWQPSKJh0c8uGFg0NevJgMOJw6dSrcunWrlrq5RCw5OOSzCBwc8uGFg0NevJgEOORF0ubeODg002ioEg4OQ1E6rR3nRxqdhijl4DAElaM2HBwWQHQ3K+VD9JqeODjkw6bJgQPXbyJ87aNzDoS35uCkdnDIZwG4MMqHF25WyosXkwEHHXyDvPv27SsOvvHYtBpZ+Rz+xrcU\/Xvyd97Pa0Yk9saFaiKhBirm\/BiI0InNTIEfkwEHNAaAwAKD+Ej6jPPnz4fNmzdncVK60BwcHBKXWb\/FprCIfcfd7xyZpfYpzKtJgIO0g23bthXpM+JHeZeWlpbySbzn4DDLmuv8nSksYgeHzqfF3BVOYV5NAhzYicfpM8RdgOPmzZuFVqFkfFmk7HZwmHsBdlHBFBaxg0MXM6HbOqYwryYBDk2aA2x\/6623wqVLl0rNTt1Oi+ba3KzUTKOhSkxhETs4DDVb0tuZwryaBDjAMnwOaAhl2VcFHjip8UmkPIAJvgo9TQn84vskFBUlx7ht08EhhQPDlJnCInZwGGautGllCvNqMuDQFK0EY8uc1WUMlwN7+\/bthQ8DoLh48WK47777VrK+xu\/h17hx48YK+ABWpAwvAysLDs+8cj2cvrbcZt5lUXYKk9+FahZTaVUnfF7lw5PJgINIWnbOAR9DqsZAPfbiINVb9p1+i++T4HsA5sKFC2HXrl13AIqDQz4LwIVRPrxwsM6LF5MDhyry2tvh6hzSVSaoOp+Fg8M4z2q4MMpLGDk\/8uKHg0PEjzJBTxHA4d13360MhbVmJfwM8f+rfA5uVlrsgnDNYbH0j1t3fuTDDweHjsBB5ig0D566q0mtWenHjq4bpc9h69at4dChQ+Ho0aPh2rVr+czolj3xcbQkWM\/FnR89Ezixem7V3Lt3bzhy5EjYv39\/2LRpU+Kb\/Rdbt7y83KmXNtWsNIvmoHe2bNmyyiF99erVUie2BYcvP\/JXw5f\/9Fz\/FO24hXvuuSccOHAgHD9+PFy\/fr3j2oerzscxHK1TWnJ+pFCp\/zJPP\/104MPj4PBNes\/icyhL3VEXPmvB4TNfWx++enVz\/9zuuAXU\/4MHD4Zjx44VBwzH+vg48uKc8yMPfhC6\/9RTT4XDhw87OFiWtI1WmgccnvvKUjh6sf6Sojymy+peuG04L644P5wfXVNgEj4HmXW6ugmu7TmHecxKDg5dT+l29blQbUevvks7P\/qmcHr9kwCH9OGml6w7IV12hiEGqLr04Nas9MLXboSXz9xM71gmJX0RZ8KIb3bD+eH86JoCowAHfNPr1q1rPXaEMKeUSdWd0+PgkA83XKjmwwt64vzIhx\/ZgwOClJPGpLEgGijlAUzOnj0brly5UoRjLToLa9xnB4cULg5TxoXRMHRObcX5kUqp\/suNAhxOnTpVaAB79uxpBAgAAZMQ4ZW53Pzm4ND\/RJ61BRdGs1Kun\/ecH\/3QdZZaswcHwiTffPPNIqkdBzHQBKo0COsnwAzFRNu9e\/csdOn1HdcceiVvq8pdGLUiV++FnR+9kzi5gezBgZGgBXCZD0CBBvHggw+uAgh+P3PmzMpJ3SYQSaZOTwUtOLz42o3wxRPukO6J1I3VujBqJNGgBZwfg5K7trFRgAMjQHPAvESqBu5KwMTEaUou8Dl37lwgjBVtgatCH3jggXwoXNITB4d82OPCKB9e0BPnRz78GA04CCDQIEhLgT+B\/EUIWkCBiCSc1gBG7o+DQz4ccmGUDy8cHPLixajAQaRDg7h8+XIBCtIW8C3MEu66CHY4OCyC6uVtOjjkwwsHh7x4MUpwgIRoEJiUFMU0Bo1BrLfgwAE4DsKN7XGhmhfHnB\/Oj64pMFpwgBD4GrhjARMTfoZ77723a\/r0Up+DQy9knalSF6ozka23l5wfvZG2dcXZgwMOaEJUq\/Im3bx5s3BW8+B3sKYlQKPN4bnW1JvxBQeHGQnXw2sujHog6hxVOj\/mIF7Hr2YPDqlJ9croMoZDcG5W6nhGt6zOhVFLgvVc3PnRM4FbVJ89OHCGAdPRLHf\/6CBczrmVHBxazNYeirow6oGoc1Tp\/JiDeB2\/mj04dDzeLKqzZiXuciBt99geX8R5ccz54fzomgIODl1TNKE+B4cEIg1UxIXqQIRObMb5kUioAYo5OAxA5LgJB4cFEL2iSRdG+fCCnjg\/8uGHg8MCeOHgsACiOzjkQ\/Sanjg45MMmB4cF8MKCw+lry+GZV64voBfzNemLeD76df2286Nris5X3xT44eAw3xyY6W0Hh5nI1stLU1jEbo7pZWrMVekU5pWDw1xTYLaXHRxmo1sfb01hETs49DEz5qtzCvNqEuCgg3Kwk7secrsWNJ5mFhz47cnfeX++mbiAt6cw+V2oLmDiNDTp8yofnjg4VPDC3ipHkR07dhSpOOqe119\/vbiQiKfudLaDQz4LwIVRPrxwsM6LFw4OJfx45513wvnz58P27dsLQAAoLl68GO67774CJOIHQDh58mTx9b59+4p7JgAKcj7p\/\/YdB4d8FoGDQz68cHDIixcODiX8QLDzPProoyu\/ln2nH8vAA4C5cOFC2LVr1x2A4uCQzyJwcMiHFw4OefHCwSHih7QAdv\/s+i0AcH9EmSaA1sB7Zb+VsdvBIZ9F4OCQDy8cHPLixSTAQcn5NLnmSbQn5zZCw\/oY0A5IAFjm8JZWwd3WvM\/TxufAOQfOO4zpcaGaF7ecH86PrikwCXDokiizggOaA1FS0jYAjKWlpVIwiTWHHzu6bnTgsHXr1nDo0KFw9OjRwJ0bY318HHlxzvmRBz927twZ9u7dG44cORL2798fNm3alEfHQgjrlmfJ0d1B92cFh9hHoXq2bNmyyjxFuRgcfn7pUHhn+a4Oej9cFVzLeuDAgXD8+PGA5jbWx8eRF+ecH3nw4+mnnw58eBwcvsmTWXwOZc7qqnrKwOHZr949Os0BM8bBgwfDsWPHVsJ385jW7Xrh42hHr75LOz\/6pnBa\/URlPvXUU+Hw4cMODpZks0Qrxc7qKg2kDBzc55A2Yfso5bb6Pqg6e53Oj9lp1\/Wb7nMooeis5xxshFMbnwOX\/XDpz5geX8R5ccv54fzomgKTAgeZcrZt29Z4mrmJkHUnpMvOMKjtmzdvFlW3iVZycGjiRn+\/u1Dtj7az1Oz8mIVq\/bwzKXDAlHP69OmAp73sJHM\/JGxfa+yQdnBoT8Ou3nBh1BUlu6nH+dENHbuoZVLgAEE4kMbu3Z5u7oJQXdYRg8MLX7sRXj5zW+MYy+OLOC9OOT+cH11TYFLgICfwrVvl9vs6U0\/XhK2rz8FhSGrXt+VCNR9e0BPnRz78mBQ45EPW+p4U4PBTnwjhnTNFQTcrLY5zLowWR\/uylp0f+fDDwWEBvIjBwc1KC2DCN5t0YbQ42js45EX7uDeTBAf8DspxpAHb1BaLZkkMDi++diN88YT7HBbBFweHRVC9uk3nRz78mBw4AAzk+rEJ8upSWSyCFQ4Oi6B6eZsujPLhhfsc8uLFpMCh7lRyXVbVoVlSgMM\/\/uEAcg1DAAAgAElEQVQQXjtWNE2kEqalMT0uVPPilvPD+dE1BRwcuqZoQn0ODglEGqiIC9WBCJ3YjPMjkVADFJsUOECv0ZiVvvDpEI7+lmsOA0zyuiZcGC2YAVHzzo98+DE5cBBAZO+QdnDIYhW4MMqCDSudcH7kw49JgkM+5C3vSWFWMuBA0j3OOozp8UWcF7ecH86PrikwKXCoy61Esry33367uN2IsNZFPg4Oi6T+6rZdqObDC3ri\/MiHH2sGHLKLVvo3L4Tw279WzATXHBa3IFwYLY72ZS07P\/LhxyTAIU6vXUVe7lvIISFfoTkYcDh9bTlw4c+YHl\/EeXHL+eH86JoCkwAHEWVUKbsdHLqeyzPV50J1JrL19pLzozfStq54UuDA6OO7m\/X\/jRs3ZqE10MdCc\/iPL4bw679QMMw1h9bztrMXXBh1RspOKnJ+dELGTiqZHDiU3f+cG0AU4PB7vxHCiz+9wsQnf+f9Thg6VCW+iIeidFo7zo80Og1Vagr8mBQ4jCpayWgOTFgHh6GW7ep2prCIGZGPYzHzp6rVKfBjcuBw6tSpYqHcf\/\/9q\/iWXbSSg0MWq3kKi9jBIYuptKoTU5hXkwIHuDOa9BluVspiRU9hETs4ZDGVHBwGYsO65eXl5VnbKgttbXufQ1zHjh077tBGqvrHgbsLFy6EXbt2Bd6LnzKfA6GsOKbH8rhQzYtTzg\/nR9cUmJzm0AWBEO7nz58P27dvLwABoLh48WK47777SoW9bVPOb+6xrirv4NAFl7qpw4VqN3TsqhbnR1eUnL+eNQEO0gLWr1+\/6hKgKvKVRTyVfVf2vsxa\/FarOfzxqyF8\/q+vVOGaw\/yTeZYaXBjNQrX+3nF+9EfbtjVPGhwQ6OzkeVJPR8fnJERQAObSpUth3759RV1lj5zeTPB33323FTiQeI80GmN5fBHnxSnnh\/OjawpMDhx0GxxmHZ4NGzbUCvSYoFW3yTVFO9n3OHDX6HM4eTKEz37\/SvPcBMeNcGN5XBjlxSnnh\/OjawpMBhysA1mAwE6\/abffFThYs1OSQzoCh8+\/tiG8fHo84LB169Zw6NChcPTo0eLO7rE+Po68OOf8yIMfO3fuLDJYHzlyJOzfvz9s2rQpj46FEFpFK1UdfksxBXUBDrFWkQwO+BzeOVN04VeWvi382fLW1gz4yY3Hw45118Pnlg61fneeF+65555w4MCBcPz48XD9+riSBtpx+zjmmQXdv+v86J6ms9T49NNPBz48owYHBqAIIyJg59EcZvE5WN9GzIiyENoiWgnNwYDDvz29MfBJfZ4\/uDF84kPvrSr+9vJtX8jPfG1972GxmDEOHjwYjh07tuLXSe17TuV8HDlx4\/ZJb59Xi+cJIfhPPfVUOHz48PjBQeSUcL9587aJpq3PgXfmiVYSUCX5HAw4vPjajfDFE2lmpec\/sjF8cs+G2hmkMxNf+L83wu+d797R7TbuxS9g2wPnh\/OjawpMxudQRhh257pLug1IzHPOoQ04nP6nfy3svXq26DrOaJzSTc+Xntwc9m5Zt1IMELD\/r3r\/U69cD6c6PGTnwqiJU8P+7vwYlt5NrU2BH5MGBzFQwn7dunVJ5xx4r+6EdJNPoel3mZXagMMT29eHIx9b7RQqS\/Udg4edxJQHgLoImZ3C5Ic2Po4mMTfs786PYeld11r24HDjxo1AWOrmzZtbUw0hfffdd4ctW7a0frfPFwQOv\/8PfzR874ZzjZoDmgFC3z5\/dmU5\/P0\/Wqr0LfyDb9sYLry\/HA7vv9OPAUhwrmKedB2+iPucIe3rdn60p1mfb0yBH9mDA4L0zJkzYffu3eHee+9N4ifRM6dPnw5LS0vJmkJSxR0VEjgc\/eyPhifCbXCou0c61gbanomo8lHMc8nQFCa\/aw4dTegOq\/F51SEx56xqFOBAGm5MQg888EAjQNioJd556KGHAhFDOT0Ch5c\/8yPhk1veKrpWJagxJWFS0jPrSWrqACTK\/BOzpO7wRZzTjHLzWF7cmAY\/sgcHzEqAAwetEPZ79uwJHJSJH8qdPXs2vPfee0U58iihbWzbti23eXP7mtCTJ0MTOMR+hn\/\/xs3wy3\/a7LSuG\/Bd60P43e+7M\/1HW4BwcMhrWjk\/nB9dUyB7cGDAnF0AIK5cuVIIfoQ+i0EP3wMMAARlMT+hZZDGIsdH4PDi3\/7hcHj7+ZUu2tvg2OGjNWinP48JqIwGm9aH8HsRSLz6zq3wd44tJZHMhVESmQYr5PwYjNRJDU2BH6MAB3EDgLh8+XJxdoF02KTUPnfuXJHkDlBAWyDFtgWOJE4OXCgFHDjHgBlITx\/XiJY5ulNBaAqTH9r6OAae\/A3NOT\/y4ceowAGyoSFwtwIaBCDBYTeAgWP3aBSzRDUNzQ6Bwws\/8cPh+T0faA7WtGN9DX\/4zq3w6cQdfduxxBoK76cAhC\/itpTut7zzo1\/6tq19CvwYHTjAJF2+I21BWkRbBi6qvMDhuR\/\/VDjy2IWVbggc4h192+iktuOiPbQU6\/imjjo\/xBQmv2sObWdK\/+V9XvVP49QWRgkODI70FHwwJY0VHJ555pnwpYNvr\/BKIGC1hpRdfCqzm8r9wuMbw\/fdvzotR1V0lC\/iJmoO+7vzY1h6N7U2BX6MFhxgDr4GfA48pJjl5rUxPNIcYnDQwTR74K1NzqUuxl7mh\/ip\/7kU\/uDt1fmZpjD5XXPoYsZ0W4fPq27pOU9t2YOD0nFjQip7dKmPsrHaMvglyEme6zkHwAGzkiKSAIIz18KKI7rLdBdtJklZqo4f+R\/Xw8mrH\/DAF3EbivZf1vnRP43btDAFfowCHIhSEgi0YVDqndFt6uyirNUc\/tb2t1YyrP7G6ZvhwbvXrdj+605Nd9GPujqaIpmmMPldc+h7FrWv3+dVe5r19Ub24IBGcPXq1ZnHT24lNIicnipwAAysU3hok1IZjV75gdUH5uQD8UWc04zykNy8uDENfmQPDrkxvYv+WHAgZXecbVVtzJoqo4s+qg40iH\/xXZvCvrtXpwn\/7P\/7UNj1Ld8RXn311dFf9vP444\/7OLqcNHPU5ZuOOYjX8asODh0TNKU6Cw5PLJ9bddhN7w8ZpZTS51\/66Kbw5K4Pcjzxzs9e\/7gL1RTiDVDGheoARG7RxBT44eDQguFdFbXgQPbYsjsYcjApxeP9\/Ec3hU9EAPFjR9eFE+9e64o0g9czhUUM0Xwcg0+d2ganwA8HhwXMqRgcfvZbN4ZP7fvgfEFuWoMl0fffvz78s8dXXzrUR2qPodgyhUXs4DDUbElvZwrzysEhnd+dlYzBgYrtwbfchW1ZqGvbrK6dEXPOiqawiB0c5pwEPbw+hXnl4NDDxGiqsgwc9A6Ct4trPJv6MO\/vBx\/YHv71t6+OIhsjQExhETs4zDubu39\/CvPKwaH7edFYYx04NL6cSQEm\/\/ce+tbwc5v+eFWPctd6YvJNYRE7OGSyKEw3pjCvsgeHec85xNMmh3MPUwEHhYD+7vespvKYNIgpLGIHBweHPiiQPTggSGc9IR0TrM2JaTK\/cuWonh07dhR3RVQ99npSypBOfN++feGuu+68dW1q4PAddy\/dcVZjLBqEg0MfYmX2Op0fs9Ou6zezB4fr16+vXOYz7+A5Kc3ka7rzQYKey4QABKUIJ\/srIBE\/cXl+f\/3114ub6coAYmrg8P7774e\/uHN9+OXv\/CCKKeeIK8s\/F0bzrqpu33d+dEvPeWrLHhzmGdys7yLYeR599NGVKsq+04\/cB42AtEAAYJBSnEyxMaBMERygxePb1oV\/9d2bV2g2BoBwYTTrKunnPedHP3SdpVYHh4hqCHmEPeYghL0etIdLly5VmorKtIm1Bg7Q4Ef3bQh\/71s\/uN40d4BwYTSL2OjvHedHf7RtW7ODQ0Qx+TiYpNbHADhwf8SDDz6YlAIcgLl27Vpp+alqDiIluZj+wo4PUm3kDBAujNqKjH7LOz\/6pW+b2rMHh6b7HNoMNuV+hy7AQc7sKif2yjWhzz0X\/uRP\/qTNELIpu3Xr1nDo0KFw9OjRAgTj5xc\/cit83AAEZzc+89XVuZlyGEzTOHLoY0offBwpVBquzNj5weVp3IVz5MiRsH\/\/\/rBp0+qsCMNR8s6W1i1\/83afsYGDgIELhqxZyg5R4MB3X\/7yl4vP2J577rknHDhwIBw\/fjwQNFD2\/OTG4+HPrb+88tOf3doafuXGt2U11JRxZNXhis74OPLi0tj58fTTTwc+PNmCw9Asn8fnkAIMjEfg8MILLxSagw2ZHXq8s7aH+n\/w4MFw7Nix2pTdnzvwjVV3VHz16ubwma\/lo0GkjmNWOg31no9jKEqntTN2fmD1eOqpp8Lhw4cdHCzL20Yr8W6TKalMc+CaULKyjvFJtQ1zH8TzH9m4CiAWecNdTOvUceTOIx9HXhyaAj+y9zksguVdnHOo6\/fUHdJlY7eJBfn9D9+5FT59bGkR7F3V5hQWMQPycSx8Kk1uXjk4VMypuhPS8RkGNA3MUWVPmVN6LYJDmQbxyoVb4TN\/tFiAcKHqQrUPCkxhXjk49DEzGupci+AgksQaxKJNTFNYxK45LGARNzQ5hXnl4LCAebWWwSE3H8QUFrGDwwIWsYPDwoi+Esq6sB702PBaBgfIWgYQizoo5+DQ40SfoWrnxwxE6+kV1xx6Iqw7pOsJmwtAuDBawAKoadL5kQ8\/Rg0OOiDHHQ2ktYifN954o\/jq4Ycfzofi5pzDWghlbSJ87INAg3juK0uBf4d4XBgNQeX0Npwf6bTqu+QowKEqw6lSXWzZsiU89NBD4erVq8U9CrpDoS6Tat+Edc0hncJf+M5N4S\/tXJ2LaSiAcGGUzqchSjo\/hqByWhujAwcEP5f\/MIk4os7fgAO5QPS30lY4OKRNgllKdb2IP\/\/RTeETu1afnB7iwqCuxzELLbt4x8fRBRW7q2MK\/MgaHC5fvlwkfEIjUPprB4fuJvA8NfUx+Z\/ctT780kdXJ\/j68T+4Hl670p+JqY9xzEPXWd\/1ccxKuX7emwI\/sgaHEydOhKWlpSJFNgDBxTkODv1M5ra19jn5X\/mB1Ver\/sbpm+Hn\/\/eNtl1MKt\/nOJI60FEhH0dHhOyominwI2twePvttwOfW7duBdJtc02ng0NHs3fOavqc\/EQyfenJD26Uo6s4qF\/42o3Aobkunz7H0WU\/m+rycTRRaNjfp8CPrMEBdpK5+8yZMwETE5fvNIED7yj3OFoHf9vrPoedIuWtrfVzDqk8ACAACvt07YeYwiKGPj6O1Fk1TLkp8CN7cICVNlqpCRwAE7QMnps3bxbOageH7hfEUJM\/DnVlJP\/oq0vhv5ztRoMYahzdc2B1jT6Ovincrv4p8GM04EAivG3btgXONPC3Ryu1m6xdlx5y8n9y74bw\/MEP7qXu0sw05Di65oGtz8fRJ3Xb1z0FfowGHM6fP1+YmPQAElzFx\/ceytp+8s77xtCTv8wPwRjwQ7x85ubMwxl6HDN3tOFFH0dflJ2t3inwYzTgQCgrYMCVlNjsZTqCdUQzbd++vfBNABR+zmG2Cd3mrUVMfgDi5w5uDN9t7qamz\/\/9wq3wi\/\/nxkynqhcxjjZ0Ti3r40il1DDlpsCP7MGBy+vPnTtX3Jdgo5Xuvffewqfw3nvvFUCxcePG4v9oFA4O\/S+ARU7+D29aF37rE6ujmRjxM69cbw0QixxHl1zycXRJzfnrmgI\/sgYHTEaEsvLUhbKuX7++AAYetIs9e\/YUf\/sJ6fkneVUNi578aBE4q+Nopra5mRY9jq445OPoipLd1DMFfmQNDiTOQ+iTJuPdd9+tPQTHjWtnz54NN27cCGgVJOJzcOhmopfVksvk\/55d68M\/j05V09+XT98MXzxxs1GTyGUc83LKxzEvBbt9fwr8yBocEPQk0rt48WJj+gxMSfgjTp8+XZigAAi0DT7SJLpl\/+y1+TmH2WlX9mZZ+m+V+7vHlsLJq8uVIDGFRcxYfRzdzql5a5sCP7IGBzEo5ZyD\/Az4KEjAxwE4Ds2RkC+3x8GhH448sX19YWoqe05dWw6fLkkFPoVF7ODQz3yap9YpzKvRgQPmIz02ZbfAgd9wUqNB4IvYu3dvEc2U0+Pg0C83tm8K4Tc\/sTo\/k1qMQWIKi9jBod\/5NEvtU5hXowCHKubUXfZDWCtaBCYlwlv7fjBlnTx5csUxjjkMwNLdErZ9B4e+uXH7KtLHP7wu\/JNvL9ckcFy\/8Y3l8B8u3hOW9n1HePXVVwtz5FifKQgjB7m8Zt+owaGJlDaVRlPZeX\/H+Y2PRBoMQEF4bVnqDgeHeamd\/r6imeJEfnENgMXP\/a+lcHHpdpK\/sT0ODnlxbAr8mAw4IJhxSG\/evLkQyjyAA7vBvjWHspvqqm6vo18ODsMvZEACoV+WzK+sN5T9r+duhZfevB0inTtgTEEYueYw\/Lqoa3HU4KAzEDidy4SxnNP4HPqMWCLXE6G2hM\/KvyEzE\/mgcIxPzaz02GOPhWeffTb86q\/+auHfGdMjbeL5j2wMOLHbPIAEn98+dyu8cuGD5H+LBo8x88PS38fRZjb2WzZrcEATePPNN1fs+CIFAjg+x1AGDghtAIRdVZ\/ggAkJMLA+BoEDPgfrLLeaw3PPPTc6wSoe4Og\/cuRIGPMYGMsT+x8Mzz93OPznf\/fF8Je\/8cf9rraBap8HqOZ5d5bhFQY8Y8Xjz49\/\/OPh6NGvhFu3bq78dPoaloAQNrbD8touxW1T+Mzdu8NXLi6Hv3LXW+E337+\/2AiwgeAekZ\/48PlwdN3u8MTy2ZV6lzeWBT7cHtAP\/dAPhZdf\/k9GfpWYK5c1\/Dt\/W92\/27+vlIpodrtDt7+8uRzCXevXhXWWsJWU4J3VafFVD8cAfvCTPxj2\/8y\/XLkGYRYed\/3OuuXl5WUJWMxD3MvAvwAG4IDAtYfcYnBAawBYeKocw111ui04EGb70ksvBZDZn8wo8M6ZEHbsCeGrvx\/Cf3sphNeOZdZB706vFNj\/XSE89TdDePGnQzj8hRAOfFcIv\/1rt7\/77Pff\/pf\/r6Hnz\/\/6qaxGuwoctPuOd+NV4MCJap11IB+TDX\/tY5RtwYE+ABB8\/HEKOAWcArlSgE25Lk\/LpY8zgwPAcOXKlULLwJy0e\/fu3sfU1ufQe4e8AaeAU8ApMFEKJIMD5iMOu9nke\/yNExhwGOJpG600RJ+8DaeAU8ApMEUKJIOD7onm31u3PogeARgeeOCBwWjT5pzDYJ3yhpwCTgGnwMQokAwOjJuDZtq9k3Dv6tWrhT0fE9NDDz00CGnanJAepEPeiFPAKeAUmCAFZgaHXbt2FbfCEamE7wFn9BB+hwnywIfkFHAKOAWyo8AqcNBFPuplSigrJ6a5DwLHNGci+j4lnR0FvUNOAaeAU2CCFCjAAVDgLocYHAhtxadQd84BmnDvNIfg0CTiU8oTpJkPySngFHAKTJ4CBTi0GeXly5cLIMHnQMpurgvlPmnMS\/gehnROV\/V7LH4Jzm2Q\/ynW1PT\/lHEQ3osfSA\/mvUUCdN1ZFG0+yjLp5jAOpaZXwAWbI5vQcQz8YC5w7a+WtbT\/scwp+s9GM74CIIX2bGKV8ZdISptmh\/EPOceqxhHzp2wt5DKOO8CBBUyHmx7KAQicpOZkNAc4Ut5rqreL38cQ0YQQJTxYE7jszoymcWiiSWNj8gPcQxxILOOT+sPCtKflxzAO9Z2NDn0fIz\/iPsfzA57lzAv1l3QSsWBv6ne8nmx5QH7ItVI1jiZ+0M+cxrEKHNAKuB8abQANABSOTU1WKPA7JiUytBKtlAM4jOEsRFU+KLvrZqFDWxz\/Onkej63s7u5F3eetMWl+CBxS+JHDOGJhwjjGxo+yQ6JjGYO0aGSI\/JdKrtk0hxCqZGrABC6tWUCp74aaY3XjKNOqq27fXPQ4mP8r4KBb3UBtdp7sRmOzR9WuHmbmEq005lPUdvJcunSpNgMtWWgpHyccZPy823eeq3guaPGxuG1yxCZ+5DCOuuSNGucYxtEEDrnOKV0mRtZnAlzizMtNtOfqgHgjJS2J+YigHWKtNI2jSttW3+W\/tRvCRYxD\/SzAQSm30RIIR9WJZwiK2QjiVmkFvAv6gfKU0x0PXZiHZqljlvxLs7TT9TsSULq4qGkcLKR4tyS7ary4uu5rXJ9dvNiLLTiMYRzaZaIxs0mStmx9DmMaBxGDZaaxMYyhCeB046MFdL4rm\/PasCCXhl4rZeMoW4fWjMSRgJzGsQIO586dK0DBpsIom0x9C5p5629aAHFa73nb6+p9JjIHCmVrbRpHLuAQq+9xv8cwDo0BDVj+Gn2HLw2n9BjGwVyMneo2QGEMY1hL4CAHuXhUBSiLArnaaCUHh65Ef309MB8NzTqSx7CQpfLyr6J6xgwO1mYtLUwOfnZ1dXeJ5ADWcUCC9QOxKeL3MYwh3j03rYUxag4CBhtNNipwYCIR1kcEx1ieJvvkIsM8y2hYBgxV5iF7610Otvp4l2rHJ99VmQ0513HE4GCdhbmPo8pvMqYxVM37pjWdi8\/Bzv86s1IZMPBu1ZXHyIghfSerfA5jEfwp\/WyKbOj7zomUPqpMbEqy76aMY6gIjDZjind5YxmHFqA1O9oFrsi8XKPHUsABPuYeAVcmVJvmUE7RSlorVeAQm5Ls2opNtNZMOHTUFW23PgTXRlAsqmxTTPSi+mXbjeOZq7QKdqwSWLwjh7V2Ghx4yuWcA30qMwE08WPIGPQq3scCKPY5yISWMz+azEoI0dx5UWdaqaN9vJ4Wec6hSgMqO+cQz8ecxjFJcEg5TblIgKgzx9iTnSnjGPLUZwrN6uzDuZ+Qjk+vjvGEdDwfxjaGKnBIWQu5nCyuAgfbv3gt2cCBXMYxSXBIEWJexingFHAKOAWqKeDg4LPDKeAUcAo4Be6ggIODTwqngFPAKeAUcHDwOeAUcAo4BZwCzRRwzaGZRpMrYR3DdnCkjsDxRzp2ypw+fTrgICepIskVUx7ex6lICgeiqIiu4uQ3J\/A5M8M5E37jcBkPZVIfDgqSCZgzFLyX+i7j4h1SytvwYb4jOIAwQcXKk3+I8ed2HiaVRl7OKdAVBRwcuqLkSOpB+CGoEYZ79uxZdXOfEi0iLMkzRD4aBGibJH7E0RNeS9oJ3QyoSBNl2wQsyP5L3bQVXy9L1BCCPAYk+n7mzJmC0vSdg4BND7m\/BCj0h3aVpoQ8UPaKW8YLkCz6ToymMfnvToEhKODgMASVM2qD3TcaAQIb4SuAsKGCpGtHuLcFB10Zi\/BFcFM3j+qW5sCOH8EMiEgL0E4d4AIc0Fhi4FCYJsD28MMPF31MeQAUgIU+0Rf6R7oL2ifUE82IOr\/+9a8Xv9OXVK0kbl9JLKGtTX6nC4Tq+htfzJMytr7K1B3QnLVNhQov6r6RWfu9Vt9zcFiDnJcAQ5hLOJLPRsK6iSRVQkyCXbdb2fvEy9KE2EtRuPkLbQHwQpjTRx6ELDt+\/kV48z31V2X\/RTvBLIQA4qE8H+4qQRvCnMW40Q5kQqMuwAgw5P34FrImetjfdc5DdehMC2a6pvvVldm4TXt9le0DHOgrPIQW8FQZVvsag9c7HwUcHOaj32jfxnyCMEcgIUjfeOONYkfNg6DUThdhGadr5x0Wt30wz6CR8F6ZWaYqhxRmKIQmwGAfgApBrkypSpGQsgOnHpmd6g4eqT3GS5\/pS9WTYmoCYDFbWa1J4EC98e1mOU+evsChjEY502Et983BYQ1zX1fCasfPDlr+BQm1FJ8DO3Hs+uzElXAvzmGlXT\/fA0ZXr14tyqMp8C8CNTblAGDs+ikv0xCmJMw+CPT4wXTEBy0DsxOgxvsACh9MWdJIGCvtaTePZoHgol6ZqwBL7fgxcTWZmhgj47Eg4OBw5wI7ceJEob0x15q0qTW8PBc+dAeHhbNg8R1AMGLiwRyDLZ6nDTgADGgOPIADyenI5Iug5INAVobfqtHSdqyNqKw1g2lXLue5TQ8hAEKQ4zfRgyBCqwGQeKQNIfjRWAQ+9FX12XTXKWYmXbGr+x\/U9izgYNvGFEbd9JV+I0zpb6xp0Q4+GcZAubqy0JMNAe1QDp5RLzRTEIA0B3iJBke9lANwoa\/MdhonWhfgCq31MAdsnfqefgLU1CO\/1OJXgfcgpoCDwxqaEwoFtfeCYyJC0CAo7EMZdu48CKh4p46QRqBgp0d46ZEjGUGB4IkfvqMMO3c+CGMEEoKkygatKCIEkwS1wIH30BKsEEYg0Wce+oEwQnOgv4xLNx5K6FEWjUOgQRs8gB5tpji\/pdmgXdjoq3nAQYIW+sAngSz0k4+GfuqKX40ROsJrwJD+69pfW5axUg6aUI7y1r8DONAebQEKCm9mPLTDJiIOIqAtyilEGM0LsIR+1kckEyTfPfbYY2toBY5rqA4O4+LXXL2Nk5dRmS5xZ9GnPjIdUR7fAMKCXSCLXoKD+iSAEDoIHwQ1f7cRCAguBG8cASVwkHNcwtlqEjJ3IeQAAAQq2oVCWQEW+swY6B\/CCuBAmDIOvqd+hFvTUxXpVJdk0dYpJz79t3yy0VSU1zgFivQXfxHvxPe403929AhoTDjQXj6YWLsSPeVbUbm4fUWMASqPPPJIMQRpGRawFLkGHQFLgTXloTXtzev8b+KJ\/z4fBRwc5qPfaN+2F46wcGPNAaGJEEVgIiCrNAeEImXYSbLTRhiUhSoqMklCMOVQHX3gww6V3TECUTtQmZDkX0DQYKpAaNnzDwADgoodLcKSevhd5jN+x+TE+3yPMEWTYrypZx4k7Bh77HROjVZiXAhRBLkFh\/g8B2CpcxsIY50ZsX4WTUoBB2WgIWV4l7HFZ1fgD3NCYCvNAV5iWtKj8SicGH4q2ED+pJRFoXfmCRtOacfLzE4BB4fZaTfqN6tuo2JQCEx2dooMsuch4kHLZm3PMpSBg72nmXMF0ljqiCgA0K5fZSWMEfoIOmtGiuvDdITGoh0r\/YQvpbQAAATNSURBVEUAC+xkQ+dfNCIJXspTFuEMsNQ9VmDHQncesxJtlh1AtIIVwa9Dg9A1fqxGgEAHIOGnrnWtGpc0B2hr75Uv80UpoAH68UHbgL86IV\/WBtqOgg1yuoBr1Iu64847OHRM0LFUVwcOcjAjTBCaCH6EKbs8KyjsWJvAQQJdZpsmgYAWQB8RNuyQrVkCgc9v8mnY0NGY\/vH9BvHv1G93rwgtOa5TTWB1zvt5wMFe7GT7bcOCAUjAoersyTzgIPObBfKqscIv+MI70FTgCrACMPG5lNiMNZZ1s5b66eCwlrhtxloFDphYcM4CBnJk4hBWNBImBmtmUJVN4EA5Cd66yCRpLgAUQFJWVoIF4KKMzCRVYZFW0xDQ4ERnrLE5Ruk\/6Id1uuamOUBvhC5j60tzaAMOog9gxfxRni6Z6+KoJNcc8hc8Dg7586iXHsbgoPBG\/i3LeRSDRrzzTwEHtYlZR8n8EG58jwlC4ZkS\/mVCX74LTF6AFKClk85VobBy4kr7oR3Ah7apQ\/4HCC2NpQ04pPgcqC\/1EJzNRRWbtWTCoj5oSNscvCtLKRL7HCgDLcrOrkBHNgFyXkOfFHCgDeW7iqO6MDehTZRpNe5z6GVZd1qpg0On5BxPZQgUbPE6IY3wl0mgKveNQkp5B8Ek8wGjTgEH65zFlINQtonxsK+z4wQsyoSzQjYVBUUfEGpK4kd9ZVoNbaARKIJKJ8ABCWunt\/XTvkxaTT4HyjZFK80CDjqHYfsIn9B6JHBTopXQjgTGcjTHkU0CUGlqVSekY7MSgKPT9THQqq+x9tf2DMl4VtW0eurgMC1+No4GwapFK+HOAsemr9QXVZFEmAzYuVr7fxuzkjUt0SY7aeVSwiSEyQetAZDi\/wg0pe5AILITRRjGQlugVZXlVX1kF6vQW77DcYpphrZs\/YAGdEIQqp9NJ3m1S45DRFOjleiPIpbgA3TQeRSdR9A5h7hPdeccFHYs7ciWJfoLXiOs+dh6U8GBvkmbYwyAFtoHfWWuUGd8iFDZdVN9Oo2T2gv0QgEHh17ImmelCHcEqQ43ITAQzghGOXdtJE\/ZKBBcCK04v1KK5kB9NgeTIoaoU05l+ojmwP8lpBG89pAaAt3u5u24JGTZxSKo+Q0NifcBHcYJuPE3faac7OT8Jq2I36RNWf9LFWcRuuy+44NdqeccqFdhvvwNOPDYE9JlJ5nVH3teQxFkCN+y09RtTkhXhebGpiklboSWNmqJ8NwYWBXdlOrTyXM1Tb9XDg7T5\/GqESJ02NGxm5OAldmF71MS2ykZnjXhpIIDnVG6bnUM4Vtmj6cd9ZeyABf9LtvFW41IwhzBqpxPSieBOUvRNwCjzj7IcarT0bQHXRD4jA2fiE3JUTZt6Cvv1IXWpkw3a3Zpc5dGSt2LLiMzmM6ENGlki+7vWm7fwWGNcR+ByO6u7pyBktXFpEE4V92hoNveWPzYs8tMT7Y+ARLlKVsV2kpfEdDKKdTELnbqOi1MWXb0fIdwLzOXlTnEbRuABuaoptBb3lHG0aZorKYxTBkcoCWaQ134cRN9\/PdhKODgMAydvZU1QoH4PodZhj1lcFD6kqlpRLPwOfd3HBxy55D3b1QUUDoOHLVVobVNA5oqOPhNcE2cz+v3\/w8HRz8O8iKgdQAAAABJRU5ErkJggg==","height":188,"width":313}}
%---
%[output:558656d0]
%   data: {"dataType":"matrix","outputData":{"columns":3,"name":"Herror","rows":3,"type":"double","value":[["20","20","20"],["20","20","20"],["20","20","20"]]}}
%---
