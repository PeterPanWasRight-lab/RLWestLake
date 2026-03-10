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

%%
% 测试自动微分的使用
% 定义函数 g(x,y) = x^2 + sin(y)
g = @(x,y) x.^2 + sin(y);

% 输入转为 dlarray
xy = dlarray([2, pi/2]); % 向量输入

% 计算梯度
[grad, val] = dlfeval(@(xy) gradient_wrapper_multi(g, xy), xy);

% 显示结果
fprintf('g(2, π/2) = %.2f\n', extractdata(val)); %[output:94a87efd]
fprintf('∂g/∂x = %.2f, ∂g/∂y = %.2f\n', extractdata(grad(1)), extractdata(grad(2))); %[output:1b6cfbb6]

% 多变量包装函数
function [grad, val] = gradient_wrapper_multi(fun, xy)
    x = xy(1);
    y = xy(2);
    val = fun(x, y);
    grad = dlgradient(val, xy); % val 必须是标量
end

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
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXIAAADfCAYAAADr0ViNAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnXusp8d512fvG693s96113ux490o0BQ2beOSinARRYHwR5EIJqUIoQq3RpVIRAFRAmmkIuqmkaq2QOM2fzgSgeYPJBIJRLgUSqLyRxq8cdKmNIlQ8NpZe2\/em9ebvZ1zFn1e53v8nNn3Mu\/1987Pz0hH55zfb96ZZ77zzHeeeeaZeTfduXPnTvDkCDgCjoAjkC0Cm5zIs+07F9wRcAQcgQIBJ3JXBEfAEXAEMkdg9kR++\/btcO7cubCyshLe\/OY3h717926A\/MaNG+H8+fPFZw888EDYuXNn8feZM2fCzZs3S58Zqs9Uxz333FPU7ckRcAS6I7C6uhouX74c+L1jx45i7JJu3bpVjPG1tbWwZ8+e9c\/b1ASH4EW+7777wvbt2+969JVXXgkvv\/xy8R1jmfqHTpbL2nLGq6++GrZu3brOb7FsvYkc4fgZIr3pTW8KmzZt2lAUZHzq1Kmic+mEmDC\/853vhJdeeql45vDhwwGASM8\/\/3xB5GXP8D2KceXKlSSxUSh+yI8yoExMGKqDOh966KGksjyTI+AIlCNgx7odU4zx06dPFxzAOHzwwQdbQSiOYCKo44NLly6FzZs3hyNHjgS4qCpB+vAHMh46dChZlqr2lRWA4Xrt2rVAXRir8A5EDsddvHgx0CabehM5DQKAvgkALRGrvDIipz5mz7b7tFY52shN5+\/atauYMKjz\/vvvLxTCibxvr\/vzb1QEGEfXr18vmo8VDElVER1WulbdGnttcMMQFPFB0EwEsVX+4osvFsTJ50ePHq0tXnll1KWSeRORYyjSVgxjJh2bqAvZ4B2scyfy71rOWkkACBPDli1bCmuf3yzDWM7t3r27sAC2bdtWgNuXyG1HlmmKZl2Ubd++faXLqKYyKJdVDe3ADUU5dQnl4Yf2SnlS5EgdSCg9A5bBA55l6dvf\/naRR5Pk\/v37U4tvzKdluZbMjQ9MlAErC8uKdmNpkmg\/uobRUOUCqBNPhBWvEKkLw+fee+\/d4JpsWrU2QSFdRNdYkULGqUlWsjWMqoju7NmzhY5SD6SpVXdKXWDMJGBX+mBMOXLDMrbRQazgGCPVoclG\/1syp68OHDjQKE4TkdNPcAxy2IS8TC5qg\/oZmVRvb4u8Tvomt0jTsyg4P3QEjROx0qHyYXV1rahuWfesCFBGyi1T8DLFa2uRp5CwxQTrIybitmWgmKx04oT1Aa4QHYn2ozAkMEe5URx+mBC67AEIWyamKteTXTZTN3kffvjhxkGRmuHkyZNFG6uW1KnlDJkPNwFLZg1McAd\/izv\/101+ZfKUETltl2sytmb7EjkyoEOQJTrSxuXRhsglJxMFxAU2ZQkcpcN8j+UK1kwWkDYyIi84UwZkzqRJX7DfVZXopxg7+AgyZzySaDuTsv4vKws5ysab9ZdD5nAQdao\/+f+RRx5ZL7Ls81GJPMU3VQWeXQ7FeazF0Xezc1FEXkYsKAFKxfJKbiMURJs+4JAyOYI7CkteFAIStpvEduZHofne1kE9KBwrEwifPG2XtCIQrPwyl5n6VC4urAtZInX52xLqEGTVts66\/FhcEIwmSHC11iJ9A6mAXxcLNK5b+kI\/tO3DlHbTZ1izlN\/GWk4lcsshTfLYMYXegqNIG0MCMrefgzsyM+boE8YcmJOsa0P6XxZowUTB5MEEw991RF4lf9Ue2+yI3C6fmjpD36Pw8gPRGVJ2QKXhgNfGN287uY1yIA+dyyCAdPr4yFNImPqsskFwWKhSsNQyIAQUAYWMFeWFF14oNlC0eaIlZp2VF8vR1I+y1FhFla0I9LwsZgYJ\/QKBsZl88ODBpiqSvp8TkctoQHDcR1VuL9t3fVcoYxM5bdFk3GbTP5XI470sa40zFvXD+MAYwSi5evVqYYSIN+IJTONLkXBY0qxAsc7Z6CSVGXhlCkcdGpuKYqtSzCaLHHlksVOGNjWtCwXew9UEftZSb2WRaxmMAkKMTUlLlrKlSdOz+t4uvS0Zq4OtSyAu0wKXE5GrE2mjLGZZA6lEThkibNvhbTeOlJ8ysDrqSF\/4yxpHMeMVhe0jBhxKSSKfdumZYJi87DI57tsLFy4UflPrT0Q2yVi1IR6TDYNFG0xyJTFwGPwMbJuYFGgbhIGsDH50jHwpUUuaVOKlctlYgBSwEiFy2iTDRSFytFVRV3LD8D\/jRW2s2tDXWKib5JrwtTJrhde0+rLPpBA51jLWvlYnkKzVPxkBdrKLjTT1qa2bzzAWkBcSRgdItl\/kl0\/pq1Qua\/KRpwRx0HeU02uz0xI5QKBUkDVg4IuNEx2gpTnfa+aqazhKaMuyjUOZse4A1y7JqzYarH+9ykeqjTarDFP4yJt8tlgJ+OBQNhty1YbIpehWGbVcSyHLVAWN84n86e86QpabQdY+bYbAIMfYHaQ6IG6eU0gWOkE9WtJqlQHJojvaxJUf2vojVT9lUwY\/Imc+i\/copBdaHSI3+Vl1NPmHIWXaVhcC14S3tegph7bKQoX0GI+WyOkHfmTlCQOdxyjT81R84wld4yh1NZVC5PSV3ai0RgH9z\/goC0lkQgML+gV+iiNAZFgy5lUG2NvQQ21mWiu9qX\/4XnXDYbFODEXk6NxgRA7I8o2lNDA1T0xwdrBRhjbv2oQP8lxdDHo8uMoUXL54HSpASfpsdjYROTKrfAbNW97ylgLCVCK3KxlhhtLLwmmroKn9R76q6AlbBuTCYEEpLRZqc5VLATLURiHuF0XC2PKaJmXkYOmtuGFWmIqUoX\/xdUK8dkPM9gd\/Q5xVUThlWFmDBLkhvLbJllHmxijDvc61UqbnbfFVG6xhZV2BVW1sInL6EN2wKy67cW83KMtWffRjfCYllgWrm4mOfiavDAj6VeMkjlsv4x1F7WAw2TJjl04TkVv5tFqFm1JWBZ1dKxABiiOgy3aSrW\/bClm160we+bn42w5O+zwdx3cKG6waUIAAICJfG3lhd51jX3GqX3VsIi\/b7GgictpFm8GGv230g7ViUiaStkRDfjtZ1NUhUmIA2c1N6zKLNz2ryF9y4g7gx4aXlfWlDTcrsyC1gUc+Sx4qKyXWOMZO7eqzgWkt8rKNy75E3gVftVPEmtq+JiJHL2RJwzWK8ca9Av5MxJCwJdEYc9pDXnBDPniCSZsxhJ5QLqtA1cPkDf+Qj8mcz+OVYRORW16JDYE2RK4JVW3S6pAxTFvi1JnI5atBeZgxYh+hFVpLUCpvs3OuJboiOLBMAZ7y6AS+txsBceOqXCsoBTMnoFM2nWtJvorIyUt7tYzXRFUV4hfL00TCcf4mIm8iWk1gcj31iSJqqkvfayXQFCWhpbhdbVCGdSnFJCuyYIBg1Za582I5y\/rSWnNV1rF8pNYFpbJS+9vKYolcYa6pmCqfJXIILY6n7kvkXfCVbOq3OreYbW8TkattIlOIlYkV\/mD1VOYfV\/mMM1byjE+MAcY72GkVKhyt6w9ixx0Dl0Hk\/M33dac8q\/rUblZTHmXUHXiK9aAqnlwRTmWHIUcjcoFFR6D48um0IXINdjWU2ZJyKFPxn6mDwVqHmjW1cRX72KkXZYAwtBuuzUZ1uJZtfA8ZxBtjZXK1JXL56ezSKiWOHLmZ4HS1QDzY6vy0Iqsy+VOs+Kq4flueBj0DTQPTfq9+jyNlUiMJbFllRG7LiTfQYtKknzWYU1dqZdhp0Ku8NodaYpnsmQdbV18i74KvJc+6qzRiTFKIXCcmGWNysTIW0ENFpZTpD3VJh8jLM1i40icmaaxvOyFLJ3mWFR3jLDYy4jbUTc6K2qK\/5fpJtcite4uJkf5WdA5jmnJk3CIT47k1kQNovHkQNxDgNQPxm8q1M5xK5PIRqSG6a4Xy8J9BxrHDn2d04U58V0K8iapNsKoojLakmzKhtC2zzF\/ctgwrV3yCrSwssC+RV7lMyqzTJsziGPguVm0TkVdZx2XtqCPyqnMPWvqjq7Kk+vrIxyLyLvgOSeS6w4QxHO8BWD6AaBm\/deGzaosimTCKSGAPkVNH7DZRlJfa1GS41OFF+XAe1r04JoXItaLVKV+eYSKizUw+GIyMW+v\/LzaEx7iPnJ1bhZUx60GqKDEplcitdUFjqi7NEuhd\/L8ogy7CiknF+tfpiCrCjyeIOnJqQ8JDRK2UyVJl7VbJ3UZmykghcjtZVO2XyG1lNy67WIxNRN5kkSuSwV6SVjbAm4ic9ujipyaCAEdFP0BWEA561tT+nCxy3B3CA5ICH2tllm3mxkRbFyGjKCGwpI917QbPgC11yt8u3ber7ZQzFm0nvhQit32IXIoXZ3VCfbiVYi4ahchtDLE2PmhAE5ErbArhrY8IJWZms0ReZzXWEWmsHG0PBpWVnTIou1gtdtPPHtVuS6yxzDaOvCrEzz7Ttr4m14osqyq3iurWUtpepiYDoco9obq1wYv1MoaPvE2fWyzL4vqr9FXtt4QyNpF3wbeLbvOM3Ia2\/SJy+reMyNEdyJ+kVUmVgaXVp\/SMiUOH4OCosn0OjCf51lOiupqInPoUbIHMyIIccUixMFAklU6c6lxAHLVSxn\/rFrnC65rcJnVEaQ\/g2I22pt12ymTzAncJg4RZBz8RnQHZKNZTA0g+bCtLHCFjj9sqH8ph3Ql1bbZtUfxtWdvL7kjva91aazzuxLbEWiZLm5Odbeuzm15lqy\/5\/+qiDZDZbkiq35tkkUUF+dXdm9MmasUSSh8fuaxsVqoM1LqTnfaOEEs4YxN5F3ylX3aTO8VAsPtfjC9cELRVY72MyC3RpRwy1AYpwQyUbaNyGLcKdNBNiHZVlRJ9Yy\/zKnPRVUXd6XR4fDkcBquiaeCpqiP6Mf9t8JH3tU7juE27UdVE5DZkTXGbkDoCKlKkzrViFR\/i0wGIVDdOGdk1KXXdhNaVyKkTixkSE54sqWx0xhByxXetgFN8jwRtAFeUi\/ykFEu0LvwwxUcv7Gw5Vpeq4pxtm+ySW5ZfbIHVxZHb1YCNVe9L5NYSrbprxcYPx8v7LkROnVVyt4kjr8JX\/SVrPoUAeQY9p62QmTZ+61wPWL\/2DiLKkOXaFL2E3qHLutvbXvehEFfFf6tcfmPt436JDzLCL+ihxgV5aQNBE\/H1uPaqEeqlTxUGGfMEOk+5cp9UEXn83GCuFTqEwhRTbkNuqLSJyK0FGpNFE3HRubhtIH35MyEfljCp5LMIIm+aCKou6mnCo6lcfV+mjFp5gKV9WQgKyFITwi97u0pcp4gwXp62desoVthuejadPIx1zxI\/sqee7BTRpoSlpmJOPiZoe3GWBjf9be+hZsDjUrPRUF2JXNYvdVAukzY\/XU52xviq7eqrLjH2KqOKyO1tkcgPR2jjGPwgUHvpGziia\/CKghpsH9EH0gWewwrWRWbgzb4Mn5EvNgDsPUiUyZhRGDP\/Q9La4Ky71lf7fow1JgTqid1EkxF5bLHREIBAAW0jtLFRLAG+ezWqBVbuHHt4paxzRfIArHu0AUSzqDo0vmISsPWKKORi0kl5O1B8WVfdgEXBm65fbQod1E41ZMNysIw0hyJytYU+lNLbzSbdYUN\/Mujb3DUtwuYZu6EkQinbaCrD1q4S45Oeuh7CHkjTsWi7o68Ji0HN5zHR6E4Re20vA8paipJtCItcZWH8YHBY8gZ\/WWxMIPHJxK5EzviDaDU5i5zq2pOKr9ojdx26knI3d5PxxBhgvDNW7RUDukcc7LTvJsIFMx0AsrHWdlzRdlnw9jwJ8ljStifKxWn0h058WtdxfCV0m4mdvFXXZcREroN+dpzyfOvwQytgHLROIwEeqy1OECJLXLsUKWts2QZDTFwMaN0xrDIgUSYPO6MxwBUvqgEha4IBwc+QKeUY7ZD1zbksuUW0x1Hmtpmz\/C5bOwRsyFyfK4jtWGcso0eafMre7IMxBtHLQJQbDJ5h7Fvfezwp2sg6vsOKtjduQpZwFm2zpA2ZsspjotFNiUIrvoAtFcWqjdWYyO0VDXHZ65uddaF4VQIBMpYOPiSAly8JYFiyxBYlIOiyo7hMni+7f6LMAgVIZsG6N+mofDoCqwtZUTLq0duBUoFOyYcFm3IzYEpZy5BHByJQ+KaVyjK0943cBkVvdDnxanGLXSsYAKwk4IUyA5Fn4ROsb61i2\/QDZA9\/YcmXHeiDwGU0WtJmEmCs172gOYVj7AtyyuSOibzqeD7PrhN5381OK4gNGWsDbNu8LA0V\/dD2Wc8\/LgKpL5YYVwovfWwE1M+QXpsXS4wt1xut\/HUipyP0MtQhQMBajpczQ5SrMtpeWDVk3V5WGgJaCuo0WtpTnisnBLTyYgU+1MtAcmr\/XGQd5WTnmI2zdzQwUbDESbnUf0yZvOxqBORntGF8jtdyIIArhE1BxmHbly8vBwLzaUV2RK6LeSCGqpsX5wOvS+IIOAKOwPgIZEfkgkQbI2VX6I4Pm9fgCDgCjsB8EFh6IrcHW+YDu0viCDgCOSNQ9z7ZRbRrqYkcEv\/MZz4THn300UVg63U6Ao7AkiJAyOO73vWu2heET9n0pSZyNkbxoz\/55JPrt6ZNCW7Xut75zneGJ554wuXuCmCH5xzzDqD1fCR3zI8dO+ZE3lMH1l9CXOcjF5E\/9thjWRH50aNHw+OPPx4++tGPrr9Wri9eUzyfq9xgk6vsucqdM+as8J966qmw9ETOiSmOvHZ5nVUq4aRsduZK5MTkHj9+PJw4cSIrIs9VbnQuV9lzlTtnzJeCyOOLn+xthU0vFUgl6ZR8TuQpKE2bx0llWrxzJsOcZV8KIudEJemRRx4pLp7iohiO5upN1VMdz08ZMm6Rp6A0XB4n8uGwTC3JMU9Farh82RO5LG6Uh4tmZBXrRctzuwHQiXw45U0pyUklBaVh8zjmw+KZUtpSEDm3gXFXNi4V0pB3NKeA2CaPE3kbtPrndVLpj2HbEhzztoj1z7+0RC5XS3+Ihi3BiXxYPJtKc1JpQmj47x3z4TFtKnFpiXyuV8k6kTep5LDfO6kMi2dKaY55CkrD5nEiHxbPxtKcyBshGjSDk8qgcCYV5pgnwTRopqUgcq6t1Ds2q9CZS+SKE\/mg+ttYmJNKI0SDZ3DMB4e0scDsibyxhTPL4EQ+bYc4qUyLN7U55tNj7kQ+MeZO5NMC7qQyLd5O5NPjTY1LReR6MaiFUnHkc3mXphP5tIruRD4t3k7k0+O9NESuQ0A0iNc76U3S9ui++8j7KViuhJir3DkTomPeb6x1eXopLHIscUjbkrjA0Mt2t2\/fXhzhX3Ryi3zaHnBSmRbvnCegnGXPnshlde\/Zs6c4oh8n3cPCCx0OHz486u2HKUPGiTwFpeHyOJEPh2VqSY55KlLD5cueyCHG+Ii+4IHkV1dXC2tdF2mNeY1tSrc4kaegNFweJ5XhsEwtyTFPRWq4fNkTeZNFDlTnz58Pr7zySqnrZTgo00oqiPxTT4YnP\/50+NyZ1bSHZpAr18GZq9w5L\/Md8+kHbPZEDmT4yLG8y3zgKXeETwl7QeQ\/\/rZw+sad8NgXb01Zda+6ch2cucrtRN5LXTs\/nKu+LAWRN0Wt0KtlG6Gde7vHgyJyinj352\/2KGnaR3NV8FzldiKfVr9VW676shRErk4oiyPHJw6JzyU5kU\/bE7kOTCfyafXEiXx4vDfduXPnzpDF2rcGzWKz88ffVjTPLfIhe7m8LCfy8TGOa3DMp8d8qSzyKvicyPsrVq6DM1e53SLvr7NdSshVX5zIu\/R2j2fctdIDvA6P5jowncg7dPYAj+SqL07kA3R+myIskRO1QvRKDilXBc9VbifyxYyKXPXFiXxifXEinxbwXAemE\/m0euKbncPj\/YbZ7HSLfHjl8Y238TFtqsEnzyaEhv8+e4scCze7NwR9N2rFiXx4hXYiHx\/TphqcyJsQGv777Il8eEjGLdFdK+Pi60Q+Lb5ltTmRT98HsydyQso3bdrUGhkIc+vWrYHra+eULJF\/4Cu3w7OX1+YkXqUsuQ7OXOV2H\/lihkWu+jJrIof0Lly4UFxPu3PnzqSehfjPnj0brl27Fg4dOrTwa2tjoZ3Ik7pxsEy5Dkwn8sFUoFVBuerL7Ikc\/zeW9cGDBxvJHPLmpsNbt26FubwRyIm81TgaPHOuA9OJfHBVSCowV32ZNZFzGdaLL74YVlZWwrZt2woLu8oyh8A5wUnCFUOHHDhwIKnzpszkFvmUaPsb3adF+7XaciXDnGWfNZEDLNY1L46A1LHMecuPJXO+P3PmTLhx40ahRE2EvwjFtnVaIn\/y6yvZ3Eme6+DMVe6cScUxn55lZk\/kQIJFjosFst6yZUvhZtm1a1fxsohz586FtbW1wgrndW8PPvjg9Ci2qNGJvAVYA2R1UhkAxJZFOOYtARsgexZELjLHMr9+\/Xrh\/96xY0eAFCFwIlPYEIXc556cyKftISeVafHOeSWRs+zZELnUEcv86tWrBYHLCscX3iVEcXoVD8XkwxuCSO5aGb8HnMjHxziuwTGfHvPsiByIsMxxqyiaJQdLXF3rRD6tkjupTIt3zlZtzrJnSeQAjm\/8ypUrhZsFv\/i99947vcZ2qNES+dPPrYRPnszjBcy5EmKucudMKo55B2Lo+cisiZzNTcIK2cwsS7xwmY1QEn5y616B4NscJOqJY\/LjTuTJUA2S0UllEBhbFeKYt4JrkMyzJvLUC7HKkMjhQJBb5IPocG0hTirjY+w+8ukxjmucNZETI477pMsrPHUoaM53rXzuzGqx4ZlDypUQc5XbXSuLGRW56susiXwxXTlurda18ukXVsPHv+VEPibiuQ5MJ\/IxtaK67Fz1xYl8Yn2xRO4W+fjg5zowncjH142yGnLVFyfyifXFiXxawHMdmE7k0+qJastVX5zIJ9YXJ\/JpAc91YDqRT6snTuTD4z34OzuHF7F7iZbIeakEL5fIIeVKiLnK7US+mFGRq764RT6xvjiRTwt4rgPTiXxaPXGLfHi83SIfHtPeJeZKiLnK7UTeW2U7FZCrvmRvkevQEL3GXeX33HNPpw6c6iG3yKdC+rV6ch2YOcvumE+r49TmRD4x5pbIT9+4Ex774q2JJehWXa6DM1e5nci76Wnfp3LVFyfyvj3f8nkn8paA9cye68B0Iu\/Z8R0fz1VfnMg7dnjXx5zIuyLX7blcB6YTebf+7vtUrvriRN6351s+70TeErCe2XMdmE7kPTu+4+O56kv2RK6LtaT4c7skK9YnS+R89+7P3+yoctM+lquC5yq3E\/m0+q3actWX7Il8Md3dvVYn8u7YdXky14HpRN6lt\/s\/k6u+OJH37\/tWJTiRt4Krd+ZcB6YTee+u71RArvriRN6pu7s\/5ETeHbsuT+Y6MJ3Iu\/R2\/2dy1Rcn8v5936oEJ\/JWcPXOnOvAdCLv3fWdCshVX5zIO3V394diIudAEAeD5p5yVfBc5XYiX8yIyFVflobIb968GU6dOhX27NlTvHR5rsmJfNqeyXVgOpFPqyeqLVd9WRoihyBPnz4d9u3bF+67777FaEFCrU7kCSANmCXXgelEPqAStCgqV31ZGiKnr7DIV1dXwyOPPNKi66bN6kQ+Ld65Dkwn8mn1xC3y4fHudI2tbkFcW1srlWjz5s2zuB0xJnJeLMELJuaeciXEXOV2Il\/MiMhVX5bKIl9M17er1Ym8HV59c+c6MJ3I+\/Z8t+dz1Rcn8m793fkpJ\/LO0HV6MNeB6UTeqbt7P5SrviwdkeMnhyxt4mUTDz30UHInnz9\/Ply6dGk9P5undZEwcX4erHLlxET+5NdXwufOrCbLtqiMuSp4rnI7kS9G03PVl6Uickj8xo0bG3zh8p3v3Lkzicwh8Jdffjns3bu3IG9I+vLly+H++++vjIahXsIfmSx27NhRq4FO5NMO0FwHphP5tHqi2nLVl6UhchE2HRFbz5DxlStXkjY7n3\/++aJPbeRL2WdWzZq+t3kLIv\/pPxPCpTPFx26Rjztgcx2YTuTj6kVV6bnqixO56VEdKsKqtq4YJoJXXnml1OKueqZKUWIif\/q5lfDJk+5aGWvY5jowncjH0oj6cnPVl6Uhcrqnr2ulyqqvs+h1EOnOnTtBoY91oY4xkf+l390Ubq7N\/4j+7t27wzve8Y7w7LPPFu6rXFKucoNvrrLnKneumHMI8tChQ+Gpp54Kx44dC9u2bZvF8OwURy7J+2x2diFy+dTf9KY3rVvxZROK5CuI\/OfeH8JzXy0+emZtf\/h3K8dmAXydELt27Qpvfetbwze\/+c3AyzxySbnKDb65yp6r3Lli\/t73vjfwQ1oaIu9DMF2IvKy+ug3WmMh\/+9K28EvPbekj9iTPsuR8+9vfHr761a8WG7u5pFzllmvFMZ9W03LUF6Lq3vOe94QnnngifyKvu2sFq\/nixYvF8oNQxKrUxUdeVlad3zwmckIP2fCce8rVd5ir3CLy48ePhxMnTmQ3eeYod86YL42PvI7Ix4xaKSu7LoKmIPJ\/8cEQnv2vBXc7kY87hTmRj4tvWemO+fSYZ0\/kZYdxymAkEiXlMq22ceRlbhTCEVdWVkqjXJzIp1VyJ5Vp8c7Zqs1Z9uyJXGo65DW2dSc7IfoLFy6E\/fv3rx8Qii\/s2rJlS+XhoJjIuTCLi7PmnnIlxFzlzplUHPPpR\/PSEDnQxb5p\/b9169Yka3wK+Asi\/82PhfBbTxfVOZGPi7qTyrj4umtlenzLalwqIi87YTk3Mi+I\/FNPhvDb\/9qJfIIx4EQ+AchRFY759JgvDZEPEbUyBfwxkfO+Tt7bOfeU6+DMVW53rSxmROSqL0tF5C+99FLoe9fK2OrjRD42whvLz3VgOpFPqyeqLVd9WRoipyP6HtGfQnUKIv+PT4fw7z+2Xt27Pz\/\/Aza5KniucjuRTzEa764jV31ZKiKnW8rCEdveRz6mCjmRj4nu8gxMJ\/Jp9cQt8uHx7nXXyvDiDFtiQeS\/859CePrvu0U+LLSlpeVqYTmRT6AcJVXkqi9LZ5E2xaMsAAAeu0lEQVTHfSMLfVYvX46InM1ONj3nnHJV8FzldiJfzGjIVV+WlsgJRdTlTqmnOqdQnTKL3Il8PORzHZhO5OPpRF3JuerLUhF5mxOWi1GTULxP9NTXToTwS39jXQQn8vF6I9eB6UQ+nk44kU+DbWsfud3g1PF43uhT9VafaZpRXktB5KdOhfDhH17PwBF9TnimpEM7N4Wn3rkt8JvEs7hlxnbN5EqIucrtRJ4yGobPk6u+ZG+RVx0Eqns92\/Ddn15iGZG3eW\/nF\/989cuddbgIkh+a2HNV8FzldiJPH1ND5sxVX7IncjpRNxbyurUcLfIUIn\/f4S3hQ9+zNVlnIXJZ68kP1WTMVcFzlduJfAitbV9GrvqyFESu7tK9Kqurr73MuO4WwvZdPMwT6xY5PvJLZ4pCm4j8s+\/evu5KkRT41T\/yvVvDo3s31wr2jat3wof\/4DX3S5+Uq4LnKrcTeR9t7f5srvqyVERuu8++u3NOhN6WyPGHx2T9Z79wM6xEvPzL37ct\/Kn91aT+U8\/eDr9\/Jc0PXzYMclXwXOV2Iu9Oxn2ezFVflpbI1Zlyu2zatCkcPny49lVvfRQg9dkyIn\/6uZXwyZOvrSJs+tXv3xb+5L7Xyfl\/X1wLH\/vmSqN1\/ZNHt4Qnjt3thsEq\/+Rzq8VbidqmXBU8V7mdyNtq6DD5c9WXWRM5b9lZW1sL27dvb91LEDhvtt+5c2frZ8d8QER++ud\/LBy6fraoqux1b48d2RJ+5o++TsZdbkl8665N4dM\/dDd2Xe5Az1XBc5XbiXzMUVhddq76Mmsih\/TOnDkTDhw4EO69996knr1161Y4ffp0uH379iws8FhoEfmzH\/7r4dFwrpTIiTrBL67UhcT1LGWV+dLbbobmquC5yu1EnjTcB8+Uq77Mnsi5mha3yIMPPthI5jZ6hWeOHDmycFdKFZH\/j3\/0o+EvbD9ffB1byP\/y+7eFH\/quS6Ut4dZpdpm\/PfXmxVwVPFe5ncgH5+ikAnPVl1kTOa4ViPzGjRsFmR88eDDs3r37rg4h39mzZ8Orr75a5ONeFaz4PXv2JHXelJlkkX\/uH\/5o+JGdrxF5bHHbWHEiTj5\/vvsmZdy2nzi6JfydyH\/+0793O+B\/r0u5KniucjuRTzkqX68rV32ZNZEDL7HhkPm1a9cKkoagAVuJzyFxyJy8uGCw3nlP5xyTiPzJn3h\/+MjBl+8i8r9yeEv4J9+NF+\/jUqlre+y6Ie9\/PrMWfv7r1S+BzlXBc5XbiXwxozdXfZk9kas7IfOrV68WseH3339\/2Lt3bzh37ly4cuVKQeBY4Q888MAGkl+MKtTXKiJ\/+u++Pzyx9zUiJ8nFYa3xLpuSqW0uI\/O6+nJV8FzldiJP1eRh8+WqL9kQOd2F5X358uXCMofQOfgDie\/atauw1LtEtwyrBs2llVnkPMUBH+LF2ZhUmuIyrfjIf9UqIFcFz1VuJ\/LmsTRGjlz1JSsip+O4QwUylxUu63yMTh2jTBH5B\/7mXwtPHb2wgbRtdMmY1rhtV1VUSzyJ5KrgucrtRD7G6GsuM1d9yY7I6YoLFy4UP7hTciXyxx57LHz27RfXNYtj+lNb46ocMmcT9C8f2rJB0y2Z56rgucrtRN5MumPkyFVfsiRyOhDfOD5y0r59+8L+\/fvH6NfBy5RFDpFjkes6WixwHcUfa5OzqTE\/uHdz+Pg7t23I9s+\/vhL+y5nVYu\/h+PHj4cSJE+sv7Ggqbw7f5yq3E\/litCdXfZk1keuKWtwoZYlTnyTdemjz4Ec\/dOjQbOPIYyK3sk\/lVinDdMfmEL7w5zZelftP\/+B2+Mqt3U7kE3NLrqSSq9w5T56zJ3KiVUTYbcbRXN7RGctsLfK\/tft84Ch+nKruXmnT\/r55yzZBf3nzu9wi7wtsi+dzJcRc5XYib6GcCVnX3xCEpX39+vWER8qzcNcKlvmckiVyrhIoI8wh7xHv0\/ZYtot3doTHvnjTXSt9QG3xbK6EmKvcTuQtlDMha+tXvSWUOZssTUS+SLdKGUjxsX7892zMpr6abtHAO6lM3wOO+fSYz9q1Mj0c49cYEzk1yvKdK0n+4vFt4YcfeP063bnKWdZ7Tirj63Rcg2M+PeZO5BNjXkbkY7xjc+hm\/cwf3xUeO7CyXuyiImvatstJpS1i\/fM75v0xbFuCE3lbxHrmLyPynkVO8jiD8\/HvOxh+bOvJrMjcSWUS9dhQiWM+PeZO5BNjnjORE0f+py9\/NfxVY5k\/c2kt\/L2vVl+2NTG8d1XnpDJ9Dzjm02PuRD4x5rkTOQeCfuWPvX54Cfi+fGktfHCmZO6kMrGCh5Dt4TGQylVfnMgn1vNlIPKbN2+GOJrlf728Fv7x1+Znmec6MHMmFcd8YlIJIcyayPvGkcdwziGufFmIHGxjMv+dl1fDh772+obo9Op8d41OKtP3gmM+PeazJnJIr+vJzhjKuZz0XCYiL7s58bfOroWf+8P5WOZOKtOTimM+PeazJnJepKwXR\/SFhhOeKNii7yxfJiKnT8rI\/DdfWA1PfWselrmTSt+R0\/55x7w9Zn2fmDWR923cHJ9fNiIXxvFx\/l\/\/1kr4ty+sLrwLnFSm7wLHfHrMncgnxnxZibzs1XFzuPzLSWViBc848iPnDWYn8on1fFmJvMoy\/\/i3VsKnF2iZO5FPrOBO5NMDPveolab7yNsgNpf7yZedyMss8089vxo+8f8W4zN3Im8zSobJ65gPg2ObUmZtkTuRt+nKcfO2HZyxz\/yf\/eHt8N\/OvvYikClTW7mnlK2prlxlz1Vud600aWS7799Q19i2g2ZxubsMzpjM\/+e51fCz\/2day7yL3ItDeWPNucqeq9xO5MNqvhP5sHgOUlqXwVnmZvnC+bXAq+OmSl3knkq2pnpylT1XuZ3ImzSy3fdO5O3wmiR318FZRua\/e3Et\/IPfm4bMu8o9CagNleQqe65yO5EPq\/VO5MPiOUhpfQfnZ9+9vTg8pDTVm5D6yj0IeB0LyVX2XOV2Iu+oqBWPOZEPi+cgpfUdnGUnQHk5xdjvJ+0r9yDgdSwkV9lzlduJvKOiOpGfHha5EUsbanBO\/Q7QoeQeEdrKonOVPVe5nciH1XK3yIfFc5DShhyc\/+oHtoV33ff6O0AREMt8jBc6Dyn3IEC2KCRX2XOV24m8hXImZHUiTwBp6ixDD873Hd4SPvQ9Wzc0431fvBXO3rgzaNOGlntQ4RoKy1X2XOV2Ih9Wu5OIXIeFuGP88OHDd0nw7W9\/u\/js4YcfHla6nqUt+8nOtvDEseZDb4I6qbTtkf75HfP+GLYtYdYnO2nMpUuXwoULF8L+\/fvDfffdt94+3VW+c+fOcOTIkXD9+vWwZcuWsGPHjiLP888\/X\/x+5JFH2mIyan4n8o3wloUnDrkJ6qQyqjqXFu6YT495VkQOSfOiCRRl165dxd8Q+b59+9b\/fuihh5zIR9CjsQdnvAlKEz74ldvhy5f7HesfW+4RoF4vMlfZc5XbXSvDanPhWrl69WrYtm1bYWnLInciHxboNqVNMTh\/6tiW8LePbvSbP\/+dO8XhIaz0LmkKubvIlfJMrrLnKrcTeYpWpucpiPzkyZPh9u3b4Z577inIHNeKE3k6iEPnnGpwPrp3c\/Eu0Dh1jWqZSu6h8c6ZVBzzMbShvszZulYuXrwY+FlbWwtcQXv\/\/fc7kU+vHwtZ5uM3\/9m3bw0\/GIUofvnSWviFb6y0ss6dVKZXGsd8esxnS+RAcefOnXDmzJmAm+WBBx5oJHKewR1Dwprnb9\/sHEapFjE4f\/iBzeEXj\/ezzhch9zCIh2Iv6Pjx4+HEiRPh5s2bQxU7ejm5yp3zKmjWRA6wNmqlybUC8WO9k1ZXV4uNUCfyYcbtIgdnfE8LLUqNbFmk3H2Rz1X2XOV2Iu+rsRuf3xBHDpGfP38+7NmzJxAzzt8etTIs4CmlLXpw\/on7Nodf+4G7rfMmQl+03CnYVuXJVfZc5XYi76Otdz97F5G\/\/PLLhZtFCULfvXt34HMPPxwW\/LmTSlmYYp2F7qQyjX7YWhzz6THPxrUCcd+6dStwoEbuE+AiqmXv3r2FLx1S9zjycZRoToPzRw5uCR\/53o1himo1FvqTX19Zv7dlTnK37ZlcZc9VbrfI22poff51i\/zGjRvh3LlzxUaPjVq59957Cx\/4q6++WpD61q1bi\/+x1BdB5Mh36tSpQgYSJ0yRQ6dMbXP9ZOdwylIVqqgaOO7\/ay\/tCm\/7I28L\/\/2Zr2W1YZgzqTiRD6fjqSXN1iLHbUL4Iaku\/HDz5s3rBIrVfvDgweKZKY\/oU9fKysr6JAKpM7mUbbQ6kaeqZno+CB0L3b64In4aS\/2j31gJL16\/0yp8MV2K4XPmSoi5yp3z5DlbIufSKyxcjuJfuXKl9kAQd7CcPXu2IFOsdS7RmorIy+6CqbofBkVxIh+e8FQihP6Tx7YEfjcliP3U9TvhY99cCWy\/dD052lRPn+9zJcRc5XYi76Otdz9buFYgZVwUly9fbjyijxsD\/\/np06eL5TNkjhXPjyz0YUV8vTSiaJhomDzw15PkaiHShvj3ZXCtHD16NDz++OPhE5\/4RIHz3FMbUrdtEaF\/6eJa+OKFtfB\/X319k53vsPrrSN9+X5c3JV9umAvHXOVG\/lxln61FLqVIiSOXXxyfOhdpcRgIAuUyrbETbhSI2\/rEReT4yCWb5JBF\/oEPfCALQpTchw4dCk899VTITe5Hjx0OH\/nIR8K\/+ZVfCH\/x2u\/Xul\/G1pW5lz\/kqgR9yWHCL+uTXGVH7mO\/\/qX1Q5GL1re7wg+brrG1ZMkGKAqE75yGyUoeq1FtiZxJ5plnnik2bz0tEIELp0L4D78awoUXNwqx97U9lnD5zGu\/+Z+\/9blyl31W1py2+drmb4LQyytHqAsudc\/Y77qUXdePieVte\/VCOPYbX2rSiMm+7\/1iCUIRsc5xqxCSOGZqS+TIApnz48kRcAQcgaEQ4EoSXVEyVJl9ykki8qYK7HH9prx9vm\/rI+9Tlz\/rCDgCjkAuCLQmcjZG2ezcvn17EfZHgsjxVY9tkbeNWsmlE1xOR8ARcAT6IJBE5IoxZ0OzjEy18YmPfOzIlTZx5H2A8WcdAUfAEcgFgYLIsbBffPHF9cM+Eh5ijuPEy4gclwdkT0zr2ETe5mRnLp3gcjoCjoAj0AeBgshFjrhIcODzG3KHyIlSsQd+YiLHGmcSIFUdle8joD\/rCDgCjoAjUI\/ABiJXLHYcm11F5HopM1EhhPhx6tOTI+AIOAKOwLQIdCZySPzatWuF9Y5L5cCBA9NKXlPbnN0vhFByUMm6r2xsforsuLJYGSkxgcanWsfsjLow0LrLzBYlN3hzeI1XGZIwWOzdPHPFnD6210prhay+naPcyIybNT5XkiIrBqPezsTZFHuCmzaPqT9Vcsd9UHZJ3yLlli4kEzkuFMC1F2fxNwQCkc8pzXVDFAIERymoCCa+ErjuUjApFtcJgz3KzdUKU62IVD99b11pTZgvSm7Vq9s6c8E8ljPGj\/E2N8wlI9d1xCTcJGs8Nmx+Jt4x9adK7ibMkWuRclvOTSZyvZeT37JsKAgSf\/DBB2fD43MNUay6SsBatwze+GRt3J6yC8qmurRMbVBni8hTMF+U3DEhIHsOmJedmZiz3FppYrGySrdE3qQf9rWSWllqIoNf+Gws\/amTu2zlWXWNydRyx4SbTOQ8yHJUDeGyrOvXrxenJnGzHDlyZBZkntuhIassr7zySu2lYFwMRv74XhnazLNjbzZrMDFY7Z03TZgvSu66e3ikrHOVvYnI56QrkC5XdRCezGoyvtiuCWPOo5RdDYK+oWuQ5Bh63yR3GaFZIuf7RchdJldnIt+\/f3\/xtiAiVvCV46edg5+8yzH+Rc1AIhrdp94kOwMFX6+slDoyGrpNdjDiA7VEPle5ZdVxdz73Asl\/b33kc5ddbrfY1TJXuZsmIL0Axk6yfBaTv1xH\/IbIx9b7MrnLxpB1pcB7i5a71EcuRY834urCD5mBuc9cy6mxT3c2EVSTgsc3JDaVN+b34MqKRsvQJtkXReTxMjeWc+5yo5vaQ1BbCLNlhTlX2dG7eJPWbmrPVe5lJnJttqofqshffDnFBLSByCFwNsxiImemxPqrI3ItL7DStAE3Jvk1ld2k4HMhcjAlVt9uUjbJvigij\/2TuRF52QpGG8RYVXVXIy8K83gT2+5PoMN8P1e5Yyu1Sa9zsMhF4jZyaHZE3kSO9vurV68WpI+PnOUqy1YiAnCx4Ctf9MZnkz9uyjC9KlzLSJy8TbIvwtccW4Ubdso3bSomojK\/qH3hxyLkthZtTOTWzzlH2at8+3OXu0qHm\/R6UT5yq8t1rpUyEufZqreTje3br\/SR2y+wytlgaErkg7yxKrEQWKqmPNdUbt\/vm3bIF31oKXan2PamyD7W7n0b3GMLa85ya1DZlZgdtJDm3CKFUohcK2H2qqTTc4hwKiPEJv1YZNSK9L6KyGN3ih0nscuxzHiYarxuuDQLa5v3cWJlY1mjULG7xTZEg4CbEIlamQORI19TzGob0hoybxxzWlZ2k+xjxtOmtrVsqTxXuWMSiX3kKfqyCMybXCuQ3xwxr3M31J2PWHQ8dtUExIGsOpfxouXe4CPnH73th2B+lssIrxjLpgHOZtJcolaQNeUUWVObhv6+zkVhT7GlyD7mCbeUdtf5POd4sjM+nZfLyc64n3OQu4rIU\/R6kScky+S28sTjwm48L1LuDUSua2gZhIQQ6qQmAxbXCX7lKmubZxkobAKQT3eUpxCC53EEHAFHwBHoj0DhWoGMz507VxC4PW5fZnn1r9JLcAQcAUfAERgSgdoXSziRDwm1l+UIOAKOwDgI1BI5fi3uVSG80JMj4Ag4Ao7APBFIetXbPEV3qRwBR8ARcARAwIl8ifTARozYZhGRxOqKQ1zk4YIjImUIGSV0NCXxPDv7XMFARBOb2lwvwN4KqzY2uvmOw2Ik8qQmvWqQiCmeS32WdvEMB9FsfD6fESXEfo8Om3DJFO2fw4GwVFw8nyOQioATeSpSM88HUUGqEBfvTbV33iiMFGLjjAAXEEF2bW5L5NAMMbUc\/OJuGMq3rwjkM4idcwiUXfayEaKbIN148kD2M2fOFAgjOydBm5JeMai7r6lX99ZwXYS9yI32QvpTv4CjqQ3+vSMwFAJO5EMhueBysGqxtCFXiFJkbuN3OeQFEbclcl2MBlFCsnrBtsqWRY4lDYlC+LKuZQEzyUDkrARikle8NJPQww8\/XMiYkiB\/JgFkQhbk424U6ifmmhUHZb7wwgvF98iSau3H9StEF2yZAOvOBcTPxm\/2SWnbWHnqThZ3rVNx+lO93KSrnMv8nBP5EvWuyAbiFZFxgZGItampVYQjEtZrrqy1X3ZvjH3jCq\/8wgpnooF4kZEEIWLF8xui5XPKrzqHwKEzXCOQBYn8\/HAamVUGLh3ajdUtNxJlMXEwcfF8\/PqxJjzs94rgUhkiclxVTTd+6oxFm\/rGyjsGkSMrfQgW9Kmuqh2rDV7u3Qg4kS+ZVuBCgHghD0iPK4axVEmQmt7uBLHFh7x4hoFoEy4KLH2eK3NNVF0AhisGgoPEbWJSgXR1tazu2bBvnarrErle6k7d6Xnai8zIUpVS3C1Mhrhu7GpERE658WvN5qxSYxF5GUZzxmHZZHMiX7YeDaGwjCBpWdJYpvKHi4BSfORYuFyMhoWrqxviS8dkTfM5EwdvjSI\/Fji\/Ib\/YncFkgzVNfrlHcKfg+oB844T7hB+sd1wvtI3nIX9+cOfI0qet1CcrGYsdkqFcuWyY2GRJ4+ZpcrfQRtpjCduJ\/O6Bc\/LkyWJVhK41rVKWcNgttElO5AuFf9zKITHcHLgk8B2T2hC53v7EcxA5N+1xpgBS4wfy1FmDqpZQd2zlK691Bcna1casvVdEk0X8flhIg9UCkwdJqwxImpWAJgpkVXn2Xu8UV4suktOLKCR7FyK3deMOomxkRW6ID3njFQz1sIdAG8hXl1cntKmHfPQZ5bI3og1mWeT0JSsjyiUfkyP4ynWldrKaYSIEayV0wJapz5GTSZVytI8yroZ76ULAiXwJdEHhe\/amStwkkAKD2ibyYBGTIJPYAoZQGfz4lSEaJW1SMqghiTjxGXmwiPmBOCEPBn2Vz1TRJJCISFVEznNY35YwIQ9kJiEHxIFFjry0S3cFiaDIiyUvgqcOEhMUdaZsrGrFgNVuX2XYh8hFiuBDP2lCBD\/tKSCnLrJTG8GRvmbiQn5dbmfz0lbygQn5yG\/3IyBy6qMuCFwhqbSHepjw4w1q6iKfwjr1Inbws3sacsPx2dGjR5dgZOXTBCfyfPqqUtL4ZjkyQhAkBmhqkvuE\/PiyGdhYVwxQDXLKE1lAEBAFpMrfbQYvJANJxpEwInJtvIpIrYUulw+EBFlDfljtCj9kEkBm2oB8EAskD\/HRDj6nfIioKVVFvKRGrWiDGPltP9moGmRQOzWBIS\/7GzwT3yyK\/FjKrBJwY4C99gziVYvw1F6A8sX1K3KICeAtb3lLAYusdzu5KIIJHJnYNLGSH6ypr+\/GclOf+Pd3I+BEvmRaYd9mwiCLLXIIDsKD3CCzKoscAiMPFhoWLAO3LLxMESoirJQDRsjAD5YfVifkJctObhT5wyEFlusQjI0vh8QhFSxFiI1y+F4uJL7H7cLzfA7xsUKhvakx5SIm2h5vaKZGrdAuCA\/StUQex8szsYEzkynEqZh8uy8gVRXJkwcMycOztC0+G0D\/oBOaGGWR05e4V5TUHoWA0p\/ayNb+R8pQ0TN9Qj1T6vE8GxFwIl8yjah6LRXNhNywmBQhYuPNYxjkY7Wx4mVELgIgP3HbWgnUwSqyljWtvCJOCBpSsq6UuDzcJ6wEZAlSP2SpiUk+X35DjiJJ8pMXImUSqEuWXGOC7ONaoc6yw1iWBCFpHaAC1zhZSxvyZTKjP3mhdF2SRQ629qbTsr0TbZaDHz9Y8fSvTvaW1cMqQhvZi34b15IN7drmOJEvWW\/XEbk2Lxn4EBwkDfFhPdlBbSFpInKRr1wXTYMX6xoZZXnapTnkzHfywdtwv7ib4pcuxN9TvrUKIRhtiqa6geo2hvsQOfiXEa4N5WQy0z3\/ZS8M70PkckHZSbeqrfQX\/cIzYKqJkEmQySCO+49dOUs2vGbbHCfy2XZNN8GqiBw3Axt\/ELc2ydhsxMVBYpltl9qqvYnIySeSrItQ0YqAyQTSL8srEmCSIY9cBVWhbNaCxx2hSBDaGrskdMUActgNvblZ5OANQdK2sSzyNkQufJhY0B\/d2yOXVRyd4hZ5t3Hb9ykn8r4Izuz5mMgVksbvsjtQYoKPLeoUIleduDZ0ERdExOcswxVSJ6IuI2j52nH7MKEwweiEZlX4ojYItaqgHr0QnDLkL6eLtBJoQ+QpPnLKSz0QZO+miV07cuNQHhhSN4eQyq4tiH3k5AGLsrMB4MiErY1R8EkhcurQ\/TdxdA8uF6z0spPA7iNfDCE4kS8G99FqZfDjO9bJTohay+KquzAUBsgzkIiW0AiZQuR24w93BgRqL7XCNYAlB7GXEanC7BQNgwwQkC7goryy1QJ1YGkrkkYnVyF061e25VO\/3DpNPnLyNkWtdCFyxblbGeknwj1FjilRK3rpuaJWwCOOcNFkpxVQ1cnO2LXC5KBTwfGkKFnjVVXbGP3RBsEbsGAn8iXpdEhQA0xEzGDE3aDj9VURJSybsQitv7qNa8W6V6gTC1V3q+AWwe2BNc6Ewv8QmK4HgLyw8CCumGA1wVTdpigZsQ4VLslnbMrhnqAuWz4ED06QluRsOoEo6zMO60uNWkEeRa7QD+CgeH\/FeyuOPJapLo5coaJaddi8RAHR1xArP7bcVCJHNq2SaAMTDFY9sqIrlBkfqNItlql7EEsy9GbRDCfyWXRDPyEgYkhPBz0Y3BApJKaNQxvRUVYbJAPBxPetpFjklGfvZFHkCGVqwxIZscj5X4QKSdoDO5CvtZJtu0SIWIeQKt+x8uB5JgjayUTE38hMPvl1+U6rDb7TKsXuF1T1AASJVRsfckmNI6dchWbyN0ROsic7y05gSh4bD69IIoiy7BRom5OdVeGUsXtGl66BpY1eIaQyngQV5ZK6B9FP6\/1pi4AT+ZLoAwSBpYSVJDKU64HPUy6l0kVW1o2RSuTAqCtsBSlEWeY\/ph7JS14mGeQus47tSkPECwnqDhgdWceloygMJjHFlmtTTqc6qQ9cIGfahg+fCaQuISvP1IVDpqiRdT20uQs+pexF55ErSDH3TSudRcu7bPU7kS9Jj0JeWE11cdy6aCpuMkRadQe43gLEQMX\/WuZ+seVp8iA\/eavCEZEVMtUdI03dgAWsU47kxVLmM4i4zGVUttlq64Dgcck0hUvyjG72a4rKaWrDMhM5WGKR14WMNuHj33dHwIm8O3b+5BsIgfg+8i5NX2Yi1xUJy7bS6NLPi3jGiXwRqHud2SGgI\/9sAlaFQzY1almJ3N8Q1NTz43\/\/\/wFfdxP\/gqac\/AAAAABJRU5ErkJggg==","height":178,"width":296}}
%---
%[output:558656d0]
%   data: {"dataType":"matrix","outputData":{"columns":3,"name":"Herror","rows":3,"type":"double","value":[["20","20","20"],["20","20","20"],["20","20","20"]]}}
%---
%[output:94a87efd]
%   data: {"dataType":"text","outputData":{"text":"g(2, π\/2) = 5.00\n","truncated":false}}
%---
%[output:1b6cfbb6]
%   data: {"dataType":"text","outputData":{"text":"∂g\/∂x = 4.00, ∂g\/∂y = 0.00\n","truncated":false}}
%---
