%% 三维空间正交投影可视化
clear; clc; close all;

%% 1. 设置空间和基向量
% 定义原始三维空间
figure('Position', [100, 100, 1200, 800]);

% 创建子图1：三维空间视图
subplot(1, 2, 1);
hold on; grid on; axis equal;
view(30, 30);  % 设置视角

% 定义投影平面（二维子空间）
% 平面由两个基向量张成
v1 = [2, 1, 0]';  % 基向量1
v2 = [0, 1, 2]';  % 基向量2

% 创建一个要被投影的向量
x = [2, -1, 3]';  % 原始向量

%% 2. 计算正交投影
% 构造平面基向量组成的矩阵A
A = [v1, v2];

% 计算投影矩阵 P = A(A^T A)^{-1} A^T
P = A * inv(A' * A) * A';

% 计算投影向量
p = P * x;  % x在平面上的投影

% 计算残差向量（与平面正交）
e = x - p;  % 垂直于平面的分量

%% 3. 验证正交性
% 检查投影向量是否在平面内
% 即检查p是否可以用v1和v2线性表示
coefficients = inv([v1, v2]' * [v1, v2]) * [v1, v2]' * p;
fprintf('投影向量p = %.2fv1 + %.2fv2\n', coefficients(1), coefficients(2));

% 检查残差是否与平面正交
% 即检查e是否与v1和v2都垂直
orth1 = dot(e, v1);
orth2 = dot(e, v2);
fprintf('e与v1的点积: %.6f\n', orth1);
fprintf('e与v2的点积: %.6f\n', orth2);

% 验证勾股定理: ||x||^2 = ||p||^2 + ||e||^2
norm_x_sq = norm(x)^2;
norm_p_sq = norm(p)^2;
norm_e_sq = norm(e)^2;
fprintf('\n||x||^2 = %.4f\n', norm_x_sq);
fprintf('||p||^2 + ||e||^2 = %.4f + %.4f = %.4f\n', ...
    norm_p_sq, norm_e_sq, norm_p_sq + norm_e_sq);

%% 4. 绘制三维空间视图
% 绘制坐标轴
quiver3(0, 0, 0, 4, 0, 0, 'k', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
quiver3(0, 0, 0, 0, 4, 0, 'k', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
quiver3(0, 0, 0, 0, 0, 4, 'k', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
text(4.2, 0, 0, 'X', 'FontSize', 12, 'FontWeight', 'bold');
text(0, 4.2, 0, 'Y', 'FontSize', 12, 'FontWeight', 'bold');
text(0, 0, 4.2, 'Z', 'FontSize', 12, 'FontWeight', 'bold');

% 绘制投影平面（由v1和v2张成的平面）
% 生成平面上的网格点
s = linspace(-2, 2, 10);
t = linspace(-2, 2, 10);
[S, T] = meshgrid(s, t);
X_plane = v1(1)*S + v2(1)*T;
Y_plane = v1(2)*S + v2(2)*T;
Z_plane = v1(3)*S + v2(3)*T;

surf(X_plane, Y_plane, Z_plane, 'FaceAlpha', 0.3, ...
    'EdgeAlpha', 0.2, 'FaceColor', [0.8, 0.9, 1]);
title('三维空间正交投影', 'FontSize', 14, 'FontWeight', 'bold');

% 绘制基向量v1和v2
quiver3(0, 0, 0, v1(1), v1(2), v1(3), 'b', ...
    'LineWidth', 3, 'MaxHeadSize', 0.5);
quiver3(0, 0, 0, v2(1), v2(2), v2(3), 'b', ...
    'LineWidth', 3, 'MaxHeadSize', 0.5);
text(v1(1)/2, v1(2)/2, v1(3)/2, 'v_1', 'FontSize', 12, 'Color', 'b', 'FontWeight', 'bold');
text(v2(1)/2, v2(2)/2, v2(3)/2, 'v_2', 'FontSize', 12, 'Color', 'b', 'FontWeight', 'bold');

% 绘制原始向量x
quiver3(0, 0, 0, x(1), x(2), x(3), 'r', ...
    'LineWidth', 3, 'MaxHeadSize', 0.5);
text(x(1)/2, x(2)/2, x(3)/2, 'x', 'FontSize', 14, ...
    'Color', 'r', 'FontWeight', 'bold');

% 绘制投影向量p
quiver3(0, 0, 0, p(1), p(2), p(3), 'g', ...
    'LineWidth', 3, 'MaxHeadSize', 0.5, 'LineStyle', '-');
text(p(1)/2, p(2)/2, p(3)/2, 'p = proj(x)', 'FontSize', 12, ...
    'Color', 'g', 'FontWeight', 'bold');

% 绘制残差向量e（从投影点到原始点）
quiver3(p(1), p(2), p(3), e(1), e(2), e(3), 'm', ...
    'LineWidth', 2, 'MaxHeadSize', 0.5, 'LineStyle', '--');
text(p(1)+e(1)/2, p(2)+e(2)/2, p(3)+e(3)/2, 'e = x - p', ...
    'FontSize', 12, 'Color', 'm', 'FontWeight', 'bold');

% 绘制从原点到投影点的辅助线
plot3([p(1), p(1)], [p(2), p(2)], [0, p(3)], 'k:', 'LineWidth', 1);
plot3([p(1), p(1)], [0, p(2)], [p(3), p(3)], 'k:', 'LineWidth', 1);
plot3([0, p(1)], [p(2), p(2)], [p(3), p(3)], 'k:', 'LineWidth', 1);

% 标记交点
plot3(p(1), p(2), p(3), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');

% 添加标注
text(-1, -1, 4.5, {'正交投影原理：', ...
    '• p是x在平面上的正交投影', ...
    '• e = x - p垂直于平面', ...
    '• ||x||² = ||p||² + ||e||²'}, ...
    'FontSize', 10, 'BackgroundColor', [0.95, 0.95, 0.95]);

xlabel('X轴'); ylabel('Y轴'); zlabel('Z轴');
xlim([-2, 5]); ylim([-2, 5]); zlim([-2, 5]);

%% 5. 绘制二维平面视图（从最佳视角看平面）
subplot(1, 2, 2);
hold on; grid on; axis equal;
view(0, 90);  % 从上往下看

% 重新绘制平面（简化）
v1_2d = [v1(1), v1(2)]';
v2_2d = [v2(1), v2(2)]';
x_2d = [x(1), x(2)]';
p_2d = [p(1), p(2)]';

% 绘制平面区域
fill([-v1_2d(1), v1_2d(1), v1_2d(1)+v2_2d(1), -v1_2d(1)+v2_2d(1)], ...
     [-v1_2d(2), v1_2d(2), v1_2d(2)+v2_2d(2), -v1_2d(2)+v2_2d(2)], ...
     [0.8, 0.9, 1], 'FaceAlpha', 0.5, 'EdgeAlpha', 0.3);

% 绘制基向量
quiver(0, 0, v1_2d(1), v1_2d(2), 'b', 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver(0, 0, v2_2d(1), v2_2d(2), 'b', 'LineWidth', 2, 'MaxHeadSize', 0.5);
text(v1_2d(1)/2, v1_2d(2)/2, 'v_1', 'FontSize', 12, 'Color', 'b', 'FontWeight', 'bold');
text(v2_2d(1)/2, v2_2d(2)/2, 'v_2', 'FontSize', 12, 'Color', 'b', 'FontWeight', 'bold');

% 绘制向量和投影
quiver(0, 0, x_2d(1), x_2d(2), 'r', 'LineWidth', 3, 'MaxHeadSize', 0.5);
quiver(0, 0, p_2d(1), p_2d(2), 'g', 'LineWidth', 3, 'MaxHeadSize', 0.5);
quiver(p_2d(1), p_2d(2), x_2d(1)-p_2d(1), x_2d(2)-p_2d(2), 'm', ...
    'LineWidth', 2, 'MaxHeadSize', 0.5, 'LineStyle', '--');

% 标记
text(x_2d(1)/2, x_2d(2)/2, 'x', 'FontSize', 14, 'Color', 'r', 'FontWeight', 'bold');
text(p_2d(1)/2, p_2d(2)/2, 'p', 'FontSize', 14, 'Color', 'g', 'FontWeight', 'bold');
text(p_2d(1)+(x_2d(1)-p_2d(1))/2, p_2d(2)+(x_2d(2)-p_2d(2))/2, 'e', ...
    'FontSize', 12, 'Color', 'm', 'FontWeight', 'bold');

% 绘制直角符号（表示正交）
plot([p_2d(1), p_2d(1)+0.2*(x_2d(2)-p_2d(2))], ...
     [p_2d(2), p_2d(2)-0.2*(x_2d(1)-p_2d(1))], 'k-', 'LineWidth', 1.5);
plot([p_2d(1)+0.1*(x_2d(2)-p_2d(2)), p_2d(1)+0.1*(x_2d(2)-p_2d(2))-0.1*(x_2d(1)-p_2d(1))], ...
     [p_2d(2)-0.1*(x_2d(1)-p_2d(1)), p_2d(2)-0.1*(x_2d(1)-p_2d(1))-0.1*(x_2d(2)-p_2d(2))], 'k-', 'LineWidth', 1.5);

title('投影平面视图（从Z轴上方看）', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('X轴'); ylabel('Y轴');
xlim([-2, 5]); ylim([-2, 5]);

% 添加图例
legend({'投影平面', '基向量v1', '基向量v2', '原始向量x', ...
    '投影p', '正交残差e'}, 'Location', 'best');

%% 6. 添加整体标题和说明
sgtitle('三维空间正交投影可视化', 'FontSize', 16, 'FontWeight', 'bold');

% 在图形底部添加数学公式说明
annotation('textbox', [0.1, 0.02, 0.8, 0.05], ...
    'String', sprintf('数学表示: p = Px = A(A^T A)^{-1} A^T x, 其中 A = [v_1, v_2], e = x - p, e ⟂ v_1, e ⟂ v_2'), ...
    'FontSize', 10, 'HorizontalAlignment', 'center', ...
    'EdgeColor', 'none', 'BackgroundColor', [0.95, 0.95, 0.95]);

disp('=== 正交投影验证 ===');
disp(['原始向量 x = [', num2str(x'), ']']);
disp(['投影向量 p = [', num2str(p'), ']']);
disp(['残差向量 e = [', num2str(e'), ']']);
disp(' ');
disp('验证正交性:');
disp(['e·v1 = ', num2str(dot(e, v1)), ' (应为0)']);
disp(['e·v2 = ', num2str(dot(e, v2)), ' (应为0)']);
disp(' ');
disp('验证勾股定理:');
disp(['||x||^2 = ', num2str(norm(x)^2)]);
disp(['||p||^2 + ||e||^2 = ', num2str(norm(p)^2 + norm(e)^2)]);