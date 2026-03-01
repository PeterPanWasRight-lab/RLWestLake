classdef TDAgent < handle
    % TDAgent: 基于函数近似的 TD(0) 学习者
    
    properties
        Alpha       % 学习率
        Gamma       % 折扣因子
        Weights     % 权重向量
        BasisType   % 基函数类型 ('linear', 'quadratic', 'rbf')
        X_Max       % 用于归一化
        Y_Max       % 用于归一化
        Dim         % 特征维度
    end
    
    methods
        function obj = TDAgent(x_len, y_len, basis_type, alpha, gamma)
            obj.X_Max = x_len;
            obj.Y_Max = y_len;
            obj.BasisType = basis_type;
            obj.Alpha = alpha;
            obj.Gamma = gamma;
            
            % 初始化维度和权重
            obj.Dim = obj.get_feature_dimension();
            obj.Weights = zeros(obj.Dim, 1);
        end
        
        function val = get_value(obj, coord)
            % V(s) = w' * phi(s)
            phi = obj.get_features(coord);
            val = phi' * obj.Weights;
        end
        
        function update(obj, s_coord, reward, next_s_coord, is_done)
            % TD(0) 更新核心
            % s_coord: 当前状态坐标 [x, y]
            % next_s_coord: 下一状态坐标
            
            % 1. 计算当前特征和价值
            phi_s = obj.get_features(s_coord);
            v_s = phi_s' * obj.Weights;
            
            % 2. 计算目标价值
            if is_done
                v_next = 0; % 终止状态价值为0
            else
                phi_next = obj.get_features(next_s_coord);
                v_next = phi_next' * obj.Weights;
            end
            
            % 3. TD 误差: delta = R + gamma * V(s') - V(s)
            td_error = reward + obj.Gamma * v_next - v_s;
            
            % 4. 权重更新: w = w + alpha * delta * phi(s)
            obj.Weights = obj.Weights + obj.Alpha * td_error * phi_s;
        end
        
        function phi = get_features(obj, coord)
            % 特征工程：将坐标映射为特征向量
            x = coord(1); y = coord(2);
            
            % 归一化到 [-1, 1] 区间 (通常比 [0,1] 收敛更好)
            nx = (x / obj.X_Max - 0.5) * 2;
            ny = (y / obj.Y_Max - 0.5) * 2;
            
            switch obj.BasisType
                case 'linear'
                    % 简单线性: [1, x, y]
                    phi = [1; nx; ny];
                    
                case 'quadratic'
                    % 二次多项式: [1, x, y, x^2, y^2, xy]
                    phi = [1; nx; ny; nx^2; ny^2; nx*ny];
                    
                case 'rbf' 
                    % 径向基函数 (RBF) - 模拟 "局部感受野"
                    % 在网格上均匀分布几个高斯核
                    phi = [1]; % 偏置
                    centers = [-0.5, -0.5; 0.5, 0.5; -0.5, 0.5; 0.5, -0.5; 0, 0];
                    sigma = 0.5;
                    for i = 1:size(centers, 1)
                        dist = norm([nx, ny] - centers(i,:));
                        phi = [phi; exp(-dist^2 / (2*sigma^2))];
                    end
            end
        end
        
        function dim = get_feature_dimension(obj)
            switch obj.BasisType
                case 'linear', dim = 3;
                case 'quadratic', dim = 6;
                case 'rbf', dim = 6; % 1 bias + 5 centers
                otherwise, dim = 1;
            end
        end
    end
end