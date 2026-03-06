我将以**线性二次型调节器（LQR）** 作为最简单的最优控制方法，详细说明其与**确定性策略梯度（DPG）** 算法的联系，并进行完整的公式推导。

---

## 第一部分：LQR问题表述与求解

### 1. 倒立摆线性化模型

倒立摆在平衡点附近线性化，得到线性状态空间方程：

**状态向量**：$x = [\theta, \omega, x_c, v]^T$
其中：$\theta$为摆杆角度，$\omega$为角速度，$x_c$为小车位置，$v$为小车速度。

**线性化系统**：

$$
\dot{x} = A x + B u
$$

**离散化**（采样时间 $\Delta t$）：

$$
x_{k+1} = A_d x_k + B_d u_k
$$

---

### 2. LQR问题定义

**代价函数**：

$$
J = \sum_{k=0}^{\infty} \left( x_k^T Q x_k + u_k^T R u_k \right)
$$

其中：

- $Q \succeq 0$：状态权重矩阵（半正定）
- $R \succ 0$：控制权重矩阵（正定）

**目标**：寻找最优控制策略 $u_k = -K x_k$ 最小化 $J$。

---

### 3. LQR求解（Riccati方程）

**最优价值函数**（二次型）：

$$
V^*(x) = x^T P x
$$

其中 $P = P^T \succ 0$ 满足**离散代数Riccati方程**：

$$
P = Q + A_d^T P A_d - A_d^T P B_d (R + B_d^T P B_d)^{-1} B_d^T P A_d
$$

**最优控制增益**：

$$
K = (R + B_d^T P B_d)^{-1} B_d^T P A_d
$$

**最优策略**：

$$
u_k^* = -K x_k
$$

---

## 第二部分：从LQR到Q函数

### 1. 定义Q函数（动作价值函数）

在LQR框架中，可以定义**精确的Q函数**：

$$
q^*(x, u) = x^T Q x + u^T R u + V^*(x')
$$

其中 $x' = A_d x + B_d u$ 是下一状态。

代入 $V^*(x') = (x')^T P x'$ 得：

$$
q^*(x, u) = x^T Q x + u^T R u + (A_d x + B_d u)^T P (A_d x + B_d u)
$$

展开为二次型：

$$
q^*(x, u) = \begin{bmatrix} x \\ u \end{bmatrix}^T 
\underbrace{\begin{bmatrix} Q + A_d^T P A_d & A_d^T P B_d \\ B_d^T P A_d & R + B_d^T P B_d \end{bmatrix}}_{H}
\begin{bmatrix} x \\ u \end{bmatrix}
$$

其中 $H$ 是**对称矩阵**。

---

### 2. Q函数的最优性条件

**性质1**：最优策略可通过Q函数得到
对 $u$ 求导并置零：

$$
\frac{\partial q^*(x, u)}{\partial u} = 2(R + B_d^T P B_d) u + 2B_d^T P A_d x = 0
$$

解得：

$$
u^* = -(R + B_d^T P B_d)^{-1} B_d^T P A_d x = -K x
$$

**性质2**：最优价值函数满足

$$
V^*(x) = \min_u q^*(x, u) = q^*(x, -Kx)
$$

---

## 第三部分：DPG算法在LQR问题中的形式

### 1. DPG框架

在DPG中，我们有：

- **Actor**（策略）：$u = \mu(x; \theta)$
- **Critic**（Q函数近似）：$q(x, u; w)$

**目标**：最大化累积回报 $J(\theta) = \mathbb{E}[\sum \gamma^k r_k]$

在LQR中，回报为负代价：$r_k = - (x_k^T Q x_k + u_k^T R u_k)$

---

### 2. Critic学习（已知模型时）

如果系统模型 $(A_d, B_d)$ 和代价 $(Q, R)$ 已知，Critic可以直接计算：

$$
q(x, u; P) = \begin{bmatrix} x \\ u \end{bmatrix}^T H(P) \begin{bmatrix} x \\ u \end{bmatrix}
$$

其中 $H(P)$ 依赖于 $P$，而 $P$ 通过求解Riccati方程得到。

---

### 3. Actor更新（策略梯度）

**策略梯度定理**（确定性策略）：

$$
\nabla_\theta J(\theta) = \mathbb{E}_x \left[ \nabla_\theta \mu(x; \theta) \nabla_u q(x, u; w) \big|_{u=\mu(x;\theta)} \right]
$$

在LQR中，设策略为线性：$\mu(x; K) = -K x$，参数 $\theta = K$

则：

$$
\nabla_K \mu(x; K) = -x^T \quad \text{(对每个元素求导)}
$$

且：

$$
\nabla_u q(x, u; P) = 2(R + B_d^T P B_d) u + 2B_d^T P A_d x
$$

**梯度更新**：

$$
K \leftarrow K + \alpha \mathbb{E}_x \left[ x \left( 2(R + B_d^T P B_d)(-Kx) + 2B_d^T P A_d x \right)^T \right]
$$

化简得：

$$
K \leftarrow K - 2\alpha \mathbb{E}_x \left[ x x^T \right] \left( (R + B_d^T P B_d)K - B_d^T P A_d \right)^T
$$

当 $\mathbb{E}[xx^T]$ 正定时，稳定点满足：

$$
(R + B_d^T P B_d)K = B_d^T P A_d
$$

这正是LQR的增益方程！因此DPG收敛到LQR解。

---

## 第四部分：从LQR理解DPG的关键概念

### 1. 价值函数与Q函数的关系

在LQR中，关系是精确的：

$$
V^*(x) = \min_u q^*(x, u)
$$

在DPG中，Critic学习 $q(x,u;w)$，然后Actor通过 $\nabla_u q$ 的符号决定动作更新方向。

### 2. 策略改进的几何解释

在LQR的Q函数中：

$$
q^*(x, u) = u^T (R + B_d^T P B_d) u + 2u^T (B_d^T P A_d) x + \text{常数项}
$$

这是关于 $u$ 的凸二次函数。当前策略 $u = -Kx$ 的梯度：

$$
\nabla_u q = 2(R + B_d^T P B_d)(-Kx) + 2B_d^T P A_d x
$$

如果 $K$ 不是最优，梯度非零，指向使 $q$ 增加的方向。DPG沿此方向更新 $K$。

### 3. 表格总结：LQR与DPG对应关系

| 概念               | LQR（模型已知）                          | DPG（无模型）                      |
| ------------------ | ---------------------------------------- | ---------------------------------- |
| **价值函数** | $V^*(x) = x^TPx$（解析解）             | $V(x;w)$（神经网络近似）         |
| **Q函数**    | $q^*(x,u) = [x;u]^T H [x;u]$（二次型） | $q(x,u;w)$（函数逼近）           |
| **策略**     | $u = -Kx$（线性）                      | $u = \mu(x;\theta)$（神经网络）  |
| **求解方法** | 求解Riccati方程                          | 策略梯度 + 值函数近似              |
| **最优条件** | $(R+B^TPB)K = B^TPA$                   | $\nabla_u q = 0$（在最优动作处） |

---

## 第五部分：公式推导实例

### 1. 简单倒立摆LQR实例

考虑一阶简化系统（只控制摆杆角度）：

$$
x = [\theta, \omega]^T, \quad A = \begin{bmatrix}0 & 1 \\ a & 0\end{bmatrix}, \quad B = \begin{bmatrix}0 \\ b\end{bmatrix}
$$

设 $Q = \text{diag}(q_1, q_2)$, $R = r$。

Riccati方程给出 $P = \begin{bmatrix}p_{11} & p_{12} \\ p_{12} & p_{22}\end{bmatrix}$，最优增益：

$$
K = \frac{1}{r + b^2 p_{22}} [b p_{12}, \ b p_{22}]
$$

Q函数矩阵：

$$
H = \begin{bmatrix}
q_1 + a^2 p_{11} & a^2 p_{12} & a b p_{11} \\
a^2 p_{12} & q_2 + p_{22} & a b p_{12} \\
a b p_{11} & a b p_{12} & r + b^2 p_{22}
\end{bmatrix}
$$

### 2. DPG更新公式的具体化

策略：$u = -[k_1, k_2] x$

Critic梯度：

$$
\nabla_u q = 2(r + b^2 p_{22}) u + 2b [p_{12}, p_{22}] x
$$

代入 $u = -Kx$：

$$
\nabla_u q = 2 \left( - (r + b^2 p_{22})K + b [p_{12}, p_{22}] \right) x
$$

策略梯度：

$$
\nabla_K J = \mathbb{E}_x \left[ -x \cdot \left( 2 \left( - (r + b^2 p_{22})K + b [p_{12}, p_{22}] \right) x \right)^T \right]
$$

更新规则：

$$
K \leftarrow K + 2\alpha \left( (r + b^2 p_{22})K - b [p_{12}, p_{22}] \right) \mathbb{E}[xx^T]
$$

收敛时：$(r + b^2 p_{22})K = b [p_{12}, p_{22}]$，与LQR一致。

---

## 第六部分：从LQR到一般DPG的推广

### 1. 非线性系统的局部线性化

在平衡点附近，非线性系统 $\dot{x} = f(x,u)$ 可线性化为：

$$
\Delta \dot{x} = A \Delta x + B \Delta u
$$

其中 $A = \frac{\partial f}{\partial x}\big|_{(0,0)}$, $B = \frac{\partial f}{\partial u}\big|_{(0,0)}$。

局部LQR给出初始控制策略，可作为DPG的初始策略。

### 2. 基于模型的Q函数初始化

利用局部线性模型构造初始Q函数：

$$
q_0(x,u) = -[x^T Q x + u^T R u + (A_d x + B_d u)^T P (A_d x + B_d u)]
$$

其中 $P$ 是局部LQR的Riccati解。这为Critic提供了良好的初始值。

### 3. 安全性保证

LQR设计的控制器是局部稳定的，结合控制李亚普诺夫函数（CLF）可确保DPG在学习过程中不破坏稳定性。

---

## 总结

1. **LQR是最简单的线性二次最优控制**，具有解析解，Q函数是状态和动作的二次型。
2. **DPG是模型无关的强化学习算法**，通过策略梯度优化确定性策略。
3. **关键联系**：

   - 两者都寻求最优策略使代价最小（回报最大）
   - 在LQR问题中，DPG收敛到LQR解
   - LQR的Q函数结构为DPG的Critic设计提供指导
   - LQR的控制增益方程对应DPG策略梯度的稳定点
4. **实践意义**：

   - 可用LQR初始化DPG的策略参数
   - 可用局部线性化模型构造初始Q函数
   - LQR的稳定性分析为DPG提供安全保证

通过理解LQR与DPG的深刻联系，我们可以在保持强化学习灵活性的同时，利用经典控制的理论保证，实现更安全、高效的学习控制。
