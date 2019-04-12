## 线性空间的基本概览

### 向量空间

如果集合 $V$ 在*矢量求和 (vector summation)*  

$$
+ : V \times V \to V
$$

以及*标量乘法 (scalar multiplication)*

$$
\cdot : \mathbb{R} \times V \to V
$$

运算下是闭合的，那么集合 $V$ 就称为在 $\mathbb{R}$ 域上的*线性空间 (linear space)* 或 *向量空间 (vector space)* 。



换言之，如果 $V​$ 是一个向量空间，那么对于任意的两个向量 $\mathbf{v_1}, \mathbf{v_2} \in V​$ 和两个标量 $\alpha, \beta \in \mathbb{R}​$ ，他们的线性组合 $\alpha \mathbf{v_1} + \beta \mathbf{v_2} \in V​$ 。此外，加法运算($+​$) 满足**交换律**和**结合律**，且存在**幺元 ($0​$)** 以及每个元素存在**逆 ($- \mathbf{v}​$)** 。标量乘法 ($\cdot​$) 在 $\mathbb{R}​$ 域上有：$\alpha (\beta \mathbf{v}) = (\alpha \beta)\mathbf{v}​$，　$1\mathbf{v} = \mathbf{v}​$ 和 $0\mathbf{v} = \mathbf{0}​$ 。加法和标量乘法满足**分配率**：$(\alpha + \beta) \mathbf{v} = \alpha \mathbf{v} + \beta \mathbf{v}​$，　$\alpha(\mathbf{v} + \mathbf{u}) = \alpha \mathbf{v} + \alpha \mathbf{u}​$ 。  
例如： $\mathbb{R}^n​$ 就是实数域 $\mathbb{R}​$ 上的线性空间。根据上述此时  $V = \mathbb{R}^n​$ ， $\mathbf{v} = [x_1, \dotsc , x_n]^\top​$ 。



一个集合 $W \subset V​$ ，$V​$ 是一个向量空间的话，如果 $0 \in W​$ 且集合 $W​$ 对于任意的 $\alpha \in \mathbb{R}​$ 在 $+​$ 和 $\cdot​$ 上是闭合的，那么 $W​$ 称为 $V​$ 的**子空间 (subspace)**。



### 线性独立与基

一组向量 $S = \{\mathbf{v_1}, \dotsc , \mathbf{v_k}\} \subset V$ 张成的子空间，是由这些向量的所有线性组合构成的子空间：

$$
span(S) = \{ \mathbf{v} \in V | \mathbf{v} = \sum_{i=1}^k \alpha_i \mathbf{v_i}\}
$$

如果，

$$
\sum_{i=1}^k \alpha_i \mathbf{v}_i = \mathbf{0} \Rightarrow \alpha_i =  0   \forall i
$$

那么集合 $S​$ 就被称为**线性独立 (linearly independent)** 。

换句话说，如果集合 $S$ 中的任意一个向量无法用其余向量的线性组合表示的话，那么称为线性独立，反之称为**线性相关 (linearly dependent)** 。

一个向量集合 $B = \{\mathbf{v_1}, \dotsc, \mathbf{v}_n\}$ 如果他是线性独立且它可以张成向量空间 $V$ ，那么称 $B$ 是 $V$ 的**基 (basis)**。基是线性无关向量的最大集合。



#### 基的性质

如果 $B​$ 与 $B^{\prime}​$ 是线性空间 $V​$ 的两个基，那么：  

- $B​$ 与 $B^{\prime}​$ 将包含相同数量的线性独立向量，这个数量 $n​$ 被称为向量空间 $V​$ 的**维度  (dimension)** 。

- 让 $B = \{ b_i \}_{i = i}^n$ 和 $B' = \{ b'_i \}_{i = i}^n$ ，那么 $B$ 中的任意一个基向量都能够利用 $B^{\prime}$ 的线性组合形式表示： 

    $$
    b_j = a_{1j} b'_1 + a_{2j} b'_2 + \dotsb + a_{nj} b'_n　　a_{ij} \in \mathbb{R}, i, j = 1, 2, \dotsc, n
    $$

    这里用于**基底变换 (basis transform)** 的系数 $a_{ij}$ 能够被组合成矩阵 $\mathbf{A}$ ，那么我们可以用矩阵的形式来表示 $B \doteq [ b_1, b_2, \dotsc, b_n ]$ 和 $B' \doteq [ b'_1, b'_2, \dotsc, b'_n ]$ 之间的变换关系了：

    $$
    [ b_1, b_2, \dotsc, b_n ] = [ b'_1, b'_2, \dotsc, b'_n ] 
    	\begin{bmatrix}
        a_{11} & a_{12} & \cdots & a_{1n} \\
        a_{21} & a_{22} & \cdots & a_{2n} \\
        \vdots  & \vdots  & \ddots & \vdots  \\
        a_{n1} & a_{n2} & \cdots & a_{nn} 
      \end{bmatrix}
    $$

    即 $B = B' A$ ，当矩阵 $\mathbf{A}$ 可逆时，有 $B’ = B A^{-1}$ 。

- 任意一个向量 $\mathbf{v} \in V​$ 都能够写成基向量的线性组合：

    $$
    \mathbf{v} = x_1 b_1 + x_2 b_2 + \dotsb + x_n b_n = x'_1 b'_1 + x'_2 b'_2 + \dotsb + x'_n b'_n
    $$

    其中系数 $\{ x_i \in \mathbb{R} \}_{i=1}^n$ 和 $\{ x'_i \in \mathbb{R} \}_{i=1}^n$ 是唯一且确定的，称为 $\mathbf{v}$ 在每一基底下的坐标。结合上一条性质，有：

    $$
    \mathbf{v} 
    = 
    [ b_1, b_2, \dotsc, b_n ]
    \begin{bmatrix}
    x_1 \\ x_2 \\ \vdots \\ x_n
    \end{bmatrix}
    = 
    [ b'_1, b'_2, \dotsc, b'_n ] 
    \begin{bmatrix}
      a_{11} & a_{12} & \cdots & a_{1n} \\
      a_{21} & a_{22} & \cdots & a_{2n} \\
      \vdots & \vdots & \ddots & \vdots \\
      a_{n1} & a_{n2} & \cdots & a_{nn} 
    \end{bmatrix}
    \begin{bmatrix}
    x_1 \\ x_2 \\ \vdots \\ x_n
    \end{bmatrix}
    $$

    由于 $\mathbf{v}$ 关于 $B^{\prime}$ 的坐标是唯一的，因此可以得到一个向量从一个基底到另一个基底的坐标变换为：

    $$
    \begin{bmatrix}
    x'_1 \\ x'_2 \\ \vdots \\ x'_n
    \end{bmatrix}
    = 
    \begin{bmatrix}
      a_{11} & a_{12} & \cdots & a_{1n} \\
      a_{21} & a_{22} & \cdots & a_{2n} \\
      \vdots & \vdots & \ddots & \vdots \\
      a_{n1} & a_{n2} & \cdots & a_{nn} 
    \end{bmatrix}
    \begin{bmatrix}
    x_1 \\ x_2 \\ \vdots \\ x_n
    \end{bmatrix}
    $$


!!! note "注意变换基底和变换坐标的区别"
    $$
    B' = B A^{-1},　　\mathbf{x}' = A \mathbf{x}
    ​$$



### 内积与正交性

#### 内积

当满足：  

1. $\langle u, \alpha v + \beta w \rangle =  \alpha \langle u, v \rangle + \beta \langle u, w \rangle，　\forall \alpha, \beta \in \mathbb{R}​$ （线性的）
2. $\langle u, v \rangle ＝ \langle v, u \rangle​$  （对称的/均匀的）
3. $\langle v, v \rangle \geq 0​$ 且 $\langle v, v \rangle = 0 \Leftrightarrow v = 0​$ 　（正定的）

那么，可以在向量空间上定义**内积 (inner product) [或点积 (dot product)]** 运算:

$$
\langle \cdot , \cdot \rangle : V \times V \to \mathbb{R}
$$

进而引申出**模 (norm)** :

$$
|\cdot | : V \to \mathbb{R}, 　| \mathbf{v} | = \sqrt{\langle \mathbf{v} , \mathbf{v} \rangle}
$$

以及**[量度 (metric)](https://en.wikipedia.org/wiki/Metric_(mathematics))** :

$$
d : V \times V \to \mathbb{R}, 　d( \mathbf{v}, \mathbf{w}) = | \mathbf{v} - \mathbf{w} | = \sqrt{\langle \mathbf{v} - \mathbf{w} , \mathbf{v} - \mathbf{w} \rangle}
$$

用于测量长度与距离，使 $V$ 称为一个**量度空间 (metric space)** 。自内积引申出量度以后，向量空间 $V$ 亦称为[**希尔伯特空间 (Hibert space)**](https://en.wikipedia.org/wiki/Hilbert_space) 。



#### 实数域中的标准内积

当 $V = \mathbb{R}$ 时，可以为标准基 $B = I_n$ 定义一个标准内积形式:

$$
\langle \mathbf{x},\mathbf{y} \rangle \doteq \mathbf{x}^\top \mathbf{y} = \sum_{i=1}^n{x_i y_i}
$$

引申出**标准模 (L~2~-norm)** 或 **欧几里得范数 (Euclidean norm)** ：

$$
\| \mathbf{x} \|_2 \doteq \sqrt{\mathbf{x}^\top \mathbf{x}} = \sqrt{x_1^2 + x_2^2 + \dotsb + x_n^2}
$$

在此基础上将利用基底变换矩阵 $\mathbf{A}​$ 将 $B = I_n​$ 转换到一个新的基底 $B​'$ ， 即 $I_n = B' A^{-1}​$ 那么内积形式可以写成：

$$
\langle \mathbf{x},\mathbf{y} \rangle = \mathbf{x}^\top \mathbf{y} = (A \mathbf{x}')^\top (A \mathbf{y}') = \mathbf{x}'^\top A^\top A \mathbf{y}' \doteq \langle \mathbf{x}',\mathbf{y}' \rangle_{A^\top A}
$$



#### 正交性 (Orthogonality)

如果两个向量 $\mathbf{x}, \mathbf{y}$ 正交，那么他们的内积为零，即 $\langle \mathbf{x}, \mathbf{y} \rangle = 0$ ，通常表示为 $\mathbf{x} \bot \mathbf{y}$ 。



#### 矩阵的克罗内克乘积 (Kronecker product) 和 堆形式(stack)

矩阵 $A \in \mathbb{R}^{m \times n}​$ 和 $B \in \mathbb{R}^{k \times l}​$ de 克罗内克乘积定义为 $A \otimes B​$ ，得到一个新的矩阵为：

$$
A \otimes B =
\begin{bmatrix}
  a_{11} B & a_{12} B & \cdots & a_{1n} B \\
  a_{21} B & a_{22} B & \cdots & a_{2n} B \\
  \vdots   & \vdots   & \ddots & \vdots \\
  a_{m1} B & a_{m2} B & \cdots & a_{mn} B 
\end{bmatrix}
\in \mathbb{R}^{mk \times nl}
$$

矩阵 $A \in \mathbb{R}^{m \times n}$ 的堆形式被定义为 $A^s$ ，它是通过矩阵 ${A}$ 的 $n$ 列向量 $a_1, \dotsc, a_n \in \mathbb{R}^n$堆积形成的。表示成：

$$
A^s \doteq 
\begin{bmatrix}
a_1 \\ a_2 \\ \vdots \\ a_n
\end{bmatrix}
\in \mathbb{R}^{mn}
$$

克罗内克乘积和矩阵堆栈形式允许我们用许多不同但等价的方式重写涉及多个向量和矩阵的代数方程。比如方程：

$$
\mathbf{u}^\top A \mathbf{v} = (\mathbf{v} \otimes \mathbf{u})^\top A^\top
$$

当矩阵 ${A}$ 是上式中唯一的未知量的时候，等号右边的形式是特别有用的。



## 线性变换与矩阵群







## 参考

1.  Multiple View Geometry (IN2228) SS 2016, TU München : Chapter 1
2.  An Invitation to 3D Vision: From Images to Geometric Models : Appendix A



---8<---
mathjax.txt
---8<---