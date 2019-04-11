## 向量空间

如果集合 $V$ 在*矢量求和 (vector summation)*

$$
+ : V \times V \to V
$$

以及*标量乘法 (scalar multiplication)*

$$
\cdot : \mathbb{R} \times V \to V
$$

运算下是闭合的，那么集合 $V$ 就称为在 $\mathbb{R}$ 域上的*线性空间 (linear space)* 或 *向量空间 (vector space)* 。



换言之，对于任意的两个向量 $\mathbf{v_1}, \mathbf{v_2} \in V$ 和两个标量 $\alpha, \beta \in \mathbb{R}$ ，他们的线性组合 $\alpha \mathbf{v_1} + \beta \mathbf{v_2} \in V$ 。此外，加法运算($+$) 满足**交换律**和**结合律**，且存在**幺元 ($0$)** 以及每个元素存在**逆 ($- \mathbf{v}$)** 。标量乘法 ($\cdot$) 在 $\mathbb{R}$ 域上有：$\alpha (\beta \mathbf{v}) = (\alpha \beta)\mathbf{v}$ ，$1\mathbf{v} = \mathbf{v}$ 和 $0\mathbf{v} = \mathbf{0}$ 。加法和标量乘法满足**分配率**：$(\alpha + \beta) \mathbf{v} = \alpha \mathbf{v} + \beta \mathbf{v}$ ，$\alpha(\mathbf{v} + \mathbf{u}) = \alpha \mathbf{v} + \alpha \mathbf{u}$ 。  
例如： $\mathbb{R}^n$ 就是实数域 $\mathbb{R}$ 上的线性空间。根据上述此时  $V = \mathbb{R}^n$ ， $\mathbf{v} = [x_1, \dotsc , x_n]^\top$ 。



一个集合 $W \subset V$ ，$V$ 是一个向量空间的话，如果 $0 \in W$ 且集合 $W$ 对于任意的 $\alpha \in \mathbb{R}$ 在 $+$ 和 $\cdot$ 上是闭合的，那么 $W$ 称为 $V​$ 的**子空间**。



## 线性独立与基

一组向量 $S = \{\mathbf{v_1}, \dotsc , \mathbf{v_k}\} \subset V$ 张成的子空间，是由这些向量的所有线性组合构成的子空间：

$$
span(S) = \{ \mathbf{v} \in V | \mathbf{v} = \sum_{i=1}^k \alpha_i \mathbf{v_i}\}
$$

如果，

$$
\sum_{i=1}^k \alpha_i \mathbf{v}_i = \mathbf{0} \Rightarrow \alpha_i =  0   \forall i
$$

那么集合 $S$ 就被称为**线性独立**。

换句话说，如果集合 $S​$ 中的任意一个向量无法用其余向量的线性组合表示的话，那么称为线性独立，反之称为**线性相关**。

一个向量集合 $B = \{\mathbf{v_1}, \dotsc, \mathbf{v}_n\}$ 如果他是线性独立且它可以张成向量空间 $V$ ，那么称 $B$ 是 $V​$ 的**基**。基是线性无关向量的最大集合。





## 参考

1.  Multiple View Geometry (IN2228) SS 2016, TU München : Chapter 1
2.  An Invitation to 3D Vision: From Images to Geometric Models : Appendix A



---8<---
mathjax.txt
---8<---