## 三维重建的起源

从一组二维视图重建世界的三维结构在计算机视觉领域有着悠久的历史。这是一个经典的*不适定问题 (ill-posed problem)*，因为重构一组一致的观察或图像相通常并不是唯一的。因此，我们需要附加一些假设。在数学上，研究三维场景与观测到的二维投影之间的几何关系是基于两种类型的变换，即，

- 用**欧几里得运动 (Euclidean motion)** 或**刚体运动 (rigid-body motion)** 来表示相机从当前帧到下一帧图像的运动
- 用**透视投影 (Perspective projection)** 来表示图像的形成过程 (如：**针孔相机 (pinhole camera)** 等)。

透视投影的概念起源于古希腊 (Euclid of Alexandria,  400 B.C.) 和文艺复兴时期 (Brunelleschi & Alberti, 1435)。透视投影的研究引出了**投影几何 (projective geometry)** 领域。

关于多视几何的第一个研究工作成果是来至于 Erwin Kruppa (1913) ，他指出五个点的两个视图足以确定两个视图之间的相对变换（运动） 和点的三维位置（结构）。Longuet-Higgins 在 1981 年提出了一种基于两视图**对极约束 (epipolar constraint)** 恢复运动结构重建的线性算法。在几本教科书中总结了一系列关于这方面的著作 (Faugeras 1993, Kanatani 1993, Maybank 1993, Weng et al. 1993) 。对三个视图的扩展由 Spetsakis 和 Aloimonos 87、90、Shashua 94 和 Hartley 95 研究发布的。多视图和正交投影的因子分解技术是由 Tomasi 和 Kanade 于 1992 年研究发布的。

相机运动与三维位置的联合估计称为**运动结构重建 (structure and motion)** 或**视觉 SLAM (visual SLAM)** 。



### 三维欧式空间

一般来说，欧几里得空间是一个集合，它的元素满足欧几里得的五个公理。三维的欧几里得空间 $\mathbb{E}^3$ 是由以下式为坐标的所有点 $P \in \mathbb{E}^3$ 组成的。

$$
\mathbf{X} \doteq [X_1, X_2, X_3]^\top = 
\begin{bmatrix} X_1 \\ X_2 \\ X_3 \end{bmatrix} \in \mathbb{R}^3
$$

通过这样一个**笛卡尔 (Cartesian) 坐标系**的赋值，可以在 $\mathbb{E}^3$ 和 $\mathbb{R}^3$ 之间建立一一对应关系。这里的 $\mathbb{E}^3$ 可以被视为等同于 $\mathbb{R}^3$ 。这意味着允许我们当讨论一个点 ( $\mathbb{E}^3$ ) 和坐标 ( $\mathbb{R}^3$ ) 犹如是一回事一样。笛卡尔坐标是使测量距离和角度成为可能的第一步。为此，必须为 $\mathbb{E}^3$ 赋予**度量标准 (metric)** 。 度量的精确定义依赖于向量的概念。

#### 向量

在欧式空间中，一个向量 $\mathbf{v}$ 由一对点 $q, p \in \mathbb{E}^3$ 确定，被定义为链接 $p$ 到 $q$ 的有向箭头记号，表示称 $\mathbf{v} = \overrightarrow{pq}$ 。这里的点 $p$ 通常被称为是向量 $\mathbf{v}$ 的基点。假使点 $p$ 的坐标为 $\mathbf{X}$ ，点 $q$ 的坐标为 $\mathbf{Y}$ ，那么向量 $\mathbf{v}$ 的坐标为：

$$
\mathbf{v} = [v_1, v_2, v_3]^\top \doteq \mathbf{Y} - \mathbf{X} \in \mathbb{R}^3
$$

以上对向量的定位被称为**有界向量 (bound vector)** 。考虑这个向量独立于它的基点 $p$ 使得它是一个**自由向量 (free vector)** 。

!!! note ""  
    需要注意：点和向量是不同的几何对象。这一点很重要，我们很快就会看到，因为刚体运动对点和向量的作用是不同的。

所有自由向量 $\mathbf{v} \in \mathbb{R}^3$ 的集合构成一个线性向量空间。通过确定 $\mathbb{E}^3$ 和 $\mathbb{R}^3$ 之间的联系， $\mathbb{E}^3$ 的欧几里德度量标准仅由向量空间 $\mathbb{R}^3$ 上的一个**内积 (inner product，或称为点积，dot product)** 定义。有了这样一个度规，我们不仅可以测量点之间的距离（向量的模）或向量之间的角度，还可以计算曲线的长度或区域的体积。

运动粒子 $p$ 在 $\mathbb{E}^3$ 中的运动轨迹可用曲线 $\gamma(\cdot) : t \mapsto \mathbf{X} \in \mathbb{R}^3, t \in [0, 1]$ 来描述，则曲线的总长度为：

$$
l(\gamma(\cdot)) = \int_0^1 \| \dot{\mathbf{X}} (t) dt \|
$$

这里的 $\dot{\mathbf{X}} (t) = \frac{\mathrm d}{\mathrm d t} \big( \mathbf{X} (t) \big) \in \mathbb{R}^3$ 被称为曲线的切向量。 









--8<--
mathjax.txt
--8<--