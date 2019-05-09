## SVO : Semi-Direct Visual Odometry

半直接视觉里程计，所谓的半直接是指对图像中提取的特征点图像块进行直接匹配来获取相机的位姿，而不同于直接匹配法那么对整个图像使用直接匹配的方式来获取相机位姿。虽然*半直接 (Semi-Direct)* 法使用了特征块，但它的基础思想还是类似于*直接法 (Direct method)* 来获取位姿信息，这点与*特征点法  (Feature-Based method)* 的提取额一组稀疏特征点，使用特征描述子匹配，通过对极约束来估计位姿是不一样的。然而，半直接法与直接法不同的是它利用了特征块匹配，通过再投影误差最小化来对直接法估计的位姿进行优化。

!!! tip  
    虽然 SVO 并不是一个标准的完整 SLAM ，它舍弃了后端优化与回环检查部分，也基本没有建图功能，但是 SVO 的代码结构清晰易于理解，很适合作为第一个入门项目。



## SVO 算法架构

SVO 算法架构主要分成两个部分：位姿估计、深度估计。如下图所示

![SVO 框架流程](image/SVO_Structure.png)

运动估计线程部分实现了相对姿态估计的半直接法。步骤如下：  

1. 通过基于稀疏的图像对齐进行姿态初始化：通过最小化对应于相同 3D 点投影位置的像素之间的光度误差，得到相对于前一帧的相机姿态
2. 通过对相应的 feature-patch 进行对齐，对重新投影点对应的 2D 坐标进行优化
3. 通过最小化前向特征对准步骤中引入的重投影误差来精炼姿态和空间特征点位置以得到运动估计的结果

深度估计部分，为每个待估计相应 3D 点的 2D 特征初始化概率深度滤波器。每当在图像中发现此时的 3D 到 2D 的特征对应少于设定阈值的时候，将选择新的关键帧提取特征点，进而初始化新的深度滤波器。这些滤波器的初始值具有很大的不确定性，在随后的每一帧中，深度估计都以贝叶斯方式更新。当深度滤波器的不确定性足够小时（即收敛），在地图中插入一个新的三维点，并立即用于运动估计。



### 运动估计

SVO 利用直接方法对相机的相对运动和特征对应进行了初步的估计，并以基于特征的非线性重投影误差最小化方法进行了优化。下面将详细介绍其中的每个步骤。

#### 基于稀疏模型的图像对齐

基于稀疏模型的图像对齐 (Sparse Model-based Image Alignment) 使用直接法最小化图像块重投影残缺来获取位姿。如下图所示：其中{==红色==}的 $\color{red}{T_{k, k-1}}$ 为相邻帧之间的位姿变换，即待优化变量。

![Image Alignment](image/image_alignment.png)

这个过程的数学表达为求一个关于刚体运动最大似然估计 $T_{k, k-1}$ ，即可以通过求在两个连续的相机姿态之间亮度残差的最小化负对数似然函数来得到：

$$
\mathrm{T}_{k,k-1} = \mathrm{arg\, \mathop{min}\limits_T} \iint_{\bar{\mathcal{R}}} \mathrm{\rho} [\delta I(\mathrm{T}, \mathbf{u})] \mathrm{d} \mathbf{u} \tag{1.1}
$$

因此这个过程可以分解为：

- **准备工作：**假设相邻帧之间的位姿 $\mathrm{T}_{k, k-1}$ 已知，一般初始化为上一相邻时刻的位姿或者假设为单位矩阵。通过之前多帧之间的特征检测以及深度估计，我们已经知道在第 k-1 帧中的特征点位置 $\mathbf{u}$ 以及它们的深度 $\mathrm{d}_\mathbf{u}$ 。 

- **重投影：**亮度残差 $\delta I$ 由观测同一个三维空间点的像素间的光度差确定。准备工作中已知了 $I_{k-1}$ 中的某个特征在图像平面中位置 $\mathbf{u}$ 以及它们的深度 $\mathrm{d}_\mathbf{u}$ ，能够将该特征投影到三维空间 $\mathrm{p}_{k-1}$ 。由于该三维空间的坐标系是定义在 $I_{k-1}$ 相机坐标系下的，因此需要通过位姿变换 $T_{k, k-1}$ 将它投影到当前帧 $I_{k}$ 中，得到该点当前帧坐标系中的三维坐标 $\mathrm{p}_{k}$ 。最后通过相机内参，投影到 $I_{k}$ 的图像平面得到坐标 $\mathbf{u}'$ ，完成重投影。亮度残差 $\delta I$ 定义为：

    $$
\delta I (\mathrm{T}, \mathbf{u}) = I_k \Big( \underbrace{ \pi \big( \underbrace{ \mathrm{T} \cdot \underbrace{\pi^{-1}(\mathbf{u}, \mathrm{d}_\mathbf{u}) \big)}_{1}}_2 }_3 \Big) - I_{k-1}(\mathbf{u}) \quad \forall \mathbf{u} \in \bar{\mathcal{R}} \tag{1.2}
    $$

    公式中第 1 步为根据前一帧图像特征位置和深度逆投影到三维空间，第 2 步将三维坐标点旋转平移到当前帧坐标系下，第 3 步再将三维坐标点投影回当前帧图像坐标。其中上一帧 $I_{k-1}$ 和当前帧  $I_{k}$ 能共视到的特征集合为 $\bar{\mathcal{R}}$ ，即
    
    $$
\bar{\mathcal{R}} = \{ \mathbf{u} | \mathbf{u} \in \mathcal{R}_{k-1} \wedge \pi (\mathrm{T} \cdot \pi^{-1}(\mathbf{u}, \mathrm{d}_\mathbf{u})) \in \Omega_k \} \tag{1.3}
    $$
    
    当然在优化过程中，亮度残差 $\delta I$ 的计算方式不止这一种形式：有**前向 (forwards)** ，**逆向 (inverse)** 之分，并且还有**叠加式 (additive)** 和**构造式 (compositional)** 之分。这方面可以读读光流法方面的论文 [^2]。选择的方式不同，在迭代优化过程中计算雅克比矩阵的时候就有差别，一般为了减小计算量，都采用的是 **inverse compositional algorithm** 。 (#TODO 抑或是参考计算机视觉基础-光流篇) 

- **迭代优化更新位姿：**按理来说极短时间内的相邻两帧拍到空间中同一个点的亮度值应该没啥变化。但由于位姿是假设的一个值，所以重投影的点不准确，导致投影前后的亮度值是不相等的。不断优化位姿使得这些以特征点为中心的 $4 \times 4$ 像素块残差最小，就能得到优化后的位姿 $\mathrm{T}_{k, k-1}$ 。

将上述过程公式化如下：为简便起见，我们假设亮度残差服从单位方差正态分布，那么负对数最小化似然估计等同于最小二乘问题，即 $\rho[\cdot] \hat{=} \frac{1}{2} \|\cdot \| ^2$ 。因此位姿 $T_{k, k-1}$ 的最小化残差**损失函数 (Cost Function)** 为：

$$
\mathrm{T}_{k,k-1} = \arg  \min_\limits{\mathrm{T}_{k,k-1}} \frac{1}{2} \sum_{i \in \bar{\mathcal{R}}} \| \delta \mathrm{I}(\mathrm{T}_{k,k-1}, \mathbf{u}_i \|^2 \tag{1.4}
$$

上面的非线性最小化二乘问题，可以用高斯牛顿迭代法求解。设位姿变换的估计值为 $\hat{T}_{k, k-1}$ 、通过**旋转坐标 (twist coordinates)** $\xi = (\omega, \upsilon)^\top \in \mathfrak{se}(3)$ 参数化估计的增量更新 $\mathrm{T}(\xi)$ 。依据图像 $I_{k-1}$ 的计算更新 $\mathrm{T}(\xi)$ ，通过 **inverse compositional** 构造亮度残差：

$$
\delta \mathrm{I}(\xi, \mathbf{u}_i) = \mathrm{I}_k \big(\pi(\hat{\mathrm{T}}_{k,k-1} \cdot \mathbf{p}_i) \big) - \mathrm{I}_k \big(\pi(\mathrm{T}(\xi) \cdot \mathbf{p}_i) \big)\, , \, \mathbf{p}_i = \pi^{-1}(\mathbf{u}_i, \mathrm{d}\mathbf{u}_i) \tag{1.5}
$$

当前的估计值通过下式跟新，

$$
\hat{\mathrm{T}}_{k,k-1} \gets \hat{\mathrm{T}}_{k,k-1} \cdot {\mathrm{T}}(\xi)^{-1} \tag{1.6}
$$

为了找到最佳的更新量 $\mathrm{T}(\xi)$ ，我们可以通过求式 (4) 的偏导数并让它等于零：

$$
\sum_{i \in \bar{\mathcal{R}}} \nabla \delta \mathrm{I} (\xi, \mathbf{u}_i)^\top \delta \mathrm{I}(\xi, \mathbf{u}_i) = 0 \tag{1.7}
$$

为了求解上式，我们对当前状态进行线性化：

$$
\delta \mathrm{I} (\xi, \mathbf{u}_i) \approx \delta \mathrm{I}(0, \mathbf{u}_i) + \nabla \delta \mathrm{I}(0, \mathbf{u}_i) \cdot \xi \tag{1.8}
$$

其中雅克比矩阵 $\mathbf{J}_i := \nabla \delta \mathrm{I}(0, \mathbf{u}_i)$ 为图像残差对李代数的求导，可以通过链式求导得到:

$$
\frac{\partial \delta \mathrm{I} (\xi, \mathbf{u}_i)}{\partial \xi} = \left. \frac{\partial \mathrm{I}_{k-1}(\mathrm{a})}{\partial \mathrm{a}} \right|_{\mathrm{a} = \mathbf{u}_i} \cdot \left. \frac{\partial \pi (\mathrm{b})}{\partial \mathrm{b}} \right|_{\mathrm{b}=\mathbf{p}_i} \cdot \left. \frac{\mathrm{T}(\xi)}{\partial \xi} \right|_{\xi=0} \cdot \mathbf{p}_i \tag{1.9}
$$

其中文章中导数的求解，请参考高博的[直接法](http://www.cnblogs.com/gaoxiang12/p/5689927.html)。（#TODO）

通过将式 (1.8) 代入式 (1.7) 并通过将雅克比堆叠成矩阵 $\mathbf{J}$ ，我们得到正规方程：

$$
\mathbf{J}^\top \mathbf{J} \xi = - \mathbf{J}^\top \delta \mathrm{I}(0) \tag{1.10}
$$

注意，通过使用 inverse compositional 构造亮度残差方法，雅可比可以预先计算，因为它在所有迭代中保持不变，因此降低了计算量。

以上通过当前帧与相邻前一帧反投影解算出了相对位姿 $\mathrm{T}_{k, k-1}$ ，由于三维点的位置不准确，这种 frame-to-frame 估计位姿的方式不可避免的会带来累计误差从而导致漂移，因而通过以上步骤求出的相机的姿态需要进一步优化。由此，进行一下步骤：Relaxation Through Feature Alignment 。

#### 通过特征点对齐优化匹配关系

为了减少偏移，相机的姿态应该通过已经建立好的地图模型，来进一步约束当前帧的位姿。利用初始的位姿 $\mathrm{T}_{k, k-1}$ 关系，可以大体的对当前图像中所有可见三维点的特征位置进行初步猜测，将能够观察到地图中已经收敛的特征点投影到当前帧中。但是由于估计位姿存在偏差，导致将地图中特征点重投影到 $\mathrm{I}_k$ 中的位置并不和真正的吻合，也就是还会有残差的存在。如下图所示：（图中 $\mathrm{I}_k$ 帧图像中的灰色特征块为真实位置，蓝色特征块为预测位置）

![Feature Alignment](image/feature_alignment.png)

对于每个地图中重投影的特征点，识别出观察角度最小关键帧 $\mathrm{I}_r$ 上的对应点 $\mathbf{u}_i$ 。由于 3D 点和相机姿态估计不准确，所有利用**特征对齐 (Feature Alignment)** 通过最小化当前图像 $\mathrm{I}_k$ 中 patch (蓝色方块) 与关键帧 $\mathrm{I}_r$ 中的参考 patch 的光度差值来{==优化当前图像中每个 patch 的 2D 位置==} $\color{red}{\mathbf{u'}_i}$ ：

$$
\mathbf{u'}_i = \arg \min\limits_{\mathbf{u'}_i} \frac{1}{2} \| \mathrm{I}_k(\mathbf{u'}_i) - \mathrm{A}_i \cdot \mathrm{I}_r{\mathbf{u}_i} \|^2, \quad \forall i. \tag{1.11}
$$

这种对齐使用 inverse compositional  Lucas-Kanade 算法来求解[^2]。并且注意，光度误差的前一部分是当前图像中 $\mathrm{I}_k$ 的亮度值，后一部分不是 $\mathrm{I}_{k-1}$ 而是 $\mathrm{I}_r$ 。由于是特征块对比，并且 3D 点所在的关键帧可能离当前帧比较远，所以光度误差和前面不一样的是还加了一个仿射变换 $\mathrm{A}_i$ ，需要对关键帧中的特征块进行旋转拉伸之类仿射变换后才能和当前帧的特征块对比。 这步不考虑极线约束，因为此时的位姿还是不准确的。这时候的迭代量计算方程和之前是一样的，只不过雅克比矩阵变了，这里的雅克比矩阵很好计算 $\mathbf{J} = \frac{\partial \mathrm{I}(\mathbf{u}_i)}{\partial \mathbf{u}_i}$ ，即为图像横纵两个方向的梯度。

通过这一步我们能够得到优化后的特征点预测位置，它比之前通过相机位姿预测的位置更准，所以反过来，我们利用这个优化后的特征位置，能够进一步去优化相机位姿以及特征点的三维坐标。所以位姿估计的最后一步就是 Pose and Structure Refinement 。

#### BA 优化

利用上一步建立的 $(\mathbf{p_i \, , \, \mathbf{u}_i})$ 的对应关系，再次优化世界坐标系下的位姿 $\mathrm{T}_{k, w}$ ，以最小化重投影残差：

$$
\mathrm{T}_{k,w} = \arg \min\limits_{\mathrm{T}_{k,w}} \frac{1}{2} \sum_i \| \mathbf{u}_i - \pi (\mathrm{T}_{k,w} \; {}_w\mathbf{p}_i) \|^2 \tag{1.12}
$$

上式中{==误差变成了像素重投影以后位置的差异 (不是像素值的差异)==} ，优化变量还是相机位姿，雅克比矩阵大小为 2×6 (坐标 $\mathbf{u}_i$ 分别对李代数变量 $\xi = (\omega, \upsilon)^\top \in \mathfrak{se}(3)$ 求导) 。这一步叫做 **motion-only Bundler Adjustment** 。同时根据这个误差定义，我们还能够对获取的三维点的坐标 $[x, y, z]^\top$ 进行优化，还是上面的误差像素位置误差形式，只不过优化变量变成三维点的坐标，这一步叫 **Structure -only Bundler Adjustment** ，优化过程中雅克比矩阵大小为 2×3 (坐标 $\mathbf{u}_i$ 分别对三维点的坐标 $[x, y, z]^\top$ 变量求导) 。



### 地图构建

地图模型通常用来存储三维空间点，在 SVO 中每一个 Key frame 通过深度估计能够得到特征点的三维坐标，这些收敛的三维坐标点通过特征点在 Key Frame 中进行保存。当新帧被选为关键帧时，它会被立即插入地图。同时，又在这个新的关键帧上检测新的特征点作为深度估计的 seed ，这些 seed 会不断融合之后的图像进行深度估计。但是，如果有些 seed 点 3D 点位置通过深度估计已经收敛了，此时 map 用一个 point_candidates 来保存这些尚未插入地图中的点。所以 SVO 地图上保存的是 Key Frame 以及还未插入地图关键帧中的已经收敛的 3D 点坐标（这些 3D 点坐标是在世界坐标系下的）。

#### 深度计算

SVO 中的每个新特征点对应一个深度估计，其初值为该帧的平均深度，并被赋予极大的不确定性。通过两帧图像的匹配点就可以计算出这一点的深度值，如果有多幅图像，那就能计算出这一点的多个深度值。这就像对同一个状态变量我们进行了多次测量，因此，可以用贝叶斯估计来对多个测量值进行融合，使得估计的不确定性缩小。如下图所示：

![Depth Estimation](image/depth_estimation.png)

一开始深度估计的不确定性较大 ($\color{cyan}{\mathbf{\text{青色部分}}}$) ，通过三角化得到一个深度估计值以后，能够极大的缩小这个不确定性 ($\color{teal}{\mathbf{\text{墨绿部分}}}$) 。 

SVO 关于三角化计算深度的过程，主要是极线搜索确定匹配点。我们知道参考帧 $\mathrm{I}_r$ 中的一个特征的图像位置，假设它的深度值在 $[\mathrm{d}_{min},\mathrm{d}_{max}]$ 之间，那么根据这两个端点深度值，利用对极几何就能够计算出特征点在当前帧 $\mathrm{I}_k$ 中的大概位置 (位于极线段附近，即上图 $\mathrm{I}_k$ 中 $\color{cyan}{\mathbf{\text{青色线段}}}$)  。确定了特征出现的极线段位置，就可以进行特征搜索匹配。如果极线段很短，小于两个像素，那直接使用前面面求位姿时提到的 Feature Alignment 光流法就可以比较准确地预测特征位置。如果极线段很长，那分两步走，第一步在极线段上间隔采样，对采样的多个特征块一一和参考帧中的特征块匹配，用 Zero mean Sum of Squared Differences 方法对各采样特征块评分，哪一个特征块得分最高，说明它和参考帧中的特征块最匹配。第二步就是在这个得分最高点附近使用 Feature Alignment 得到次像素精度的特征点位置。

像素点位置确定了，就可以三角化计算深度了。 SVO 使用三角化计算特征点深度，使用的是中点法，关于这个三角化代码算法的推导见 [Github Issue](https://github.com/uzh-rpg/rpg_svo/issues/62) 。这是多视角几何的基础内容，可以参考《Multiple View Geometry in Computer Vision》，或者白巧克力亦唯心的博客[^4] （or #TODO）。

#### 深度值的不确定性计算

在三角化计算深度的时候，还有一个很重要的量需要计算，那就是这个深度值的不确定度。它在后续的利用贝叶斯概率模型更新深度的过程中被用来确定更新权重 (就像卡尔曼滤波器中的协方差矩阵扮演的角色) 。SVO 中对特征点定位不准确导致的三角化深度误差分析如下图所示：

![Depth Uncertainty](image/depth_uncertainty.png)

它是通过假设特征点定位差一个像素偏差，来计算深度估计的不确定性。下面给出 SVO 代码算法推导，也可见参考 5[^5]、6[^6] 。

已知量：$C_r$ 坐标系下的单位长度特征 $\mathbf{f}$ ，位移量 $\overrightarrow{C_r C_k} : \mathbf{t}$  ，特征 $\mathbf{f}$  的计算深度 $z$ ，以及一个像素偏差的误差角度 $\angle{err\_angle} = \arctan (1 /(2.0 * focal\_length))*2.0$ ，则：

$$
向量 \: \overrightarrow{C_k \: {}_r\mathrm{p}} : \mathbf{a} = \mathbf{f} \cdot z - \mathbf{t} \\
\alpha = \arccos \big(\mathbf{f} \cdot \mathbf{t} \div ( \|\mathbf{f}\| \times \|\mathbf{t}\|) \big) \\
\beta = \arccos \big(\mathbf{a} \cdot (-\mathbf{t}) \div ( \|\mathbf{a}\| \times \|\mathbf{t}\|) \big) \\
\beta^+ = \beta + \angle{err\_angle} \\
\gamma^+ =  \pi - \alpha - \beta^+ \\
\frac{z^+}{\sin(\beta^+)} = \frac{\|\mathbf{t}\|}{\sin(\gamma^+)} \quad (正弦定理) \\
\tau = z^+ - z
$$

#### 深度值估计更新

有了新的深度估计值和估计不确定量以后，就可以根据贝叶斯概率模型对深度值进行更新。SVO 对深度值的估计分布采用了高斯与均匀混合分布来表示 (见参考 7 [^7]) 。

高斯与均匀混合分布给出：一个好的测量值是在真实深度 $Z$ 为均值的正态分布附近，而一个离群值的测量值是在范围为 $[Z_{min}, Z_{max}]$ 的均匀分布的区间内，因此 $x$ 概率密度函数为：

$$
p(x_n | Z, \pi) = \pi \mathcal{N}(x_n | Z, \tau_n^2) + (1-\pi) \mathcal{U}(x_n | Z_{min}, Z_{max}) \tag{2.1}
$$

注意，这里的 $\pi$ 与上文中的不是同一个，其中这里 $\pi$ 是为有效测量的概率，$\tau$ 是上一步计算的深度估计值的不确定量。当我们得到同一 seed 的一系列测量值 $x_1, \dotsc, x_n$ 假设这些测量值独立。我们想从式 (2.1) 求出 $Z, \pi$ ，最为直观的做法是求解最大似然估计，然而参考 7 [^7] 作者 G. Vogiatzis 认为最大似然估计容易被局部极大值干扰，其结果并不准确，于是选择从最大后验概率求解，等价求解 $\arg \max_\limits{Z,\pi}p(Z, \pi| x_1, \dotsc, x_n)$ 。

下面结合 G. Vogiatzis 论文 (参考 7 [^7] ) 中的 Supplementary material 以及引用参考 8[^8]、9[^9] ，整理出该概率模型的后验概率 $p(Z, \pi| x_1, \dotsc, x_n)$ 可以用 Gaussian×Beta 分布来近似的证明推导，以及后验概率迭代形式的推导。

首先假设 $Z , \pi$ 的先验分布在没有任何其他信息的情况下，这些量在概率上是独立的，因此满足：

$$
p(Z,\pi) = p(Z) p(\pi) \tag{2.2}
$$

引入**二进制潜变量 (binary latent variables)** $y_1 \dotsc y_n$ ，那么有：

$$
p(x_n | Z, \pi, y_n) = \mathcal{N}(x_n|Z, \tau_n^2)^{y_n} \mathcal{U}(x_n)^{1-y_n} \tag{2.3}
$$

和

$$
p(y_n | \pi) = \pi^{y_n}(1-\pi)^{1-y_n} \tag{2.4}
$$

其中，当 $y_n = 1$ 的时候表示第 n 个测量值为内点 (inlier) (即，此次测量满足高斯分布) ，反之，当 $y_n = 0$ 的时候表示此次测量值为离群值 (outlier) (即，此次测量值属于均匀分布) 。当这些潜变量被 (2.3) 和 (2.4) 所描述的模型边缘化时，我们回到 (2.1) 的简单混合模型。令 $\mathcal{X} = [x_1, \dotsc, x_n] , \mathcal{Y} = [y_1, \dotsc, y_n]$ ，那么 $\mathcal{X, Y}, Z, \pi$ 的联合分布为：

$$
\begin{align*}
p(\mathcal{X, Y},Z,\pi) &= p(\mathcal{X}|Z, \pi , \mathcal{Y}) p(Z, \pi, \mathcal{Y}) \\
&= p(\mathcal{X}|Z, \pi , \mathcal{Y}) p(\mathcal{Y} |Z, \pi) p(Z,\pi) \\
带入式子(2.2)得 \: &= p(\mathcal{X}|Z, \pi , \mathcal{Y}) p(\mathcal{Y} |Z, \pi) p(Z) p(\pi) \\
由于\mathcal{Y}与Z无关 \: &= p(\mathcal{X}|Z, \pi , \mathcal{Y}) p(\mathcal{Y} | \pi) p(Z) p(\pi) \\
&= \Bigg[ \prod\limits_{n=1}^N p(x_n|Z,\pi,y_n)p(y_n|\pi) \Bigg] p(Z)p(\pi)
\end{align*} \tag{2.5}
$$

由于并不知道要求解的后验估计 $p(\mathcal{Y}, Z, \pi | \mathcal{X})$ 是怎么样的形式，因此令 $q(\mathcal{Y}, Z, \pi)$ 是后验估计 $p(\mathcal{Y}, Z, \pi | \mathcal{X})$ 的一个近似推断，且满足以下因式分解形式：

$$
q(\mathcal{Y}, Z, \pi) = q_{\mathcal{Y}}(\mathcal{Y}) \, q_{Z,\pi}(Z, \pi) \tag{2.6}
$$

由变分 (calculus of variations) 推断理论，求解后验估计 $p(\mathcal{Y}, Z, \pi | \mathcal{X})$ 的最佳近似分布等价于最小化 $q(\mathcal{Y}, Z, \pi)$ 与 $p(\mathcal{X, Y},Z,\pi)$ 的 Kullback-Leibler 散度，由此推出 $q_{\mathcal{Y}}(\mathcal{Y}) , \: q_{Z,\pi}(Z, \pi)$ 需要满足：（这步未仔细研究，读者可以先看参考10[^10] 中的 10.1.1 章节（变分推断之分解分布）、参考 11[^11] ，#TODO）

$$
\ln q_{Z,\pi}(Z, \pi) = E_\mathcal{Y}[\ln p(\mathcal{X, Y}, Z, \pi)] + const \tag{2.7}
$$

和

$$
\ln q_{\mathcal{Y}}(\mathcal{Y}) = E_{Z,\pi}[\ln p(\mathcal{X, Y}, Z, \pi)] + const \tag{2.8}
$$

其中 $E_\mathcal{Y}, \, E_{Z,\pi}$ 分别表示 $q_{\mathcal{Y}}(\mathcal{Y}, \, q_{Z,\pi}(Z, \pi)$ 的期望，这里我们只关心 $Z, \pi$ 的估计，将式 (2.3) (2.4) (2.5) 带入式 (2.7) 中：

$$
\scriptsize {
\begin{align*}
& \small{ \ln q_{Z,\pi}(Z, \pi) } \\
&= E_\mathcal{Y} \Bigg[ \ln \bigg( \Big( \prod_\limits{n=1}^N p(x_n|Z,\pi,y_n) p(y_n|\pi) \Big)p(Z)p(\pi) \bigg) \Bigg] + const \\
&= E_\mathcal{Y} \Bigg[ \ln \Big( \prod_\limits{n=1}^N \mathcal{N}(x_n|Z,\tau_n^2)^{y_n} \mathcal{U}(x_n)^{1-y_n} \pi^{y_n}(1-\pi)^{1-y_n} \Big) + \ln(Z) + \ln(\pi) \Bigg] + const \\
&= E_\mathcal{Y} \Bigg[ \sum_\limits{n=1}^N y_n \ln \mathcal{N}(x_n|Z,\tau_n^2) + \sum_\limits{n=1}^N (1-y_n) \ln \mathcal{U}(x_n) + \sum_\limits{n=1}^N y_n \ln \pi + \sum_\limits{n=1}^N (1-y_n)\ln(1-\pi) \Bigg] \\
& \quad + \ln(Z) + \ln(\pi) + const \\
&= \sum_{n=1}^N E_\mathcal{Y}[y_n] \big(\ln \mathcal{N}(x_n|Z,\tau_n^2) + \ln \pi \big) + \sum_{n=1}^N E_\mathcal{Y}[1-y_n] \big(\ln \mathcal{U}(x_n) + \ln(1-\pi) \big) \\ 
& \quad + \ln(Z) + \ln(\pi) + const
\end{align*}
\tag{2.9}
}
$$

对上式取两边取指数 (exponentiating) 得：

$$
q_{Z,\pi}(Z, \pi) = \Bigg[\prod_{n=1}^N \mathcal{N}(x_n|Z,\tau_n^2)^{r_n} \pi^S (1-\pi)^{N-S} p(Z) p(\pi) \Bigg] \tag{2.10} \\
其中, \quad r_n = E_\mathcal{Y}[y_n], \, S = \sum_{n=1}^N r_n
$$

如果我们为 Z 和 π 选择共轭先验，就可以证明的近似分布式 (10) 具有 Gaussian×Beta 形式。因此给出了一个近似真实后验的概密度函数函数：

$$
q(Z,\pi|a,b,\mu,\sigma^2) \doteq \mathcal{N}(Z|\mu, \sigma^2)Beta(\pi|a,b) \tag{2.11}
$$

这里的 $\mathcal{N}(Z|\mu, \sigma_n^2)$ 是高斯分布，以及[**贝塔分布**](https://zh.wikipedia.org/wiki/%CE%92%E5%88%86%E5%B8%83) $Beta(\pi|a,b)$ ：

$$
Beta(\pi|a,b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \pi^{a-1}(1-\pi)^{b-1} \tag{2.12}
$$

其中 $\Gamma(\cdot)$ 是 [**Gamma 函数**](https://zh.wikipedia.org/wiki/%CE%93%E5%87%BD%E6%95%B0)，Gamma 函数具有 $\Gamma(x+1) = x\,\Gamma(x)$ 的递归形式。并且给出了该模型的迭代式：

$$
q(Z,\pi|a',b',\mu',\sigma') \approx p(x|Z,\pi) q(Z,\pi|a,b,\mu,\sigma^2) \tag{2.13}
$$

其中，这里的 $(\cdot)'$ 表示 n 时刻的参数， $(\cdot)$ 表示 n-1 时刻的参数。由迭代式可以看出新时刻的参数是通过上一时刻的参数和新的深度测量值来更新的。{==之所以是**约等于**是因为上式的右端形式并不是 Gaussian×Beta 分布，而是用 $q(Z,\pi|a',b',\mu',\sigma')$ 关于 $Z$ 和 $\pi$ 的一阶矩和二阶矩相等去近似右端项，且两端的全概率等于 1，进而更新参数（这里类似于 EKF 中的更新，通常一个高斯分布通过非线性变换后分布并非高斯，但是通常利用高斯分布的一阶矩和二阶矩去近似这个变换结果）==}。将式 (2.1) (2.11) 代入上式的右端得：

$$
\big(\pi \mathcal{N}(x | Z, \tau^2) + (1-\pi) \mathcal{U}(x) \big) \mathcal{N}(Z|\mu, \sigma^2)Beta(\pi|a,b) \tag{2.14}
$$

由于要求关于 $Z$ 和 $\pi$ 的一阶矩和二阶矩，因此需要构造的关于 $Z$ 和 $\pi$ 为变量的式子，因此将上式变换如下 ({==这一步骤的思路是让式 (2.14) 转变成便于求关于 $Z$ 和 $\pi$ 的一阶矩和二阶矩的形式==}) :

$$
\fbox{$
\scriptsize{
  \begin{align*}
    & \quad \big(\pi \mathcal{N}(x | Z, \tau^2) + (1-\pi) \mathcal{U}(x) \big) \mathcal{N}(Z|\mu, \sigma^2)Beta(\pi|a,b) \\
    &= \pi \mathcal{N}(x | Z, \tau^2) \mathcal{N}(Z|\mu, \sigma^2)Beta(\pi|a,b) + (1-\pi) \mathcal{U}(x) \mathcal{N}(Z|\mu, \sigma^2) Beta(\pi|a,b) \\
    & 根据 \Gamma(x+1) = x\,\Gamma(x) 的性质有：\\
    & \quad \quad Beta(\pi|a,b) = \frac{1}{\pi} \frac{a}{a+b} Beta(\pi|a+1,b) = \frac{1}{1-\pi} \frac{b}{a+b} Beta(\pi|a,b+1) \\
    &= \underbrace{\frac{a}{a+b} \mathcal{N}(x | Z, \tau^2) \mathcal{N}(Z|\mu, \sigma^2) Beta(\pi|a+1,b)}_{1} + \underbrace{\frac{b}{a+b} \mathcal{U}(x) \mathcal{N}(Z|\mu, \sigma^2) Beta(\pi|a,b+1)}_{2} \\
    & 我们发现上式中的第1项中存在以 Z 或者 \pi 为参数的项 \mathcal{N}(x | Z, \tau^2)，\\
    & 因此将上式第1项中高斯分布乘积中的Z与其他参数分离，进行变形 \\
    & 为了方便只取 \mathcal{N}(x | Z, \tau^2) \mathcal{N}(Z|\mu, \sigma^2) 的指数项做推导: \\
    & \fbox{$ \small{ \begin{align*}
        & \overbrace{-\frac{(x-Z)^2}{2 \tau^2}}^{\mathcal{N}(x | Z, \tau^2)} \overbrace{-\frac{(Z-\mu)^2}{2\sigma^2}}^{\mathcal{N}(Z|\mu, \sigma^2)} \\
        &= -\frac{(\sigma^2+\tau^2)Z^2 - 2(x \sigma^2 + \mu \tau^2)Z + \sigma^2 x^2 + \mu^2\tau^2}{2\tau^2\sigma^2} \\
        &= -\frac{Z^2-2(\frac{x\sigma^2+\mu\tau^2}{\sigma^2+\tau^2})Z + (\frac{x\sigma^2+\mu\tau^2}{\sigma^2+\tau^2})^2 - (\frac{x\sigma^2+\mu\tau^2}{\sigma^2+\tau^2})^2 + \frac{\sigma^2x^2+\tau^2\mu^2}{\sigma^2+\tau^2}}{2\frac{\tau^2\sigma^2}{\sigma^2+\tau^2}} \\
        &=-\frac{Z^2-2(\frac{x\sigma^2+\mu\tau^2}{\sigma^2+\tau^2})Z + (\frac{x\sigma^2+\mu\tau^2}{\sigma^2+\tau^2})^2}{2\frac{\tau^2\sigma^2}{\sigma^2+\tau^2}}-\frac{- (\frac{x\sigma^2+\mu\tau^2}{\sigma^2+\tau^2})^2 + \frac{\sigma^2x^2+\tau^2\mu^2}{\sigma^2+\tau^2}}{2\frac{\tau^2\sigma^2}{\sigma^2+\tau^2}} \\
        &=\underbrace{-\frac{(Z-\frac{x\sigma^2+\mu\tau^2}{\sigma^2+\tau^2})^2}{2\frac{\tau^2\sigma^2}{\sigma^2+\tau^2}}}_{\mathcal{N}\big(Z \big| \color{red}{ \frac{x\sigma^2+\mu\tau^2}{\sigma^2+\tau^2}},\color{blue}{ \frac{\tau^2\sigma^2}{\sigma^2+\tau^2}}\big)} \underbrace{-\frac{(x-\mu)^2}{2(\sigma^2+\tau^2)}}_{\mathcal{N}(x|\mu, \sigma^2+\tau^2)}
      \end{align*}}$}
  \end{align*}
} $}
$$

根据上面的推导我们可以将式 (2.14) 转变成：

$$
\small{\begin{align*}\frac{a}{a+b} \mathcal{N}(x | u, \sigma^2 + \tau^2) \mathcal{N}(Z| \color{red}{m}, \color{blue}{s^2}) Beta(\pi|a+1,b) \\ + \frac{b}{a+b} \mathcal{U}(x) \mathcal{N}(Z|\mu, \sigma^2) Beta(\pi|a,b+1)\end{align*}} \tag{2.15}
$$

其中有：

$$
s^2 = \frac{\tau^2\sigma^2}{\sigma^2+\tau^2} \to \frac{1}{s^2} = \frac{1}{\sigma^2} + \frac{1}{\tau^2} \tag{2.16}
$$

和

$$
m = \frac{x\sigma^2+\mu\tau^2}{\sigma^2+\tau^2} \to m = s^2 (\frac{\mu}{\sigma^2} + \frac{x}{\tau^2}) \tag{2.17}
$$

式 (2.15) 令

$$
C_1 = \frac{a}{a+b} \mathcal{N}(x | u, \sigma^2 + \tau^2), \quad C_2 = \frac{b}{a+b} \mathcal{U}(x) \tag{2.18}
$$

易知 $C_1, C_2$ 是与 $Z, \pi$ 无关的系数，因此式 (2.15) 变为

$$
\fbox{$
\scriptsize{ \begin{align*}
& C_1 \mathcal{N}(Z| m, s^2) Beta(\pi|a+1,b)  + C_2 \mathcal{N}(Z|\mu, \sigma^2) Beta(\pi|a,b+1) \\
& 上式的全概率为 \\
& \int C_1 \mathcal{N}(Z| m, s^2) Beta(\pi|a+1,b)  + C_2 \mathcal{N}(Z|\mu, \sigma^2) Beta(\pi|a,b+1) dZ\, d\pi = C_1 +C_2 \\
& 由于全概率需要为1，因此转变成 \\
& \frac{C_1}{C_1 + C_2} \mathcal{N}(Z| m, s^2) Beta(\pi|a+1,b)  + \frac{C_2}{C_1 + C_2} \mathcal{N}(Z|\mu, \sigma^2) Beta(\pi|a,b+1)
\end{align*}
} $}
$$

我们令

$$
C'_1 = \frac{C_1}{C_1 + C_2}, \quad C'_2 = \frac{C_2}{C_1 + C_2} \tag{2.19}
$$

由此归一化后的概率密度函数形式为：

$$
\small{ C'_1 \mathcal{N}(Z| m, s^2) Beta(\pi|a+1,b)  + C'_2 \mathcal{N}(Z|\mu, \sigma^2) Beta(\pi|a,b+1) } \tag{A}
$$

分别对上式 (A) 和 $q(Z,\pi|a',b',\mu',\sigma')$ 求关于 $Z$ 和 $\pi$ 的一阶矩和二阶矩：

$$
\fbox{$
\scriptsize{\begin{align*}
  & 先对 q(Z,\pi|a',b',\mu',\sigma') =\mathcal{N}(Z|\mu', \sigma'^2)Beta(\pi|a',b') 求关于 Z 的一阶矩和二阶矩: \\
  & E_Z = \int Z \, \mathcal{N}(Z|\mu',\sigma'^2)Beta(\pi|a',b') dZ \\
  & \quad = \int Z \, \mathcal{N}(Z|\mu',\sigma'^2) dZ = E[\mathcal{N}(Z|\mu',\sigma'^2)] = \mu' \\
  & D_Z = \int Z^2 \, \mathcal{N}(Z|\mu',\sigma'^2)Beta(\pi|a',b') dZ \\
  & \quad = \underbrace{\int (Z-u')^2 \, \mathcal{N}(Z|\mu',\sigma'^2) dZ}_{\small{D[\mathcal{N}(Z|\mu',\sigma'^2)]=\sigma'^2}} + 2\mu' \underbrace{\int Z \, \mathcal{N}(Z|\mu',\sigma'^2) dZ}_{\small{E[\mathcal{N}(Z|\mu',\sigma'^2)]=\mu'}} - \mu^2 \underbrace{\int \mathcal{N}(Z|\mu',\sigma'^2) dZ}_{\small{=1}}  \\
  & \qquad = \sigma'^2 + \mu'^2 \\ \\
  & 对 q(Z,\pi|a',b',\mu',\sigma') 求关于 \pi 的一阶矩和二阶矩: \\
  & E_\pi = \int \pi \, \mathcal{N}(Z|\mu',\sigma'^2)Beta(\pi|a',b') d\pi \\
  & \quad\! = \underbrace{\int \pi \, Beta(\pi|a',b') d\pi}_{\small{E[Beta(\pi|a',b')] = \frac{a'}{a'+b'}}} = \frac{a'}{a'+b'} \underbrace{\int Beta(\pi|a'+1,b') d\pi}_{\small{=1}} = \frac{a'}{a'+b'} \\
  & D_\pi = \int \pi^2 \, \mathcal{N}(Z|\mu',\sigma'^2)Beta(\pi|a',b') d\pi \\
  & \quad\! = \int \pi^2 \, Beta(\pi|a',b') d\pi = \frac{a'}{a'+b'} \frac{a'+1}{a'+b'+1} \underbrace{\int Beta(\pi|a'+1+1,b') d\pi}_{\small{=1}} \\
  & 或 \; = \underbrace{\int (\pi-E_\pi)^2 \, Beta(\pi|a',b') d\pi}_{\small{D[Beta(\pi|a',b')] = \frac{a'b'}{(a'+b')^2(a'+b'+1)}}} - \underbrace{ 2E_\pi \int \pi \, Beta(\pi|a',b') d\pi}_{\small{2 E^2_\pi = 2\frac{a'^2}{(a'+b')^2}}} + \underbrace{E_\pi^2 \int Beta(\pi|a',b') d\pi}_{\small{E_\pi^2\cdot1 = \frac{a'^2}{(a'+b')^2}}} \\
  & \quad\! = \frac{a'(a'+1)}{(a'+b')(a'+b'+1)} \\ \\ \\
  & 根据上面求解的方式对 式(A) 求关于 Z 和 \pi 的一阶矩和二阶矩有: \\
  & E'_Z = C'_1 m + C'_2\mu \quad D'_Z = C'_1(m^2+s^2) + C'_2(\mu^2+\sigma^2) \\ \\
  & E'_\pi = C'_1\frac{a+1}{a+b+1} + C'_2\frac{a}{a+b+1} \quad D'_\pi = C'_1\frac{(a+1)(a+2)}{(a+b+1)(a+b+2)} + C'_2\frac{a(a+1)}{(a+b+1)(a+b+2)}
\end{align*}}
$}
$$

为了使迭代式 (式 2.13) 成立，需要令式子两端关于 $Z$ 和 $\pi$ 的一阶矩和二阶矩相等，所以有：

令 $E_Z = E'_Z$ 得：

$$
\mu' = C'_1 m + C'_2\mu \tag{2.20}
$$

令 $D_Z = D'_Z$ 得：

$$
\sigma'^2+\mu'^2 = C'_1(m^2+s^2) + C'_2(\mu^2+\sigma^2) \tag{2.21}
$$

令 $E_\pi = E'_\pi$ 得：

$$
\frac{a'}{a'+b'} = C'_1\frac{a+1}{a+b+1} + C'_2\frac{a}{a+b+1} \tag{2.22}
$$

令 $D_\pi = D'_\pi$ 得：

$$
\begin{align*}
  \frac{a'(a'+1)}{(a'+b')(a'+b'+1)} &= C'_1\frac{(a+1)(a+2)}{(a+b+1)(a+b+2)} \\
  &\quad + C'_2\frac{a(a+1)}{(a+b+1)(a+b+2)} 
\end{align*} \tag{2.23}
$$

由上面四个等式就可以通过 n-1 时刻的参数 $a, b, \mu, \sigma$ 求解 n 时刻的参数 $a', b', \mu', \sigma'$ 了。因此在加入新的测量时，通过迭代式 (2.14) 近似后验概率分布会得到更新，当 $\sigma$ 小于给定阈值时，认为估计的深度值已经收敛，计算对应特征点的三维坐标，并加入地图。



## 总结

SVO 相对于目前其他开源的方案，它的最大优势是速度极快。这是因为它使用了稀疏的直接法，既不用费力的去计算描述子用于特征匹配，也不必像稠密或半稠密法那么处理大量的信息，因此 SVO 在低端的计算平台上也可以打到实时性。但也由于 SVO 针对目标应用平台选为了无人机的俯视相机，围绕该特点应用场景设计也使得 SVO 的局限性比较大，并且为了计算速度，直接舍弃了后端优化月回环检查，也基本没有重定位能力。因此我们可以称它仅仅是个里程计，而非完整的 SLAM 。



关于代码中的其他细节，请观看下一篇 「 SVO 代码解析」。









## 参考

[^1]: Forster, Christian, Matia Pizzoli, and Davide Scaramuzza. "[SVO: Fast semi-direct monocular visual odometry.](http://rpg.ifi.uzh.ch/docs/ICRA14_Forster.pdf)" 2014 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2014.

[^2]: S. Baker and I. Matthews, “[Lucas-Kanade 20 Years On: A Unifying Framework](https://www.cs.cmu.edu/afs/cs/academic/class/15385-s12/www/lec_slides/Baker&Matthews.pdf): Part 1,” International Journal of Computer Vision, vol. 56, no. 3, pp. 221–255, 2002.

[^3]: [白巧克力亦唯心 - svo : semi-direct visual odometry 论文解析](https://blog.csdn.net/heyijia0327/article/details/51083398) 

[^4]: [Monocular slam 中的理论基础 (2) ](https://blog.csdn.net/heyijia0327/article/details/50774104) 

[^5]: [REMODE: Probabilistic, Monocular Dense Reconstruction in Real Time](https://files.ifi.uzh.ch/rpg/website/rpg.ifi.uzh.ch/html/docs/ICRA14_Pizzoli.pdf) 

[^6]: 《视觉 SLAM 十四讲》高翔等著：13.2.3 小节
[^7]:  [G. Vogiatzis](http://www.george-vogiatzis.org/) and C. Hern´ andez, “[Video-based, Real-Time Multi View Stereo](http://www.george-vogiatzis.org/publications/ivcj2010.pdf),” Image and Vision Computing, vol. 29, no. 7, 2011.  [Supplementary matterial](http://www.george-vogiatzis.org/publications/ivcj2010supp.pdf) 

[^8]: [路游侠 - SVO 解析](http://www.cnblogs.com/luyb/p/5773691.html) 
[^9]: [可爱的小蚂蚁 - svo 的 Supplementary material 推导过程](https://blog.csdn.net/u013004597/article/details/52069741)

[^10]: Christopher M. Bishop. Pattern Recognition and Machine Learning (Information Science and Statistics). Springer, 1 edition, October 2007.
[^11]: [参考 10 的翻译版本](https://mqshen.gitbooks.io/prml/content/Chapter10/variational/factorized_distributions.html)，膜拜大神



--8<--
mathjax.txt
--8<--