### 前言

随着 SLAM 技术在市场中的应用越来越广泛，越来越多的学者投身于 SLAM 的研究中。国内外的诸多大学相关实验室发布了许多研究成果，其中不少成果进行开源，并发布了许多相关的数据集用于算法的验证。在此，借用网络上大神 [^1] [^2] [^3] 对各大学实验室的介绍引出研究成果进行归类。后续的篇幅主要是针对目前主要流行的的开源项目进行梳理。个人能力有限，如果哪些讲解和推导有问题劳请大神指出。

- [香港科技大学的 Aerial Robotics Group](http://uav.ust.hk/ "http://uav.ust.hk/")
    - [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) ：一种鲁棒且通用的实时单目视觉惯性状态估计框架
    - [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)：一种基于优化的多传感器状态框架，可实现自主应用（无人机，汽车和  AR / VR）的精确自定位。VINS-Fusion 是 VINS-Mono 的扩展，支持多种视觉惯性传感器类型（单声道摄像机 + IMU，立体摄像机 + IMU，甚至仅限立体声摄像机
- [浙江大学CAD＆CG国家重点实验室的CVG（Computer Vision Group）](http://www.zjucvg.net/)
    - RKSLAM：用于AR的基于关键帧的鲁棒单目SLAM系统
    - LS-ACTS：大型自动相机跟踪系统，可以处理大型视频/序列数据集 https://github.com/zju3dv/ENFT ，https://github.com/zju3dv/SegmentBA ， https://github.com/zju3dv/ENFT-SfM
    - ACTS：自动相机跟踪系统
    - RDSLAM：是一个实时同步定位和建图系统，它允许场景的一部分是动态的或整个场景逐渐变化。与PTAM相比，RDSLAM不仅可以在动态环境中稳健地工作，而且还可以处理更大规模的场景（重建的3D点的数量可以是数万个）
- [清华大学自动化系宽带网络与数字媒体实验室 BBNC](http://media.au.tsinghua.edu.cn/index/index/index)
- [中科院自动化研究所国家模式识别实验室 Robot Vision Group](http://vision.ia.ac.cn/)
- [英国伦敦大学帝国理工学院 Dyson 机器人实验室](http://www.imperial.ac.uk/dyson-robotics-lab)
    - [ElasticFusion](https://bitbucket.org/dysonroboticslab/elasticfusionpublic/src/master/)：一个实时的稠密的视觉 SLAM 系统，可以利用 RGB-D 相机来对房间进行全局一致的三维稠密重建
    - CodeSLAM：一种生成室内场景轨迹的大规模照片级真实渲染的系统
    - [SceneNet RGB-D](https://bitbucket.org/dysonroboticslab/scenenetrgb-d/src/master/)：一种生成室内场景轨迹的大规模照片级真实渲染的系统，数据集地址：https://robotvault.bitbucket.io/scenenet-rgbd.html
    - [SemanticFusion](https://bitbucket.org/dysonroboticslab/semanticfusion/src/master/)：一种实时可视 SLAM 系统，能够使用卷积神经网络在语义上注释密集的 3D 场景
- [英国牛津大学 Active Vision Laboratory](http://www.robots.ox.ac.uk/ActiveVision/index.html)
    - [PTAM](https://github.com/Oxford-PTAM/PTAM-GPL)：（并行跟踪和建图）用于增强现实的相机跟踪系统
- [英国牛津大学 Torr Vision Group](http://www.robots.ox.ac.uk/~tvg/)
    - [交互式实时 3D 场景分割的框架](https://github.com/torrvision/spaint/tree/collaborative)：创建了一个只需使用廉价的硬件，就可以在半小时内捕获并重建整个房屋或实验室的建图系统
- [苏黎世联邦理工学院 Autonomous System Lab](https://asl.ethz.ch/)
    - [libpointmatcher](https://github.com/ethz-asl/libpointmatcher)：一个模块化库，它实现了迭代最近点（ICP）算法，用于配准点云
    - [libnabo](https://github.com/ethz-asl/libnabo)：用于低维空间的快速K最近邻库
    - [ethzasl_sensor_fusion](https://github.com/ethz-asl/ethzasl_sensor_fusion)：基于EKF的时延补偿单传感器和多传感器融合框架
    - [ethzasl_ptam](https://github.com/ethz-asl/ethzasl_ptam)：用于单目SLAM的框架PTAM
- [苏黎世 Robotics and Perception Group](http://rpg.ifi.uzh.ch/)
    - [视觉（惯性）里程计轨迹定量评估方法](https://github.com/uzh-rpg/rpg_trajectory_evaluation)：通过视觉（ 惯性）里程计（VO / VIO）定量评估估计轨迹的质量
    - [基于高效数据的分布式视觉SLAM](https://github.com/uzh-rpg/dslam_open)：该算法可实现使用便宜，轻便和多功能的相机进行分布式通信多机器人建图
    - [Kalibr](https://github.com/ethz-asl/kalibr)：相机惯导标定
- [慕尼黑工业大学 The Computer Vision Group](https://vision.in.tum.de/research)
    - [dvo_slam](https://github.com/tum-vision/dvo_slam)：提供了来自连续图像的RGB-D相机的刚体运动估计的实现方案
    - [LSD-SLAM](https://github.com/tum-vision/lsd_slam): Large-Scale Direct Monocular SLAM：一种直接单目SLAM建图技术
    - [DSO](https://github.com/JakobEngel/dso): Direct Sparse Odometry：一种用于视觉里程计的新的直接稀疏建图方法
    - [Basalt](https://gitlab.com/VladyslavUsenko/basalt): Visual-Inertial Mapping with Non-Linear Factor Recovery：使用非线性因子恢复法从视觉 - 惯性里程计提取信息来进行视觉 - 惯性建图
- [德国弗莱堡大学 Autonomous Intelligent Systems](http://ais.informatik.uni-freiburg.de/index_en.php)
    - [GMapping](https://github.com/OpenSLAM-org/openslam_gmapping) : 基于Bpf粒子滤波算法的滤波SLAM框架
    - [RGBD SLAM2](https://github.com/felixendres/rgbdslam_v2)：是一个非常全面优秀的系统，将SLAM领域的图像特征、优化、闭环检测、点云、octomap等技术融为一体，非常适合RGBD SLAM初学者，也可以在其基础上继续开发
- [西班牙萨拉戈萨大学RoPeRT机器人，感知和实时组SLAM实验室](http://robots.unizar.es/slamlab/)
- [明尼苏达大学 Multiple Autonomous Robotic Systems Laboratory（MARS）](http://mars.cs.umn.edu/)
- [卡内基梅隆大学 Robot Perception Lab](http://rpl.ri.cmu.edu/)
    - [isam](https://github.com/ori-drs/isam)：增量平滑和建图（iSAM），这是一种基于快速增量矩阵分解的同时定位和建图问题方法，通过更新自然稀疏平滑信息矩阵的QR分解来实现
- [斯坦福大学人工智能实验室自动驾驶团队](http://driving.stanford.edu/)
- [麻省理工大学计算机科学与人工智能实验室（CSAIL）海洋机器人组](https://marinerobotics.mit.edu/)
- [宾夕法尼亚大学机械工程与应用力学系Vijay Kumar实验室](https://www.kumarrobotics.org/)
- [华盛顿大学 UW Robotics and State Estimation Lab](http://rse-lab.cs.washington.edu/)
- [加拿大谢布鲁克大学 IntRoLab](https://introlab.3it.usherbrooke.ca/mediawiki-introlab/index.php/Main_Page)



### 开源项目列表





### 目录





### 参考

[^1]: https://zhuanlan.zhihu.com/p/70066976 
[^2]: https://blog.csdn.net/mulinb/article/details/53421864 
[^3]: http://www.computervisionblog.com/2016/01/why-slam-matters-future-of-real-time.html 