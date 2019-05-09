## SVO 运行实践

SVO 的下载编译可以参照 SVO 作者写的教程：https://github.com/uzh-rpg/rpg_svo/wiki 。如果手上没有摄像头的话，作者提供了数据集，可以参照文档先跑跑试试。

下面给出利用 [**MYNT-EYE(S)**](https://mynt-eye-s-sdk.readthedocs.io/zh_CN/latest/index.html) 摄像头运行 SVO 程序的简单实践过程。作者提供的数据集选用的是 ATAN 相机模型，然而 MYNT-EYE 选用 Pinhole 模型。利用 MYNT-EYE 官方提供的 SDK 运行摄像头，关于摄像头标定这部分这里就不展开。根据 live.launch 我们知道需要提供摄像头的 Mono 图像数据，以及该摄像头的标定参数。#TODO



## 初始化

在介绍初始化前，先简单介绍一下关于 `SVO_ROS` 的节点函数。当运行 `roslaunch` 调用 svo 之后，来到 `rpg_svo/svo_ros/ src/vo_node.cpp` 文件下运行 main 函数。

1. 初始化 `ROS` ，接着创建一个节点句柄和节点 `VoNode` ，在创建 `VoNode` 的构造函数中开辟了一个线程用于监听控制台输入，然后加载摄像头参数，并初始化可视化的初始位姿，最后创建视觉里程计，并完成一系列初始化操作
2. 订阅摄像头消息，每当获取到更新图像信息后回调 `svo::VoNode::imgCb` 函数，进入循环之后，接下来的所有工作都会在这个函数内完成
3. 订阅远程输入消息（应该指的就是键盘输入）

紧接着我们还是进入正题。在节点创建的过程中，程序创建了一个里程计算法入口 `svo::FrameHandlerMono` 类的变量 ，在整个构造函数运行过程中，以及并调用了 initialize 函数来完成一系列算法部件的初始化。其中有几个比较重要的过程：

- 进行重投影的初始化，由 `Reprojector` (定义在 `reprojector.cpp` 中) 构造函数以及 `initialize` 函数完成，`grid_` 为 `Grid` 类变量，`Grid` 中定义了 `CandidateGrid` 型变量 `cells` ，而 `CandidateGrid` 是一个 `Candidate` 型的list（双向链表）组成的 vector（向量）向量。`grid_.cells.resize` 是设置了 `cells` 的大小，即将图像划分成多少个格子。然后通过 `for_each` 函数对 `cells` 每个链表（即图像每个格子）申请一块内存。之后通过 `for` 函数给每个格子编号，最后调用 `random_shuffle` 函数将格子的编号顺序打乱。
- 特征检测初始化，创建一个 `FastDetector` 类变量，该类继承于 `AbstractDetector` 类，初始化特征检测格子大小及需要多少行多少列格子，设置特征提取的金字塔层数，和每个格子是否已经存在特征点的 `std::vector<bool> grid_occupancy_` 变量。
- 通过 `DepthFilter`（深度滤波器）构造函数完成初始化，设置了特征检测器指针、回调函数指针、线程、新关键帧深度的初值，并启动深度滤波器线程。其中这里的回调函数指针需要注意，它的本意为：`depth_filter_cb(_1, _2) => (&map_.point_candidates_)->newCandidatePoint(_1, _2)`



### 第一帧

图像是通过 `FrameHandlerMono::addImage` 函数加载进里程计算法中的。在每一新建 `Frame` 类指针变量时，对图像进行提取图像金字塔，默认 5 层、尺度因子为 2 。接着进入 `FrameHandlerMono::processFirstFrame` 函数处理第一帧图像。

1. 创建一个位姿变换矩阵赋给第一帧的 `T_f_w_` (表示从世界坐标到相机坐标的变换矩阵) 
2. 创建一个特征检测器，默认设置特征提取的金字塔为 0~3 层，栅格大小为 30 。调用特征检测函数提取当前帧中的 Fast 特征角点。为了适应多尺度变化，对图像金字塔的多层图像提取 Fast 角点，紧接通过小窗口做非最大值抑制。最后对同一层图像坐落在同一网格中的 Fast 角点求 shiTomasi 得分，取得分最高的强角点。				

