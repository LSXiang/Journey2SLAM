## 简介

加速鲁棒特征 (Speed Up Robust Feature, SURF) [^1][^2] 和 SIFT 特征类似，同样是一个用于检测、描述、匹配图像局部特征点的特征描述子。SIFT 是被广泛应用的特征点提取算法，但其实时性较差，如果不借助于硬件的加速和专用图形处理器 (GPUs) 的配合，很难达到实时的要求。对于一些实时应用场景，如基于特征点匹配的实时目标跟踪系统，每秒要处理数十帧的图像，需要在毫秒级完成特征点的搜索定位、特征向量的生成、特征向量的匹配以及目标锁定等工作，SIFT 特征很难满足这种需求。SURF 借鉴了 SIFT 中近似简化 (DoG 近似替代 LoG) 的思想，将 Hessian 矩阵的高斯二阶微分模板简化，使得模板对图像的滤波只需要进行几次简单的加减法运算，并且这种运算与滤波模板的尺寸无关。SURF 相当于 SIFT 的加速改进版本，在特征点检测取得相似性能的条件下，提高了运算速度。整体来说，SURF 比 SIFT 在运算速度上要快数倍，综合性能更优。



## 参考

[^1]: [SURF: Speeded Up Robust Features](http://www.vision.ee.ethz.ch/~surf/eccv06.pdf) 
[^2]: [The Website of SURF: Speeded Up Robust Features](http://www.vision.ee.ethz.ch/~surf/index.html) 