### Animation

#### 思路

大体上用第二题的数据，然后对点的世界坐标进行随机变换和移动相机位置和朝向来实现拍摄

我全部采用的最简单的线性动画实现，简单封装了一个类，并手写了一个多线程处理器，加速生成动画

#### 效果

见视频



Update 2023-10-15

1. 优化计算相机坐标系的点的算法（合成大矩阵再一次性矩阵乘法），减小了移动滑动条的延迟

2. 优化计算动画功能：可以按照关键帧来生成

3. 添加了优化编译选项 `-O3 -Wall
