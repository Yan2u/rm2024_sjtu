### Car Plate

#### 效果：

![result](./build/result/1.jpg)

#### 思路

转 hsv 用颜色筛选二值化，然后用形态学闭运算修补轮廓缝隙，然后根据轮廓特点筛选

先定一个基准蓝色范围，然后用滑动条调参，最后只剩第 5 张图片蓝色路牌有一点干扰；考虑到车牌形状，先求一个最小矩形，再拟合一个多边形，比较两者面积差距，取小的，再结合基础的面积大小筛选（代码没写）