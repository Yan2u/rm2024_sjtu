### Apple

#### 效果：

![result](./build/apple_res.png)

#### 思路

一开始是想像笔试题那样直接转 hsv 然后 `inRange` 筛选红色，发现底部和顶部的树枝不好排除（实在不会调参。。。），最后是先用的通道相减（`G -> R - G`），然后再用 `inRage` 筛选的；最后一个简单的 `findContours` 和取最大面积得到目标轮廓；但是遗憾的是还是没能排除树枝的干扰。。
