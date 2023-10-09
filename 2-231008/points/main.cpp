#include <fstream>
#include <iostream>
#include <set>
#include <stdio.h>
#include <stdlib.h>

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "opencv2/opencv.hpp"

#define MATRIX(N, M) Eigen::Matrix<double, N, M>

// 姿态四元数
Eigen::Quaterniond POSE_QUATERNIOND(-0.5, 0.5, 0.5, -0.5);

// 四元数转为旋转矩阵
MATRIX(3, 3) ROTATION_MAT = POSE_QUATERNIOND.matrix().transpose();

// 内参矩阵
MATRIX(3, 4) INTRINSTICS_MAT;

// 平移矩阵
MATRIX(3, 1) CAMERA_POS;
MATRIX(3, 1) TRANSLATION_MAT;

// 外参矩阵
MATRIX(3, 4) EXTRINSTICS_MAT;

// 世界坐标系的点
std::vector<MATRIX(3, 1)> WORLD_POINTS;

// 输出图像大小
const int WIDTH = 1200, HEIGHT = 800;

// 将世界坐标系下的点转换到图像坐标系下
MATRIX(2, 1)
world_to_pixel(MATRIX(3, 1) world_pos, MATRIX(3, 4) extrinstics_mat, MATRIX(3, 4) intrinstics_mat) {
    // step 1: 先转换到相机坐标系下
    MATRIX(4, 1) world_ex;
    world_ex << world_pos, 1.;
    MATRIX(3, 1) camera_pos = extrinstics_mat * world_ex;

    MATRIX(4, 1) camera_ex;
    camera_ex << camera_pos, 1.;

    // step 2: 相机坐标系转化到图像坐标系, 不考虑畸变
    // [x, y, 1].T = F * [Xc, Yc, Zc, 1].T / Zc
    MATRIX(3, 1) pixel_ex = intrinstics_mat * camera_ex;
    pixel_ex /= pixel_ex[2];

    return pixel_ex.block(0, 0, 2, 1);
}

cv::Mat image;
int camera_x = 1200, camera_y = 1200, camera_z = 1200;
void draw(int, void*) {
    // 重新计算外参矩阵
    double new_cam_x = (double)(camera_x) / 100. - 10.;
    double new_cam_y = (double)(camera_y) / 100. - 10.;
    double new_cam_z = (double)(camera_z) / 100. - 10.;
    CAMERA_POS << (double)new_cam_x, (double)new_cam_y, (double)new_cam_z;
    TRANSLATION_MAT = -ROTATION_MAT * CAMERA_POS;
    EXTRINSTICS_MAT << ROTATION_MAT, TRANSLATION_MAT;

    image = cv::Mat::zeros({ WIDTH, HEIGHT }, CV_8UC3);

    for (auto&& world_pos: WORLD_POINTS) {
        MATRIX(2, 1) pixel_pos = world_to_pixel(world_pos, EXTRINSTICS_MAT, INTRINSTICS_MAT);

        if (pixel_pos[0] >= 0 && pixel_pos[0] <= WIDTH && pixel_pos[1] >= 0
            && pixel_pos[1] <= HEIGHT)
        {
            cv::circle(
                image,
                cv::Point((int)(pixel_pos[0]), (int)(pixel_pos[1])),
                1,
                cv::Scalar(0, 255, 0)
            );
        }
    }

    cv::imshow("image", image);
    cv::imwrite("points.jpg", image);
}

int main() {
    // 初始化
    INTRINSTICS_MAT << 400., 0., 190., 0., 0., 400., 160., 0., 0., 0., 1., 0.;

    // 读入数据
    std::ifstream ifile("points.txt");
    if (!ifile) {
        std::cerr << "points.txt not found." << std::endl;
        return 1;
    }

    int n;
    double xw, yw, zw;
    ifile >> n;
    for (int i = 0; i < n; ++i) {
        ifile >> xw >> yw >> zw;
        MATRIX(3, 1) world_pos;
        world_pos << xw, yw, zw;
        WORLD_POINTS.push_back(world_pos);
    }

    ifile.close();

    cv::namedWindow("image");

    cv::createTrackbar("x", "image", &camera_x, 2000, draw);
    cv::createTrackbar("y", "image", &camera_y, 2000, draw);
    cv::createTrackbar("z", "image", &camera_z, 2000, draw);

    draw(0, 0);
    cv::waitKey(0);

    return 0;
}