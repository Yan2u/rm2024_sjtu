#include <fstream>
#include <iostream>
#include <set>
#include <stdio.h>
#include <stdlib.h>

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "opencv2/opencv.hpp"

// 长边和宽边上内角点的个数
const int CORNERS_W = 9;
const int CORNERS_H = 6;

// 亚像素精化时的 region_size
const cv::Size REGION_SIZE = { 5, 5 };

// 每个方格的尺寸
const cv::Size SQUARE_SIZE = { 10, 10 };

// 记录成功检测出的图片的角点坐标
std::vector<std::vector<cv::Point2f>> corner_points;

// 图片的尺寸
cv::Size image_size;

void find_corners(const std::string& filename) {
    cv::Mat src = cv::imread(filename);
    assert(src.channels() == 3);

    image_size.width = src.cols;
    image_size.height = src.rows;

    // 找寻角点
    std::vector<cv::Point2f> corners;
    if (cv::findChessboardCorners(src, { CORNERS_W, CORNERS_H }, corners)) {
        if (corners.size() != CORNERS_H * CORNERS_W) {
            std::cout << filename << ": not enough corners detected (" << corners.size() << " / "
                      << CORNERS_H * CORNERS_W << ")\n";
            return;
        }

        // 转灰度图, 亚像素精化
        cv::Mat src_gray;
        cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
        cv::find4QuadCornerSubpix(src_gray, corners, REGION_SIZE);

        corner_points.push_back(corners);
        cv::drawChessboardCorners(src, { CORNERS_W, CORNERS_H }, corners, true);

        // 从图片可以看出角点的排列顺序
        cv::imshow("image", src);
        cv::waitKey(50);
        std::cout << filename << ": success" << std::endl;
    } else {
        std::cout << filename << ": no corners found." << std::endl;
        return;
    }
}

int main() {
    // 读取图片, 寻找角点
    for (int i = 0; i < 41; ++i) {
        std::string filename = "chess/" + std::to_string(i) + ".jpg";
        find_corners(filename);
    }

    // 准备参数矩阵
    int n = corner_points.size();

    // 角点世界坐标
    std::vector<cv::Point3f> point_world_pos;
    // 相机旋转矩阵
    std::vector<cv::Mat> rotation_mats;
    // 相机平移矩阵
    std::vector<cv::Mat> translation_mats;
    // 内参矩阵
    cv::Mat intrinsics_mat = cv::Mat::zeros({ 3, 3 }, CV_32FC1);
    // 畸变系数
    cv::Mat dist_coeffs = cv::Mat::zeros({ 1, 5 }, CV_32FC1);

    // 构造世界坐标
    // 构造时, 内层循环行，外层列 (要和 findChessboard 找的角点顺序一致)
    for (int j = 0; j < CORNERS_H; ++j) {
        for (int i = 0; i < CORNERS_W; ++i) {
            cv::Point3f p;
            p.x = i * SQUARE_SIZE.width;
            p.y = j * SQUARE_SIZE.height;
            p.z = 0;
            point_world_pos.push_back(p);
        }
    }

    // 复制 n 份, 传入 calibrateCamera 用
    std::vector<std::vector<cv::Point3f>> point_world_pos_n(n, point_world_pos);

    double delta = cv::calibrateCamera(
        point_world_pos_n,
        corner_points,
        image_size,
        intrinsics_mat,
        dist_coeffs,
        rotation_mats,
        translation_mats
    );

    std::cout << "calibrate delta = " << delta << std::endl;
    std::cout << "intrinsics:\n" << intrinsics_mat << std::endl;
    std::cout << "dist coeffs:\n" << dist_coeffs << std::endl;

    return 0;
}