#include <iostream>

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"

// 四个点的顺序: 左上角 - 左下角 - 右下角 - 右上角

// 像素坐标
std::vector<cv::Point2f> pixel_points{
    {575.508, 282.175},
    {573.93, 331.819},
    {764.518, 337.652},
    {765.729, 286.741}};

// 旋转四元数: 相机 -> 世界坐标系
Eigen::Quaterniond quad{
    -0.0816168,
    0.994363,
    -0.0676645,
    -0.00122528};

// 装甲板的世界坐标
std::vector<cv::Point3f> world_points{
    {-0.115, 0.0265, 0.},
    {-0.115, -0.0265, 0.},
    {0.115, -0.0265, 0.},
    {0.115, 0.0265, 0.}};

// 相机内参矩阵
cv::Mat intrinsics_mat;

// 相机畸变矩阵
cv::Mat distortion_mat;

// 读入畸变矩阵和内参矩阵
void read_mat() {
    cv::FileStorage fs("f_mat_and_c_mat.yml", cv::FileStorage::READ);
    fs["F"] >> intrinsics_mat;
    fs["C"] >> distortion_mat;
    fs.release();
}

int main() {
    read_mat();
    cv::Mat tvec, rvec;
    cv::solvePnP(world_points, pixel_points, intrinsics_mat, distortion_mat, rvec, tvec);
    std::cout << rvec << std::endl;
    std::cout << tvec << std::endl;

    Eigen::Vector3d eigen_tvec;
    cv::cv2eigen(tvec, eigen_tvec);

    std::cout << quad.matrix() * eigen_tvec << std::endl;

    return 0;
}