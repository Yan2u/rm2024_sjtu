#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <thread>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "opencv2/opencv.hpp"

#include "Converter.hpp"
#include "LinearAnimation.hpp"
#include "Task.hpp"

// 原始图像尺寸
const int SRC_WIDTH = 944, SRC_HEIGHT = 534;

// 绘图图像尺寸
const int IMG_WIDTH = 1200, IMG_HEIGHT = 800;

// 相机能移动坐标范围
const int MAX_CAMERA_POS[3] = { 2000, 2000, 2000 };

// 相机欧拉角转动范围
const int MAX_CAMERA_EULER_1000[3] = { 6280, 6280, 6280 };

// 动画参数: 总帧数
const int FRAMES = 150;

// 动画参数: 每秒帧数
const int FPS = 30;

// 同时用来计算动画的线程数
const int THREADS_COUNT = 12;

// 白色点坐标
std::vector<Eigen::Vector3d> points;

// 世界坐标轴 (debug)
std::vector<Eigen::Vector3d> axis_points[3];

// 坐标轴颜色
cv::Scalar axis_colors[3] = { cv::Scalar(255, 255, 0),
                              cv::Scalar(255, 255, 255),
                              cv::Scalar(255, 0, 0) };

// 坐标轴文字
const char* axis_string[3] = { "x", "y", "z" };

// 相机世界坐标
int camera_pos[3];

// 像素点随机深度最大值, 随机数生成范围为 (-rand_max, rand_max) 均匀分布
int rand_max = 2000;

// 相机欧拉角的 1000 倍, 这是为了方便用滑动条
int camera_euler_1000[3] = { 0, 0, 0 };

// 相机姿态 欧拉角, 范围 (0, 2pi)
Eigen::Vector3d camera_euler;

// 坐标转换器
rm::Converter cter;

// 初始化, 读入 logo, 构造初始状态参数
void init() {
    srand((unsigned)time(nullptr));

    cv::Mat src = cv::imread("logo.png");
    assert(src.channels() == 3);

    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
    cv::adaptiveThreshold(src, src, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 201, 0);

    // 一开始默认 z = 0 平面上
    cv::Mat yellow = cv::Mat::zeros({ src.cols, src.rows }, CV_8UC3);

    for (int j = 0; j < SRC_HEIGHT; ++j) {
        for (int i = 0; i < SRC_WIDTH; ++i) {
            if (src.at<uint8_t>({ i, j }) == 255) {
                points.push_back(Eigen::Vector3d { (double)j, (double)i, 0. });
            }
        }
    }

    for (int i = 0; i < 200; ++i) {
        axis_points[0].push_back({ (double)i, 0., 0. });
        axis_points[1].push_back({ 0., (double)i, 0. });
        axis_points[2].push_back({ 0., 0., (double)i });
    }

    // 初始相机世界坐标 (0, 0, 1000)
    camera_pos[0] = SRC_WIDTH / 2;
    camera_pos[1] = SRC_HEIGHT / 2;
    camera_pos[2] = 1000;

    // 初始相机姿态欧拉角
    camera_euler = Eigen::Vector3d { 0., 0., 0. };

    // 初始化内参矩阵
    cter.intrinsics_mat << 400., 0., 190., 0., 0., 400., 160., 0., 0., 0., 1., 0.;
}

// 欧拉角转四元数, 顺序 rpy (Z-Y-X)
Eigen::Quaterniond euler_to_quad(Eigen::Vector3d euler_angle) {
    Eigen::AngleAxisd roll(Eigen::AngleAxisd(euler_angle[0], Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitch(Eigen::AngleAxisd(euler_angle[1], Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yaw(Eigen::AngleAxisd(euler_angle[2], Eigen::Vector3d::UnitZ()));

    return Eigen::Quaterniond(roll * pitch * yaw);
}

// 绘制图片, 滑动条调试用
void draw_image() {
    cv::Mat image = cv::Mat::zeros({ IMG_WIDTH, IMG_HEIGHT }, CV_8UC3);

    // 计算像素坐标, 并绘制在 image 上
    for (auto&& point: points) {
        auto pixel_point = cter.to_pixel_pos(point);
        int x = pixel_point[0], y = pixel_point[1];
        if (x > 0 && x < IMG_WIDTH && y > 0 && y < IMG_HEIGHT) {
            cv::circle(image, { x, y }, 1, cv::Scalar(0, 0, 255), 1);
        }
    }

    // 为了调试方便, 把坐标轴也画出来
    for (int i = 0; i < 3; ++i) {
        int len = axis_points[i].size();
        for (int j = 0; j < len; ++j) {
            auto&& point = axis_points[i][j];
            auto pixel_point = cter.to_pixel_pos(point);
            int x = pixel_point[0], y = pixel_point[1];
            if (x > 0 && x < IMG_WIDTH && y > 0 && y < IMG_HEIGHT) {
                cv::circle(image, { x, y }, 1, axis_colors[i], 1);
            }
        }

        auto pixel_point = cter.to_pixel_pos(axis_points[i][len - 1]);
        int x = pixel_point[0], y = pixel_point[1];
        cv::putText(
            image,
            axis_string[i],
            { x + 2, y - 2 },
            cv::FONT_HERSHEY_PLAIN,
            1,
            axis_colors[i],
            1
        );
    }

    cv::imshow("image", image);
}

// 随机改变 z = 0 上的像素点的深度 (z 值)
void apply_random() {
    static int len = points.size();

    Eigen::Matrix<double, Eigen::Dynamic, 1> coeff(len, 1);

    coeff = Eigen::Matrix<double, Eigen::Dynamic, 1>::Random(len, 1);

    rand_max = std::max(rand_max, 1);
    for (int i = 0; i < len; ++i) {
        points[i][2] = (double)rand_max * coeff[i];
    }
}

// 滑动条回调函数
void update_trackbar(int, void*) {
    static int last_randmax = 199;

    if (rand_max != last_randmax) {
        last_randmax = rand_max;
        apply_random();
    }

    // 欧拉角除以 1000 化到 [0, 2pi]
    camera_euler[0] = (double)camera_euler_1000[0] / 1000.;
    camera_euler[1] = (double)camera_euler_1000[1] / 1000.;
    camera_euler[2] = (double)camera_euler_1000[2] / 1000.;

    // 更新参数
    cter.camera_pos[0] = (double)camera_pos[0];
    cter.camera_pos[1] = (double)camera_pos[1];
    cter.camera_pos[2] = (double)camera_pos[2];

    cter.pose_quad = euler_to_quad(camera_euler);
    cter.calc_extrinsics_mat();
    draw_image();
}

// 创建滑动条窗口, 方便调参和调试
void create_trackbars() {
    cv::namedWindow("bars");

    cv::createTrackbar("x", "bars", camera_pos, MAX_CAMERA_POS[0], update_trackbar);
    cv::createTrackbar("y", "bars", camera_pos + 1, MAX_CAMERA_POS[1], update_trackbar);
    cv::createTrackbar("z", "bars", camera_pos + 2, MAX_CAMERA_POS[2], update_trackbar);
    cv::createTrackbar(
        "euler x",
        "bars",
        camera_euler_1000,
        MAX_CAMERA_EULER_1000[0],
        update_trackbar
    );
    cv::createTrackbar(
        "euler y",
        "bars",
        camera_euler_1000 + 1,
        MAX_CAMERA_EULER_1000[1],
        update_trackbar
    );
    cv::createTrackbar(
        "euler z",
        "bars",
        camera_euler_1000 + 2,
        MAX_CAMERA_EULER_1000[2],
        update_trackbar
    );

    cv::createTrackbar("rd_max", "bars", &rand_max, 1000, update_trackbar);

    cv::Mat temp = cv::Mat::zeros({ 1200, 10 }, CV_8UC1);
    cv::imshow("bars", temp);
}

void test_animation() {
    rm::LinearAnimationInfo st, ed;
    st.camera_pos = Eigen::Vector3d { 0, 800, 500 };
    st.camera_euler_1000 = Eigen::Vector3i { 0, 4600, 1570 };
    st.rand_max = 1000;

    ed.camera_pos = Eigen::Vector3d { 0, 1008, 500 };
    ed.camera_euler_1000 = Eigen::Vector3i { 0, 6280, 1570 };
    ed.rand_max = 0;

    rm::LinearAnimation la(st, ed, points, { IMG_WIDTH, IMG_HEIGHT }, FRAMES);

    std::vector<cv::Mat> frames;
    la.generate_frames(THREADS_COUNT, frames);
    la.generate_video(frames, FPS, "animation");
}

int main() {
    srand((unsigned)time(nullptr));
    init();

    // create_trackbars();
    // update_trackbar(0, 0);
    test_animation();

    cv::waitKey(0);
    return 0;
}
