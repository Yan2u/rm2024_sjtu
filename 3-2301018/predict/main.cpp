#include <fstream>
#include <iostream>

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "opencv2/opencv.hpp"

#include "KalmanFilter.hpp"

std::vector<double> rates;

void read_data() {
    std::ifstream ifs("dollar.txt");

    rates.clear();
    double rate;
    while (ifs >> rate) {
        rates.push_back(rate);
    }

    ifs.close();
}

int main() {
    read_data();
    int n = rates.size();

    // 预测向量 [x, v] = [汇率, 增长速度]
    // 测量向量 [x]    = [汇率]
    rm::KalmanFilter<2, 1> kf;

    kf.predict = Eigen::Vector2d::Zero();

    // 不确定度
    Eigen::Matrix2d uncertainty = Eigen::Matrix2d::Identity();
    uncertainty << 10., 0., 0., 1.;
    kf.uncertainty = uncertainty;

    // 状态转移矩阵
    // x[i] = x[i - 1] + v[i - 1]
    // v[i] = v[i - 1]
    kf.pred_transition_mat << 1., 1., 0., 1.;

    // 预测转测量矩阵
    kf.measure_transition_mat << 1., 0.;

    // 过程噪声
    kf.progress_cov_mat = Eigen::Matrix2d::Identity();

    // 测量噪声
    kf.measure_cov_mat = Eigen::Matrix<double, 1, 1>::Identity();

    std::cout << "id rate speed" << std::endl;
    Eigen::Vector2d result;

    double speed = 0.;
    int speed_n = 0;

    for (int i = 0; i < n; ++i) {
        result = kf.update(Eigen::Vector<double, 1>{rates[i]});
        if (i > 15) {
            ++speed_n;
            speed += result[1];
        }
        std::cout << i + 1 << ' ' << result[0] << ' ' << result[1] << std::endl;
    }

    speed /= (double)speed_n;
    double last_rate = result[0];
    std::cout << "speed = " << speed << std::endl;
    std::cout << "predict" << std::endl;

    for (int i = 0; i < 10; ++i) {
        result = kf.update(Eigen::Vector<double, 1>(last_rate + speed * (i + 1)));
        std::cout << n + i + 1 << ' ' << result[0] << ' ' << result[1] << std::endl;
    }

    return 0;
}