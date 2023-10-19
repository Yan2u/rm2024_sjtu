#include "Eigen/Core"
#include "Eigen/Dense"

namespace rm {
template <int PredictN, int MeasureN>
class KalmanFilter {
   public:
    using PredictVec = Eigen::Vector<double, PredictN>;
    using MeasureVec = Eigen::Vector<double, MeasureN>;
    using PredictMat = Eigen::Matrix<double, PredictN, PredictN>;
    using MeasureMat = Eigen::Matrix<double, MeasureN, MeasureN>;

    // 当前的预测向量
    PredictVec predict;

    // 状态转移矩阵
    PredictMat pred_transition_mat;

    // 转移方程其他项
    PredictVec pred_transition_constant;

    // 不确定度
    PredictMat uncertainty;

    // 过程噪声 (协方差) 矩阵
    PredictMat progress_cov_mat;

    // 观测转预测向量矩阵
    Eigen::Matrix<double, MeasureN, PredictN> measure_transition_mat;

    // 测量噪声 (协方差) 矩阵
    MeasureMat measure_cov_mat;

    // 卡尔曼增益
    Eigen::Matrix<double, PredictN, MeasureN> kalman_gain;

    PredictVec update(MeasureVec new_measure) {
        // predict
        // x' = F * x + U
        auto&& new_predict = pred_transition_mat * predict + pred_transition_constant;

        // P' = F * P * F^T + Q
        auto&& new_uncertainty = pred_transition_mat * uncertainty * pred_transition_mat.transpose() + progress_cov_mat;

        // measure & kalman gain
        // y = z - H * x'
        auto&& delta = new_measure - measure_transition_mat * new_predict;

        // S = H * P' * H^T + R
        MeasureMat temp = measure_transition_mat * new_uncertainty * measure_transition_mat.transpose() + measure_cov_mat;

        // K = P' * H^T * S^-1
        kalman_gain = new_uncertainty * measure_transition_mat.transpose() * temp.inverse();

        // update & get result
        // x = x' + K * y
        predict = new_predict + kalman_gain * delta;

        // P = (I - KH) * P'
        uncertainty = (PredictMat::Identity() - kalman_gain * measure_transition_mat) * new_uncertainty;

        return new_predict;
    }

    KalmanFilter() {
        uncertainty = PredictMat::Identity();
    }
};
}  // namespace rm