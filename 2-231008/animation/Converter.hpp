#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"

namespace rm {
class Converter {
    template<int N, int M>
    using Matrix = Eigen::Matrix<double, N, M>;

private:
    Matrix<4, 4> extrinsics_mat_ex;

public:
    Matrix<3, 1> camera_pos;
    Matrix<3, 4> intrinsics_mat;
    Eigen::Quaterniond pose_quad;

    void calc_extrinsics_mat() {
        Matrix<3, 3> rotation_mat = pose_quad.toRotationMatrix().transpose();
        extrinsics_mat_ex = Matrix<4, 4>::Zero();
        extrinsics_mat_ex.block(0, 0, 3, 3) = rotation_mat;
        extrinsics_mat_ex.block(0, 3, 3, 1) = (-rotation_mat) * camera_pos;
        extrinsics_mat_ex(3, 3) = 1.;
    }

    Matrix<2, 1> to_pixel_pos(const Matrix<3, 1>& world_pos) {
        Matrix<4, 1> world_pos_ex;
        world_pos_ex << world_pos, 1.;
        Matrix<3, 1> pixel_pos_ex = intrinsics_mat * (extrinsics_mat_ex * world_pos_ex);
        pixel_pos_ex /= pixel_pos_ex[2];
        return pixel_pos_ex.block(0, 0, 2, 1);
    }

    void
    to_pixel_pos(const std::vector<Eigen::Vector3d>& world_pos_array, Eigen::MatrixXd& result) {
        int n = world_pos_array.size();
        auto&& world_pos_array_noconst = const_cast<std::vector<Eigen::Vector3d>&>(world_pos_array);

        Eigen::Map<Eigen::MatrixXd> map(world_pos_array_noconst.data()->data(), 3, n);
        Eigen::MatrixXd world_pos_mat_ex(4, n);
        world_pos_mat_ex.block(0, 0, 3, n) = map;
        world_pos_mat_ex.block(3, 0, 1, n).setConstant(1.);

        result = intrinsics_mat * (extrinsics_mat_ex * world_pos_mat_ex);

        auto&& last_row = result.block(2, 0, 1, n).array();
        result.block(0, 0, 1, n).array() /= last_row;
        result.block(1, 0, 1, n).array() /= last_row;
    }
};
} // namespace rm