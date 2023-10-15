#ifndef RM_LINEAR_ANIMATION_HPP
#define RM_LINEAR_ANIMATION_HPP
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Task.hpp"
#include "opencv2/opencv.hpp"

namespace rm {

struct LinearAnimationInfo {
    Eigen::Vector3d camera_pos;
    Eigen::Vector3i camera_euler_1000;
    int rand_max;
};

class LinearAnimation {
private:
    LinearAnimationInfo start;
    LinearAnimationInfo end;
    std::vector<LinearAnimationInfo> key_frames;
    std::vector<int> frames_between_keyframes;
    cv::Size2i image_size;
    std::vector<Eigen::Vector3d> points;
    int total_frames;

    Eigen::VectorXd random_values;

    // 欧拉角转四元数, 顺序 rpy (Z-Y-X)
    static Eigen::Quaterniond euler_to_quad(Eigen::Vector3d euler_angle) {
        Eigen::AngleAxisd roll(Eigen::AngleAxisd(euler_angle[0], Eigen::Vector3d::UnitX()));
        Eigen::AngleAxisd pitch(Eigen::AngleAxisd(euler_angle[1], Eigen::Vector3d::UnitY()));
        Eigen::AngleAxisd yaw(Eigen::AngleAxisd(euler_angle[2], Eigen::Vector3d::UnitZ()));

        return Eigen::Quaterniond(roll * pitch * yaw);
    }

    LinearAnimationInfo info_between(
        const LinearAnimationInfo& start,
        const LinearAnimationInfo& end,
        int total_steps,
        int current_step
    ) {
        LinearAnimationInfo result;

        double percent = (double)current_step / total_steps;

        result.camera_pos = start.camera_pos + (end.camera_pos - start.camera_pos) * percent;
        result.camera_euler_1000 = start.camera_euler_1000
            + ((end.camera_euler_1000 - start.camera_euler_1000).cast<double>() * percent)
                  .cast<int>();
        result.rand_max = start.rand_max + (double)(end.rand_max - start.rand_max) * percent;

        return result;
    }

    // 运行在每个线程上的, 计算指定范围内的动画帧的函数
    void apply_image_thread(
        const LinearAnimationInfo* st,
        const LinearAnimationInfo* ed,
        cv::Mat* rst,
        cv::Mat* red
    ) {
        auto&& ptr = st;
        auto&& rptr = rst;

        int n = points.size();
        std::vector<Eigen::Vector3d> points_local(n);
        std::copy(points.begin(), points.end(), points_local.begin());

        Eigen::Vector3d camera_pos_local;
        Eigen::Vector3d camera_euler_local;
        rm::Converter cter_local;
        cter_local.intrinsics_mat << 400., 0., 190., 0., 0., 400., 160., 0., 0., 0., 1., 0.;

        auto&& apply_ai_local = [&](const LinearAnimationInfo& ai) -> void {
            for (int i = 0; i < n; ++i) {
                points_local[i][2] = (double)ai.rand_max * random_values[i];
            }
            camera_euler_local[0] = (double)ai.camera_euler_1000[0] / 1000.;
            camera_euler_local[1] = (double)ai.camera_euler_1000[1] / 1000.;
            camera_euler_local[2] = (double)ai.camera_euler_1000[2] / 1000.;
            cter_local.camera_pos[0] = (double)ai.camera_pos[0];
            cter_local.camera_pos[1] = (double)ai.camera_pos[1];
            cter_local.camera_pos[2] = (double)ai.camera_pos[2];
            cter_local.pose_quad = LinearAnimation::euler_to_quad(camera_euler_local);
            cter_local.calc_extrinsics_mat();
        };

        for (; ptr != ed && rptr != red; ++ptr, ++rptr) {
            apply_ai_local(*ptr);
            *rptr = cv::Mat::zeros(image_size, CV_8UC1);
            Eigen::MatrixXd result;
            cter_local.to_pixel_pos(points_local, result);

            for (int i = 0; i < n; ++i) {
                int x = result(0, i), y = result(1, i);
                if (x > 0 && x < image_size.width && y > 0 && y < image_size.height) {
                    cv::circle(*rptr, { x, y }, 1, cv::Scalar(255), 1);
                }
            }
        }
    }

public:
    /**
     * @brief 生成全部动画帧
     * 
     * @param thread_count 多线程数
     * @param frames 用来接收结果的帧数组
     */
    void generate_frames(int thread_count, std::vector<cv::Mat>& frames) {
        std::vector<LinearAnimationInfo> infos {};

        random_values = Eigen::VectorXd::Random(points.size());

        int n = key_frames.size();
        for (int i = 0; i < n - 1; ++i) {
            for (int j = 0; j < frames_between_keyframes[i] + 1; ++j) {
                infos.push_back(
                    info_between(key_frames[i], key_frames[i + 1], frames_between_keyframes[i], j)
                );
            }
        }

        auto&& callable = std::bind(
            &LinearAnimation::apply_image_thread,
            this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4
        );

        rm::Task::run_task_ref_await(callable, infos, frames, thread_count);
    }

    /**
     * @brief 写入视频, 固定 mp4 格式
     * 
     * @param frames 动画帧数组
     * @param fps fps
     * @param filename_prefix 视频文件前缀名
     */
    void generate_video(
        const std::vector<cv::Mat>& frames,
        int fps,
        const std::string& filename_prefix
    ) {
        cv::VideoWriter writer(
            filename_prefix + ".mp4",
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            fps,
            image_size,
            false
        );
        for (auto&& frame: frames) {
            writer << frame;
        }
        writer.release();
    }

    /**
     * @brief 调用 `cv::imshow` 播放全部动画帧
     * 
     * @param frames 动画帧数组
     * @param fps 每秒帧数
     * @param window_name 窗口名称
     */
    static void
    show_frames(std::vector<cv::Mat>& frames, int fps, std::string window_name = "animation") {
        for (auto&& frame: frames) {
            cv::imshow(window_name, frame);
            cv::waitKey(1000 / fps);
        }
    }

    explicit LinearAnimation(
        const std::vector<LinearAnimationInfo>& key_frames,
        const std::vector<int>& frames_between_keyfames,
        const std::vector<Eigen::Vector3d>& points,
        cv::Size2i image_size
    ):
        key_frames(key_frames),
        frames_between_keyframes(frames_between_keyfames),
        image_size(image_size),
        points(points) {
        assert(key_frames.size() > frames_between_keyfames.size());
    }
};
} // namespace rm

#endif