#include <iostream>

#include "opencv2/opencv.hpp"

#include "Eigen/Core"

using Contour = std::vector<cv::Point>;
using ContourArray = std::vector<Contour>;

int hsv_low[3] = { 9, 0, 128 };
int hsv_high[3] = { 50, 255, 255 };
int iteration_cnt = 1;
int kernel_size = 3;
int img_id = 1;
int last_img_id = 1;

double hw_limit = 2.0;

int dh_limit = 6;

cv::Mat src;

cv::VideoWriter writer;

void filter() {
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    cv::Mat bin;
    cv::inRange(
        hsv,
        cv::Scalar(hsv_low[0], hsv_low[1], hsv_low[2]),
        cv::Scalar(hsv_high[0], hsv_high[1], hsv_high[2]),
        bin
    );

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, { kernel_size, kernel_size });

    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel, { -1, -1 }, iteration_cnt);

    ContourArray contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    cv::Mat img_contours = cv::Mat::zeros({ bin.cols, bin.rows }, CV_8UC3);
    cv::drawContours(img_contours, contours, -1, cv::Scalar(0, 255, 0));

    std::vector<cv::Rect> qualified_rects;

    for (auto&& contour: contours) {
        cv::Rect rect = cv::boundingRect(contour);
        cv::Size size = rect.size();
        double hw = (double)size.height / size.width;
        if (hw < hw_limit || size.height * size.width < 50) {
            continue;
        }
        qualified_rects.push_back(rect);
        cv::rectangle(img_contours, rect, cv::Scalar(0, 255, 255));
        cv::putText(
            img_contours,
            std::to_string(hw),
            { rect.x, rect.y },
            cv::FONT_HERSHEY_PLAIN,
            1.0,
            cv::Scalar(0, 0, 255)
        );
        cv::putText(
            img_contours,
            std::to_string(size.height * size.width),
            { rect.x + size.width, rect.y + size.height },
            cv::FONT_HERSHEY_PLAIN,
            1.0,
            cv::Scalar(255, 0, 0)
        );
    }

    cv::imshow("bin", bin);
    cv::imshow("contours", img_contours);

    int len = qualified_rects.size();
    if (len) {
        std::sort(
            qualified_rects.begin(),
            qualified_rects.end(),
            [](const cv::Rect& r1, const cv::Rect& r2) -> bool { return r1.x < r2.x; }
        );

        cv::Mat img_contours_2 = src.clone();
        cv::rectangle(img_contours_2, qualified_rects[0], cv::Scalar(0, 255, 255));

        int str_y = 30;
        for (int i = 1; i < len; ++i) {
            cv::rectangle(img_contours_2, qualified_rects[i], cv::Scalar(0, 255, 255));
            int dx = qualified_rects[i].x - qualified_rects[i - 1].x;
            int dy = qualified_rects[i].y - qualified_rects[i - 1].y;
            cv::putText(
                img_contours_2,
                std::to_string(i) + "->" + std::to_string(i + 1) + ": " + std::to_string(dx) + ", "
                    + std::to_string(dy),
                { 0, str_y },
                cv::FONT_HERSHEY_PLAIN,
                1,
                cv::Scalar(0, 0, 255)
            );
            str_y += 20;
        }

        for (int i = 1; i < len; ++i) {
            if (std::abs(qualified_rects[i].y - qualified_rects[i - 1].y) < dh_limit) {
                cv::Rect new_rect(
                    cv::Point(qualified_rects[i - 1].x, qualified_rects[i - 1].y),
                    cv::Point(
                        qualified_rects[i].x + qualified_rects[i].size().width,
                        qualified_rects[i].y + qualified_rects[i].size().height
                    )
                );
                cv::rectangle(img_contours_2, new_rect, cv::Scalar(0, 0, 255));
                i += 2;
            }
        }
        cv::imshow("contours_2", img_contours_2);
        writer.write(img_contours_2);
    }
}

int main() {
    writer.open("car_plate.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, { 1920, 1080 });
    for (int i = 1; i < 442; ++i) {
        std::string filename = "frames/" + std::to_string(i) + ".jpg";
        src = cv::imread(filename);

        cv::namedWindow("src");
        cv::imshow("src", src);
        filter();
        cv::waitKey(50);
    }
    cv::waitKey(0);
    writer.release();
    return 0;
}