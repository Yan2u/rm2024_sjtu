#include "opencv2/opencv.hpp"
#include <iostream>

int hl = 100;
int sl = 158;
int vl = 120;
int hr = 124;
int sr = 255;
int vr = 255;
int it = 20;

// 100, 43, 46
// 124, 255, 255

// 100, 158. 164
// 124, 255, 255

// 100, 158, 120

cv::Mat src;

using Contour = std::vector<cv::Point>;
using ContourArray = std::vector<Contour>;

void task(int img_id) {
    std::string file_name = "plates/00" + std::to_string(img_id) + ".jpg";
    // std::string file_name = "plates/002.jpg";
    cv::Mat src = cv::imread(file_name);
    assert(src.channels() == 3);

    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    cv::Mat bin;

    cv::inRange(hsv, cv::Scalar(hl, sl, vl), cv::Scalar(hr, sr, vr), bin);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, { 3, 3 });
    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel, { -1, -1 }, it);

    ContourArray contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    int max_i = -1;
    int len = contours.size();
    double maxarea = 0.0;
    ContourArray targets;

    for (int i = 0; i < len; ++i) {
        // minarea rect
        cv::RotatedRect rect = cv::minAreaRect(contours[i]);
        cv::Point2f pts[4];
        rect.points(pts);
        std::vector<cv::Point2f> pts_vec(pts, pts + 4);

        double minarea = cv::contourArea(pts_vec, false);

        ContourArray ca(1);
        cv::approxPolyDP(contours[i], ca[0], 1, true);

        double polyarea = cv::contourArea(ca[0], false);

        double percentage = (minarea - polyarea) / minarea;

        if (percentage >= 0.15) {
            continue;
        }

        if (minarea > maxarea) {
            maxarea = minarea;
            max_i = i;
            targets.clear();
            targets.push_back(Contour());
            for (auto&& p: pts_vec) {
                targets[0].push_back(p);
            }
        }
    }

    if (max_i == -1) {
        std::cout << "failed" << std::endl;
    } else {
        cv::drawContours(src, targets, -1, cv::Scalar(255, 255, 0), 3);
        cv::imshow(file_name, src);
    }

    cv::imwrite("result/" + std::to_string(img_id) + ".jpg", src);
}

int main() {
    for (int i = 1; i < 6; ++i) {
        task(i);
    }
    cv::waitKey(0);
    return 0;
}
