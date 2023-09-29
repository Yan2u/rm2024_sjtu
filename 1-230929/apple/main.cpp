#include <filesystem>
#include <iostream>
#include <set>
#include <stdio.h>
#include <stdlib.h>

#include "opencv2/opencv.hpp"

using Contour = std::vector<cv::Point>;
using ContourArray = std::vector<Contour>;

template<typename TFirst, typename... TRest>
inline void show_images(TFirst first, TRest... rest) {
    static_assert(std::is_same_v<TFirst, cv::Mat>, "not cv::Mat object");

    cv::imshow(std::to_string(sizeof...(rest)), first);
    cv::waitKey(1);

    if constexpr (sizeof...(rest) > 0) {
        show_images(rest...);
    } else {
        cv::waitKey(0);
    }
}

int argmax_counter_area(const std::vector<std::vector<cv::Point>>& contours) {
    int res = 0, len = contours.size();
    double max_area = 0;
    for (int i = 0; i < len; ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            res = i;
        }
    }
    return res;
}

void task() {
    cv::Mat src = cv::imread("apple.png");
    assert(src.channels() == 3);

    cv::Mat channels[3]; // b, g, r
    cv::split(src, channels);
    channels[1] = channels[2] - channels[1];

    cv::Mat res;
    cv::merge(channels, 3, res);

    cv::Mat hsv;
    cv::cvtColor(res, hsv, cv::COLOR_BGR2HSV);

    cv::Mat bin;
    cv::inRange(hsv, cv::Scalar(6, 100, 100), cv::Scalar(36, 255, 255), bin);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, { 3, 3 });

    cv::Mat morph;
    cv::morphologyEx(bin, morph, cv::MORPH_CLOSE, kernel, { -1, -1 }, 1);

    ContourArray contours;
    cv::findContours(morph, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    int max_i = argmax_counter_area(contours);

    cv::drawContours(src, contours, max_i, cv::Scalar(255, 255, 255), 1);

    cv::Rect rect = cv::boundingRect(contours[max_i]);

    cv::rectangle(src, rect, cv::Scalar(0, 255, 255));

    cv::imshow("src", src);
    cv::imshow("res", res);
    cv::imshow("bin", bin);
    cv::imshow("morph", morph);
    cv::waitKey(0);
    cv::imwrite("apple_res.png", src);
}

int main() {
    task();
    return 0;
}