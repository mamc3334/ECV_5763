#pragma once

#include <opencv2/opencv.hpp>

struct Frame
{
    cv::Mat frame;
    int frame_number;
};
