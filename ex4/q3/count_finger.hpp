/**
 * @file count_finger.hpp
    * @brief Header file for finger counting functionality.
*/

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>
#include <queue>
#include <set>

#include <cmath>
#include <string>


using namespace cv;
using namespace std;

int drawBoxesAroundFingers(const cv::Mat& skel, cv::Mat& canvas);

int highlightAndCountFingers(const cv::Mat& skel, cv::Mat& canvas);

int countFingers(const cv::Mat &src, cv::Mat &debugOut);

int contorFingers(const cv::Mat &src, cv::Mat &debugOut);

int improvedCountFingers(const cv::Mat& skel, const cv::Mat& binaryMask, cv::Mat& canvas);

int countCircle(const cv::Mat& skel, const cv::Mat& binaryMask, cv::Mat& canvas);