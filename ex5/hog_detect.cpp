#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Load input image
    cv::Mat img = cv::imread("images.jpeg");
    if (img.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }

    // Initialize Native OpenCV HOG Descriptor
    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    // Storage for detected regions and weights
    std::vector<cv::Rect> found_regions;
    std::vector<double> weights;

    // Run multi-scale HOG detection
    hog.detectMultiScale(img, found_regions, weights, 0, cv::Size(8,8), cv::Size(), 1.05, 2, false);

    // Draw bounding boxes on the image
    for (size_t i = 0; i < found_regions.size(); i++) {
        cv::rectangle(img, found_regions[i], cv::Scalar(0, 255, 0), 2);
        std::cout << "Detected object at: " << found_regions[i] << std::endl;
    }

    // Display the output window
    cv::imshow("OpenCV HOG Detection", img);
    cv::waitKey(0);
    return 0;
}