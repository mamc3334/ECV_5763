// Very simple example found on Stackoverflow
// Adapted by Mason McGaffin for ECV 5763 OpenCV 4
// Based on ECV-ECEE-5763-mcgaffin/sift/sift.cpp
//

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{        
  Mat image = imread(argv[1]);

  // Create smart pointer for SIFT feature detector.
  Ptr<Feature2D> dector_extractor = SIFT::create();
  vector<KeyPoint> keypoints;
  // Compute the 128 dimension SIFT descriptor at each keypoint.
  // Each row in "descriptors" correspond to the SIFT descriptor for each keypoint
  Mat descriptors;
  dector_extractor->detectAndCompute(image, noArray(), keypoints, descriptors);

  // If you would like to draw the detected keypoint just to check
  Mat outputImage;
  Scalar keypointColor = Scalar(255, 0, 0);     // Blue keypoints.
  drawKeypoints(image, keypoints, outputImage, keypointColor, DrawMatchesFlags::DEFAULT);

  imshow("Output", outputImage);

  for(;;)
  {
    char key = (char)waitKey(0);
    if(key == 27) //esc
    {
        syslog(LOG_INFO, "ESC Pressed. Closing application...\n");
        break;
    }
  }

  return 0;

}
