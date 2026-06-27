/**
* Interactive Edge Detection - Canny - Sobel - None
* Based off the examples of canny and sobel edge detection in the path: 
              ECV-ECEE-5763-mcgaffin/computer_vision_cv4_tested/capture-transformer

* Author: Mason McGaffin
* RTES
* Exercise 2 - Question 5

Using the OpenCV camera capture code from previous work and your USB webcam, 
create a viewer where you can turn on/off edge detection for Canny and/or Sobel 
by keystroke (e.g., “c” for Canny, “s” for Sobel, “n” for None). You should be 
able to build this by combining work on Sobel, Canny, and OpenCV camera capture 
code you have already completed. This example may however help - 
capture-transformer which was written for OpenCV 3.x (so update as needed or 
write your own!). Again, compute the frame average rate for Canny and Sobel 
test runs using posix_clock_gettime time stamping. Add code for this analysis 
as needed to update and display frame rate updated periodically with OpenCV 
“put text” on an imshow window or as an update in a syslog (syslog_example). 
Consider making a slider or other interactive features for thresholds used for 
Canny and/or Sobel, but this is not required.

*/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <syslog.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

//canny
#define CANNY_RATIO 3U
#define MAX_THRESHOLD 100U

//sobel
#define MAX_KSIZE 16U
#define MAX_SCALE 200U
#define MAX_DELTA 200U

#define FIVE_SEC_MS 5000.0f

#define KERNEL_SIZE 3U

#define WINDOW_NAME "Interactive Edge Detector"

enum State{
  STATE_RESET,
  STATE_CANNY,
  STATE_SOBEL,
  STATE_GRAY
};

struct SobelData {
  Mat img_gray;
  Mat img_dst;
  int ksize = 1;
  int scale = 100;
  int delta = 100;
};

struct CannyData {
  Mat img_gray;
  Mat img_src;
  Mat img_dst;
  int low_threshold = 0;
};

State state;

//canny callback
void CannyImage(int, void* myData)
{
  if(state != STATE_CANNY) return;

  CannyData* data = static_cast<CannyData*>(myData);

  Mat detected_edges;

  /// Reduce noise with a kernel 3x3
  blur(data->img_gray, detected_edges, Size(KERNEL_SIZE,KERNEL_SIZE) );

  /// Canny detector
  Canny( detected_edges, detected_edges, data->low_threshold, data->low_threshold*CANNY_RATIO, KERNEL_SIZE );

  /// Using Canny's output as a mask, we display our result
  data->img_dst = Scalar::all(0);

  data->img_src.copyTo( data->img_dst, detected_edges);
}


//sobel callback
void SobelImage(int, void* myData)
{
  if(state != STATE_SOBEL) return;

  SobelData* data = static_cast<SobelData*>(myData);

  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  int ddepth = CV_16S;
  int ksize = data->ksize * 2 - 1;
  int scale = data->scale - 100;
  int delta = data->delta - 100;
  
  Sobel(data->img_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
  Sobel(data->img_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
  // converting back to CV_8U
  convertScaleAbs(grad_x, abs_grad_x);
  convertScaleAbs(grad_y, abs_grad_y);
  addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, data->img_dst);
}

void processImage(Mat image, CannyData *cd, SobelData *sd)
{
  Mat src, src_gray;

  // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
  GaussianBlur(image, src, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_DEFAULT);
  // Convert the image to grayscale
  cvtColor(src, src_gray, COLOR_BGR2GRAY);

  sd->img_gray = src_gray;
  cd->img_gray = src_gray;
  cd->img_src = src;
}

int main( int argc, char** argv )
{
  Mat image, src, dst;
  State old_state;
  state = STATE_RESET;
  old_state = STATE_CANNY;
  SobelData sobel_data;
  CannyData canny_data;
  unsigned int framecnt=0;
  struct timespec start_time;
  struct timespec curr_time;
  double time_elapsed_ms;
  char frame_txt[32] = {0};
  unsigned int lognum = 5;

  openlog("interactive_edge", LOG_PID, LOG_USER);

  cv::CommandLineParser parser(argc, argv,
                               "{@input   |../Jeep-Sideview.png|input image}"
                               "{live    l|false|live video feed using webcam}"
                               "{help    h|false|show help message}");
  cout << "The sample uses Canny and Sobel or Scharr OpenCV functions for edge detection\n\n";
  
  if(parser.get<bool>("help"))
  {
    parser.printMessage();
    return 0;
  }
  
  cout << "\nPress 'ESC' to exit program.\nPress 'C' for Canny Edge Detection.\nPress 'S' for Sobel Edge Detection.\nPress 'R' to see Original Image.\nPress 'G' to see Grayscale Image\n)";
  
  String imageName = parser.get<String>("@input");
  bool live = parser.get<bool>("live");

  VideoCapture cam0(0);
  namedWindow(WINDOW_NAME);

  if(live)
  {
    syslog(LOG_INFO, "Interactive Edge Application started using live webcam.");

    if (!cam0.isOpened())
    {
        syslog(LOG_CRIT, "Error opening web camera.\n");
        perror("webcam");

        return 1;
    }

    cam0.set(CAP_PROP_FRAME_WIDTH, 640);
    cam0.set(CAP_PROP_FRAME_HEIGHT, 480);

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    cam0.read(image);
  }
  else{
    syslog(LOG_INFO, "Interactive Edge Application started using static image: %s", imageName.c_str());
    // As usual we load our source image (src)
    image = imread( imageName, IMREAD_COLOR ); // Load an image
    // Check if image is loaded fine
    if( image.empty() )
    {
      syslog(LOG_CRIT, "Error opening image: %s\n", imageName.c_str());
      perror("input image");

      return 1;
    }
  }
  
  processImage(image, &canny_data, &sobel_data);

  //create canny trackbar
  createTrackbar( "Canny: Min Threshold:", WINDOW_NAME, &canny_data.low_threshold, MAX_THRESHOLD, CannyImage, &canny_data);

  //create sobel trackbar
  createTrackbar("Sobel: ksize", WINDOW_NAME, &sobel_data.ksize, MAX_KSIZE, SobelImage, &sobel_data);
  createTrackbar("Sobel: scale", WINDOW_NAME, &sobel_data.scale, MAX_SCALE, SobelImage, &sobel_data);
  createTrackbar("Sobel: delta", WINDOW_NAME, &sobel_data.delta, MAX_DELTA, SobelImage, &sobel_data);

  CannyImage(0, &canny_data);
  SobelImage(0, &sobel_data);

  for (;;)
  {
    if(live || (state != old_state))
    {
      old_state = state;

      switch(state)
      {
        case STATE_CANNY:
          CannyImage(0, &canny_data);
          dst = canny_data.img_dst;
          
          break;
        
        case STATE_SOBEL:
          SobelImage(0, &sobel_data);
          dst = sobel_data.img_dst;

          break;

        case STATE_GRAY:
          dst = sobel_data.img_gray;

          break;

        case STATE_RESET: default:
          dst = image;

          break;
      }
    }

    putText(dst, frame_txt, Point(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(200,200,250), 1, LINE_AA);
    imshow(WINDOW_NAME, dst);

    char key = (char)waitKey(10);
    if(key == 27) //esc
    {
      syslog(LOG_INFO, "Closing application...\n");
      break;
    }
    if ((key == 'c') || (key == 'C'))
    {
      state = STATE_CANNY;
    }
    if ((key == 's') || (key == 'S'))
    {
      state = STATE_SOBEL;
    }
    if ((key == 'g') || (key == 'G'))
    {
      state = STATE_GRAY;
    }
    if ((key == 'r') || (key == 'R')) // reset - none
    {
      state = STATE_RESET;
    }

    if(live)
    {
      clock_gettime(CLOCK_MONOTONIC, &curr_time);
      time_elapsed_ms = (curr_time.tv_sec - start_time.tv_sec) * 1000 + 
                      ((curr_time.tv_nsec - start_time.tv_nsec) / 1e6);
      
      framecnt++;

      float frame_rate = (framecnt / time_elapsed_ms) * 1000.0f;
      snprintf(frame_txt, sizeof(frame_txt), "FRAME RATE: %.4f fps", frame_rate);

      //log every 5 seconds
      if (time_elapsed_ms > FIVE_SEC_MS)
      {
          syslog(LOG_INFO, "[%ds] Calculated frame rate: %.4f fps", lognum, frame_rate);
	  lognum+=5;
	  framecnt=0;
	  clock_gettime(CLOCK_MONOTONIC, &start_time);
      }
      
      cam0.read(image);
      processImage(image, &canny_data, &sobel_data);
    }
  }

  //cleanup
  destroyWindow(WINDOW_NAME);

  return 0;
}
