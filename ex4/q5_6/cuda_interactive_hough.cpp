/**
* Interactive Hough Detection - lines: standard, probabailistic - circles
* Uses CUDA

* Based off the examples of canny and sobel edge detection in the path: 
              ECV-ECEE-5763-mcgaffin/computer_vision_cv4_tested/capture-transformer

* Author: Mason McGaffin
* ECV
* Exercise 4 - Question 6
*/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <syslog.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

using namespace cv;
using namespace std;

#define WINDOW_NAME "Interactive Hough Detection"

#define MAX_THRESHOLD 300
#define MAX_LENGTH 100
#define MAX_GAP 100
#define MAX_RADIUS 200

enum State{
    STATE_RESET,
    STATE_S,
    STATE_P,
    STATE_C,
    STATE_GRAY
};

State state;

struct Data {
  Mat src;
//   Mat src_gray;
  cuda::GpuMat gpu_src_gray;
  Mat dst;
  int threshold = 100;
  int length = 10;
  int gap = 10;
  int min_r = 10;
  int max_r = 50;
};

unsigned char trackbar_update;

// Hough Line Transform
void Hough_L(int, void* myData)
{
    cuda::GpuMat gpu_edges, gpu_lines;
    Mat dst;
    vector<Vec2f> lines; // will hold the results of the detection
    vector<Vec4i> linesP;

    if(!(state == STATE_S || state == STATE_P)) return;

    trackbar_update = 1;

    Data* data = static_cast<Data*>(myData);

    int threshold = data->threshold;
    int length = data->length;
    int gap = data->gap;

    // gpu_src.upload(data->src_gray);

    // Edge detection - use CUDA
    Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector(50, 200, 3);
    canny->detect(data->gpu_src_gray, gpu_edges);

    // Canny(data->src, edges, 50, 200, 3); // runs the actual detection

    // Copy edges to the images that will display the results in BGR
    // cuda::cvtColor(gpu_edges, gpu_dst, COLOR_BGR2GRAY);
    // cvtColor(edges, dst, COLOR_GRAY2BGR);

    // hough
    if(state == STATE_S)
    {
        Ptr<cuda::HoughLinesDetector> hough = cv::cuda::createHoughLinesDetector(1.0f, CV_PI / 180, threshold, true);
        hough->detect(gpu_edges, gpu_lines);
        hough->downloadResults(gpu_lines, lines);
    }
    else // STATE_P
    {
        Ptr<cuda::HoughSegmentDetector> houghP = cv::cuda::createHoughSegmentDetector(1.0f, CV_PI / 180, length, gap, 4096, threshold);
        houghP->detect(gpu_edges, gpu_lines);
        gpu_lines.download(linesP);
    }
    
    cuda::cvtColor(gpu_edges, gpu_edges, COLOR_GRAY2BGR); // convert to BGR for display
    gpu_edges.download(dst); // download edges to dst for display
    
    // Draw the lines
    if(state==STATE_S)
    {
        for( size_t i = 0; i < lines.size(); i++ )
        {
            float rho = lines[i][0], theta = lines[i][1];
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
            line( dst, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
        }
    }  
    else
    {
        // Draw the lines
        for( size_t i = 0; i < linesP.size(); i++ )
        {
            Vec4i l = linesP[i];
            line( dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, LINE_AA);
        }
    }

    data->dst = dst;
}

void Hough_C(int, void* myData)
{
    cuda::GpuMat gpu_circles;
    Mat dst;
    vector<Vec3f> circles;
    Data* data = static_cast<Data*>(myData);

    // this was causing a race condition with cuda hough circle code
    // if(data->min_r < 1)
    // {
    //     data->min_r = 1;
    //     setTrackbarPos("Circle: min r", WINDOW_NAME, data->min_r);
    // }
    // if(data->max_r < 1)
    // {
    //     data->max_r = 1;
    //     setTrackbarPos("Circle: max r", WINDOW_NAME, data->max_r);
    // }
    // if(data->min_r > data->max_r) 
    // {
    //     data->min_r = data->max_r;
    //     setTrackbarPos("Circle: min r", WINDOW_NAME, data->min_r);
    // }
    //replace with getting instead and updating data
    if (data->min_r < 1) data->min_r = 1;
    if (data->max_r <= data->min_r) data->max_r = data->min_r + 1;

    //update trackbar
    trackbar_update = 1;

    if (state != STATE_C)
        return;


    int md = data->gpu_src_gray.rows/16; // change this value to detect circles with different distances to each other
    int threshold = data->threshold;
    int min_r = data->min_r;
    int max_r = data->max_r;

    dst = data->src.clone();

    // gpu_src.upload(data->src_gray);

    Ptr<cuda::HoughCirclesDetector> houghCircles = cuda::createHoughCirclesDetector(1, md, threshold, 30, min_r, max_r);
    houghCircles->detect(data->gpu_src_gray, gpu_circles);

    // HoughCircles(data->src_gray, circles, HOUGH_GRADIENT, 1,
    //              data->src_gray.rows/16,  // change this value to detect circles with different distances to each other
    //              100, 30, data->min_r, data->max_r // change the last two parameters // (min_radius & max_radius) to detect larger circles
    // );
    // for( size_t i = 0; i < circles.size(); i++ )
    // {
    //     Vec3i c = circles[i];
    //     Point center = Point(c[0], c[1]);
    //     // circle center
    //     circle( dst, center, 1, Scalar(0,100,100), 3, LINE_AA);
    //     // circle outline
    //     int radius = c[2];
    //     circle( dst, center, radius, Scalar(255,0,255), 3, LINE_AA);
    // }

    // houghCircles->downloadResults(gpu_circles, dst); // this function does not exist for OpenCV 4.10.0
    Mat cpu_circles;
    gpu_circles.download(cpu_circles);

    circles.assign((Vec3f*)cpu_circles.data, (Vec3f*)cpu_circles.data + cpu_circles.cols);

    for( size_t i = 0; i < circles.size(); i++ )
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // circle center
        circle( dst, center, 1, Scalar(0,100,100), 3, LINE_AA);
        // circle outline
        int radius = c[2];
        circle( dst, center, radius, Scalar(255,0,255), 3, LINE_AA);
    }

    data->dst = dst;
}


int main( int argc, char** argv )
{
    Mat image, image_gray, dst;
    cuda::GpuMat gpu_image, gpu_blur;
    State old_state;
    state = STATE_RESET;
    old_state = STATE_GRAY;
    Data data;

    openlog("interactive_hough", LOG_PID, LOG_USER);

    cv::CommandLineParser parser(argc, argv,
                                "{@input   |../Jeep-Sideview.png|input image}"
                                "{live    l|false|live video feed using webcam}"
                                "{help    h|false|show help message}");
    cout << "The sample uses Hough line and circle detection\n\n";
    
    if(parser.get<bool>("help"))
    {
        parser.printMessage();
        return 0;
    }
    
    cout << "\nPress 'ESC' to exit program.\nPress 'C' for Hough Circles.\nPress 'S' for Standard Hough Lines.\nPress 'P' for Probabalistic Hough Lines.\nPress 'R' to see Original Image.\nPress 'G' to see Grayscale Image.\n)";
    
    String imageName = parser.get<String>("@input");
    bool live = parser.get<bool>("live");

    VideoCapture cam0;
    namedWindow(WINDOW_NAME);
    //create gaussian filter on GPU
    Ptr<cuda::Filter> gaussFilter = cuda::createGaussianFilter(CV_8UC3, CV_8UC3, Size(3, 3), 0, 0, BORDER_DEFAULT);

    if(live)
    {
        syslog(LOG_INFO, "Interactive Hough Detection started using live webcam.");

        cam0.open(0);
        if (!cam0.isOpened())
        {
            syslog(LOG_CRIT, "Error opening web camera.\n");
            perror("webcam");

            return 1;
        }

        cam0.set(CAP_PROP_FRAME_WIDTH, 640);
        cam0.set(CAP_PROP_FRAME_HEIGHT, 480);
    }
    else{
        syslog(LOG_INFO, "Interactive Hough Detection started using static image: %s", imageName.c_str());
        // As usual we load our source image (src)
        image = imread( imageName, IMREAD_COLOR ); // Load an image
        // Check if image is loaded fine
        if( image.empty() )
        {
            syslog(LOG_CRIT, "Error opening image: %s\n", imageName.c_str());
            perror("input image");

            return 1;
        }

        gpu_image.upload(image);

        // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
        // GaussianBlur(image, image_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);
        gaussFilter->apply(gpu_image, gpu_blur);

        // Convert the image to grayscale
        // cvtColor(image_gray, image_gray, COLOR_BGR2GRAY);
        cuda::cvtColor(gpu_blur, data.gpu_src_gray, COLOR_BGR2GRAY);

        data.gpu_src_gray.download(image_gray);

        data.src = image;
    }

    //create hough line trackbar
    createTrackbar( "Line/Circle threshold:", WINDOW_NAME, &data.threshold, MAX_THRESHOLD, Hough_L, &data);
    setTrackbarMin("Line/Circle threshold:", WINDOW_NAME, 1);

    createTrackbar( "LineP length:", WINDOW_NAME, &data.length, MAX_LENGTH, Hough_L, &data);
    createTrackbar( "LineP gap:", WINDOW_NAME, &data.gap, MAX_GAP, Hough_L, &data);

    //create hough circletrackbar
    createTrackbar("Circle: min r", WINDOW_NAME, &data.min_r, MAX_RADIUS, Hough_C, &data);
    createTrackbar("Circle: max r", WINDOW_NAME, &data.max_r, MAX_RADIUS, Hough_C, &data);
    
    Hough_L(0, &data);
    Hough_C(0, &data);

    for (;;)
    {
        if(live)
        {
            cam0.read(image);
            // // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
            // GaussianBlur(image, image_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);
            // // Convert the image to grayscale
            // cvtColor(image_gray, image_gray, COLOR_BGR2GRAY);

            // data.src = image;
            // data.src_gray = image_gray;
            
            gpu_image.upload(image);

            // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
            // GaussianBlur(image, image_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);
            gaussFilter->apply(gpu_image, gpu_blur);

            // Convert the image to grayscale
            // cvtColor(image_gray, image_gray, COLOR_BGR2GRAY);
            cuda::cvtColor(gpu_blur, data.gpu_src_gray, COLOR_BGR2GRAY);

            data.gpu_src_gray.download(image_gray);

            data.src = image;
        }

        if(live || (state != old_state) || trackbar_update)
        {
            old_state = state;

            setTrackbarPos("Circle: min r", WINDOW_NAME, data.min_r);
            setTrackbarPos("Circle: max r", WINDOW_NAME, data.max_r);

            switch(state)
            {
            case STATE_S: case STATE_P:
                Hough_L(0, &data);
                dst = data.dst;
            
                break;

            case STATE_C:
                Hough_C(0, &data);
                dst = data.dst;

                break;
            
            case STATE_GRAY:
                dst = image_gray;

                break;

            case STATE_RESET: default:
                dst = image;

                break;
            }
            trackbar_update = 0;
        }

        imshow(WINDOW_NAME, dst);

        char key = (char)waitKey(10);
        if(key == 27) //esc
        {
            syslog(LOG_INFO, "Closing application...\n");
            break;
        }
        if ((key == 'c') || (key == 'C'))
        {
            state = STATE_C;
            syslog(LOG_INFO, "Key pressed...Hough Circle Detection\n");
        }
        if ((key == 's') || (key == 's'))
        {
            state = STATE_S;
            syslog(LOG_INFO, "Key pressed...Hough Standard Line Detection\n");
        }
        if ((key == 'p') || (key == 'P'))
        {
            state = STATE_P;
            syslog(LOG_INFO, "Key pressed...Hough Probabalistic (Segment) Line Detection\n");
        }
        if ((key == 'g') || (key == 'G'))
        {
            state = STATE_GRAY;
            syslog(LOG_INFO, "Key pressed...Grayscale Mode\n");
        }
        if ((key == 'r') || (key == 'R')) // reset - none
        {
            state = STATE_RESET;
            syslog(LOG_INFO, "Key pressed...Original Image\n");
        }
    }

    //cleanup
    destroyWindow(WINDOW_NAME);

    return 0;
}
