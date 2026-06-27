/**
* Interactive Hough Detection - lines: standard, probabailistic - circles
* Based off the examples of canny and sobel edge detection in the path: 
              ECV-ECEE-5763-mcgaffin/computer_vision_cv4_tested/capture-transformer

* Author: Mason McGaffin
* ECV
* Exercise 3 - Question 2/3

Starting with the OpenCV example for Hough Lines, adapt the code so you can compute 
Hough lines for a continuous camera stream (houghlines.cpp) and refer to class example
for OpenCV 3.x as it may help you too, but you will have to update the code to get 
it to build and run (simple-hough-interactive/). Capture an image showing detection
of lines for an object you hold up to the camera. What could you use this for?

Starting with the OpenCV example for Hough Circles, adapt the code so you can compute
Hough circles for a continuous camera stream (houghcircles.cpp) and refer to class 
example for OpenCV 3.x as it may help you too if you update it 
(simple-hough-ellipticalinteractive/). Capture an image showing detection of circles 
for an object you hold up to the camera. What could you use this for?

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

struct HL_Data {
  Mat src;
  Mat dst;
  int threshold = 100;
  int length = 10;
  int gap = 10;
};

struct HC_Data {
  Mat src;
  Mat src_gray;
  Mat dst;
  int min_r = 10;
  int max_r = 50;
};

unsigned char trackbar_update;

// Hough Line Transform
void Hough_L(int, void* myData)
{
    Mat edges, dst;
    vector<Vec2f> lines; // will hold the results of the detection
    vector<Vec4i> linesP;

    if((state != STATE_S) && (state != STATE_P))
        return;

    trackbar_update = 1;

    HL_Data* data = static_cast<HL_Data*>(myData);

    // Edge detection
    Canny(data->src, edges, 50, 200, 3);

    // Copy edges to the images that will display the results in BGR
    cvtColor(edges, dst, COLOR_GRAY2BGR);
    
    // Draw the lines
    if(state==STATE_S)
    {
        // Standard Hough Line Transform
        HoughLines(edges, lines, 1, CV_PI/180, data->threshold, 0, 0 ); // runs the actual detection

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
        HoughLinesP(edges, linesP, 1, CV_PI/180, data->threshold, data->length, data->gap ); // runs the actual detection
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
    vector<Vec3f> circles;
    HC_Data* data = static_cast<HC_Data*>(myData);

    if(data->min_r < 1)
    {
        data->min_r = 1;
        setTrackbarPos("Circle: min r", WINDOW_NAME, data->min_r);
    }
    if(data->max_r < 1)
    {
        data->max_r = 1;
        setTrackbarPos("Circle: max r", WINDOW_NAME, data->max_r);
    }
    if(data->min_r > data->max_r) 
    {
        data->min_r = data->max_r;
        setTrackbarPos("Circle: min r", WINDOW_NAME, data->min_r);
    }

    if (state != STATE_C)
        return;

    trackbar_update = 1;



    Mat dst = data->src.clone();

    HoughCircles(data->src_gray, circles, HOUGH_GRADIENT, 1,
                 data->src_gray.rows/16,  // change this value to detect circles with different distances to each other
                 100, 30, data->min_r, data->max_r // change the last two parameters // (min_radius & max_radius) to detect larger circles
    );
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
    State old_state;
    state = STATE_RESET;
    old_state = STATE_GRAY;
    HL_Data hl_data;
    HC_Data hc_data;

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
        // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
        GaussianBlur(image, image_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);
        // Convert the image to grayscale
        cvtColor(image_gray, image_gray, COLOR_BGR2GRAY);

        hl_data.src = image;
        hc_data.src = image;
        hc_data.src_gray = image_gray;
    }

    //create hough line trackbar
    createTrackbar( "Line(P) threshold:", WINDOW_NAME, &hl_data.threshold, MAX_THRESHOLD, Hough_L, &hl_data);
    createTrackbar( "LineP length:", WINDOW_NAME, &hl_data.length, MAX_LENGTH, Hough_L, &hl_data);
    createTrackbar( "LineP gap:", WINDOW_NAME, &hl_data.gap, MAX_GAP, Hough_L, &hl_data);

    //create hough circletrackbar
    createTrackbar("Circle: min r", WINDOW_NAME, &hc_data.min_r, MAX_RADIUS, Hough_C, &hc_data);
    createTrackbar("Circle: max r", WINDOW_NAME, &hc_data.max_r, MAX_RADIUS, Hough_C, &hc_data);
    
    Hough_L(0, &hl_data);
    Hough_C(0, &hc_data);

    for (;;)
    {
        if(live)
        {
            cam0.read(image);
            // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
            GaussianBlur(image, image_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);
            // Convert the image to grayscale
            cvtColor(image_gray, image_gray, COLOR_BGR2GRAY);

            hl_data.src = image;
            hc_data.src = image;
            hc_data.src_gray = image_gray;
        }

        if(live || (state != old_state) || trackbar_update)
        {
            old_state = state;
            trackbar_update = 0;

            switch(state)
            {
            case STATE_S: case STATE_P:
                Hough_L(0, &hl_data);
                dst = hl_data.dst;
            
                break;

            case STATE_C:
                Hough_C(0, &hc_data);
                dst = hc_data.dst;

                break;
            
            case STATE_GRAY:
                dst = image_gray;

                break;

            case STATE_RESET: default:
                dst = image;

                break;
            }
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
        if ((key == 'p') || (key == 'P'))
        {
            state = STATE_P;
            syslog(LOG_INFO, "Key pressed...Hough Probabistic Line Detection\n");
        }
        if ((key == 's') || (key == 'S'))
        {
            state = STATE_S;
            syslog(LOG_INFO, "Key pressed...Hough Standard Line Detection\n");
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
