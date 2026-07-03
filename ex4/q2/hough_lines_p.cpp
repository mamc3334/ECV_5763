/**
* Hough lane detection - probabailistic hough lines detection
* Based off the examples of canny and sobel edge detection in the path: 
              ECV-ECEE-5763-mcgaffin/computer_vision_cv4_tested/capture-transformer
              ECV_5763/ex3/interactive-hough.cpp

* Author: Mason McGaffin
* ECV
* Exercise 4 - Question 2

Build and run the basic Hough lines example provided by OpenCV and adapt it for use with 
a camera or video source with the ability to specify the source as camera or video on the 
command line. Determine the best method (HoughLines, HoughLinesP) and parameters to use 
to find the edges of a roadway from the one of the following challenge set videos – 
Self-Driving-Car-Video-1, Self-Driving-Car-Video-2, or Colorado-Challenge-Set. Paste a 
transformed image showing your best detection of roadway edges and lanes in your report.
*/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

#define WINDOW_NAME "Hough Line Detection"

// Parameter limits and defaults
#define MIN_THRESHOLD 1
#define MIN_LENGTH 0
#define MIN_GAP 0
#define MIN_RHO 1
#define MAX_RHO 200
#define MIN_THETA (1e-5)
#define MAX_THETA (CV_PI/2)

#define DEFAULT_THRESHOLD 15
#define DEFAULT_LENGTH 10
#define DEFAULT_GAP 10
#define DEFAULT_RHO 1
#define DEFAULT_THETA (CV_PI/180)

//tmp
int cannyLow = 30;
int cannyHigh = 100;

int thresh = 0;
int length = 0;
int gap = 0;
int rho = 0;
double theta = 0.0;

// draw the left and right lanes
void drawAverageLine(const int y_top, const int y_bot, const vector<double>& slopes, const vector<double>& intercepts, Mat &dst)
{
    if (slopes.empty())
        return;

    double sum_m = 0.0;
    double sum_b = 0.0;
    for (size_t i = 0; i < slopes.size(); ++i)
    {
        sum_m += slopes[i];
        sum_b += intercepts[i];
    }

    double avg_m = sum_m / slopes.size();
    double avg_b = sum_b / intercepts.size();

    int x1 = cvRound((y_bot - avg_b) / avg_m); // x = (y-b)/m
    int x2 = cvRound((y_top - avg_b) / avg_m); 

    line(dst, Point(x1, y_bot), Point(x2, y_top), Scalar(0, 255, 0), 6, LINE_AA);
};

// Hough Line Transform
void HoughLines_P(Mat src, Mat &dst)
{
    Mat gray, edges;
    vector<Vec4i> linesP;

    // Gray and Blur the image
    cvtColor(src, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(11, 11), 0, 0, BORDER_DEFAULT);
    // imshow("blur", gray);

    // Edge detection
    Canny(gray, edges, cannyLow, cannyHigh, 3);

    //only care about edges in area of interest
    Mat mask = Mat::zeros(edges.size(), edges.type());
    vector<Point> roi_poly;

    // define poly as trap in lower screen
    roi_poly.push_back(Point(edges.cols *0.2, edges.rows * 0.9)); // Bottom Left
    roi_poly.push_back(Point(edges.cols *0.4, edges.rows * 0.6)); // Top Left
    roi_poly.push_back(Point(edges.cols *0.6, edges.rows * 0.6)); // Top Right
    roi_poly.push_back(Point(edges.cols *0.8, edges.rows * 0.9)); // Bottom Left

    fillPoly(mask, roi_poly, Scalar(255));
    bitwise_and(edges, mask, edges);
    // imshow("edges", edges);

    dst = src.clone();
    
    HoughLinesP(edges, linesP, rho, theta, thresh, length, gap); // runs the actual detection

    // filter the lines
    vector<double> left_slopes;
    vector<double> left_intercepts;
    vector<double> right_slopes;
    vector<double> right_intercepts;

    int mid_x = edges.cols / 2;

    for( size_t i = 0; i < linesP.size(); i++ )
    {
        Vec4i l = linesP[i];
        double m;
        if (l[0] == l[2]) // vertical lines - avoid divide by zero
            m = 1e6;
        else
            m = double(l[3] - l[1]) / double(l[2] - l[0]);
    
        if (fabs(m) < 0.2)
            continue; // skip flat lines

        double x_mid = double(l[0] + l[2]) / 2.0;
        double b = l[1] - m * l[0]; // b=y-mx

        if (x_mid < mid_x && m < 0)
        {
            left_slopes.push_back(m);
            left_intercepts.push_back(b);
        }
        else if (x_mid > mid_x && m > 0)
        {
            right_slopes.push_back(m);
            right_intercepts.push_back(b);
        }
    }

    // average the lines and draw them
    drawAverageLine(edges.rows * 0.5, edges.rows, left_slopes, left_intercepts, dst);
    drawAverageLine(edges.rows * 0.5, edges.rows, right_slopes, right_intercepts, dst);
}

bool validParams(int threshold, int length, int gap, int rho, double theta)
{
    if(threshold < MIN_THRESHOLD)
    {
        cerr << "Threshold: Got: " << threshold << ". Must be greater than or equal to " << MIN_THRESHOLD << endl;
        return false;
    }
    if(length < MIN_LENGTH)
    {
        cerr << "Length: Got: " << length << ". Must be greater than or equal to " << MIN_LENGTH << endl;
        return false;
    }
    if(gap < MIN_GAP)
    {
        cerr << "Gap: Got: " << gap << ". Must be greater than or equal to " << MIN_GAP << endl;
        return false;
    }
    if(rho < MIN_RHO || rho > MAX_RHO)
    {
        cerr << "Rho: Got: " << rho << ". Must be between " << MIN_RHO << " and " << MAX_RHO << endl;
        return false;
    }
    if(theta < MIN_THETA || theta > MAX_THETA)
    {
        cerr << "Theta: Got: " << theta << ". Must be between " << MIN_THETA << " and " << MAX_THETA << endl;
        return false;
    }

    return true;
}


int main( int argc, char** argv )
{
    Mat image, dst;

    cv::CommandLineParser parser(argc, argv,
                                "{@input   |video/02260001_copy.mp4|input video}"
                                "{webcam    w|false|live video feed using webcam}"
                                "{cannyLow  c|30|Canny low threshold}"
                                "{cannyHigh  C|100|Canny high threshold}"
                                "{threshold t|15|Hough threshold}"
                                "{length   l|10|Hough line length}"
                                "{gap      g|10|Hough line gap}"
                                "{rho      r|1|Hough line rho}"
                                "{theta    T| <CV_PI/180> |Hough line theta}"
                                "{help    h|false|show help message}");
    cout << "The sample uses Hough line detection. If a video input is specified, then the modified video will saved as hough_<input_file>.mp4\n\n";
    
    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    if(parser.get<bool>("help"))
    {
        parser.printMessage();
        return 0;
    }
    
    String fileName = parser.get<String>("@input");

    //used google search to extract base name from file path - i changed it to use mp4
    String baseName = fileName;
    size_t pos_slash = fileName.find_last_of("/\\");
    if (pos_slash != String::npos)
    {
        baseName = fileName.substr(pos_slash + 1);
    }
    size_t pos_dot = baseName.find_last_of(".");
    baseName = baseName.substr(0, pos_dot);
    String outfileName = "hough_" + baseName + ".mp4";
    
    bool live = parser.get<bool>("webcam");

    // Get canny params from command line
    cannyLow = parser.get<int>("cannyLow");
    cannyHigh = parser.get<int>("cannyHigh");

    // Get Hough parameters from command line
    thresh = parser.get<int>("threshold");
    length = parser.get<int>("length");
    gap = parser.get<int>("gap");
    rho = parser.get<int>("rho");
    theta = parser.get<double>("theta");
    if(!theta) // if theta is 0, then set it to CV_PI/180
    {
        theta = DEFAULT_THETA;
    }

    if(!validParams(thresh, length, gap, rho, theta))
    {
        cerr << "Invalid Hough parameters specified.\n";
        parser.printMessage();
        return -1;
    }

    cout << "Canny Parameters: \n\tlow threshold=" << cannyLow << "\n\thighThreshold=" << cannyHigh <<endl;
    cout << "Hough Parameters: \n\tthreshold=" << thresh << "\n\tlength=" << length << "\n\tgap=" << gap << "\n\trho=" << rho << "\n\ttheta=" << theta << endl;

    VideoCapture cam0;
    VideoWriter writer;
    namedWindow(WINDOW_NAME);

    //tmp
    // createTrackbar("Canny Low", WINDOW_NAME, &cannyLow, 255);
    // createTrackbar("Canny High", WINDOW_NAME, &cannyHigh, 255);

    // createTrackbar("Rho", WINDOW_NAME, &rho, 255);
    // createTrackbar("Threshold", WINDOW_NAME, &thresh, 255);
    // createTrackbar("Length", WINDOW_NAME, &length, 255);
    // createTrackbar("Gap", WINDOW_NAME, &gap, 255);

    if(!fileName.empty())
    {
        fileName = "video/02260001_copy.mp4";
    }

    cout << "Hough Line Detection started using video input: " << fileName.c_str() << endl;

    cam0.open(fileName);
    if (!cam0.isOpened())
    {
        cerr << "Error opening video: " << fileName.c_str() << endl;
        perror("input video");

        return 1;
    }

    cam0.read(image);
    if (image.empty())
    {
        cerr << "Error reading first frame of video:  " << fileName.c_str() << endl;
        perror("input video");
        
        return 1;
    }

    int codec = VideoWriter::fourcc('m', 'p', '4', 'v');  // select desired codec (must be available at runtime)
    double fps = cam0.get(CAP_PROP_FPS); // framerate of the created video stream
    cout << "fps: " << fps <<endl;
    writer.open(outfileName, codec, fps, image.size(), true);

    if (!writer.isOpened()) 
    {
        cerr << "Could not open the output video file: " << outfileName.c_str() << endl;
        perror("output video");

        return -1;
    }

    for (;;)
    {
        if (image.empty())
        {
            cout <<  "End of video stream or error reading frame.\n";
            break;
        }

        // HoughLines_P(image, dst, threshold, length, gap, rho, theta);
        HoughLines_P(image, dst);

        if(live)
        {
            imshow(WINDOW_NAME, dst);

            char key = (char)waitKey(1);
            if(key == 27) //esc
            {
                cout << "ESC Pressed. Closing application...\n";
                break;
            }
        }
        else
        {
            // imshow(WINDOW_NAME, dst);
            // char key = (char)waitKey(1);
            // if(key == 27) //esc
            // {
            //     cout << "ESC Pressed. Closing application...\n";
            //     break;
            // }

            writer.write(dst);
        }

        cam0.read(image); //read at end so first frame is processed
    }

    //cleanup
    destroyWindow(WINDOW_NAME);

    return 0;
}
