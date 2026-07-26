
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <syslog.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

#define CANNY_RATIO 3
#define KERNEL_SIZE 3
#define THRESHOLD 220

struct COM{
    int x;
    int y;
};

void getCOM(COM *com, Mat frame)
{
    int xMin = INT_MAX;
    int xMax = -1;
    int yMin = INT_MAX;
    int yMax = -1;

    for(int r=0; r < frame.rows; r++)
    {
        for(int c=0; c < frame.cols; c++)
        {
            unsigned char val = frame.at<unsigned char>(r, c);
            if(THRESHOLD < val)
            {
                xMin = min(xMin, c);
                xMax = max(xMax, c);
                yMin = min(yMin, r);
                yMax = max(yMax, r);
            }
        }
    }

    if(xMax == -1 || xMin == INT_MAX) // pick any - if original state - invalid
        return;

    com->x = (xMin+xMax)/2;
    com->y = (yMin+yMax)/2;
}

void drawCx(Mat frame, COM *com)
{    
    int xS = max(com->x - 100, 0);
    int xE = min(com->x + 100, frame.cols);
    int yS = max(com->y - 100, 0);
    int yE = min(com->y + 100, frame.rows);
    
    line(frame, Point(xS, com->y), Point(xE, com->y), Scalar(255,255,255), 1);
    line(frame, Point(com->x, yS), Point(com->x, yE), Scalar(255,255,255), 1);
}

int main(int argc, char** argv)
{
    Mat frame, prev, diff, frame_bands[3];
    COM com;

    cv::CommandLineParser parser(argc, argv,
                                "{@input   |Dark-Room-Laser-Spot.mpeg|input video}"
                                "{@output  ||output video}"
                                "{live    l|false|live video feed}"
                                "{band    b|0|band(defaut red): R=0, G=1, B=2}"
                                "{help    h|false|show help message}");
    cout << "The sample removes the background of the input video only keeping the primary moving foreground object\n\n";
    
    if(parser.get<bool>("help"))
    {
        parser.printMessage();
        return 0;
    }

    String input = parser.get<String>(0);
    String output = parser.get<String>(1);
    bool live = parser.get<bool>("live");
    int band = parser.get<int>("band");

    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    
    input = samples::findFileOrKeep(input);
    cout << "Input file: " << input <<endl;

    if (!live)
    {
        if(output.empty())
        {
            size_t idx = input.find_last_of(".");
            output = "com_" + input.substr(0, idx) + ".mp4";
        }
        cout << "Output file: " << output <<endl;
    }

    if(band>2)
    {
        cout << "Invalid Color band" <<endl;
        return -1;
    }

    //open input
    VideoCapture cap(input);
    if (!cap.isOpened())
    {
        cerr << "Can not open input video: '" <<  input << "'" << endl;
        return 2;
    }
    
    //set up initial frame
    cap >> frame;
    if (frame.empty())
    {
        cerr << "Empty input video: '" << input << "'" << endl;
        return 3;
    }

    //setup output
    VideoWriter writer;
    if(!live)
    {
        int codec = VideoWriter::fourcc('m', 'p', '4', 'v');  // select desired codec (must be available at runtime)
        double fps = cap.get(CAP_PROP_FPS); // framerate of the created video stream
        writer.open(output, codec, fps, frame.size(), true);
        // check if we succeeded
        if (!writer.isOpened()) {
            cerr << "Could not open the output video file:" << output << endl;
            return -1;
        }
    }

    prev = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    while(1)
    {
        if (frame.empty()) break;

        split(frame, frame_bands);

        // imwrite("frame_diff.pgm", diff);
        // break;

        absdiff(frame_bands[2-band], prev, diff);
        frame_bands[2-band].copyTo(prev);

        // /// Reduce noise with a kernel 3x3
        // blur(diff, diff, Size(KERNEL_SIZE,KERNEL_SIZE));

        /// Canny detector
        // Canny( diff, diff, THRESHOLD, THRESHOLD*CANNY_RATIO, KERNEL_SIZE );

        // Sobel(diff, grad_x, CV_16S, 1, 0, 1, 1, 0, BORDER_DEFAULT);
        // Sobel(diff, grad_y, CV_16S, 0, 1, 1, 1, 0, BORDER_DEFAULT);
        // // converting back to CV_8U
        // convertScaleAbs(grad_x, abs_grad_x);
        // convertScaleAbs(grad_y, abs_grad_y);
        // addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, diff);
        
        //compute COM based on grayband
        getCOM(&com, diff);

        //draw COM cross-hair on original image
        drawCx(frame, &com);

        if(live)
        {
            imshow("Diff", diff);
            imshow("COM Crosshairs", frame);
            char c = (char)waitKey(1);
            if (c==27) break; //esc
        }
        else
        {
            writer.write(frame); //BGR
        }

        cap >> frame;
    }

    cout << "Done" << endl;

    return 0;
}