
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <syslog.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat frame, frame_bands[3], band_prev, band_diff;

    cv::CommandLineParser parser(argc, argv,
                                "{@input   |Dark-Room-Laser-Spot.mpeg|input video}"
                                "{band    b|1|band (default green): R=0, G=1, B=2}"
                                "{help    h|false|show help message}");
    cout << "The sample removes the background of the input video only keeping the primary moving foreground object\n\n";
    
    if(parser.get<bool>("help"))
    {
        parser.printMessage();
        return 0;
    }

    String input = parser.get<String>(0);
    int band = parser.get<int>("band");

    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }
    
    input = samples::findFileOrKeep(input);
    cout << "Input file: " << input <<endl;

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

    size_t idx = input.find_last_of(".");
    string input_strip = input.substr(0, idx);
    int count = 0;
    int buf_size = 25 + input_strip.length();
    char file_name[buf_size];

    while(1)
    {
        if (frame.empty()) break;

        split(frame, frame_bands);

        snprintf(file_name, buf_size, "frames/gray_%s_%06d.pgm", input_strip.c_str(), count);
        count++;

        imwrite(file_name, frame_bands[2-band]);

        cap >> frame;
    }

    cout << "Done" << endl;

    return 0;
}