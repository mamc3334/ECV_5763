
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
                                "{@output  ||output video}"
                                "{band    b|1|band (default green): R=0, G=1, B=2}"
                                "{help    h|false|show help message}");
    cout << "The sample removes the background of the input video only keeping the primary moving foreground object\n\n";
    
    if(parser.get<bool>("help"))
    {
        parser.printMessage();
        return 0;
    }

    String input = parser.get<String>(0);
    String output = parser.get<String>(1);
    int band = parser.get<int>("band");

    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    
    input = samples::findFileOrKeep(input);
    cout << "Input file: " << input <<endl;

    if (output.empty())
    {
        size_t idx = input.find_last_of(".");
        output = "cv_ob_" + input.substr(0, idx) + ".mp4";
    }
    cout << "Output file: " << output <<endl;

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
    int codec = VideoWriter::fourcc('m', 'p', '4', 'v');  // select desired codec (must be available at runtime)
    double fps = cap.get(CAP_PROP_FPS); // framerate of the created video stream
    writer.open(output, codec, fps, frame.size(), false);
    // check if we succeeded
    if (!writer.isOpened()) {
        cerr << "Could not open the output video file:" << output << endl;
        return -1;
    }

    while(1)
    {
        if (frame.empty()) break;

        split(frame, frame_bands);

        writer.write(frame_bands[2-band]);

        cap >> frame;
    }

    cout << "Done" << endl;

    return 0;
}