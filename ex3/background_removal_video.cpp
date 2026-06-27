/*
References: 
    https://o365coloradoedu-my.sharepoint.com/shared?listurl=%2Fpersonal%2Fsiewerts%5Fcolorado%5Fedu%2FDocuments&viewid=352a79ce%2Ddcf4%2D42ac%2D896f%2D903d069f746d&csf=1&FolderCTID=0x0120000AC3DE8CB97B0647B9FAC11DCE7EAC00&id=%2Fpersonal%2Fsiewerts%5Fcolorado%5Fedu%2FDocuments%2FESEE%2DWeb%2DResources%2FECV%2D5763%2DCU%2Fcode%2Fdiff%2Dinteractive%2Fcapture%2Ecpp&parent=%2Fpersonal%2Fsiewerts%5Fcolorado%5Fedu%2FDocuments%2FESEE%2DWeb%2DResources%2FECV%2D5763%2DCU%2Fcode%2Fdiff%2Dinteractive
    https://docs.opencv.org/4.8.0/df/d94/samples_2cpp_2videowriter_basic_8cpp-example.html#a9

Process the video, Dark-Room-Laser-Spot-with-Clutter.mpeg and use color frame differencing for R,G & B to remove the bookshelf background
and preserve the moving laser spot foreground. 

Author: Mason McGaffin
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

int main(int argc, char** argv)
{
    Mat frame, frame_prev, frame_diff;

    cv::CommandLineParser parser(argc, argv,
                                "{@input   |./Dark-Room-Laser-Spot.mpeg|input video}"
                                "{@output  ||output video}"
                                "{help    h|false|show help message}");
    cout << "The sample removes the background of the input video only keeping the primary moving foreground object\n\n";
    
    if(parser.get<bool>("help"))
    {
        parser.printMessage();
        return 0;
    }

    String input = parser.get<String>(0);
    String output = parser.get<String>(1);

    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    input = samples::findFileOrKeep(input);
    cout << "Input file: " << input <<endl;

    bool file_out = false;
    if (!output.empty())
    {
        file_out = true;
        cout << "Output file: " << output <<endl;
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
    if(file_out)
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

    frame_prev = frame.clone();
    
    for (;;)
    {
        cap >> frame;
        if (frame.empty())
        {
            cout << "Finished reading video: '" << input << "'" << endl;
            break;
        }

        absdiff(frame_prev, frame, frame_diff);

        frame.copyTo(frame_prev);

        if(file_out)
        {
            writer.write(frame_diff);
        }
        else //live
        {
            imshow("Original", frame);
            imshow("Diff", frame_diff);

            char c = (char)waitKey(1);
            if(c == 27) //esc
            {
                cout << "Stopping application" << endl;
                break;
            }
        }
    }

    return 0;
}
