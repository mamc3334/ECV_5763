/**
* Skeletal Transform - probabailistic hough lines detection
* Based off the example: 
    https://o365coloradoedu-my.sharepoint.com/shared?listurl=https%3A%2F%2Fo365coloradoedu%2Dmy%2Esharepoint%2Ecom%2Fpersonal%2Fsiewerts%5Fcolorado%5Fedu%2FDocuments&id=%2Fpersonal%2Fsiewerts%5Fcolorado%5Fedu%2FDocuments%2FESEE%2DWeb%2DResources%2FECV%2D5763%2DCU%2Fcode%2Fva%2Dopencv%2Dexamples%2Fcapture%2Dtransformer%2Fskeletal%2Ecpp&parent=%2Fpersonal%2Fsiewerts%5Fcolorado%5Fedu%2FDocuments%2FESEE%2DWeb%2DResources%2FECV%2D5763%2DCU%2Fcode%2Fva%2Dopencv%2Dexamples%2Fcapture%2Dtransformer
    ECV_5763/ECV-ECEE-5763-mcgaffin/va-opencv-examples/capture-transformer/skeletal.cpp
    
* Author: Mason McGaffin
* ECV
* Exercise 4 - Question 3

Using a top-down OpenCV approach, adapt the example code for OpenCV version used
in class based on examples found in capture_transformer to use the skeletal.cpp
transform on continuous frames (like captureskel.cpp) from your camera, but use
the much simpler approach of simpler-capture rather than V4L2 camera capture. 
Gesture in front or your camera and see if you can get a reasonable continuous 
skeletal transform of your arm and hand, holding up 1, 2, 3, 4, or 5 fingers to 
see if you could distinguish each and count fingers held up. Record example frames 
(at least 300 frame and no more than 3000 for 10 to 100 seconds of video – JPEG 
frames are fine) and encode your results to an MPEG video. Upload the modified 
code with your report.
*/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <syslog.h>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "count_finger.hpp"

using namespace cv;
using namespace std;

// Parameter limits and defaults

//global for max frames to capture, live for contiunours interactive gui
int maxFrames = 300;
bool live = false;
bool debug = false;

// Hough Line Transform
void skeletal(Mat src, Mat &bin, Mat &dst)
{
    Mat gray, binary, mfblur, hist;

    // Gray and Blur the image
    // medianBlur(binary, mfblur, 3);
    GaussianBlur(src, gray, Size(7, 7), 0, 0, BORDER_DEFAULT);
    cvtColor(gray, gray, COLOR_BGR2GRAY);
    
    if (debug) imshow("Gray", gray);

    int threshVal = 100; //original param

    // Compute a histogram for grayscale intensities
    // Choose threshold at the first peak
    // int histSize = 256;
    // float range[] = { 0, 256 };
    // const float* ranges[] = { range };
    // calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

    // for (int i = 11; i < histSize - 1; i++) //skip first 10 bins to avoid noise
    // {
    //     float curr = hist.at<float>(i);
    //     if ((curr > hist.at<float>(i-1)) && (curr > hist.at<float>(i+1))) //first peak
    //     {
    //         threshVal = i;
    //         break;
    //     }
    // }

    // cout << "Threshold: " << threshVal << endl;

    threshold(gray, binary, threshVal, 255, THRESH_BINARY);
    binary = 255 - binary;

    if (debug) imshow("Binary", binary);

    bin = binary.clone();

    binary.copyTo(mfblur); // no blur

    if (debug) imshow("Median Blur", mfblur);


    // Skeletonization
    Mat skel(mfblur.size(), CV_8UC1, Scalar(0));
    Mat temp;
    Mat eroded;
    Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));

    bool done;
    int iter = 0;
    do
    {
        erode(mfblur, eroded, element);
        dilate(eroded, temp, element); // temp = open(mfblur)
        subtract(mfblur, temp, temp);
        bitwise_or(skel, temp, skel);
        eroded.copyTo(mfblur);

        done = (countNonZero(mfblur) == 0);
        iter++;
    } while (!done && iter < 100);

    dst = skel.clone();
}


int main( int argc, char** argv )
{
    Mat image, dst;

    openlog("Skeletal Transform", LOG_PID, LOG_USER);

    cv::CommandLineParser parser(argc, argv,
                                "{input   i|""|input video}"
                                "{webcam  w|false|live video feed using webcam output to mp4 video up to -f frames}"
                                "{live    l|false|Continuous live video feed using webcam - interactive gui}"
                                "{frames  f|300|Max frames to capture - from input or live webcam}"
                                "{debug   d|false|Debug mode - show intermediate images}"
                                "{help    h|false|show help message}");
    cout << "The sample uses Skeletal Transform. \nIf a video input is specified using the -i flag, then the modified video up to -f frames will be saved as skeletal <input_file>.mp4\n\n";
    
    if(parser.get<bool>("help"))
    {
        parser.printMessage();
        return 0;
    }

    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }
    
    String fileName = parser.get<String>("input");
    String outfileName, frameDir;

    if(!fileName.empty())
    {
        //used google search to extract base name from file path - i changed it to use mp4
        String baseName = fileName;
        size_t pos_slash = fileName.find_last_of("/\\");
        if (pos_slash != String::npos)
        {
            baseName = fileName.substr(pos_slash + 1);
        }
        size_t pos_dot = baseName.find_last_of(".");
        baseName = baseName.substr(0, pos_dot);
        outfileName = "skeletal_" + baseName + ".mp4";

        frameDir = "skeletal_" + baseName + "_frames";
    }

    maxFrames = parser.get<int>("frames");
    
    bool webcam = parser.get<bool>("webcam");
    live = parser.get<bool>("live");
    debug = parser.get<bool>("debug");

    VideoCapture cam0;
    VideoWriter writer;

    // cout << cv::getBuildInformation() << endl;

    if(live || webcam)
    {
        if(!fileName.empty())
        {
            cout << "Ignoring input video file, using live webcam feed instead.\n";
        }

        if(webcam)
        {
            syslog(LOG_INFO, "Skeletal Transform started using live webcam to mp4 video up to %d frames.", maxFrames);
            live = false; // webcam mode is not interactive gui, just capture to mp4

            outfileName = "skeletal_webcam.mp4";
            frameDir = "skeletal_webcam_frames";
        }
        else
        {
            syslog(LOG_INFO, "Interactive Skeletal Transform started using live webcam.");
        }

        cam0.open(0, cv::CAP_V4L2);
        if (!cam0.isOpened())
        {
            syslog(LOG_CRIT, "Error opening web camera.\n");
            perror("webcam");

            return 1;
        }

        cam0.set(CAP_PROP_FRAME_WIDTH, 640);
        cam0.set(CAP_PROP_FRAME_HEIGHT, 480);
        cam0.set(cv::CAP_PROP_FPS, 15); //30 was too fast

        cout << "Webcam properties: \n\tWidth=" << cam0.get(CAP_PROP_FRAME_WIDTH) << "\n\tHeight=" << cam0.get(CAP_PROP_FRAME_HEIGHT) << "\n\tFPS=" << cam0.get(CAP_PROP_FPS) << endl;
    }
    else{
        syslog(LOG_INFO, "Skeletal Transform started using video input: %s", fileName.c_str());

        cout << "Input file: " << fileName <<endl;
        cam0.open(fileName);
        if (!cam0.isOpened())
        {
            syslog(LOG_CRIT, "Error opening video: %s\n", fileName.c_str());
            perror("input video");

            return 1;
        }
    }

    //read first frame
    cam0.read(image);

    // create writer
    if(!live)
    {
        // create output directory for frames
        if (!std::filesystem::exists(frameDir))
        {
            std::filesystem::create_directory(frameDir);
        }

        if (image.empty())
        {
            syslog(LOG_CRIT, "Error reading first frame of video: %s\n", fileName.c_str());
            perror("input video");
            
            return 1;
        }

        int codec = VideoWriter::fourcc('m', 'p', '4', 'v');  // select desired codec (must be available at runtime)
        double fps = cam0.get(CAP_PROP_FPS); // framerate of the created video stream
        writer.open(outfileName, codec, fps, image.size(), false); //gray

        if (!writer.isOpened()) 
        {
            syslog(LOG_CRIT, "Could not open the output video file: %s\n", outfileName.c_str());
            perror("output video");

            return -1;
        }
    }

    int frame_count = 0;
    Mat bin;

    while (live || (frame_count < maxFrames))
    {
        if (image.empty())
        {
            syslog(LOG_INFO, "End of video stream or error reading frame.\n");
            break;
        }

        skeletal(image, bin, dst);

        // Mat debugOut1, debugOut2, debugOut3, debugOut4;
        
        // int finger_count =  improvedCountFingers(dst, bin, debugOut1);

        // cout << "Debug 1 - " << frame_count << ": Detected " << finger_count << " fingers." << endl;
        // imshow("Debug 1", debugOut1);

        // // finger_count = highlightAndCountFingers(dst, debugOut2);
        // // cout << "Debug 2 - " << frame_count << ": Detected " << finger_count << " fingers." << endl;
        // // imshow("Debug 2", debugOut2);
        // finger_count = countCircle(dst, bin, debugOut2);
        // cout << "Debug 2 - " << frame_count << ": Detected " << finger_count << " fingers." << endl;
        // imshow("Debug 2", debugOut2);

        // finger_count = countFingers(dst, debugOut3);
        // cout << "Debug 3 - " << frame_count << ": Detected " << finger_count << " fingers." << endl;
        // imshow("Debug 3", debugOut2);
        
        // finger_count = contorFingers(dst, debugOut4);
        // cout << "Debug 4 - " << frame_count << ": Detected " << finger_count << " fingers." << endl;
        // imshow("Debug 4", debugOut2);

        // putText(dst, std::to_string(finger_count), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        // imshow("Fingers", debugOut);

        if(live || debug)
        {
            imshow("Skeletal", dst);

            char key = (char)waitKey(1);
            if(key == 27) //esc
            {
                syslog(LOG_INFO, "ESC Pressed. Closing application...\n");
                break;
            }
        }
        else
        {
            // cvtColor(dst, dst, COLOR_GRAY2BGR); // convert to 3 channel for video writer

            writer.write(dst);
            // save individual frames as JPEG
            std::string frame_filename = "./" + frameDir + "/frame_" + std::to_string(frame_count) + ".jpg";
            cout << frame_filename << endl;
            
            imwrite(frame_filename, dst);
        }

        cam0.read(image); //read at end so first frame is processed
        frame_count++;
    }

    return 0;
}
