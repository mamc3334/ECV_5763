#include <iostream>
#include <stdlib.h>
#include <syslog.h>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaobjdetect.hpp>

#define SCALE_FACTOR 1.1 // increase to speed up - trade off accuracy
#define MIN_NEIGHBORS 4  // precision - may miss in low quality

using namespace std;
using namespace cv;

void detect_draw_faces(Mat frame, Ptr<cuda::CascadeClassifier> &cascade_gpu)
{
    vector<Rect> faces;
    double ticks;

    if( frame.empty() )
    {
        syslog(LOG_CRIT, "Frame empty");
        perror("input image");

        return;
    }

    //start stopwatch
    ticks = (double)getTickCount();

    //Upload to GPU
    cuda::GpuMat frame_gpu(frame);
    cuda::GpuMat gray_gpu;
    cuda::cvtColor(frame_gpu, gray_gpu, COLOR_BGR2GRAY);

    //detect
    cuda::GpuMat faces_buf;
    cascade_gpu->detectMultiScale(gray_gpu, faces_buf);
    cascade_gpu->convert(faces_buf, faces);

    //stop stopwatch
    ticks = (double)getTickCount() - ticks;
    syslog(LOG_INFO, "detection time = %gms\n", ticks*1000.0/getTickFrequency());

    for(auto face : faces)
    {
        //this can also be converted to use cuda but we only are comparing the detection algo
        rectangle(frame, face, Scalar(0,255,255), 3);
    }
}

void print_cuda()
{
    cuda::DeviceInfo devInfo;
    syslog(LOG_INFO, "Using CUDA device: %s", devInfo.name());
}

int main(const int argc, const char **argv)
{
    Mat frame;
    VideoCapture cam0;
    string filename;

    openlog("cuda_detect", LOG_PID, LOG_USER);

    cv::CommandLineParser parser(argc, argv,
                                "{@input   |sam_scaled.png|input image}"
                                "{frames    f|0|frames from live webcam}"
                                "{help    h|false|show help message}");

    if(parser.get<bool>("help"))
    {
        parser.printMessage();
        return 0;
    }
    
    String imageName = parser.get<String>("@input");
    unsigned int frames = parser.get<unsigned int>("frames");

    cuda::setDevice(0);

    Ptr<cuda::CascadeClassifier> cascade_gpu = cuda::CascadeClassifier::create("haarcascade_frontalface_alt.xml");
    if (cascade_gpu.empty()) {
        syslog(LOG_CRIT, "ERROR: Could not load face cascade");

        return -1;
    }

    cascade_gpu->setScaleFactor(SCALE_FACTOR);
    cascade_gpu->setMinNeighbors(MIN_NEIGHBORS);

    if(frames > 0)
    {
        syslog(LOG_INFO, "cuda_detect started using live webcam - %u frames", frames);
        print_cuda();

        // use live webcam - process #frames
        cam0.open(0, cv::CAP_V4L2);
        if (!cam0.isOpened())
        {
            syslog(LOG_CRIT, "Error opening web camera.\n");
            perror("webcam");

            return 1;
        }

        cam0.set(CAP_PROP_FRAME_WIDTH, 640);
        cam0.set(CAP_PROP_FRAME_HEIGHT, 480);
        // cam0.set(cv::CAP_PROP_FPS, 15); //configure frame rate if desired
        
        for(unsigned int i=0; i<frames; i++)
        {
            cam0 >> frame;

            detect_draw_faces(frame, cascade_gpu);

            filename = "frames/frame_" + std::to_string(i) + ".jpg";
            imwrite(filename, frame);
        }
    }
    else
    {
        syslog(LOG_INFO, "cuda_detect started using static image - %s", imageName.c_str());
        print_cuda();

        // process static image
        frame = imread(imageName);

        detect_draw_faces(frame, cascade_gpu);

        filename = imageName;
        filename = filename.substr(0,filename.length() - 4) + "_cuda_faces.jpg";
        imwrite(filename, frame);
    }

    syslog(LOG_INFO, "cuda_detect closing");

    return 0;
}