#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <vector>
#include <syslog.h>

#define SCALE_FACTOR 1.1 // increase to speed up - trade off accuracy
#define MIN_NEIGHBORS 4  // precision - may miss in low quality

using namespace std;
using namespace cv;

CascadeClassifier cascade;

void detect_faces(Mat frame, vector<Mat> &cropped, vector<Rect> &faces)
{
    Mat gray;

    if( frame.empty() )
    {
        syslog(LOG_CRIT, "Frame empty");
        perror("input image");

        return;
    }

    cvtColor(frame, gray, COLOR_BGR2GRAY);
    //detect
    cascade.detectMultiScale(gray, faces, SCALE_FACTOR, MIN_NEIGHBORS); // can adjust scales and such too

    for(auto face:faces)
    {
        Mat crop = gray(face);
        resize(crop, crop, cv::Size(200, 200));
        cropped.push_back(crop);
    }
}

void recognize_faces(Ptr<face::LBPHFaceRecognizer> model, vector<Mat> cropped, vector<int> &labels, vector<double> &confidences)
{
    int label;
    double confidence;

    for(auto face : cropped)
    {
        model->predict(face, label, confidence);
        labels.push_back(label);
        confidences.push_back(confidence);
    }
}

void draw_face(Mat frame, Rect face, int label, double confidence)
{
    Scalar color;

    if(confidence < 30)
        color = Scalar(0,255,0); //green
    else if (confidence < 70)
        color = Scalar(0,255,255); //yellow
    else
        color = Scalar(0,0,255); //red

    rectangle(frame, face, color, 3);
    string text = to_string(label) + "  -  " + to_string(confidence);
    putText(frame, text, Point(face.x, face.y-10), FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
}   

int main(const int argc, const char **argv)
{
    Mat frame;
    VideoCapture cam0;
    string filename;

    openlog("lbp_recognize", LOG_PID, LOG_USER);

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

    cascade.load("haarcascade_frontalface_alt.xml");

    //load ref
    // Mat train_image = imread("mason_ref.png");
    // // imshow("train image", train_image);
    // if(train_image.empty())
    // {
    //     cerr << "no training image" <<endl;
    //     syslog(LOG_CRIT, "NO TRAINING IMAGE");
    //     return -1;
    // }
    // vector<Rect> train_face;
    // vector<Mat> train_cropped;

    // detect_faces(train_image, train_cropped, train_face);
    // if(train_cropped.empty())
    // {
    //     cerr << "bad training image" <<endl;
    //     syslog(LOG_CRIT, "BAD TRAINING IMAGE");
    //     return -1;
    // }
    // imshow("train face", train_cropped[0]);
    // waitKey(0);
    // return 0;
    // Train the LBPH Model on first face and arbitrary ID label
    Ptr<face::LBPHFaceRecognizer> model = face::LBPHFaceRecognizer::create();
    // vector<int> train_label = {123}; //arbitrary
    // model->train(train_cropped, train_label);
    // syslog(LOG_INFO, "Successfully trained LBPH model on 1 reference image: mason_ref.png");
    model->read("lbp_trained_model.yml");

    vector<Mat> cropped;
    vector<Rect> faces;
    vector<int> labels;
    vector<double> confidences;
    double ticks;

    if(frames > 0)
    {
        syslog(LOG_INFO, "lbp_recognize started using live webcam - %u frames", frames);
        // use live webcam - process #frames
        cam0.open(0, CAP_V4L2);
        if (!cam0.isOpened())
        {
            syslog(LOG_CRIT, "Error opening web camera.\n");
            perror("webcam");

            return 1;
        }

        cam0.set(CAP_PROP_FRAME_WIDTH, 640);
        cam0.set(CAP_PROP_FRAME_HEIGHT, 480);
        // cam0.set(cv::CAP_PROP_CONVERT_RGB, false); // grayscale
        // cam0.set(cv::CAP_PROP_FPS, 15); //configure frame rate if desired
        
        for(unsigned int i=0; i<frames; i++)
        {
            cropped.clear();
            faces.clear();
            labels.clear();
            confidences.clear();
            cam0 >> frame;

            ticks = (double)getTickCount();
            detect_faces(frame, cropped, faces);
            recognize_faces(model, cropped, labels, confidences);
            ticks = (double)getTickCount() - ticks;

            syslog(LOG_INFO, "---------------------------------");
            syslog(LOG_INFO, "recognized %lu faces in %gms\n", cropped.size(), ticks*1000.0/getTickFrequency());
            for(size_t j=0; j<cropped.size(); j++)
            {
                syslog(LOG_INFO, "    Label: %d - Confidence: %f", labels[j], confidences[j]);
                draw_face(frame, faces[j], labels[j], confidences[j]);
            }
            
            filename = "frames/frame_" + std::to_string(i) + ".jpg";
            imwrite(filename, frame);
        }
    }
    else
    {
        syslog(LOG_INFO, "lbp_recognize started using static image - %s", imageName.c_str());
        // process static image
        frame = imread(imageName);

        ticks = (double)getTickCount();
        detect_faces(frame, cropped, faces);
        recognize_faces(model, cropped, labels, confidences);
        ticks = (double)getTickCount() - ticks;

        syslog(LOG_INFO, "---------------------------------");
        syslog(LOG_INFO, "recognized %lu faces in %gms\n", cropped.size(), ticks*1000.0/getTickFrequency());
        for(size_t i=0; i<cropped.size(); i++)
        {
            syslog(LOG_INFO, "    Label: %d - Confidence: %f", labels[i], confidences[i]);
            draw_face(frame, faces[i], labels[i], confidences[i]);
        }

        int start = imageName.find_last_of("/")+1;
        filename = imageName.substr(start, imageName.find_last_of(".")-start) + "_lbp.jpg";
        imwrite(filename, frame);
    }

    syslog(LOG_INFO, "lbp_recognize closing");

    return 0;
}
