#include "deepFace.hpp"
#include "input.hpp"
#include "trace.hpp"
#include "writer.hpp"

#include <cstdlib>
#include <map>
#include <sys/syslog.h>

int exit_failure()
{
    syslog(LOG_INFO, "DEEP RECOGNIZER EXITING WITH FAILURE");
    closelog();
    return EXIT_FAILURE;
}

int main(int argc, char **argv) {
    InputType type;
    std::string path;
    Input cap;
    DeepFaceSystem face_system;
    Writer wb;
    TraceLogger trace;

    openlog("deep-recognizer", LOG_PID | LOG_CONS, LOG_USER);
    syslog(LOG_INFO, "DEEP LEARNING RECOGNIZER STARTED");

    const std::string keys =
        "{help h ?     | false  | Show help message }"
        "{frames f     |  0     | Frames to process. Default file-length or 500(live)}"
        "{live l       | false  | Read from webcam }"
        "{input i      |        | Read from video file }"
        "{display d    | false  | Display using imshow }"
        "{save s       | true    | Save frames to outdir }" 
        "{outdir o     | frames  | path of directory to save frames }";
    
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Deep Learning Face Recognizer");

    if (parser.get<bool>("help")) {
        parser.printMessage();
        closelog();
        return EXIT_SUCCESS;
    }

    if (!parser.check()) {
        parser.printErrors();
        return exit_failure();
    }

    bool live = parser.get<bool>("live");
    std::string file = parser.get<std::string>("input");

    if (live == !file.empty()) {
        std::cerr << "Error: You must specify only one input source: --live OR --input.\n\n";
        parser.printMessage();
        return exit_failure();
    }

    int frames = parser.get<int>("frames");

    if (live) {
        type = InputType::WEBCAM;
        path = "";

        
        if(frames == 0){
            frames = 500;
        }
    } else {
        type = InputType::FILE_VIDEO;
        path = file;
    }

    bool limit_frames = (frames > 0) ? true : false;
    
    bool display = parser.get<bool>("display");
    bool save = parser.get<bool>("save");
    std::string outdir = parser.get<std::string>("outdir");

    if (display)
        syslog(LOG_DEBUG, "Displaying live frames  ...");

    syslog(LOG_DEBUG, "Initializing and training model. Be patient ...");

    // Initialize CUDA models
    if (!face_system.init("models/face_detection_yunet_2023mar.onnx", 
                         "models/face_recognition_sface_2021dec.onnx", 
                         cv::Size(640, 480))) {
        std::cerr << "Failed to initialize CUDA DNN models!\n";
        return exit_failure();
    }

    // Train CUDA models
    if(!face_system.train())
    {
        std::cerr << "Training failed\n";
        return exit_failure();
    }

    syslog(LOG_DEBUG, "Model is trained. Starting inference now!");

    // Initialize Write-back
    if(!wb.start(save, display, outdir,&trace))
    {
        std::cerr << "Failed to initialize WB thread!\n";
        return exit_failure();
    }

    syslog(LOG_DEBUG, "Initialized WB thread");

    // Start Input
    if (!cap.start(type, path, frames, &trace)) {
        std::cerr << "Failed to open input stream." << std::endl;
        return exit_failure();
    }

    syslog(LOG_DEBUG, "Initialized Input thread");

    Frame frame;
    // Main Inference Loop
    while (!cap.isFinished()) {
        if(limit_frames && frame.frame_number > frames)
        {
            syslog(LOG_DEBUG, "Captured %u frames - stopping", frames);
            break;
        }

        // Pull the most recent frame instantly without waiting on hardware driver delays
        if (cap.getLatestFrame(frame)) {
            trace.log(frame.frame_number, TraceEvent::PREDICT_START);
            
            // Execute DNN inference asynchronously on GPU
            face_system.processFrame(frame.frame);
            
            trace.log(frame.frame_number, TraceEvent::PREDICT_END);

            // send to WB
            wb.unloadFrame(frame);
        }
    }

    cap.stop();
    wb.stop();
    trace.writeCSV("deep-recognizer-trace.csv");

    syslog(LOG_INFO, "DEEP RECOGNIZER FINISHED SUCCESSFULLY");
    closelog();
    return EXIT_SUCCESS;
}