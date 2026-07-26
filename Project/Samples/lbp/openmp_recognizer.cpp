#include <iostream>
#include <sys/syslog.h>
#include <vector>
#include <string>
#include <atomic>
#include <queue>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <chrono>

#include <dirent.h>
#include <omp.h>
#include <syslog.h>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/face/facerec.hpp>

#define HRES 640
#define VRES 480
#define FPS 30

#define TRAIN_SCALE_FACTOR 1.05 // decreased to detect glasses
#define RUN_SCALE_FACTOR 1.1
#define MIN_NEIGHBORS 4

enum class InputType {
    WEBCAM,
    FILE_VIDEO
};

struct AcqConfig {
    InputType type{InputType::WEBCAM};
    std::string path;
};

struct FrameTask {
    unsigned int frame_number;
    cv::Mat frame;
};

// Thread-safe queue
template <typename T>
class SafeQueue {
private:
    std::queue<T> q;
    size_t max_size;

public:
    SafeQueue(size_t max_s = 10) : max_size(max_s) {}

    void push(T item) {
        bool pushed = false;
        while (!pushed) {
            #pragma omp critical(queue_lock)
            {
                if (q.size() < max_size) {
                    q.push(item);
                    pushed = true;
                }
            }
            if (!pushed) {
                #pragma omp taskyield
            }
        }
    }

    bool pop(T &item, std::atomic<bool> &finished) {
        while (true) {
            bool empty = true;
            #pragma omp critical(queue_lock)
            {
                empty = q.empty();
                if (!empty) {
                    item = q.front();
                    q.pop();
                }
            }
            if (!empty) return true;
            if (finished && empty) return false;

            #pragma omp taskyield
        }
    }
};

// Global models
static cv::CascadeClassifier face_cascade;
static cv::Ptr<cv::face::LBPHFaceRecognizer> lbp_model;
static const std::string default_cascade = "haarcascade_frontalface_alt.xml";
static const std::string default_lbp = "lbp_trained_model.yml";

static void draw_face(cv::Mat &frame, const cv::Rect &face, int label, double confidence) {
    cv::Scalar color;
    if (confidence < 30.0)      
        color = cv::Scalar(0, 255, 0);   // Green
    else if (confidence < 70.0) 
        color = cv::Scalar(0, 255, 255); // Yellow
    else                        
        color = cv::Scalar(0, 0, 255);   // Red

    cv::rectangle(frame, face, color, 3);
    std::string text = std::to_string(label) + " - " + std::to_string(confidence);
    cv::putText(frame, text, cv::Point(face.x, std::max(face.y - 10, 20)), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
}

void detect_faces(cv::Mat frame, std::vector<cv::Mat> &cropped, std::vector<cv::Rect> &faces) {
    if (frame.empty()) {
        syslog(LOG_CRIT, "Frame empty");
        return;
    }

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    face_cascade.detectMultiScale(gray, faces, TRAIN_SCALE_FACTOR, MIN_NEIGHBORS);

    for (auto face : faces) {
        cv::Mat crop = gray(face);
        cropped.push_back(crop);
    }
}

static int train_model() {
    std::vector<cv::Mat> faces;
    std::vector<int> labels;

    for (int label = 1; label <= 9; label++) {
        std::string dirName = "./data/" + std::to_string(label);
        DIR *dir = opendir(dirName.c_str());
        if (!dir) {
            std::cerr << "Could not open data directory: " << dirName << "\n";
            return EXIT_FAILURE;
        }

        struct dirent *entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            if (filename == "." || filename == "..") 
                continue;

            std::string path = dirName + "/" + filename;
            cv::Mat image = cv::imread(path);

            if (image.empty()) {
                std::cerr << "No training image: " << path << std::endl;
                closedir(dir);
                return EXIT_FAILURE;
            }

            std::vector<cv::Rect> train_face;
            std::vector<cv::Mat> train_cropped;

            std::cout << "Detecting face in: " << path << std::endl; 
            detect_faces(image, train_cropped, train_face);

            for (auto face : train_cropped) {
                faces.push_back(face);
                labels.push_back(label);
            }
        }
        closedir(dir);
    }

    lbp_model = cv::face::LBPHFaceRecognizer::create();
    lbp_model->train(faces, labels);
    std::cout << "Successfully trained LBPH model on " << faces.size() << " reference images\n";

    lbp_model->save("lbp_trained_model.yml");
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    AcqConfig acq_config{};

    openlog("lbp_recognize", LOG_PID | LOG_CONS, LOG_USER);
    syslog(LOG_INFO, "OPENMP RECOGNIZER STARTED");

    // Command-line parser
    cv::CommandLineParser parser(argc, argv,
                                "{train    t|false|train model}"
                                "{cascade  c||load cascade model (xml)}"
                                "{lbp       ||load LBP model (yml)}"
                                "{live      |false|live webcam}"
                                "{file      ||challenge video}"
                                "{frames   f|500|frames to process}"
                                "{help     h|false|show help message}");

    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    cv::String cascade = parser.get<cv::String>("cascade");
    std::string cascade_path = cascade.empty() ? default_cascade : std::string(cascade);

    if (!face_cascade.load(cascade_path)) {
        std::cerr << "Failed to load Haar cascade classifier: " << cascade_path << std::endl;
        return EXIT_FAILURE;
    }

    if (parser.get<bool>("train")) {
        return train_model();
    }

    bool live = parser.get<bool>("live");
    cv::String file = parser.get<cv::String>("file");

    if ((!live && file.empty()) || (live && !file.empty())) {
        std::cerr << "Specify to use live webcam OR challenge video\n";
        parser.printMessage();
        return 1;
    }

    if (live) {
        acq_config.type = InputType::WEBCAM;
    } 
    else if (!file.empty()) {
        acq_config.type = InputType::FILE_VIDEO;
        acq_config.path = file;
    }

    if (!face_cascade.load(cascade_path)) {
        syslog(LOG_CRIT, "Failed to load cascade classifier: %s", cascade_path.c_str());
        return EXIT_FAILURE;
    }

    cv::String lbp = parser.get<cv::String>("lbp");
    std::string lbp_path = lbp.empty() ? default_lbp : std::string(lbp);

    lbp_model = cv::face::LBPHFaceRecognizer::create();
    try {
        lbp_model->read(lbp_path);
    } catch (const cv::Exception &e) {
        syslog(LOG_CRIT, "Failed to load LBP model from %s: %s", lbp_path.c_str(), e.what());
        return EXIT_FAILURE;
    }

    unsigned int total_frames = parser.get<unsigned int>("frames");

    SafeQueue<FrameTask> acq_to_process_q(15);
    SafeQueue<FrameTask> process_to_write_q(50);

    std::atomic<bool> acq_done{false};
    std::atomic<bool> process_done{false};

    // Set thread budget
    omp_set_max_active_levels(2);
    omp_set_num_threads(6);

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel sections
    {
        // Frame Read Thread (Dedicated Core)
        #pragma omp section
        {
            syslog(LOG_INFO, "[READ] Reader thread #%d started on Core %d", omp_get_thread_num(), sched_getcpu());
            
            cv::VideoCapture cap;
            if (acq_config.type == InputType::WEBCAM) {
                if (!cap.open(0, cv::CAP_V4L2)) {
                    cap.open(0);
                }
                cap.set(cv::CAP_PROP_FRAME_WIDTH, HRES);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT, VRES);
                cap.set(cv::CAP_PROP_FPS, FPS);
            } else {
                cap.open(acq_config.path);
            }

            if (!cap.isOpened()) {
                syslog(LOG_CRIT, "[READ] Error opening video capture source.");
                acq_done = true;
            } else {
                unsigned int frame_idx = 1;
                cv::Mat raw_frame;
                while (frame_idx <= total_frames) {
                    // auto t_start = std::chrono::steady_clock::now();

                    cap >> raw_frame;
                    if (raw_frame.empty()) {
                        if (acq_config.type == InputType::WEBCAM) {
                            syslog(LOG_CRIT, "[READ] Unable to read frame from webcam!");
                            break;
                        } else {
                            syslog(LOG_INFO, "[READ] Video complete before desired frames captured.");
                            break;
                        }
                    }

                    if (raw_frame.cols != HRES || raw_frame.rows != VRES) {
                        cv::resize(raw_frame, raw_frame, cv::Size(HRES, VRES));
                    }

                    acq_to_process_q.push({frame_idx, raw_frame.clone()});
                    
                    // std::chrono::duration<double, std::milli> t_done = std::chrono::steady_clock::now() - t_start;
                    // double t_ms = t_done.count(); 

                    auto now = std::chrono::high_resolution_clock::now();
                    double elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();

                    syslog(LOG_INFO, "[READ] Frame %d read at %.04f ms", frame_idx, elapsed_ms);
                    frame_idx++;
                }
                cap.release();
                acq_done = true;
            }
        }

        // Multithreaded Processing - 4 Workers
        #pragma omp section
        {
            #pragma omp parallel num_threads(4)
            {
                int thread_id = omp_get_thread_num();
                syslog(LOG_INFO, "[WORKER %d] Processing Thread #%d started on Core %d", thread_id, thread_id, sched_getcpu());
                
                #pragma omp single nowait
                {
                    FrameTask frame_task;
                    while (acq_to_process_q.pop(frame_task, acq_done)) {
                        #pragma omp task firstprivate(frame_task)
                        {
                            // auto t_start = std::chrono::steady_clock::now();

                            cv::Mat gray;
                            cv::cvtColor(frame_task.frame, gray, cv::COLOR_BGR2GRAY);

                            std::vector<cv::Rect> faces;
                            face_cascade.detectMultiScale(gray, faces, RUN_SCALE_FACTOR, MIN_NEIGHBORS);

                            for (const auto &face : faces) {
                                cv::Mat cropped = gray(face);
                                cv::resize(cropped, cropped, cv::Size(200, 200));

                                int label = -1;
                                double confidence = 0.0;
                                lbp_model->predict(cropped, label, confidence);

                                draw_face(frame_task.frame, face, label, confidence);
                            }

                            // std::chrono::duration<double, std::milli> t_end = std::chrono::steady_clock::now() - t_start;
                            // double t_ms = t_elapsed.count();

                            auto now = std::chrono::high_resolution_clock::now();
                            double elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();

                            syslog(LOG_INFO, "[WORKER %d] Frame %d: %zu faces detected at %.04f ms", 
                                   omp_get_thread_num(), frame_task.frame_number, faces.size(), elapsed_ms);

                            process_to_write_q.push(frame_task);
                        }
                    }
                }
            }
            process_done = true;
        }

        // SECTION 3: Writer Thread (Dedicated Core)
        #pragma omp section
        {
            syslog(LOG_INFO, "[WB] Write-back thread #%d started on Core %d", omp_get_thread_num(), sched_getcpu());
            
            FrameTask write_task;
            while (process_to_write_q.pop(write_task, process_done)) {
                char filename[64];
                std::snprintf(filename, sizeof(filename), "frames/frame_%04d.jpg", write_task.frame_number);
                cv::imwrite(filename, write_task.frame);

                auto now = std::chrono::high_resolution_clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();

                syslog(LOG_INFO, "[WB] Frame #%d written to %s at %.04f ms", 
                       write_task.frame_number, filename, elapsed_ms);
            }
        }
    } // Implicit OpenMP barrier

    syslog(LOG_INFO, "OpenMP Video processing completed successfully.");
    return EXIT_SUCCESS;
}