/**
* @file ensemble_recognize.cpp
* 
* Doorbell facial recognition: LBPH + Eigenface + Fisherface soft-vote ensemble
* with open-set stranger rejection, 3-stage OMP pipeline, lock-free ring buffers,
* and syslog telemetry.
* 
* Build (on Jetson w/ OpenCV contrib):
*   g++ -O2 -fopenmp -std=c++17 \
*       $(pkg-config --cflags --libs opencv4) \
*       ensemble_recognize.cpp -o ensemble_recognize
* 
* Usage:
*   Train:   ./ensemble_recognize --train
*   Webcam:  ./ensemble_recognize --live --frames=300
*   File:    ./ensemble_recognize --file=video.mp4
*/
#include "opencv2/highgui.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <sched.h>       // sched_yield()

#include <dirent.h>
#include <omp.h>
#include <syslog.h>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/face/facerec.hpp>

#define HRES 640
#define VRES 480
#define FPS  30

#define FACE_SIZE 200

#define DETECT_SCALE_FACTOR 1.05
#define MIN_NEIGHBORS       4

#define STRANGER_THRESHOLD 0.40


enum class InputType { WEBCAM, FILE_VIDEO };

struct AcqConfig {
    InputType   type{InputType::WEBCAM};
    std::string path;
};

struct FrameTask {
    unsigned int frame_number;
    cv::Mat      frame;
};

struct EnsembleResult {
    int    label;
    double norm_score;   // [0,1]: higher = more confident match
    bool   is_known;
    int    lbp_label,   eigen_label,   fisher_label;
    double lbp_conf,    eigen_conf,    fisher_conf;
};

// ─── Thread-Safe Bounded Ring Buffer ─────────────────────────────────────────

template <typename T, size_t N>
class RingBuffer {
private:
    std::vector<T>       buffer;
    alignas(64) std::atomic<size_t> head{0};
    alignas(64) std::atomic<size_t> tail{0};

public:
    RingBuffer() : buffer(N) {}

    void push(T item) {
        bool pushed = false;
        while (!pushed) {
            size_t current_tail = tail.load(std::memory_order_relaxed);
            size_t current_head = head.load(std::memory_order_acquire);

            if ((current_tail + 1) % N == current_head) {
                sched_yield();
            }
            else {
                #pragma omp critical(ring_buffer_push_lock)
                {
                    current_tail = tail.load(std::memory_order_relaxed);
                    current_head = head.load(std::memory_order_acquire);
                    
                    if ((current_tail + 1) % N != current_head) {
                        buffer[current_tail] = std::move(item);
                        tail.store((current_tail + 1) % N, std::memory_order_release);
                        pushed = true;
                    }
                }
                if (!pushed) sched_yield(); 
            }
        }
    }

    bool pop(T &item, std::atomic<bool> &finished) {
        while (true) {
            size_t current_head = head.load(std::memory_order_relaxed);
            size_t current_tail = tail.load(std::memory_order_acquire);

            if (current_head == current_tail) {
                if (finished.load(std::memory_order_acquire)) {
                    current_tail = tail.load(std::memory_order_acquire);
                    if (current_head == current_tail) {
                        return false;
                    }
                }
                sched_yield();
                continue;
            }
            
            bool popped = false;
            #pragma omp critical(ring_buffer_pop_lock)
            {
                current_head = head.load(std::memory_order_relaxed);
                current_tail = tail.load(std::memory_order_acquire);

                if (current_head != current_tail) {
                    item = std::move(buffer[current_head]);
                    head.store((current_head + 1) % N, std::memory_order_release);
                    popped = true;
                }
            }

            if (popped) return true;
            
            if (finished.load(std::memory_order_acquire) && head.load() == tail.load()) {
                return false;
            }

            sched_yield();
        }
    }
};

static cv::CascadeClassifier train_cascade; // used only in train_models()

static const std::string DEFAULT_CASCADE = "haarcascade_frontalface_alt.xml";
static const std::string DEFAULT_LBP     = "lbp_trained_model.yml";
static const std::string DEFAULT_EIGEN   = "eigen_trained_model.yml";
static const std::string DEFAULT_FISHER  = "fisher_trained_model.yml";

static cv::Mat preprocess_face(const cv::Mat &gray_crop) {
    cv::Mat out;
    cv::resize(gray_crop, out, cv::Size(FACE_SIZE, FACE_SIZE));
    cv::equalizeHist(out, out);

    return out;
}

static inline double dist_to_score(double dist, double scale) {
    return 1.0 / (1.0 + dist / scale);
}

static EnsembleResult ensemble_predict(
    const cv::Mat &face_preprocessed,
    cv::face::LBPHFaceRecognizer   *lbp,
    cv::face::EigenFaceRecognizer  *eigen,
    cv::face::FisherFaceRecognizer *fisher)
{
    EnsembleResult r{};

    lbp->predict   (face_preprocessed, r.lbp_label,    r.lbp_conf);
    eigen->predict (face_preprocessed, r.eigen_label,  r.eigen_conf);
    fisher->predict(face_preprocessed, r.fisher_label, r.fisher_conf);

    double s_lbp    = dist_to_score(r.lbp_conf,    100.0);
    double s_eigen  = dist_to_score(r.eigen_conf,  10000.0);
    double s_fisher = dist_to_score(r.fisher_conf,  3000.0);

    syslog(LOG_DEBUG, "[CONF] lbp=%.1f eigen=%.1f fisher=%.1f | scores lbp=%.3f eigen=%.3f fisher=%.3f",
           r.lbp_conf, r.eigen_conf, r.fisher_conf, s_lbp, s_eigen, s_fisher);

    std::map<int, double> vote;
    vote[r.lbp_label]    += s_lbp;
    vote[r.eigen_label]  += s_eigen;
    vote[r.fisher_label] += s_fisher;

    int    best_label = -1;
    double best_score =  0.0;
    for (const auto &kv : vote) {
        if (kv.second > best_score) {
            best_score = kv.second;
            best_label = kv.first;
        }
    }

    r.label      = best_label;
    r.norm_score = best_score / 3.0;
    r.is_known   = (r.norm_score >= STRANGER_THRESHOLD);

    return r;
}

static void draw_result(cv::Mat &frame, const cv::Rect &face, const EnsembleResult &res) {
    cv::Scalar color;
    std::string id_text;

    if (!res.is_known) {
        color   = cv::Scalar(0, 0, 255);
        id_text = "STRANGER: " + std::to_string(res.label);
    } else if (res.norm_score >= 0.65) {
        color   = cv::Scalar(0, 255, 0);
        id_text = "ID:" + std::to_string(res.label);
    } else {
        color   = cv::Scalar(0, 255, 255);
        id_text = "ID:" + std::to_string(res.label) + "?";
    }

    cv::rectangle(frame, face, color, 3);

    std::string top = id_text + "  score:" +
                      std::to_string(static_cast<int>(res.norm_score * 100)) + "%";
    cv::putText(frame, top, cv::Point(face.x, std::max(face.y - 10, 20)),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2);

    std::string dbg = "L:" + std::to_string(res.lbp_label)    +
                      " E:" + std::to_string(res.eigen_label)  +
                      " F:" + std::to_string(res.fisher_label);
    cv::putText(frame, dbg, cv::Point(face.x, std::min(face.y + face.height + 18, frame.rows - 5)),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 200, 200), 1);
}

static void detect_and_crop(const cv::Mat &bgr,
                             std::vector<cv::Mat>  &crops_out,
                             std::vector<cv::Rect> &rects_out)
{
    if (bgr.empty()) return;
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    train_cascade.detectMultiScale(gray, rects_out, DETECT_SCALE_FACTOR, MIN_NEIGHBORS);
    for (const auto &r : rects_out)
        crops_out.push_back(preprocess_face(gray(r)));
}

static int train_models() {
    std::vector<cv::Mat> faces;
    std::vector<int>     labels;

    for (int label = 1; label <= 9; label++) {
        std::string dir_path = "../../data/raw_data/" + std::to_string(label);
        DIR *dir = opendir(dir_path.c_str());
        if (!dir) {
            std::cerr << "[TRAIN] Cannot open directory: " << dir_path << "\n";
            return EXIT_FAILURE;
        }

        struct dirent *entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string fname = entry->d_name;
            if (fname == "." || fname == "..") continue;

            std::string path = dir_path + "/" + fname;
            cv::Mat img = cv::imread(path);
            if (img.empty()) {
                std::cerr << "[TRAIN] Cannot read image: " << path << " — skipping\n";
                continue;
            }

            std::vector<cv::Mat>  crops;
            std::vector<cv::Rect> rects;
            std::cout << "[TRAIN] Detecting: " << path << "\n";
            detect_and_crop(img, crops, rects);

            if (crops.empty()) {
                std::cerr << "[TRAIN] WARN: no face detected in " << path << " — skipping\n";
                continue;
            }

            for (const auto &crop : crops) {
                faces.push_back(crop);
                labels.push_back(label);
                std::string out_path = "./training_data/" + std::to_string(label) + "/" + fname;
                cv::imwrite(out_path, crop);
            }
        }
        closedir(dir);
    }

    if (faces.empty()) {
        std::cerr << "[TRAIN] No training faces collected. Aborting.\n";
        return EXIT_FAILURE;
    }

    std::cout << "[TRAIN] Training on " << faces.size() << " face crops across identities...\n";

    auto lbp_model    = cv::face::LBPHFaceRecognizer::create();
    auto eigen_model  = cv::face::EigenFaceRecognizer::create();
    auto fisher_model = cv::face::FisherFaceRecognizer::create();

    lbp_model->train(faces, labels);
    eigen_model->train(faces, labels);
    fisher_model->train(faces, labels);

    lbp_model->save(DEFAULT_LBP);
    eigen_model->save(DEFAULT_EIGEN);
    fisher_model->save(DEFAULT_FISHER);

    std::cout << "[TRAIN] Models successfully trained and saved.\n";
    return EXIT_SUCCESS;
}

// ─── Main Pipeline ────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    openlog("ensemble_recognize", LOG_PID | LOG_CONS, LOG_USER);
    syslog(LOG_INFO, "ENSEMBLE RECOGNIZER STARTED");

    AcqConfig acq_config{};

    cv::CommandLineParser parser(argc, argv,
        "{train    t|false|train LBPH + Eigenface + Fisherface models}"
        "{cascade  c||Haar cascade XML path}"
        "{lbp       ||LBPH model YML path}"
        "{eigen     ||Eigenface model YML path}"
        "{fisher    ||Fisherface model YML path}"
        "{live      |false|read from webcam}"
        "{file      ||read from video file}"
        "{frames   f|500|max frames to process}"
        "{help     h|false|show this help}");

    if (parser.get<bool>("help")) {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    cv::String cascade_arg = parser.get<cv::String>("cascade");
    std::string cascade_path = cascade_arg.empty() ? DEFAULT_CASCADE : std::string(cascade_arg);
    if (!train_cascade.load(cascade_path)) {
        std::cerr << "Failed to load Haar cascade: " << cascade_path << "\n";
        return EXIT_FAILURE;
    }

    if (parser.get<bool>("train")) {
        int rc = train_models();
        closelog();
        return rc;
    }

    bool       live = parser.get<bool>("live");
    cv::String file = parser.get<cv::String>("file");

    if (live == !file.empty()) {
        std::cerr << "Specify exactly one of --live OR --file=<path>.\n";
        parser.printMessage();
        return EXIT_FAILURE;
    }

    if (live) {
        acq_config.type = InputType::WEBCAM;
    } else {
        acq_config.type = InputType::FILE_VIDEO;
        acq_config.path = std::string(file);
    }

    auto resolve = [&](const std::string &arg, const std::string &def) {
        cv::String v = parser.get<cv::String>(arg);
        return v.empty() ? def : std::string(v);
    };

    const std::string lbp_path    = resolve("lbp",    DEFAULT_LBP);
    const std::string eigen_path  = resolve("eigen",  DEFAULT_EIGEN);
    const std::string fisher_path = resolve("fisher", DEFAULT_FISHER);

    // Validate that files exist by doing a single probe load before spawning threads.
    {
        auto probe_lbp    = cv::face::LBPHFaceRecognizer::create();
        auto probe_eigen  = cv::face::EigenFaceRecognizer::create();
        auto probe_fisher = cv::face::FisherFaceRecognizer::create();
        try {
            probe_lbp->read(lbp_path);
            probe_eigen->read(eigen_path);
            probe_fisher->read(fisher_path);
            syslog(LOG_INFO, "Model files validated: %s, %s, %s",
                   lbp_path.c_str(), eigen_path.c_str(), fisher_path.c_str());
        } catch (const cv::Exception &e) {
            syslog(LOG_CRIT, "Model load failed: %s", e.what());
            std::cerr << "Model load failed: " << e.what() << "\n";
            closelog();
            return EXIT_FAILURE;
        }
    }

    unsigned int total_frames = parser.get<unsigned int>("frames");

    RingBuffer<FrameTask, 16> acq_to_proc_q;
    RingBuffer<FrameTask, 64> proc_to_write_q;

    std::atomic<bool> acq_done{false};
    std::atomic<bool> proc_done{false};

    omp_set_max_active_levels(2);

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel sections
    {
        // Frame Acquisition
        #pragma omp section
        {
            syslog(LOG_INFO, "[READ] started on core %d", sched_getcpu());

            cv::VideoCapture cap;
            if (acq_config.type == InputType::WEBCAM) {
                cap.open(0, cv::CAP_V4L2);
                if (!cap.isOpened()) cap.open(0);
                cap.set(cv::CAP_PROP_FRAME_WIDTH,  HRES);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT, VRES);
                cap.set(cv::CAP_PROP_FPS,          FPS);
            } else {
                cap.open(acq_config.path);
            }

            if (!cap.isOpened()) {
                syslog(LOG_CRIT, "[READ] Failed to open video source.");
            } else {
                cv::Mat      frame;
                unsigned int idx = 1;
                while (idx <= total_frames && cap.read(frame) && !frame.empty()) {
                    // cv::imshow("frame", frame);
                    acq_to_proc_q.push({idx, frame.clone()});

                    auto   now = std::chrono::high_resolution_clock::now();
                    double ms  = std::chrono::duration<double, std::milli>(now - start_time).count();
                    syslog(LOG_INFO, "[READ] Frame %u queued at %.4f ms", idx, ms);
                    ++idx;
                }
                cap.release();
            }
            acq_done.store(true, std::memory_order_release);
        }

        #pragma omp section
        {
            #pragma omp parallel num_threads(4)
            {
                const int tid = omp_get_thread_num();
                syslog(LOG_INFO, "[WORKER %d] started on core %d", tid, sched_getcpu());

                // Per-thread model copies — fully independent, no locking needed.
                auto t_lbph   = cv::face::LBPHFaceRecognizer::create();
                auto t_eigen  = cv::face::EigenFaceRecognizer::create();
                auto t_fisher = cv::face::FisherFaceRecognizer::create();
                cv::CascadeClassifier t_cascade;

                try {
                    t_lbph->read(lbp_path);
                    t_eigen->read(eigen_path);
                    t_fisher->read(fisher_path);
                    if (!t_cascade.load(cascade_path)) {
                        syslog(LOG_CRIT, "[WORKER %d] Failed to load cascade", tid);
                        exit(EXIT_FAILURE);
                    }
                } catch (const cv::Exception &e) {
                    syslog(LOG_CRIT, "[WORKER %d] Model load error: %s", tid, e.what());
                    exit(EXIT_FAILURE);
                }

                {
                    FrameTask task;
                    while (acq_to_proc_q.pop(task, acq_done)) {
                        cv::Mat gray;
                        cv::cvtColor(task.frame, gray, cv::COLOR_BGR2GRAY);
                        // cv::imshow("gray", gray);
                        // cv::waitKey(0);

                        std::vector<cv::Rect> face_rects;
                        t_cascade.detectMultiScale(gray, face_rects, DETECT_SCALE_FACTOR, MIN_NEIGHBORS);

                        for (const auto &rect : face_rects) {
                            cv::Mat crop = preprocess_face(gray(rect));
                            // cv::imshow("crop", crop);
                            // cv::waitKey(0);
                            EnsembleResult res = ensemble_predict(
                                crop,
                                t_lbph.get(),
                                t_eigen.get(),
                                t_fisher.get());
                            draw_result(task.frame, rect, res);

                            auto   now = std::chrono::high_resolution_clock::now();
                            double ms  = std::chrono::duration<double, std::milli>(now - start_time).count();
                            syslog(LOG_INFO,
                                   "[WORKER %d] Frame %u face: label=%d score=%.2f known=%d | %.4f ms",
                                   tid, task.frame_number, res.label, res.norm_score,
                                   (int)res.is_known, ms);
                        }

                        proc_to_write_q.push(task);
                    }
                }
            }
            proc_done.store(true, std::memory_order_release);
        }

        // Frame Writer
        #pragma omp section
        {
            syslog(LOG_INFO, "[WB] started on core %d", sched_getcpu());

            FrameTask wt;
            while (proc_to_write_q.pop(wt, proc_done)) {
                char fname[64];
                std::snprintf(fname, sizeof(fname), "frames/frame_%04u.jpg", wt.frame_number);
                cv::imwrite(fname, wt.frame);
                auto   now = std::chrono::high_resolution_clock::now();
                double ms  = std::chrono::duration<double, std::milli>(now - start_time).count();
                syslog(LOG_INFO, "[WB] Frame #%u written at %.4f ms", wt.frame_number, ms);
            }
        }
    }

    syslog(LOG_INFO, "Ensemble processing complete.");
    closelog();
    return EXIT_SUCCESS;
}