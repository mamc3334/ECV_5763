#pragma once

#include <opencv2/opencv.hpp>
#include <ratio>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <iostream>
#include <syslog.h>
#include "trace.hpp"
#include "util.h"

enum class InputType { WEBCAM, FILE_VIDEO };

class Input {
private:
    InputType type;
    std::string path;
    cv::VideoCapture cap;
    std::thread capture_thread;
    std::mutex mtx;
    std::condition_variable cv_frame;
    std::condition_variable cv_space;
    int frame_cnt;

    std::queue<Frame> frame_queue;
    static constexpr size_t MAX_QUEUE_SIZE = 10;

    std::atomic<bool> running{false};

    TraceLogger *trace;

    void captureLoop() {
        int idx = 1;
        cv::Mat raw_frame;
        while (running) {
            if (!cap.read(raw_frame) && raw_frame.empty()) {
                syslog(LOG_DEBUG, "[INPUT] Frame capture ended");
                running = false;
                cv_frame.notify_all();
                cv_space.notify_all();
                break;
            }
            else {
                {
                    std::unique_lock<std::mutex> lock(mtx);

                    cv_space.wait(lock, [this]() {
                        return frame_queue.size() < MAX_QUEUE_SIZE || !running.load();
                    });

                    if (!running.load())
                        break;
		    
		    if (type==InputType::FILE_VIDEO)
		    	cv::resize(raw_frame, raw_frame, cv::Size(640, 480));

                    frame_queue.push({raw_frame.clone(), idx});
                }

                cv_frame.notify_one(); // Signal inference thread
                
                trace->log(idx++, TraceEvent::CAPTURE);
            }
        }
    }

public:
    bool start(InputType t, std::string p, int f, TraceLogger *tr) {
        type = t;
        path = p;
        frame_cnt = f;
        trace = tr;

        if(type == InputType::WEBCAM)
        {
            cap.open(0);
            if (!cap.isOpened()) return false;
            cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        }
        else if(type == InputType::FILE_VIDEO)
        {
            cap.open(path);
            if (!cap.isOpened()) return false;
        }
        cap.set(cv::CAP_PROP_FPS, 30);
        
        running = true;
        capture_thread = std::thread(&Input::captureLoop, this);
        return true;
    }

    // Non-blocking or timeout get for the main/inference loop
    bool getLatestFrame(Frame &out_frame) {
        std::unique_lock<std::mutex> lock(mtx);

        // Wait until queue is non-empty
        cv_frame.wait(lock,[this]() {
            return !frame_queue.empty() || !running.load();
        });

        if (frame_queue.empty()) {
            return false;
        }

        out_frame = std::move(frame_queue.front());
        frame_queue.pop();
        cv_space.notify_one();

        return true;
    }

    bool isFinished() {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mtx));
        return !running && frame_queue.empty();
    }

    void stop() {
        running = false;
        cv_frame.notify_all();
        cv_space.notify_all();
        if (capture_thread.joinable()) {
            capture_thread.join();
        }
        cap.release();
    }

    ~Input() {
        stop();
    }
};
