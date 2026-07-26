#pragma once

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <string>
#include "util.h"
#include "trace.hpp"

class Writer
{
private:
    bool save;
    bool display;
    std::string window_name {"DEEP RECOGNIZER"};
    std::string outdir;

    std::thread writer_thread;

    std::mutex mtx;
    std::condition_variable cv_queue;
    std::queue<Frame> frame_queue;

    std::atomic<bool> running{false};

    TraceLogger *trace = nullptr;

    void writerLoop()
    {
        while (running || !frame_queue.empty())
        {
            Frame packet;
            {
                std::unique_lock<std::mutex> lock(mtx);

                cv_queue.wait(lock, [this]
                {
                    return !frame_queue.empty() || !running;
                });

                if (frame_queue.empty())
                    continue;

                packet = std::move(frame_queue.front());
                frame_queue.pop();
            }

            trace->log(packet.frame_number, TraceEvent::WB_START);

            if (display)
            {
                cv::imshow(window_name, packet.frame);
                cv::waitKey(1);
            }

            if (save)
            {
                char buffer[5];
                std::snprintf(buffer, sizeof(buffer), "%04u", packet.frame_number);
                std::string filename = outdir + "/frame_" + (std::string)(buffer) + ".jpg";

                cv::imwrite(filename, packet.frame);
            }

            trace->log(packet.frame_number, TraceEvent::WB_END);
        }

        if (display)
        {
            cv::destroyWindow(window_name);
        }
    }

public:

    bool start(bool s, bool d, const std::string& o, TraceLogger *t)
    {
        save = s;
        display = d;
        outdir = o;
        trace = t;

        running = true;

        writer_thread = std::thread(&Writer::writerLoop, this);

        return true;
    }

    void unloadFrame(Frame frame)
    {
        if (!running)
            return;

        {
            std::unique_lock<std::mutex> lock(mtx);


            frame_queue.push({frame});
        }

        cv_queue.notify_one();
    }

    void stop()
    {
        running = false;
        cv_queue.notify_all();

        if (writer_thread.joinable())
            writer_thread.join();
    }

    ~Writer()
    {
        stop();
    }
};
