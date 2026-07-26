#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <mutex>

enum class TraceEvent : uint8_t {
    CAPTURE,
    PREDICT_START,
    PREDICT_END,
    WB_START,
    WB_END
};

struct TraceRecord {
    uint64_t timestamp_ns;
    uint32_t frame;
    TraceEvent event;
};

class TraceLogger {
private:
    std::chrono::steady_clock::time_point start;
    std::vector<TraceRecord> records;
    std::mutex mtx;
public:
    explicit TraceLogger(size_t reserve = 100000)
    {
        records.reserve(reserve);
        start = std::chrono::steady_clock::now();
    }

    inline void log(uint32_t frame, TraceEvent event)
    {
        auto now = std::chrono::steady_clock::now();

        std::lock_guard<std::mutex> lock(mtx);

        records.push_back({
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    now - start).count()),
            frame,
            event
        });
    }

    void writeCSV(const std::string &filename)
    {
        std::ofstream out(filename);
        out << "time_ns,frame,event\n";

        for (const auto &r : records)
        {
            out << r.timestamp_ns << ","
                << r.frame << ","
                << static_cast<int>(r.event)
                << "\n";
        }
    }
};
