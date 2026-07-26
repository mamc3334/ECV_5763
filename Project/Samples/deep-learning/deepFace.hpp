#pragma once

#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <map>
#include <cmath>

// Structure to store pre-calculated feature vectors for registered users
struct SubjectProfile {
    int id;
    std::string name;
    std::vector<cv::Mat> embeddings; // Embeddings from the 6 training images
};

class DeepFaceSystem {
private:
    cv::Ptr<cv::FaceDetectorYN> detector;
    cv::Ptr<cv::FaceRecognizerSF> recognizer;
    std::vector<SubjectProfile> database;

    std::map<int, std::string> Names = {
        {1, "JONNY"},
        {2, "ROWEN"},
        {3, "ROSIE"},
        {4, "NATE"},
        {5, "JAMES"},
        {6, "HIBBETT"},
        {7, "CY"},
        {8, "MASON"},
        {9, "ELI"}
    };

    // Thresholds recommended by OpenCV SFace documentation
    const float COSINE_THRESHOLD = 0.363f; // > 0.363 indicates SAME identity

    float computeSimilarity(const cv::Mat& feat1, const cv::Mat& feat2) {
        return recognizer->match(feat1, feat2, cv::FaceRecognizerSF::DisType::FR_COSINE);
    }

public:
    bool init(const std::string& yunet_model, const std::string& sface_model, cv::Size input_size) {
        // 1. Initialize YuNet Face Detector
        detector = cv::FaceDetectorYN::create(
            yunet_model, "", input_size, 0.6f, 0.3f, 5000,
            cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA // Hardware acceleration
        );

        // 2. Initialize SFace Embedder
        recognizer = cv::FaceRecognizerSF::create(
            sface_model, "",
            cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA // Hardware acceleration
        );

        return !detector.empty() && !recognizer.empty();
    }

    void registerUser(int user_id, const std::string& name, const std::vector<cv::Mat>& user_images) {
        SubjectProfile profile{user_id, name, {}};

        for (const auto& img : user_images) {
            cv::Mat faces, aligned_face, feature;
            
            // Set detector size dynamically for input image
            detector->setInputSize(img.size());
            detector->detect(img, faces);

            if (faces.rows > 0) {
                // Align face using facial landmarks automatically computed by YuNet
                recognizer->alignCrop(img, faces.row(0), aligned_face);
                // Extract 128D embedding
                recognizer->feature(aligned_face, feature);
                profile.embeddings.push_back(feature.clone());
            }
        }
        database.push_back(profile);
    }

    bool train()
    {
        std::vector<cv::Mat> train_imgs;
        // Train CUDA models
        for (int label = 1; label <= 9; label++) {
            std::string dir_path = "../../data/raw_data/" + std::to_string(label);
            DIR *dir = opendir(dir_path.c_str());
            if (!dir) {
                std::cerr << "Cannot open directory: " << dir_path << "\n";
                return false;
            }

            struct dirent *entry;
            while ((entry = readdir(dir)) != nullptr) {
                std::string fname = entry->d_name;
                if (fname == "." || fname == "..") continue;

                std::string path = dir_path + "/" + fname;
                cv::Mat img = cv::imread(path);
                if (img.empty()) {
                    std::cerr << "Cannot read image: " << path << " — skipping\n";
                    continue;
                }

                train_imgs.push_back(img);
            }

            registerUser(label, Names.at(label), train_imgs);

            closedir(dir);
        }

        return true;
    }

    // Process live video frame
    void processFrame(cv::Mat& frame) {
        detector->setInputSize(frame.size());
        
        cv::Mat faces;
        detector->detect(frame, faces); // Fast CUDA detection

        for (int i = 0; i < faces.rows; i++) {
            cv::Mat aligned_face, feature;
            
            // Auto-align face using 5 landmarks
            recognizer->alignCrop(frame, faces.row(i), aligned_face);
            recognizer->feature(aligned_face, feature);

            std::string best_name;
            float max_sim = -1.0f;

            // Compare against database
            for (const auto& user : database) {
                for (const auto& db_feature : user.embeddings) {
                    float sim = computeSimilarity(feature, db_feature);
                    if (sim > max_sim) {
                        max_sim = sim;
                        best_name = user.name;
                    }
                }
            }

            // Evaluate Match
            bool is_known = (max_sim >= COSINE_THRESHOLD);
            
            // Draw result
            int x = static_cast<int>(faces.at<float>(i, 0));
            int y = static_cast<int>(faces.at<float>(i, 1));
            int w = static_cast<int>(faces.at<float>(i, 2));
            int h = static_cast<int>(faces.at<float>(i, 3));

            cv::Scalar color = is_known ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            std::string label = is_known ? (best_name + " - " + cv::format("%.2f", max_sim)) : "STRANGER";

            cv::rectangle(frame, cv::Rect(x, y, w, h), color, 2);
            cv::putText(frame, label, cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }
    }
};

