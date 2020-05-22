#pragma once

// https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.cpp
// https://github.com/argman/EAST

#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "Config.hpp"

namespace tppocr {

class TextDetector {
    cv::Mat blob;
    cv::dnn::Net network;
    std::vector<cv::Mat> outputBlobs;
    std::vector<std::string> outputBlobNames;
    std::vector<cv::RotatedRect> detections;
    std::vector<float> confidences;
    float confidenceMinimumThreshold = 0.5; // [0.0, 1.0]
    float nonmaximumSuppressionThreshold = 0.4; // [0.0, 1.0]
    std::vector<int> indices;

public:
    explicit TextDetector(std::shared_ptr<Config> config);

    const std::vector<cv::RotatedRect> & getDetections();
    const std::vector<float> & getConfidences();
    const std::vector<int> & getIndices();

    void processImage(const cv::Mat & image);

};

}
