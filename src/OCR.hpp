#pragma once

#include <string>
#include <memory>
#include <vector>

#include <tesseract/baseapi.h>
#include <opencv2/core.hpp>

#include "Config.hpp"

namespace tppocr {

class TesseractDeleter {
public:
    void operator()(tesseract::TessBaseAPI * tesseract) {
        tesseract->End();
        delete tesseract;
    }
};

class OCR {
    std::unique_ptr<tesseract::TessBaseAPI,TesseractDeleter> tesseract;
    std::string text;
    float meanConfidence = 0;

public:
    explicit OCR(std::shared_ptr<Config> config);

    const std::string & getText();
    float getMeanConfidence();

    void processImage(const cv::Mat & image);

    cv::Mat getThresholdedImage();
    std::vector<cv::Rect> getLineBoundaries();
};


}
