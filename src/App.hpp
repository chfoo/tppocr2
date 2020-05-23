#pragma once

#include <iostream>
#include <memory>
#include <functional>
#include <unordered_map>

#include <opencv2/core.hpp>
#include <opencv2/freetype.hpp>

#include "Config.hpp"
#include "OCR.hpp"
#include "InputStream.hpp"
#include "TextDetector.hpp"

namespace tppocr {

class App {
    std::shared_ptr<Config> config;
    std::unordered_map<std::string,OCR> textRecognizers;
    std::unordered_map<std::string,TextDetector> textDetectors;
    InputStream inputStream;
    cv::Mat image;
    cv::Mat debugImage;
    std::shared_ptr<cv::freetype::FreeType2> freetype;
    unsigned int frameSkip = 0;
    unsigned int frameSkipCounter = 0;

public:
    explicit App(std::shared_ptr<Config> config);

    void run();

private:
    void frameCallback();
    void processRegions();
    void processRegion(const Region & region);
    void drawRegion(const Region & region);
    void drawDetection(const Region & region, const cv::RotatedRect & box,
        float confidence);
    void processTextBlock(const Region & region, const cv::Rect & box);
    void drawTextBlock(const Region & region, const cv::Rect & box);
    void drawOCRText(const Region & region, const cv::Rect & box);
    void drawOCRThresholdImage(const Region & region, const cv::Rect & box);
    void drawOCRLineBoundaries(const Region & region, const cv::Rect & box);
};

}
