#pragma once

#include <opencv2/core.hpp>
#include <opencv2/freetype.hpp>

#include "Region.hpp"
#include "TextDetector.hpp"
#include "OCR.hpp"

namespace tppocr {

struct WorkUnit {
    unsigned int id;
    unsigned int frameID;
    Region & region;
    cv::Mat image;
    cv::Mat debugImage;
    TextDetector * textDetector = nullptr;
    OCR * ocr = nullptr;
    std::shared_ptr<cv::freetype::FreeType2> freetype = nullptr;

    explicit WorkUnit(unsigned int id, unsigned int frameID, Region & region, cv::Mat image, cv::Mat debugImage);
};

}
