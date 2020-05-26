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
    cv::Mat image;
    cv::Mat debugImage;

    explicit WorkUnit(unsigned int id, unsigned int frameID, cv::Mat image, cv::Mat debugImage);
};

}
