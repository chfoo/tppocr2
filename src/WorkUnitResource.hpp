#pragma once

#include <memory>

#include <opencv2/freetype.hpp>

#include "OCR.hpp"
#include "TextDetector.hpp"
#include "Config.hpp"

namespace tppocr {

class WorkUnitResource {
public:
    TextDetector textDetector;
    OCR ocr;
    std::shared_ptr<cv::freetype::FreeType2> freetype;

    explicit WorkUnitResource(std::shared_ptr<Config> config, const Region & region);

};

}
