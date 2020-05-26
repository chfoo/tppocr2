#pragma once

#include <memory>
#include <unordered_map>

#include <opencv2/freetype.hpp>

#include "OCR.hpp"
#include "TextDetector.hpp"
#include "Config.hpp"

namespace tppocr {

class WorkUnitResource {
public:
    std::unordered_map<std::string,TextDetector> textDetectors;
    std::unordered_map<std::string,OCR> textRecognizers;
    std::shared_ptr<cv::freetype::FreeType2> freetype;

    explicit WorkUnitResource(std::shared_ptr<Config> config);

};

}
