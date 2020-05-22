#pragma once

#include <string>
#include <vector>

#include "Region.hpp"

namespace tppocr {

class Config {
public:
    std::string url;
    std::string tessdataPath;
    std::string detectorModelPath;
    bool realTime = false;
    double processingFPS = 60;
    std::vector<Region> regions;
    float detectorConfidenceThreshold = 0.5;
    float detectorNonmaximumSuppressionThreshold = 0.4;

    void parseFromTOML(const std::string path);
};

}
