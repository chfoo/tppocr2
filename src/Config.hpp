#pragma once

#include <string>
#include <vector>

#include <toml++/toml.h>

#include "Region.hpp"

namespace tppocr {

class Config {
public:
    bool debugWindow = false;
    bool frameStepping = false;
    bool preferCPU = false;
    bool preferOpenCL = false;
    bool preferCUDA = false;
    bool preferInference = false;
    bool profiling = false;

    std::string url;
    std::string tessdataPath;
    std::string detectorModelPath;
    bool realTime = false;
    double processingFPS = 60;
    std::vector<Region> regions;
    float detectorConfidenceThreshold = 0.5;
    float detectorNonmaximumSuppressionThreshold = 0.4;
    float recognizerConfidenceThreshold = 0.7;

    void parseFromTOML(const std::string path);

private:
    toml::node_view<toml::node> getTOMLNode(toml::table & table, const std::string key);

};

}
