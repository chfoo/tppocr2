#pragma once

#include <iostream>
#include <memory>
#include <functional>
#include <unordered_map>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

#include <opencv2/core.hpp>
#include <opencv2/freetype.hpp>

#include "Config.hpp"
#include "OCR.hpp"
#include "InputStream.hpp"
#include "TextDetector.hpp"
#include "WorkUnit.hpp"
#include "WorkUnitResource.hpp"

namespace tppocr {

class App {
    std::shared_ptr<Config> config;
    std::vector<std::shared_ptr<std::thread>> workers;
    std::queue<WorkUnit> workUnits;
    std::mutex workUnitsMutex;
    std::condition_variable workUnitsConditionVar;
    InputStream inputStream;
    cv::Mat frameImage;
    cv::Mat debugImage;
    std::mutex debugImageMutex;
    std::condition_variable debugImageConditionVar;
    bool debugImageHasContent = false;
    unsigned int frameSkip = 0;
    unsigned int frameSkipCounter = 0;
    unsigned int processedFrameCounter = 0;
    bool running = false;

public:
    explicit App(std::shared_ptr<Config> config);

    void run();

private:
    void frameCallback();
    void startWorkers();
    void workerEntry();
    void drawDebugWindow();
};

}
