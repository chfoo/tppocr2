#include "App.hpp"

#include <stdio.h>
#include <limits>
#include <cmath>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/freetype.hpp>

#include "AppWorker.hpp"

namespace tppocr {

App::App(std::shared_ptr<Config> config) :
    config(config),
    inputStream(config) {

    inputStream.callback = std::bind(&App::frameCallback, this);

    frameImage = cv::Mat(
        inputStream.videoFrameHeight(),
        inputStream.videoFrameWidth(),
        CV_8UC3,
        inputStream.videoFrameData()
    );
    debugImage = cv::Mat(
        inputStream.videoFrameHeight(),
        inputStream.videoFrameWidth(),
        CV_8UC3
    );

    frameSkip = std::floor(inputStream.fps() / config->processingFPS);

    std::cerr << "Skipping every " << frameSkip << " frame(s)." << std::endl;
}

void App::run() {
    running = true;
    startWorkers();

    while (inputStream.isRunning()) {
        inputStream.runOnce();
    }

    std::cerr << "Stream ended" << std::endl;

    running = false;
    workUnitsConditionVar.notify_all();

    for (auto & thread : workers) {
        thread->join();
    }
}

void App::frameCallback() {
    if (frameSkipCounter > 0) {
        frameSkipCounter--;
        return;
    }

    auto image = cv::Mat(
        inputStream.videoFrameHeight(),
        inputStream.videoFrameWidth(),
        CV_8UC3
    );
    auto debugImage = cv::Mat(
        inputStream.videoFrameHeight(),
        inputStream.videoFrameWidth(),
        CV_8UC3
    );

    inputStream.convertFrameToBGR();
    frameImage.copyTo(image);
    frameImage.copyTo(debugImage);

    std::unique_lock<std::mutex> workUnitsLock(workUnitsMutex);
    workUnitsConditionVar.wait(workUnitsLock, [&]{ return workUnits.empty(); });
    workUnitsLock.unlock();

    workUnitsMutex.lock();
    workUnits.emplace(processedFrameCounter, inputStream.frameCounter(),
        image, debugImage);
    workUnitsMutex.unlock();
    workUnitsConditionVar.notify_one();

    processedFrameCounter += 1;

    frameSkipCounter = frameSkip;

    if (config->debugWindow) {
        if (!debugImageHasContent) {
            std::unique_lock<std::mutex> debugImageLock(debugImageMutex);
            debugImageConditionVar.wait(debugImageLock);
            debugImageLock.unlock();
        }
        drawDebugWindow();
    }
}

void App::startWorkers() {
    auto count = std::thread::hardware_concurrency();

    std::cerr << "Worker count: " << count << std::endl;

    for (size_t index = 0; index < count; index++) {
        auto thread = std::make_shared<std::thread>(std::bind(&App::workerEntry, this));
        workers.push_back(thread);
    }
}

void App::workerEntry() {
    std::cerr << "Worker started" << std::endl;

    AppWorker worker(config);

    while (running) {
        workUnitsMutex.lock();

        if (workUnits.empty()) {
            workUnitsMutex.unlock();
            std::unique_lock<std::mutex> workUnitsLock(workUnitsMutex);
            workUnitsConditionVar.wait(workUnitsLock);
            workUnitsLock.unlock();
            continue;
        }

        auto workUnit = workUnits.front();
        workUnits.pop();
        workUnitsMutex.unlock();

        workUnitsConditionVar.notify_all();

        worker.processWorkUnit(workUnit);
        debugImageMutex.lock();
        workUnit.debugImage.copyTo(debugImage);
        debugImageHasContent = true;
        debugImageMutex.unlock();
        debugImageConditionVar.notify_all();
    }

    std::cerr << "Worker stopped" << std::endl;
}

void App::drawDebugWindow() {
     const std::string windowName = "tppocr";

    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    debugImageMutex.lock();
    cv::imshow(windowName, debugImage);
    debugImageMutex.unlock();

    if (config->frameStepping) {
        cv::waitKey(0);
    } else {
        cv::waitKey(10);
    }
}

}
