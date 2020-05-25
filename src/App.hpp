#pragma once

#include <iostream>
#include <memory>
#include <functional>
#include <unordered_map>

#include <opencv2/core.hpp>
#include <opencv2/freetype.hpp>
#include <Poco/ObjectPool.h>

#include "Config.hpp"
#include "OCR.hpp"
#include "InputStream.hpp"
#include "TextDetector.hpp"
#include "WorkUnit.hpp"
#include "WorkUnitResource.hpp"

namespace tppocr {

class WorkUnitResourceFactory {
    std::shared_ptr<Config> config;
    const Region & region;

public:
    explicit WorkUnitResourceFactory(std::shared_ptr<Config> config, const Region & region) :
        config(config), region(region) {}
    WorkUnitResourceFactory(const WorkUnitResourceFactory & other) :
        config(other.config), region(other.region)  {}

    WorkUnitResource * createObject() { return new WorkUnitResource(config, region); }
    void activateObject(WorkUnitResource * object) {}
    bool validateObject(WorkUnitResource * object) { return true; }
    void deactivateObject(WorkUnitResource * object) {}
    void destroyObject(WorkUnitResource * object) { delete object; }
};

class App {
    std::shared_ptr<Config> config;
    std::unordered_map<std::string,Poco::ObjectPool<WorkUnitResource,WorkUnitResource*,WorkUnitResourceFactory>*> workUnitResources;
    InputStream inputStream;
    cv::Mat frameImage;
    std::vector<WorkUnit> workUnits;
    unsigned int frameSkip = 0;
    unsigned int frameSkipCounter = 0;
    unsigned int processedFrameCounter = 0;

public:
    explicit App(std::shared_ptr<Config> config);

    void run();

private:
    void frameCallback();
    void updateDebugWindow();
    void processWorkUnits();
    void processRegion(const WorkUnit & workUnit);
    void drawRegion(const WorkUnit & workUnit);
    void drawDetection(const WorkUnit & workUnit, const cv::RotatedRect & box,
        float confidence);
    void processTextBlock(const WorkUnit & workUnit, const cv::Rect & box);
    void drawTextBlock(const WorkUnit & workUnit, const cv::Rect & box);
    void drawOCRText(const WorkUnit & workUnit, const cv::Rect & box);
    void drawOCRThresholdImage(const WorkUnit & workUnit, const cv::Rect & box);
    void drawOCRLineBoundaries(const WorkUnit & workUnit, const cv::Rect & box);
    void drawFrameInfo(const WorkUnit & workUnit);
};

}
