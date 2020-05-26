#pragma once

#include <memory>

#include "WorkUnitResource.hpp"
#include "WorkUnit.hpp"
#include "Region.hpp"

namespace tppocr {

class AppWorker {
    std::shared_ptr<Config> config;
    WorkUnitResource resource;
    WorkUnit dummyWorkUnit;
    WorkUnit & workUnit;

public:
    AppWorker(std::shared_ptr<Config> config);

    void processWorkUnit(const WorkUnit & workUnit);
private:
    void processRegion(const Region & region);
    void drawRegion(const Region & region);
    void drawDetection(const Region & region, const cv::RotatedRect & box,
        float confidence);
    void processTextBlock(const Region & region, const cv::Rect & box);
    void drawTextBlock(const Region & region, const cv::Rect & box);
    void drawOCRText(const Region & region, const cv::Rect & box);
    void drawOCRThresholdImage(const Region & region, const cv::Rect & box);
    void drawOCRLineBoundaries(const Region & region, const cv::Rect & box);
    void drawFrameInfo(const WorkUnit & workUnit);
};

}
