#include "App.hpp"

#include <stdio.h>
#include <limits>
#include <cmath>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "mathutil.hpp"

namespace tppocr {

App::App(std::shared_ptr<Config> config) :
    config(config),
    inputStream(config) {

    for (auto & region : config->regions) {
        WorkUnitResourceFactory workUnitResourcePoolFactory(config, region);
        workUnitResources.emplace(region.name,
            new Poco::ObjectPool<WorkUnitResource,WorkUnitResource*,WorkUnitResourceFactory>(workUnitResourcePoolFactory, 32, 32));
    }

    inputStream.callback = std::bind(&App::frameCallback, this);

    frameImage = cv::Mat(
        inputStream.videoFrameHeight(),
        inputStream.videoFrameWidth(),
        CV_8UC3,
        inputStream.videoFrameData()
    );

    frameSkip = std::floor(inputStream.fps() / config->processingFPS);

    std::cerr << "Skipping every " << frameSkip << " frame(s)." << std::endl;
}

void App::run() {
    while (inputStream.isRunning()) {
        inputStream.runOnce();
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

    for (size_t index = 0; index < config->regions.size(); index++) {
        workUnits.emplace_back(processedFrameCounter, inputStream.frameCounter(),
            config->regions.at(index), image, debugImage);
    }

    processedFrameCounter += 1;

    if (workUnits.size() >= 4) {
        processWorkUnits();
    }

    frameSkipCounter = frameSkip;
}

void App::updateDebugWindow() {
    if (config->debugWindow) {
        const std::string windowName = "tppocr";

        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
        drawFrameInfo(workUnits.back());
        cv::imshow(windowName, workUnits.back().debugImage);

        if (config->frameStepping) {
            cv::waitKey(0);
        } else {
            cv::waitKey(1);
        }
    }
}

void App::processWorkUnits() {
    // std::cerr << "processWorkUnits size=" << workUnits.size() << std::endl;

    // for (auto & workUnit : workUnits) {
    //     std::cerr << "   id=" << workUnit.id << " " << workUnit.frameID << std::endl;
    // }

    cv::parallel_for_(
        cv::Range(0, workUnits.size()),
        [&](const cv::Range & range) {
            for (int index = range.start; index < range.end; index++) {
                auto & workUnit = workUnits.at(index);
                auto workUnitResource = workUnitResources.at(workUnit.region.name)->borrowObject();
                workUnit.freetype = workUnitResource->freetype;
                workUnit.ocr = &workUnitResource->ocr;
                workUnit.textDetector = &workUnitResource->textDetector;
                processRegion(workUnit);
                workUnitResources.at(workUnit.region.name)->returnObject(workUnitResource);
            }
        }
    );

    updateDebugWindow();

    workUnits.clear();
}

void App::processRegion(const WorkUnit & workUnit) {
    drawRegion(workUnit);

    auto & region = workUnit.region;

    cv::Mat regionImage(
        roundUp2(region.height, 32),
        roundUp2(region.width, 32),
        workUnit.image.type(),
        cv::Scalar(0, 0)
    );

    if (region.alwaysHasText) {
        processTextBlock(workUnit,
            cv::Rect(region.x, region.y, region.width, region.height));
        return;
    }

    cv::Mat subImage = cv::Mat(workUnit.image,
        cv::Rect(region.x, region.y, region.width, region.height));

    subImage.copyTo(regionImage(cv::Rect(0, 0, region.width, region.height)));

    auto textDetector = workUnit.textDetector;

    cv::TickMeter tickMeter;
    if (config->profiling) {
        tickMeter.start();
    }
    textDetector->processImage(regionImage);

    if (config->profiling) {
        tickMeter.stop();
        std::cerr << "Detecting region " << region.name
            << " tick time: " << tickMeter.getTimeSec() << std::endl;
    }

    auto & detections = textDetector->getDetections();
    auto & confidences = textDetector->getConfidences();
    auto & indices = textDetector->getIndices();

    if (indices.empty()) {
        return;
    }

    int minX = std::numeric_limits<int>::max();
    int minY = std::numeric_limits<int>::max();
    int maxX = std::numeric_limits<int>::min();
    int maxY = std::numeric_limits<int>::min();

    for (auto index : indices) {
        auto & box = detections.at(index);
        auto confidence = confidences.at(index);
        auto boundingBox = box.boundingRect();

        minX = std::min(minX, region.x + boundingBox.x);
        minY = std::min(minY, region.y + boundingBox.y);
        maxX = std::max(maxX, region.x + boundingBox.x + boundingBox.width);
        maxY = std::max(maxY, region.y + boundingBox.y + boundingBox.height);

        drawDetection(workUnit, box, confidence);
    }

    minX = std::max(minX - 5, 0);
    minY = std::max(minY - 5, 0);
    maxX = std::min(maxX + 5, workUnit.image.cols);
    maxY = std::min(maxY + 5, workUnit.image.rows);

    cv::Rect boundingBox(minX, minY, maxX - minX, maxY - minY);

    processTextBlock(workUnit, boundingBox);
}

void App::drawRegion(const WorkUnit & workUnit) {
    auto & region = workUnit.region;

    cv::rectangle(workUnit.debugImage,
        cv::Rect(region.x, region.y, region.width, region.height),
        CV_RGB(255, 0, 255));
    cv::putText(workUnit.debugImage, region.name, cv::Point(region.x, region.y),
        cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 255));
}

void App::drawDetection(const WorkUnit & workUnit, const cv::RotatedRect & box,
        float confidence) {
    auto & region = workUnit.region;
    cv::Point2f points[4];
    box.points(points);

    for (size_t index = 0; index < 4; index++) {
        points[index].x += region.x;
        points[index].y += region.y;
    }

    for (size_t index = 0; index < 4; index++) {
        cv::line(workUnit.debugImage, points[index], points[(index + 1) % 4],
            CV_RGB(0, 255, 0));
    }

    char confidenceString[10];
    snprintf(confidenceString, 10, "%0.2f", confidence);

    cv::putText(workUnit.debugImage, confidenceString,
        points[1],
        cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0));
}

void App::processTextBlock(const WorkUnit & workUnit, const cv::Rect & box) {
    auto & region = workUnit.region;
    cv::Mat regionImage = cv::Mat(workUnit.image, box);
    auto ocr = workUnit.ocr;

    cv::TickMeter tickMeter;
    if (config->profiling) {
        tickMeter.start();
    }
    ocr->processImage(regionImage);

    if (config->profiling) {
        tickMeter.stop();
        std::cerr << "Recognizing box " << region.name
            << " tick time: " << tickMeter.getTimeSec() << std::endl;
    }

    auto text = ocr->getText();

    auto confidence = ocr->getMeanConfidence();

    if (confidence >= config->recognizerConfidenceThreshold) {
        // TODO: emit text
        // (in thread safe manner if threading)
        // emitText();
    }

    drawTextBlock(workUnit, box);
    drawOCRThresholdImage(workUnit, box);
    drawOCRLineBoundaries(workUnit, box);
    drawOCRText(workUnit, box);
}

void App::drawTextBlock(const WorkUnit & workUnit, const cv::Rect & box) {
    cv::rectangle(workUnit.debugImage,
        cv::Rect(box.x, box.y, box.width, box.height),
        CV_RGB(255, 255, 0));
}

void App::drawOCRText(const WorkUnit & workUnit, const cv::Rect & box) {
    auto & region = workUnit.region;
    auto ocr = workUnit.ocr;
    auto confidence = ocr->getMeanConfidence();
    char confidenceString[10];
    snprintf(confidenceString, 10, "%0.2f", confidence);

    int offsetY = 0;

    if (box.y + box.height < workUnit.image.rows) {
        offsetY = box.height * 2;
    } else if (box.y - box.height >= 0) {
        offsetY = -box.height;
    }

    cv::putText(workUnit.debugImage, confidenceString,
        cv::Point(box.x + box.width, box.y + offsetY * 0.75),
        cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 255));

    auto & text = ocr->getText();

    // cv::putText(debugImage, text,
    //     cv::Point(box.x, box.y + offsetY),
    //     cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 255));
    workUnit.freetype->putText(workUnit.debugImage, text,
        cv::Point(box.x, box.y + offsetY),
        16, CV_RGB(255, 127, 0), -1, cv::LINE_8, true);
}

void App::drawOCRThresholdImage(const WorkUnit & workUnit, const cv::Rect & box) {
    auto & region = workUnit.region;
    auto ocr = workUnit.ocr;
    auto thresholdImage = ocr->getThresholdedImage();
    int offsetY = 0;

    if (box.y + box.height < workUnit.image.rows) {
        offsetY = box.height;
    } else if (box.y - box.height >= 0) {
        offsetY = -box.height;
    }

    auto drawingRect = cv::Rect(box.x, box.y + offsetY, box.width, thresholdImage.rows);

    thresholdImage.copyTo(workUnit.debugImage(drawingRect));
}

void App::drawOCRLineBoundaries(const WorkUnit & workUnit, const cv::Rect & box) {
    auto & region = workUnit.region;
    auto ocr = workUnit.ocr;
    auto lineBoundaries = ocr->getLineBoundaries();
    int offsetY = 0;

    if (box.y + box.height < workUnit.image.rows) {
        offsetY = box.height;
    } else if (box.y - box.height >= 0) {
        offsetY = -box.height;
    }

    for (auto & lineBoundary : lineBoundaries) {
        auto drawingRect = cv::Rect(
            box.x + lineBoundary.x,
            box.y + lineBoundary.y + offsetY,
            lineBoundary.width,
            lineBoundary.height
        );
        cv::rectangle(workUnit.debugImage, drawingRect, CV_RGB(0, 255, 255));
    }
}

void App::drawFrameInfo(const WorkUnit & workUnit) {
    char text[50];
    snprintf(text, 50, "Frame %d, id %d", workUnit.frameID, workUnit.id);

    cv::putText(workUnit.debugImage, text,
        cv::Point(0, 40),
        cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 127, 0));
}


}
