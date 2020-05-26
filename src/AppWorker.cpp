#include "AppWorker.hpp"

#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "mathutil.hpp"

namespace tppocr {

AppWorker::AppWorker(std::shared_ptr<Config> config) :
    config(config), resource(config),
    dummyWorkUnit(0, 0, cv::Mat(), cv::Mat()), workUnit(dummyWorkUnit) {}

void AppWorker::processWorkUnit(const WorkUnit & workUnit) {
    this->workUnit = workUnit;

    for (auto & region : config->regions) {
        processRegion(region);
    }

    drawFrameInfo(workUnit);
}

void AppWorker::processRegion(const Region & region) {
    drawRegion(region);

    cv::Mat regionImage(
        roundUp2(region.height, 32),
        roundUp2(region.width, 32),
        workUnit.image.type(),
        cv::Scalar(0, 0)
    );

    if (region.alwaysHasText) {
        processTextBlock(region,
            cv::Rect(region.x, region.y, region.width, region.height));
        return;
    }

    cv::Mat subImage = cv::Mat(workUnit.image,
        cv::Rect(region.x, region.y, region.width, region.height));

    subImage.copyTo(regionImage(cv::Rect(0, 0, region.width, region.height)));

    auto & textDetector = resource.textDetectors.at(region.name);

    cv::TickMeter tickMeter;
    if (config->profiling) {
        tickMeter.start();
    }
    textDetector.processImage(regionImage);

    if (config->profiling) {
        tickMeter.stop();
        std::cerr << "Detecting region " << region.name
            << " tick time: " << tickMeter.getTimeSec() << std::endl;
    }

    auto & detections = textDetector.getDetections();
    auto & confidences = textDetector.getConfidences();
    auto & indices = textDetector.getIndices();

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

        drawDetection(region, box, confidence);
    }

    minX = std::max(minX - 5, 0);
    minY = std::max(minY - 5, 0);
    maxX = std::min(maxX + 5, workUnit.image.cols);
    maxY = std::min(maxY + 5, workUnit.image.rows);

    cv::Rect boundingBox(minX, minY, maxX - minX, maxY - minY);

    processTextBlock(region, boundingBox);
}

void AppWorker::drawRegion(const Region & region) {
    cv::rectangle(workUnit.debugImage,
        cv::Rect(region.x, region.y, region.width, region.height),
        CV_RGB(255, 0, 255));
    cv::putText(workUnit.debugImage, region.name, cv::Point(region.x, region.y),
        cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 255));
}

void AppWorker::drawDetection(const Region & region, const cv::RotatedRect & box,
        float confidence) {
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

void AppWorker::processTextBlock(const Region & region, const cv::Rect & box) {
    cv::Mat regionImage = cv::Mat(workUnit.image, box);
    auto & ocr = resource.textRecognizers.at(region.name);

    cv::TickMeter tickMeter;
    if (config->profiling) {
        tickMeter.start();
    }
    ocr.processImage(regionImage);

    if (config->profiling) {
        tickMeter.stop();
        std::cerr << "Recognizing box " << region.name
            << " tick time: " << tickMeter.getTimeSec() << std::endl;
    }

    auto text = ocr.getText();

    auto confidence = ocr.getMeanConfidence();

    if (confidence >= config->recognizerConfidenceThreshold) {
        // TODO: emit text
        // (in thread safe manner if threading)
        // emitText();
    }

    drawTextBlock(region, box);
    drawOCRThresholdImage(region, box);
    drawOCRLineBoundaries(region, box);
    drawOCRText(region, box);
}

void AppWorker::drawTextBlock(const Region & region, const cv::Rect & box) {
    cv::rectangle(workUnit.debugImage,
        cv::Rect(box.x, box.y, box.width, box.height),
        CV_RGB(255, 255, 0));
}

void AppWorker::drawOCRText(const Region & region, const cv::Rect & box) {
    auto & ocr = resource.textRecognizers.at(region.name);
    auto confidence = ocr.getMeanConfidence();
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

    auto & text = ocr.getText();

    // cv::putText(debugImage, text,
    //     cv::Point(box.x, box.y + offsetY),
    //     cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 255));
    resource.freetype->putText(workUnit.debugImage, text,
        cv::Point(box.x, box.y + offsetY),
        16, CV_RGB(255, 127, 0), -1, cv::LINE_8, true);
}

void AppWorker::drawOCRThresholdImage(const Region & region, const cv::Rect & box) {
    auto & ocr = resource.textRecognizers.at(region.name);
    auto thresholdImage = ocr.getThresholdedImage();
    int offsetY = 0;

    if (box.y + box.height < workUnit.image.rows) {
        offsetY = box.height;
    } else if (box.y - box.height >= 0) {
        offsetY = -box.height;
    }

    auto drawingRect = cv::Rect(box.x, box.y + offsetY, box.width, thresholdImage.rows);

    thresholdImage.copyTo(workUnit.debugImage(drawingRect));
}

void AppWorker::drawOCRLineBoundaries(const Region & region, const cv::Rect & box) {
    auto & ocr = resource.textRecognizers.at(region.name);
    auto lineBoundaries = ocr.getLineBoundaries();
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

void AppWorker::drawFrameInfo(const WorkUnit & workUnit) {
    char text[50];
    snprintf(text, 50, "Frame %d, id %d", workUnit.frameID, workUnit.id);

    cv::putText(workUnit.debugImage, text,
        cv::Point(0, 40),
        cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 127, 0));
}


}
