#include "App.hpp"

#include <stdio.h>
#include <limits>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "mathutil.hpp"

namespace tppocr {

App::App(std::shared_ptr<Config> config) :
    config(config),
    ocr(config),
    inputStream(config) {

    for (auto & region : config->regions) {
        textDetectors.emplace(region.name, config);
    }

    inputStream.callback = std::bind(&App::frameCallback, this);

    image = cv::Mat(
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

    freetype = cv::freetype::createFreeType2();
    freetype->loadFontData("/usr/share/fonts/truetype/unifont/unifont.ttf", 0);
}

void App::run() {
    while (inputStream.isRunning()) {
        inputStream.runOnce();
    }
}

void App::frameCallback() {
    image.copyTo(debugImage);

    processRegions();

    const std::string windowName = "tppocr";

    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, debugImage);
    cv::waitKey(0);
}


void App::processRegions() {
    for (const auto & region : config->regions) {
        // cv::TickMeter tickMeter;
        // tickMeter.start();
        processRegion(region);
        // tickMeter.stop();
        // std::cerr << "Procesing region " << region.name
        //     << " tick time: " << tickMeter.getTimeSec() << std::endl;
    }
}

void App::processRegion(const Region & region) {
    drawRegion(region);

    cv::Mat regionImage(
        roundUp2(region.height, 32),
        roundUp2(region.width, 32),
        image.type(),
        cv::Scalar(0, 0)
    );

    cv::Mat subImage = cv::Mat(image,
        cv::Rect(region.x, region.y, region.width, region.height));

    subImage.copyTo(regionImage(cv::Rect(0, 0, region.width, region.height)));

    auto & textDetector = textDetectors.at(region.name);

    textDetector.processImage(regionImage);

    auto & detections = textDetector.getDetections();
    auto & confidences = textDetector.getConfidences();
    auto & indices = textDetector.getIndices();

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
    maxX = std::min(maxX + 5, image.cols);
    maxY = std::min(maxY + 5, image.rows);

    cv::Rect boundingBox(minX, minY, maxX - minX, maxY - minY);

    processTextBlock(region, boundingBox);
}

void App::drawRegion(const Region & region) {
    cv::rectangle(debugImage,
        cv::Rect(region.x, region.y, region.width, region.height),
        CV_RGB(255, 0, 255));
    cv::putText(debugImage, region.name, cv::Point(region.x, region.y),
        cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 255));
}

void App::drawDetection(const Region & region, const cv::RotatedRect & box,
        float confidence) {
    cv::Point2f points[4];
    box.points(points);

    for (size_t index = 0; index < 4; index++) {
        points[index].x += region.x;
        points[index].y += region.y;
    }

    for (size_t index = 0; index < 4; index++) {
        cv::line(debugImage, points[index], points[(index + 1) % 4],
            CV_RGB(0, 255, 0));
    }

    char confidenceString[10];
    snprintf(confidenceString, 10, "%0.2f", confidence);

    cv::putText(debugImage, confidenceString,
        points[1],
        cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0));
}

void App::processTextBlock(const Region & region, const cv::Rect & box) {
    cv::Mat regionImage = cv::Mat(image, box);
    ocr.processImage(regionImage);

    auto text = ocr.getText();

    drawTextBlock(region, box);
    drawOCRThresholdImage(box);
    drawOCRLineBoundaries(box);
    drawOCRText(box);
}

void App::drawTextBlock(const Region & region, const cv::Rect & box) {
    cv::rectangle(debugImage,
        cv::Rect(box.x, box.y, box.width, box.height),
        CV_RGB(255, 255, 0));
}

void App::drawOCRText(const cv::Rect & box) {
    auto confidence = ocr.getMeanConfidence();
    char confidenceString[10];
    snprintf(confidenceString, 10, "%0.2f", confidence);

    int offsetY = 0;

    if (box.y + box.height < image.rows) {
        offsetY = box.height * 2;
    } else if (box.y - box.height >= 0) {
        offsetY = -box.height;
    }

    cv::putText(debugImage, confidenceString,
        cv::Point(box.x + box.width, box.y + offsetY * 0.75),
        cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 255));

    auto & text = ocr.getText();

    // cv::putText(debugImage, text,
    //     cv::Point(box.x, box.y + offsetY),
    //     cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 255));
    freetype->putText(debugImage, text,
        cv::Point(box.x, box.y + offsetY),
        16, CV_RGB(0, 255, 255), -1, cv::LINE_8, true);
}

void App::drawOCRThresholdImage(const cv::Rect & box) {
    auto thresholdImage = ocr.getThresholdedImage();
    int offsetY = 0;

    if (box.y + box.height < image.rows) {
        offsetY = box.height;
    } else if (box.y - box.height >= 0) {
        offsetY = -box.height;
    }

    auto drawingRect = cv::Rect(box.x, box.y + offsetY, box.width, thresholdImage.rows);

    thresholdImage.copyTo(debugImage(drawingRect));
}

void App::drawOCRLineBoundaries(const cv::Rect & box) {
    auto lineBoundaries = ocr.getLineBoundaries();
    int offsetY = 0;

    if (box.y + box.height < image.rows) {
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
        cv::rectangle(debugImage, drawingRect, CV_RGB(0, 255, 255));
    }
}

}
