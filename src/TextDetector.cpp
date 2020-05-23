#include "TextDetector.hpp"

#include <iostream>

namespace tppocr {

TextDetector::TextDetector(std::shared_ptr<Config> config) {
    network = cv::dnn::readNet(config->detectorModelPath);

    if (config->preferInference) {
        std::cerr << "Set network backend preference to Intel Inference Engine" << std::endl;
        network.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
    }

    if (config->preferCPU) {
        std::cerr << "Set network target preference to CPU" << std::endl;
        network.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } else if (config->preferOpenCL) {
        std::cerr << "Set network target preference to OpenCL" << std::endl;
        network.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    } else if (config->preferCUDA) {
        std::cerr << "Set network backend and target preference to CUDA" << std::endl;
        network.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        network.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }

    outputBlobNames.push_back("feature_fusion/Conv_7/Sigmoid");
    outputBlobNames.push_back("feature_fusion/concat_3");
}

const std::vector<cv::RotatedRect> & TextDetector::getDetections() {
    return detections;
}

const std::vector<float> & TextDetector::getConfidences() {
    return confidences;
}

const std::vector<int> & TextDetector::getIndices() {
    return indices;
}


void TextDetector::processImage(const cv::Mat & image) {
    outputBlobs.clear();
    blob = cv::Mat();

    // Magic values copied from sample
    cv::dnn::blobFromImage(image, blob, 1.0,
        cv::Size(image.cols, image.rows),
        cv::Scalar(123.68, 116.78, 103.94),
        true, false);

    network.setInput(blob);
    network.forward(outputBlobs, outputBlobNames);

    auto & scores = outputBlobs[0];
    auto & geometry = outputBlobs[1];

    detections.clear();
    confidences.clear();
    indices.clear();

    // -------
    // Rest of code mostly copied from sample.

    CV_Assert(scores.dims == 4);
    CV_Assert(geometry.dims == 4);
    CV_Assert(scores.size[0] == 1);
    CV_Assert(geometry.size[0] == 1);
    CV_Assert(scores.size[1] == 1);
    CV_Assert(geometry.size[1] == 5);
    CV_Assert(scores.size[2] == geometry.size[2]);
    CV_Assert(scores.size[3] == geometry.size[3]);

    const int height = scores.size[2];
    const int width = scores.size[3];

    for (int y = 0; y < height; ++y) {
        const float* scoresData = scores.ptr<float>(0, 0, y);
        const float* x0_data = geometry.ptr<float>(0, 0, y);
        const float* x1_data = geometry.ptr<float>(0, 1, y);
        const float* x2_data = geometry.ptr<float>(0, 2, y);
        const float* x3_data = geometry.ptr<float>(0, 3, y);
        const float* anglesData = geometry.ptr<float>(0, 4, y);
        for (int x = 0; x < width; ++x)
        {
            float score = scoresData[x];
            if (score < confidenceMinimumThreshold)
                continue;

            // Decode a prediction.
            // Multiple by 4 because feature maps are 4 time less than input image.
            float offsetX = x * 4.0f, offsetY = y * 4.0f;
            float angle = anglesData[x];
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);
            float h = x0_data[x] + x2_data[x];
            float w = x1_data[x] + x3_data[x];

            cv::Point2f offset(
                offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
            cv::Point2f p1 = cv::Point2f(-sinA * h, -cosA * h) + offset;
            cv::Point2f p3 = cv::Point2f(-cosA * w, sinA * w) + offset;
            cv::RotatedRect r(0.5f * (p1 + p3),
                cv::Size2f(w, h), -angle * 180.0f / (float)CV_PI);
            detections.push_back(r);
            confidences.push_back(score);
        }
    }

    cv::dnn::NMSBoxes(detections, confidences, confidenceMinimumThreshold,
        nonmaximumSuppressionThreshold, indices);
}

}
