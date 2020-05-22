#include "OCR.hpp"

#include <stdexcept>
#include <assert.h>

#include <leptonica/allheaders.h>

namespace tppocr {

OCR::OCR(std::shared_ptr<Config> config) {
    tesseract.reset(new tesseract::TessBaseAPI());\

    auto errorCode = tesseract->Init(config->tessdataPath.c_str(),
        "eng+jpn+chi_sim+chi_tra+kor+spa+deu+ita");

    if (errorCode) {
        throw std::runtime_error("Tesseract error " + std::to_string(errorCode));
    }

    tesseract->SetVariable("classify_enable_learning", "0");
    tesseract->SetVariable("user_defined_dpi", "90");
}

const std::string & OCR::getText() {
    return text;
}

float OCR::getMeanConfidence() {
    return meanConfidence;
}

void OCR::processImage(const cv::Mat & image) {
    auto pix = pixCreate(image.cols, image.rows, 32);

    assert(pix);

    for (int heightIndex = 0; heightIndex < image.rows; heightIndex++) {
        for (int widthIndex = 0; widthIndex < image.cols; widthIndex++) {
            // The byte ordering is probably wrong??
            auto & vec = image.at<cv::Vec3b>(heightIndex, widthIndex);
            uint32_t pixValue = (vec[0] << 16) | (vec[1] << 8) | (vec[2] << 0);
            auto errorCode = pixSetPixel(pix, widthIndex, heightIndex, pixValue);
            assert(!errorCode);
        }
    }

    tesseract->SetImage(pix);
    pixDestroy(&pix);

    auto cText = tesseract->GetUTF8Text();
    text = std::string(cText);
    delete cText;

    meanConfidence = tesseract->MeanTextConf() / 100.0;
}

cv::Mat OCR::getThresholdedImage() {
    auto pixThresholdedImage = tesseract->GetThresholdedImage();
    const auto width = pixGetWidth(pixThresholdedImage);
    const auto height = pixGetHeight(pixThresholdedImage);
    auto thresholdedImage = cv::Mat(height, width, CV_8UC3);

    for (int heightIndex = 0; heightIndex < height; heightIndex++) {
        for (int widthIndex = 0; widthIndex < width; widthIndex++) {
            uint32_t pixel;
            pixGetPixel(pixThresholdedImage, widthIndex, heightIndex, &pixel);

            uint8_t value = pixel ? 255 : 0;

            auto & vec = thresholdedImage.at<cv::Vec3b>(heightIndex, widthIndex);
            vec[0] = value;
            vec[1] = value;
            vec[2] = value;
        }
    }


    pixDestroy(&pixThresholdedImage);

    return thresholdedImage;
}

std::vector<cv::Rect> OCR::getLineBoundaries() {
    std::vector<cv::Rect> lineBoundaries;

    Pixa * pixArray;
    int * blockIDs;
    Boxa * boxArray = tesseract->GetTextlines(&pixArray, &blockIDs);

    for (int32_t index = 0; index < boxArray->n; index++) {
        Box * box = boxArray->box[index];
        lineBoundaries.emplace_back(box->x, box->y, box->w, box->h);
    }

    return lineBoundaries;
}

}
