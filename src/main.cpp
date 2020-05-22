#include <string>

#include <iostream>
#include <memory>

#include <opencv2/core/utility.hpp>

#include "Config.hpp"
#include "OCR.hpp"
#include "App.hpp"
#include "TextDetector.hpp"

namespace tppocr {

int main(int argc, char const *argv[]) {
    const std::string keys =
        "{help h | | Help message}"
        "{@config | | Path to toml configuration file}"
        "{@url | | URL or path to image/video file}"
        "{real-time | | Process stream as live stream}"
        "{processing-fps | 60 | FPS at which frames are sampled and processed}"
        "{tessdata | ./tessdata_fast/ | Path to directory of trained Tesseract data}"
        "{detector-model | ./frozen_east_text_detection.pb | Path to EAST .pb file (TensorFlow trained model)}"
    ;

    cv::CommandLineParser argParser(argc, argv, keys);

    if (argParser.has("help")) {
        argParser.printMessage();
        return 0;
    }

    std::shared_ptr<Config> config = std::make_shared<Config>();

    auto configPath = argParser.get<std::string>(0);
    config->url = argParser.get<std::string>(1);

    if (configPath == "" || config->url == "") {
        std::cerr << "Missing config or url" << std::endl;
        return 1;
    }

    config->parseFromTOML(configPath);

    App app(config);

    app.run();

    std::cerr << "Done." << std::endl;

    return 0;
}

}

int main(int argc, char const *argv[]) {
    return tppocr::main(argc, argv);
}
