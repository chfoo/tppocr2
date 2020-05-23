#include <string>

#include <iostream>
#include <memory>

#include <opencv2/core/utility.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/ocl.hpp>

#include "Config.hpp"
#include "OCR.hpp"
#include "App.hpp"
#include "TextDetector.hpp"

namespace tppocr {

void printOpenCLInfo() {
    // Based on https://stackoverflow.com/a/28644721/1524507
    if (!cv::ocl::haveOpenCL()) {
        std::cerr << "Don't have OpenCL" << std::endl;
        return;
    }

    cv::ocl::Context context;

    if (!context.create(cv::ocl::Device::TYPE_GPU)) {
        std::cerr << "OpenCL create context failed." << std::endl;
        return;
    }

    std::cerr << "OpenCL device count: " << context.ndevices() << std::endl;
    for (size_t i = 0; i < context.ndevices(); i++) {
        cv::ocl::Device device = context.device(i);

        std::cerr << "name: " << device.name() << std::endl
            << "available: " << device.available() << std::endl
            << "imageSupport: " << device.imageSupport() << std::endl
            << "OpenCL_C_Version: " << device.OpenCL_C_Version() << std::endl;
    }
}

int main(int argc, char const *argv[]) {
    const std::string keys =
        "{help h | | Help message}"
        "{@config | | Path to toml configuration file}"
        "{@url | | URL or path to image/video file}"
        "{debug-window | | Show a GUI window with debugging image}"
        "{frame-stepping | | Whether the GUI window waits for a keypress before continuing}"
        "{cpu | | Tell OpenCV to prefer CPU target}"
        "{opencl | | Tell OpenCV to prefer OpenCL target}"
        "{cuda | | Tell OpenCV to prefer CUDA backend and target}"
        "{inference | | Tell OpenCV to prefer Intel Inference Engine backend}"
        "{profiling | | Print timer and profile statistics}"
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

    config->debugWindow = argParser.get<bool>("debug-window");
    config->frameStepping = argParser.get<bool>("frame-stepping");
    config->preferCPU = argParser.get<bool>("cpu");
    config->preferOpenCL = argParser.get<bool>("opencl");
    config->preferCUDA = argParser.get<bool>("cuda");
    config->preferInference = argParser.get<bool>("inference");
    config->profiling = argParser.get<bool>("profiling");
    config->parseFromTOML(configPath);

    if (config->debugWindow) {
        std::cerr << "debug window enabled" << std::endl;
    }

    std::cerr << "Available OpenCV backends: ";

    for (auto & item : cv::dnn::getAvailableBackends()) {
        std::cerr << " " << item.first << " : " << item.second << std::endl;
    }

    printOpenCLInfo();

    App app(config);

    app.run();

    std::cerr << "Done." << std::endl;

    return 0;
}

}

int main(int argc, char const *argv[]) {
    return tppocr::main(argc, argv);
}
