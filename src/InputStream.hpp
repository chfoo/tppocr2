#pragma once

#include <string>
#include <limits>
#include <stdint.h>
#include <vector>
#include <memory>
#include <functional>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

#include "Config.hpp"

namespace tppocr {

class InputStream {
    AVFormatContext * formatContext = nullptr;
    AVCodec * videoCodec = nullptr;
    AVCodecParameters * videoCodecParameters = nullptr;
    AVCodecContext * videoCodecContext = nullptr;
    AVFrame * frame = nullptr;
    AVPacket * packet = nullptr;
    int videoStreamIndex = std::numeric_limits<unsigned int>::max();
    unsigned int videoWidth = 0;
    unsigned int videoHeight = 0;
    AVFrame * frameBGR = nullptr;
    uint8_t * frameBGRBuffer = nullptr;
    SwsContext * scalerContext = nullptr;
    double fps_ = 0;

    bool running = false;

public:
    std::function<void()> callback;

    explicit InputStream(std::shared_ptr<Config> config);
    ~InputStream();

    bool isRunning();
    unsigned int videoFrameWidth();
    unsigned int videoFrameHeight();
    uint8_t * videoFrameData();
    double fps();

    void runOnce();
    void convertFrameToBGR();

private:
    void checkError(int errorCode, const std::string errorMessage);
    void findVideoStream();
    void createVideoBuffers();
};

}
