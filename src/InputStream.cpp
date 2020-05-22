#include "InputStream.hpp"

#include <stdexcept>
#include <iostream>

namespace tppocr {

InputStream::InputStream(std::shared_ptr<Config> config) {
    formatContext = avformat_alloc_context();

    if (!formatContext) {
        throw std::runtime_error("avformat_alloc_context failed");
    }

    checkError(
        avformat_open_input(&formatContext, config->url.c_str(), nullptr, nullptr),
        "avformat_open_input failed"
    );
    checkError(
        avformat_find_stream_info(formatContext, nullptr),
        "avformat_find_stream_info failed"
    );

    std::cerr << "ffmpeg format: " << formatContext->iformat->name << " "
        << "duration: " << formatContext->duration << " "
        << "bit rate: " << formatContext->bit_rate << " "
        << "number of streams: " << formatContext->nb_streams << std::endl;

    frame = av_frame_alloc();

    if (!frame) {
        throw std::runtime_error("av_frame_alloc failed");
    }

    packet = av_packet_alloc();

    if (!packet) {
        throw std::runtime_error("av_packet_alloc failed");
    }

    findVideoStream();
    createVideoBuffers();

    running = true;
}

InputStream::~InputStream() {
    if (formatContext != nullptr) {
        avformat_free_context(formatContext);
    }
}

void InputStream::checkError(int errorCode, const std::string errorMessage) {
    if (errorCode < 0) {
        throw std::runtime_error("InputStream error: " + std::to_string(errorCode) + errorMessage);
    }
}

void InputStream::findVideoStream() {
    for (unsigned int index = 0; index < formatContext->nb_streams; index++) {
        auto currentCodecParams = formatContext->streams[index]->codecpar;
        auto currentCodec = avcodec_find_decoder(currentCodecParams->codec_id);

        if (!currentCodec) {
            throw std::runtime_error("avcodec_find_decoder failed");
        }

        std::cerr << "ffmpeg stream: " << index << " "
            << "codec type: " << currentCodecParams->codec_type << " "
            << std::endl;

        if (currentCodecParams->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = index;
            videoCodec = currentCodec;
            videoCodecParameters = currentCodecParams;
            break;
        }
    }

    if (!videoCodec) {
        throw std::runtime_error("failed to find video stream");
    }

    videoCodecContext = avcodec_alloc_context3(videoCodec);

    if (!videoCodecContext) {
        throw std::runtime_error("avcodec_alloc_context3 failed");
    }

    checkError(
        avcodec_parameters_to_context(videoCodecContext, videoCodecParameters),
        "avcodec_parameters_to_context failed"
    );

    checkError(
        avcodec_open2(videoCodecContext, videoCodec, nullptr),
        "avcodec_open2 failed"
    );
}

void InputStream::createVideoBuffers() {
    videoWidth = videoCodecContext->width;
    videoHeight = videoCodecContext->height;

    frameBGR = av_frame_alloc();

    if (!frameBGR) {
        throw std::runtime_error("av_frame_alloc failed (bgr)");
    }

    frameBGR->width = videoWidth;
    frameBGR->height = videoHeight;

    const int align = 32;

    auto size = av_image_get_buffer_size(AV_PIX_FMT_BGR24, videoWidth, videoHeight, align);
    frameBGRBuffer = static_cast<uint8_t*>(av_malloc(size));

    if (!frameBGRBuffer) {
        throw std::runtime_error("av_malloc failed (frameBGRBuffer)");
    }

    av_image_fill_arrays(frameBGR->data, frameBGR->linesize,
        frameBGRBuffer, AV_PIX_FMT_BGR24,
        frameBGR->width, frameBGR->height, align);

    scalerContext = sws_getContext(
        videoCodecContext->width,
        videoCodecContext->height,
        videoCodecContext->pix_fmt,
        frameBGR->width, frameBGR->height, AV_PIX_FMT_BGR24,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );
}

bool InputStream::isRunning() {
    return running;
}

unsigned int InputStream::videoFrameWidth() {
    return videoWidth;
}

unsigned int InputStream::videoFrameHeight() {
    return videoHeight;
}

uint8_t * InputStream::videoFrameData() {
    return frameBGRBuffer;
}

void InputStream::runOnce() {
    auto errorCode = av_read_frame(formatContext, packet);

    if (errorCode < 0) {
        running = false;
        return;
    }

    if (packet->stream_index != videoStreamIndex) {
        return;
    }

    checkError(
        avcodec_send_packet(videoCodecContext, packet),
        "avcodec_send_packet failed"
    );

    while (true) {
        errorCode = avcodec_receive_frame(videoCodecContext, frame);

        if (errorCode == AVERROR(EAGAIN) || errorCode == AVERROR_EOF) {
            return;
        } else if (errorCode < 0) {
            throw std::runtime_error("avcodec_receive_frame failed");
        }

        sws_scale(scalerContext, frame->data, frame->linesize, 0, frame->height,
            frameBGR->data, frame->linesize);

        callback();
    }
}

}
