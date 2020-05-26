cmake_minimum_required(VERSION 3.15)
project(tppocr2)

set(USE_INFERENCE_ENGINE false CACHE BOOL "Use Intel OpenVINO Inference Engine")

file(GLOB SRC_FILES src/*.cpp)

add_executable(tppocr_exe ${SRC_FILES})

find_path(TESSERACT_INCLUDE_PATH "tesseract/baseapi.h")
find_path(LEPTONICA_INCLUDE_PATH "leptonica/allheaders.h")
find_path(TOMLPLUSPLUS_INCLUDE_PATH "toml++/toml.h")

find_library(TESSERACT_LIBRARY_PATH "tesseract")
find_library(LEPTONICA_LIBRARY_PATH "leptonica")

target_include_directories(tppocr_exe PRIVATE
    "${TESSERACT_INCLUDE_PATH}"
    "${LEPTONICA_INCLUDE_PATH}"
    "${TOMLPLUSPLUS_INCLUDE_PATH}"
)

target_link_libraries(tppocr_exe PRIVATE
    "${TESSERACT_LIBRARY_PATH}"
    "${LEPTONICA_LIBRARY_PATH}"
)


find_path(FFMPEG_AVCODEC_INCLUDE_PATH libavcodec/avcodec.h)
find_path(FFMPEG_AVFORMAT_INCLUDE_PATH libavformat/avformat.h)
find_path(FFMPEG_AVUTIL_INCLUDE_PATH libavutil/avutil.h)
find_path(FFMPEG_AVDEVICE_INCLUDE_PATH libavdevice/avdevice.h)
find_path(FFMPEG_SWSCALE_INCLUDE_PATH libswscale/swscale.h)
find_library(FFMPEG_AVCODEC_LIBRARY_PATH avcodec)
find_library(FFMPEG_AVFORMAT_LIBRARY_PATH avformat)
find_library(FFMPEG_AVUTIL_LIBRARY_PATH avutil)
find_library(FFMPEG_AVDEVICE_LIBRARY_PATH avdevice)
find_library(FFMPEG_SWSCALE_LIBRARY_PATH swscale)

target_include_directories(tppocr_exe PRIVATE
    "${FFMPEG_AVCODEC_INCLUDE_PATH}"
    "${FFMPEG_AVFORMAT_INCLUDE_PATH}"
    "${FFMPEG_AVUTIL_INCLUDE_PATH}"
    "${FFMPEG_AVDEVICE_INCLUDE_PATH}"
    "${FFMPEG_SWSCALE_INCLUDE_PATH}"
)
target_link_libraries(tppocr_exe PRIVATE
    "${FFMPEG_AVCODEC_LIBRARY_PATH}"
    "${FFMPEG_AVFORMAT_LIBRARY_PATH}"
    "${FFMPEG_AVUTIL_LIBRARY_PATH}"
    "${FFMPEG_AVDEVICE_LIBRARY_PATH}"
    "${FFMPEG_SWSCALE_LIBRARY_PATH}"
)


find_package(OpenCV REQUIRED)
target_include_directories(tppocr_exe PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tppocr_exe PRIVATE ${OpenCV_LIBS} )

if(USE_INFERENCE_ENGINE)
    find_package(InferenceEngine)
    target_include_directories(tppocr_exe PRIVATE ${InferenceEngine_INCLUDE_DIRS})
    target_link_libraries(tppocr_exe PRIVATE ${InferenceEngine_LIBRARIES} dl)
endif()

set_property(TARGET tppocr_exe PROPERTY C_STANDARD 11)
set_property(TARGET tppocr_exe PROPERTY CXX_STANDARD 17)

if(MSVC)
    target_compile_options(tppocr_exe PRIVATE /W4)
else()
    target_compile_options(tppocr_exe PRIVATE -Wall -Wextra -pedantic)
    target_link_options(tppocr_exe PRIVATE -pthread)
endif()

set_target_properties(tppocr_exe PROPERTIES OUTPUT_NAME "tppocr")

install(TARGETS tppocr_exe DESTINATION bin)
