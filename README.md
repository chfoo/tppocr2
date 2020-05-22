# tppocr2

Tesseract OCR of Pokemon (or video game) dialog text on streaming video (modern version).

This project reads streaming video and runs OCR on defined regions for streams such as TwitchPlaysPokemon. For previous
information, see [version 1](https://github.com/chfoo/tppocr) of this project.

Note: **work in progress**

## Quick start

### Requirements

* [Tesseract](https://github.com/tesseract-ocr/tesseract) 4
* [tessdata_best](https://github.com/tesseract-ocr/tessdata_best) or [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast)
  * Which should contain: eng jpn chi_sim chi_tra kor spa deu ita
* [Leptonica](http://www.leptonica.org/)
* [ffmpeg](https://ffmpeg.org/download.html)
* [OpenCV](https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html) 4
* [EAST](https://github.com/argman/EAST) trained model
  * Download the Tensorflow trained model from [this link](https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1)

Compiling:

* C++17 compiler and associated C++ runtime
* CMake 1.15 or newer
* [tomlplusplus](https://marzer.github.io/tomlplusplus/) headers

### Building

    mkdir -p build
    cd build
    cmake .. -D CMAKE_BUILD_TYPE=Release
    cmake --build . --config Release

Optional:

    cmake --install --config Release --prefix install_prefix

### Running

Basic usage:

    ./build/tppocr CONFIG_FILE URL_OR_FILE_PATH

Example:

    ./build/tppocr data/tpp-sword-720p.toml sample_images/sword_720p_narrator_dialog.png
