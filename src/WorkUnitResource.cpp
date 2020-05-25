#include "WorkUnitResource.hpp"

namespace tppocr {

WorkUnitResource::WorkUnitResource(std::shared_ptr<Config> config,
        const Region & region) : ocr(config, region), textDetector(config) {
    freetype = cv::freetype::createFreeType2();
    freetype->loadFontData("/usr/share/fonts/truetype/unifont/unifont.ttf", 0);
}

}
