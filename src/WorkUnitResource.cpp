#include "WorkUnitResource.hpp"

namespace tppocr {

WorkUnitResource::WorkUnitResource(std::shared_ptr<Config> config) {
    for (auto & region : config->regions) {
        textDetectors.emplace(region.name, config);
        textRecognizers.try_emplace(region.name, config, region);
    }

    freetype = cv::freetype::createFreeType2();
    freetype->loadFontData("/usr/share/fonts/truetype/unifont/unifont.ttf", 0);
}

}
