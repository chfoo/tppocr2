#pragma once

#include <string>

namespace tppocr {

struct Region {
    std::string name;
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

}
