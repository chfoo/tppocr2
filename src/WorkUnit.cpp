#include "WorkUnit.hpp"

namespace tppocr {

WorkUnit::WorkUnit(unsigned int id, unsigned int frameID,
        Region & region, cv::Mat image, cv::Mat debugImage) :
    id(id),
    frameID(frameID),
    region(region),
    image(image),
    debugImage(debugImage) {}

}
