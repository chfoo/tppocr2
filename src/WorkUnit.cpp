#include "WorkUnit.hpp"

namespace tppocr {

WorkUnit::WorkUnit(unsigned int id, unsigned int frameID,
        cv::Mat image, cv::Mat debugImage) :
    id(id),
    frameID(frameID),
    image(image),
    debugImage(debugImage) {}

}
