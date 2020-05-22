#include "mathutil.hpp"

#include <assert.h>
#include <cmath>

namespace tppocr {


int roundUp2(int value, int multiple) {
    // https://stackoverflow.com/a/9194117/1524507
    assert(multiple && ((multiple & (multiple - 1)) == 0));
    return (value + multiple - 1) & -multiple;
}

}
