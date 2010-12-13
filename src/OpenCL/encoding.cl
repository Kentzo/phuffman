#include "Code.h"

__kernel void EncodeWithCodelengths(__global uchar *data,
                                    __global uint *codelengths,
                                    __constant Code* codes,
                                    size_t count,
                                    size_t data_size)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1), dy = get_global_size(1);
    size_t z = get_global_id(2), dz = get_global_size(2);
    size_t i = min((x*dy*dz + y*dz + z) * count, data_size);
    size_t size = min(i + count, data_size);
    for (; i<size; ++i) {
        uchar c = data[i];
        codelengths[i] = codes[c].codelength;
    }
}
