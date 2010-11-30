#include "Code.h"

__kernel void EncodeWithCodelengths(__constant Code *codes, __global uchar *data) {
    int id = (int)get_global_id(0);
    __global uchar4 *data4 = (__global uchar4*)data;
    uchar4 code = data4[id];
    code = (uchar4)(codes[code.s0].codelength,
                    codes[code.s1].codelength,
                    codes[code.s2].codelength,
                    codes[code.s3].codelength);
    data4[id] = code;
}
