#pragma once

#include <climits>

#define UINT_BIT (sizeof(uint) * CHAR_BIT)

#define TYPE_BIT(x) (sizeof(x) * CHAR_BIT)

#ifdef __CUDACC__
#define UINT2_BIT (sizeof(uint2) * CHAR_BIT)
#endif // __CUDACC__
