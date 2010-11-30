#pragma once
#ifndef _FREQUENCY_H_
#define _FREQUENCY_H_

#include "stddef.h"

typedef struct {
    unsigned char symbol;
    size_t frequency;
} Frequency;

#ifndef __OPENCL_VERSION__

Frequency FrequencyMake(unsigned char symbol, size_t frequency);

#endif /* __OPENCL_VERSION__ */

#endif /* _FREQUENCY_H_ */
