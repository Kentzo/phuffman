#pragma once
#ifndef _FREQUENCY_H_
#define _FREQUENCY_H_

#include "stddef.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned char symbol;
    size_t frequency;
} Frequency;

#ifndef __OPENCL_VERSION__

Frequency FrequencyMake(unsigned char symbol, size_t frequency);

#endif /* __OPENCL_VERSION__ */

#ifdef __cplusplus
}
#endif

#endif /* _FREQUENCY_H_ */
