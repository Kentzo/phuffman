#pragma once
#ifndef _FREQUENCY_H_
#define _FREQUENCY_H_

#include "stddef.h"

struct Frequency {
    unsigned char symbol;
    size_t frequency;
};

struct Frequency FrequencyMake(unsigned char symbol, size_t frequency);

#endif /* _FREQUENCY_H_ */
