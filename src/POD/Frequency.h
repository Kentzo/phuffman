#pragma once
#ifndef _FREQUENCY_H_
#define _FREQUENCY_H_

#include <cstddef>

struct Frequency {
    unsigned char symbol;
    size_t frequency;
};

Frequency FrequencyMake(unsigned char symbol, size_t frequency) {
    Frequency f = {symbol, frequency};
    return f;
}

#endif /* _FREQUENCY_H_ */
