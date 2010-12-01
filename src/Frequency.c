#include "Frequency.h"

Frequency FrequencyMake(unsigned char symbol, size_t frequency) {
    Frequency f = {symbol, frequency};
    return f;
}
