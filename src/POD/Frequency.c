#include "Frequency.h"

struct Frequency FrequencyMake(unsigned char symbol, size_t frequency) {
    struct Frequency f = {symbol, frequency};
    return f;
}
