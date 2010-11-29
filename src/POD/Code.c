#include "Code.h"

struct Code CodeMake(unsigned char codelength, unsigned char code) {
    struct Code c = {codelength, code};
    return c;
}
