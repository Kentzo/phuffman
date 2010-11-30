#include "Code.h"

Code CodeMake(unsigned char codelength, unsigned char code) {
    Code c = {codelength, code};
    return c;
}
