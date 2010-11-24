#pragma once
#ifndef _CODE_H_
#define _CODE_H_

struct Code {
    unsigned char codelength;
    unsigned char code;
};

Code CodeMake(unsigned char codelength, unsigned char code) {
    Code c = {codelength, code};
    return c;
}

#endif /* _CODE_H_ */
