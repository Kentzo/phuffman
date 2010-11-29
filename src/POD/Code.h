#pragma once
#ifndef _CODE_H_
#define _CODE_H_

struct Code {
    unsigned char codelength;
    unsigned char code;
};

struct Code CodeMake(unsigned char codelength, unsigned char code);

#endif /* _CODE_H_ */
