#pragma once

typedef struct {
    unsigned char codelength;
    unsigned char code;
} Code ;

Code CodeMake(unsigned char codelength, unsigned char code);
