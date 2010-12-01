#pragma once
#ifndef _CODE_H_
#define _CODE_H_

typedef struct {
    unsigned char codelength;
    unsigned char code;
} Code ;

#ifndef __OPENCL_VERSION__

Code CodeMake(unsigned char codelength, unsigned char code);

#endif /* __OPENCL_VERSION__ */

#endif /* _CODE_H_ */
