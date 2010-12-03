#pragma once
#ifndef _CODE_H_
#define _CODE_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned char codelength;
    unsigned char code;
} Code ;

#ifndef __OPENCL_VERSION__

Code CodeMake(unsigned char codelength, unsigned char code);

#endif /* __OPENCL_VERSION__ */

#ifdef __cplusplus
}
#endif

#endif /* _CODE_H_ */
