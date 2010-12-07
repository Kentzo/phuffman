#pragma once
#ifndef CODESTABLE_H_
#define CODESTABLE_H_

#include "CodeTableInfo.h"
#include "Code.h"
#include "constants.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    Code codes[ALPHABET_SIZE];
    CodeTableInfo info;
} CodeTable;

#ifdef __cplusplus
}
#endif

#endif /* CODESTABLE_H_ */
