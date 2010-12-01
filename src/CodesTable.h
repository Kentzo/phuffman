#pragma once
#ifndef CODESTABLE_H_
#define CODESTABLE_H_

#include "CodesTableInfo.h"
#include "Code.h"
#include "constants.h"

typedef struct {
    Code codes[ALPHABET_SIZE];
    CodesTableInfo info;
} CodesTable;

#endif /* CODESTABLE_H_ */
