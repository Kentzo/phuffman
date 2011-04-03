#pragma once

#include "constants.h"
#include "CodesTableInfo.h"
#include "Code.h"

typedef struct {
    Code codes[ALPHABET_SIZE];
    CodesTableInfo info;
} CodesTable;
