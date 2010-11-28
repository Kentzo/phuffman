#pragma once
#ifndef CODESTABLE_H_
#define CODESTABLE_H_

#include "CodesTableInfo.h"
#include "Code.h"
#include "stdint.h"

struct CodesTable {
    uint16_t size;
    Code *codes;
    CodesTableInfo info;
};

#endif /* CODESTABLE_H_ */
