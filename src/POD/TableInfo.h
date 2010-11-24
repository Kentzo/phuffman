#pragma once
#ifndef _TABLEINFO_H_
#define _TABLEINFO_H_

struct TableInfo {
    unsigned char maximum_codelength;
};

TableInfo TableInfoMake(unsigned char maximum_codelength) {
    TableInfo t = {maximum_codelength};
    return t;
}

#endif /* _TABLEINFO_H_ */
