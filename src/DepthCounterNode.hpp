#pragma once
#ifndef _DEPTHCOUNTERNODE_H_
#define _DEPTHCOUNTERNODE_H_

#include <cstddef>

struct DepthCounterNode {
    bool isLeaf;
    union {
        struct {
            DepthCounterNode* left;
            DepthCounterNode* right;
        };
        struct {
            unsigned char element;
            size_t depth;
        };
    };

    DepthCounterNode(DepthCounterNode* l, DepthCounterNode* r);
    DepthCounterNode(unsigned char elem);
    ~DepthCounterNode();
    void incDepth();
    void decDepth();
};

#endif /* _DEPTHCOUNTERNODE_H_ */
