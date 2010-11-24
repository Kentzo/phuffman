#include "DepthCounterNode.hpp"

DepthCounterNode::DepthCounterNode(DepthCounterNode* l, DepthCounterNode* r) : isLeaf(false), left(l), right(r) {
    left->incDepth();
    right->incDepth();
}

DepthCounterNode::DepthCounterNode(unsigned char elem) : isLeaf(true), element(elem), depth(0) {}
    
DepthCounterNode::~DepthCounterNode() {
    if (!isLeaf) {
        delete left;
        delete right;
    }
}

void DepthCounterNode::incDepth() {
    if (!isLeaf) {
        left->incDepth();
        right->incDepth();
    }
    else {
        ++depth;
    }
}

void DepthCounterNode::decDepth() {
    if (!isLeaf) {
        left->decDepth();
        right->decDepth();
    }
    else {
        --depth;
    }
}
