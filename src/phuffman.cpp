#include "phuffman.hpp"

using namespace phuffman;

bool utility::TreeComparator(const DepthCounterNode* left, const DepthCounterNode* right) {
    // Compare nodes first by codelength then by value
    return ((left->depth > right->depth) || ((left->depth == right->depth) && (left->element > right->element)));
}
