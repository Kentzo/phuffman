#include "HuffmanTable.h"
#include "FreqCounter.h"
#include "DepthCounterNode.h"
#include <map>
#include <vector>
#include <algorithm>
#include <cassert>


using namespace std;

bool _comparator_depth_gr(const DepthCounterNode<unsigned char>* left, const DepthCounterNode<unsigned char>* right) {
    // Compare nodes first by codelength then by value
    return ((left->depth > right->depth) || ((left->depth == right->depth) && (left->element > right->element)));
}

bool _comparator_orderTable_gr(const pair<unsigned char, unsigned char>& left, const pair<unsigned char, unsigned char>& right) {
    return ((left.first > right.first) || ((left.first == right.first) && (left.second > right.second)));
}

//----

HuffmanTable::HuffmanTable(const unsigned char* array, size_t num) {
    _buildTable(array, num);
}

HuffmanTable::HuffmanTable(const char binaryTable[ALPHABET_SIZE]) {
    memset(_table, 0, ALPHABET_SIZE * sizeof(Row));
    // Read bytes into vector. Each i-th byte represents length of codeword of i-th alphabet symbol
    vector<pair<unsigned char, unsigned char> > orderTable;
    for (size_t i=0; i<ALPHABET_SIZE; ++i) {
        orderTable.push_back(make_pair(binaryTable[i], i));
    }
    // Sort pairs first by codelength then by values. Descending order
    sort(orderTable.begin(), orderTable.end(), _comparator_orderTable_gr);
   
    // Build table
    vector<pair<unsigned char, unsigned char> >::iterator cur = orderTable.begin(), prev = orderTable.begin(), end = orderTable.end();
    unsigned char value = 0;
    Row row = {cur->first, value};
    _table[cur->second] = row;
    ++cur;
    while (cur != end) {
        if (cur->first != 0) {
            // If current codeword and next codeword have equal length
            if (cur->first == prev->first) {
                // Just increase codeword by 1
                value += 1;
                row.codelength = cur->first;
                row.code = value;
                _table[cur->second] = row;
            }
            // Otherwise
            else {
            // Increase codeword by 1 and _after_ that shift codeword right
                assert((prev->first - cur->first) == 1);
                value = (value + 1) >> 1;
                row.codelength = cur->first;
                row.code = value;
            }
            ++prev;
        }
        ++cur;
    }
}

Row HuffmanTable::operator[](unsigned char index) const {
    return _table[index];
}

void HuffmanTable::_buildTable(const unsigned char* array, size_t num) {
    memset(_table, 0, ALPHABET_SIZE * sizeof(Row));
    typedef FreqCounter<unsigned char> Counter;
    typedef Counter::Frequencies Frequencies;
    typedef DepthCounterNode<unsigned char> Node;
    typedef multimap<size_t, Node*> HuffmanTree;
    typedef vector<Node*> Nodes;

    if (num == 0)
        return;

    // Elements are sorted first by freq then by value
    Frequencies frequencies = Counter(array, array + num).mostCommon();

    // Initialize tree
    HuffmanTree tree;
    Nodes leafs;
    Nodes nodes;
    Frequencies::iterator cur_freq = frequencies.begin(), end_freq = frequencies.end();
    while (cur_freq != end_freq) {
        Node* leaf = new Node(cur_freq->first);
        tree.insert(make_pair(cur_freq->second, leaf));
        leafs.push_back(leaf);
        nodes.push_back(leaf);
        ++cur_freq;
    }
    
    // Build tree
    for (size_t i=0, size=tree.size(); i<size-1; ++i) {
        HuffmanTree::iterator first_least = tree.begin(), second_least = tree.begin();
        ++second_least;
        size_t freq = first_least->first + second_least->first;
        Node* node = new Node(first_least->second, second_least->second);
        ++second_least;
        tree.erase(first_least, second_least);
        tree.insert(make_pair(freq, node));
        nodes.push_back(node);
    }
    assert(tree.size() == 1);

    // Count codelengths
    // In fact, codelengths are already counted in the 'depth' member of a node
    // There is only one exception: if the tree contains only one object, we need to set its depth manually
    Node* root = tree.begin()->second;
    root->depth = 1;
   
    // Sort nodes by codelength
    sort(leafs.begin(), leafs.end(), _comparator_depth_gr);

    // Build table
    Nodes::iterator cur_leaf = leafs.begin(), end_leaf = leafs.end();
    Node *currentNode = *cur_leaf;
    Row currentRow = {currentNode->depth, 0};
    _table[currentNode->element] = currentRow;
    ++cur_leaf;
    while (cur_leaf != end_leaf) {
        currentNode = *cur_leaf;
        // If current codeword and next codeword have equal lengths
        if (currentNode->depth == currentRow.codelength) {
            // Just increase codeword by 1
            currentRow.codelength = currentNode->depth;
            currentRow.code = currentRow.code + 1;
            _table[currentNode->element] = currentRow;
        }
        // Otherwise
        else {
            // Increase codeword by 1 and _after_ that shift codeword right
            //assert((currentRow.codelength - currentNode->depth) == 1);
            currentRow.codelength = currentNode->depth;
            currentRow.code = (currentRow.code + 1) >> 1;
            _table[currentNode->element] = currentRow;
        }
        ++cur_leaf;
    }

    // Clean up
    Nodes::iterator cur_node = nodes.begin(), end_node = nodes.end();
    while (cur_node != end_node) {
        delete *cur_node;
        ++cur_node;
    }
}

std::string HuffmanTable::binaryTable() const {
    string result;
    result.reserve(ALPHABET_SIZE);
    for (size_t i=0; i<ALPHABET_SIZE; ++i) {
        result += _table[i].codelength;
    }
    return result;
}
