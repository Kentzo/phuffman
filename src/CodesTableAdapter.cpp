#include "CodesTableAdapter.hpp"
#include "constants.h"
#include <cassert>
#include <cstring>
#include <algorithm>
#include <ostream>

namespace phuffman {

    bool _LeafComparator(const DepthCounterNode* left, const DepthCounterNode* right) {
        // Compare nodes first by codelength then by value
        return ((left->depth > right->depth) ||
                ((left->depth == right->depth) && (left->element < right->element)));
    }

    bool _FrequencyComparator(const Frequency& left, const Frequency& right) {
        return left.frequency < right.frequency;
    }

    CodesTableAdapter::CodesTableAdapter(const char *file_data, size_t size) {
        assert(size == ALPHABET_SIZE);
        assert(file_data != NULL);

        memset(adaptee.codes, 0, sizeof(Code)*ALPHABET_SIZE);

        unsigned char lengths[ALPHABET_SIZE];
        for (size_t i=0; i<size; ++i) {
            lengths[i] = file_data[i];
        }

        Nodes leaves;
        for (size_t i=0; i<ALPHABET_SIZE; ++i) {
            if (lengths[i] > 0) {
                assert(lengths[i] < MAXIMUM_CODELENGTH);
                DepthCounterNode* leaf = new DepthCounterNode(i);
                leaf->depth = lengths[i];
                leaves.push_back(leaf);
            }
        }

        // Leaves contain code length for each symbol in data
        sort(leaves.begin(), leaves.end(), _LeafComparator);

        _buildTable(leaves);

        Nodes::const_iterator first = leaves.begin(), last = leaves.end();
        while (first != last) {
            delete *first;
            ++first;
        }
    }

    CodesTableAdapter::CodesTableAdapter(const unsigned char *data, size_t size) {
        assert(size <= MAXIMUM_DATABLOCK_SIZE);
        assert(data != NULL);

        memset(adaptee.codes, 0, sizeof(Code)*ALPHABET_SIZE);

        Tree tree;
        Nodes leaves;

        // Build huffman tree
        Frequencies frequencies = _countFrequencies(data, size);
        Frequencies::const_iterator first = frequencies.begin(), last = frequencies.end();
        while (first != last) {
            DepthCounterNode* leaf = new DepthCounterNode(first->symbol);
            tree.insert(std::make_pair(first->frequency, leaf));
            leaves.push_back(leaf);
            ++first;
        }

        DepthCounterNode* root = NULL;
        if (tree.size() > 1) {
            root = _buildTree(&tree);
        }
        else {
            root = tree.begin()->second;
            assert(root->isLeaf);
            root->depth = 1;
        }

        // Leaves contain code length for each symbol in data
        sort(leaves.begin(), leaves.end(), _LeafComparator);

        // Build codes table
        _buildTable(leaves);

        delete root;
    }

    CodesTableAdapter::~CodesTableAdapter() {}

    CodesTableInfo CodesTableAdapter::info() const {
        return adaptee.info;
    }

    Code CodesTableAdapter::operator[](size_t index) const {
        return at(index);
    }

    Code CodesTableAdapter::at(size_t index) const {
        assert(index < ALPHABET_SIZE);
        return adaptee.codes[index];
    }

    const CodesTable* CodesTableAdapter::c_table() const {
        return &adaptee;
    }

    std::vector<Frequency> CodesTableAdapter::_countFrequencies(const unsigned char* data, size_t size) const {
        size_t freqs[ALPHABET_SIZE] = {0};
        for (size_t i=0; i<size; ++i) {
            unsigned char symbol = data[i];
            ++freqs[symbol];
        }
        Frequencies frequencies;
        frequencies.reserve(ALPHABET_SIZE);
        for (size_t i=0; i<ALPHABET_SIZE; ++i) {
            if (freqs[i] > 0) {
                frequencies.push_back(FrequencyMake(i, freqs[i]));
            }
        }
        std::sort(frequencies.begin(), frequencies.end(), _FrequencyComparator);

        return frequencies;
    }

    DepthCounterNode* CodesTableAdapter::_buildTree(Tree *tree) const {
        // 1. Get two rarest element
        // 2. Create new node that points to these elements and has sum of their frequencies
        // 3. Repeat until tree size is equal to 1
        for (size_t i=0, size=tree->size(); i<size-1; ++i) {
            Tree::iterator first = tree->begin(), second = tree->begin();
            ++second;
            size_t freq = first->first + second->first; // Calculate frequency for a node
            DepthCounterNode* node = new DepthCounterNode(first->second, second->second);
            ++second;
            tree->erase(first, second); // Remove two nodes with the smallest frequency
            tree->insert(std::make_pair(freq, node)); // Add node that points to previously removed nodes
        }
        assert(tree->size() == 1);
        return tree->begin()->second;
    }

    void CodesTableAdapter::_buildTable(const Nodes& leaves) {
        Nodes::const_iterator currentNode = leaves.begin(), lastNode = leaves.end();
        // First longest element always has 0 code
        adaptee.info.maximum_codelength = (*currentNode)->depth;
        Code lastCode = CodeMake((*currentNode)->depth, 0);
        adaptee.codes[(*currentNode)->element] = lastCode;
        ++currentNode;

        while (currentNode != lastNode) {
            // If current codeword and next codeword have equal lengths
            if ((*currentNode)->depth == lastCode.codelength) {
                // Just increase codeword by 1
                lastCode.code += 1;
            }
            // Otherwise
            else {
                // We are iterating from longest to shortest code lengths
                assert(lastCode.codelength > (*currentNode)->depth);
                // Increase codeword by 1 and _after_ that shift codeword right
                lastCode.code = (lastCode.code + 1) >> (lastCode.codelength - (*currentNode)->depth);
            }
            lastCode.codelength = (*currentNode)->depth;
            assert(lastCode.codelength < MAXIMUM_CODELENGTH);
            adaptee.codes[(*currentNode)->element] = lastCode;
            ++currentNode;
        }
    }

    std::ostream& operator<<(std::ostream& os, const CodesTableAdapter& table) {
        for (size_t i=0, size=ALPHABET_SIZE; i<size; ++i) {
            os << table[i].codelength;
        }
        return os;
    }

    bool operator==(const CodesTableAdapter& left, const CodesTableAdapter& right) {
		const CodesTable* leftTable = left.c_table();
		const CodesTable* rightTable = right.c_table();
		return (memcmp(leftTable->codes, rightTable->codes, sizeof(Code)*ALPHABET_SIZE) == 0);
    }

}
