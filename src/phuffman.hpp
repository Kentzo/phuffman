#pragma once
#ifndef _PHUFFMAN_H_
#define _PHUFFMAN_H_

#include "Code.h"
#include "Frequency.h"
#include "TableInfo.h"
#include "DepthCounterNode.hpp"
#include <vector>
#include "constants.hpp"

namespace phuffman {
    typedef std::vector<Code> Codes;
    namespace constants {
        const size_t ALPHABET_SIZE = 256;
        const size_t MAXIMUM_CODELENGTH = 48; // bits
        const size_t MAXIMUM_DATABLOCK_SIZE = 32951280098; // bytes
    }
    namespace utility {
        typedef std::vector<Frequency> Frequencies;

        template <typename InputIterator>
        Frequencies CountFrequencies(InputIterator first, InputIterator last);

        template <typename InputIterator>
        TableInfo BuildTable(InputIterator first, InputIterator last, Codes& table);

        bool TreeComparator(const DepthCounterNode* left, const DepthCounterNode* right);
    }
    namespace opencl {
        int EncodeCPU();
        int EncodeGPU();
    }

    template <typename InputIterator>
    int Encode(Codes& table, InputIterator first, InputIterator last, unsigned char** out, size_t* size);

    template <typename InputIterator>
    int Decode(Codes& table, InputIterator first, InputIterator last, unsigned char** out, size_t* size);
}


/*
####################################################################################################
                                            Implementation
####################################################################################################
*/

#include <cassert>
#include <iterator>
#include <map>
#include <algorithm>

template <typename InputIterator>
phuffman::utility::Frequencies phuffman::utility::CountFrequencies(InputIterator first, InputIterator last) {
    size_t freqs[constants::ALPHABET_SIZE] = {0};
    while (first != last) {
        freqs[(unsigned char)first] += 1;
        ++first;
    }
    Frequencies frequencies;
    frequencies.reserve(constants::ALPHABET_SIZE);
    for (size_t i=0; i<constants::ALPHABET_SIZE; ++i) {
        if (freqs[i] > 0) {
            frequencies.push_back(FrequencyMake(i, freqs[i]));
        }
    }
    std::sort(frequencies.begin(), frequencies.end());

    return frequencies;
}

template <typename InputIterator>
TableInfo phuffman::utility::BuildTable(InputIterator first, InputIterator last, Codes& table) {
    using namespace std;
    typedef DepthCounterNode Node;
    typedef multimap<size_t, Node*> HuffmanTree;
    typedef vector<Node*> Nodes;

    assert(distance(first, last) <= constants::MAXIMUM_DATABLOCK_SIZE);
    assert(table.size() >= constants::ALPHABET_SIZE);

    HuffmanTree tree;
    Nodes leafs;
    TableInfo info;

    // Initialize tree
    {
        Frequencies frequencies = CountFrequencies(first, last);
        Frequencies::const_iterator first = frequencies.begin(), last = frequencies.end();
        while (first != last) {
            Node* leaf = new Node(first->symbol);
            tree.insert(make_pair(first->frequency, leaf));
            leafs.push_back(leaf);
            ++first;
        }
    }

    // Build tree
    {
        for (size_t i=0, size=tree.size(); i<size-1; ++i) {
            HuffmanTree::iterator first = tree.begin(), second = tree.begin();
            ++second;
            size_t freq = first->first + second->first; // Calculate freq for a node
            Node* node = new Node(first->second, second->second); 
            ++second;
            tree.erase(first, second); // Remove two nodes with the smallest frequency
            tree.insert(make_pair(freq, node)); // Add node that points to previosly removed nodes
        }
        assert(tree.size() == 1);
    }

    // Count codelengths
    // In fact, codelengths are already counted in the 'depth' member of a node
    // There is only one exception: if the tree contains only one object, we need to set it's depth manually
    Node* root = tree.begin()->second;
    root->depth = 1;

    // Sort nodes by codelength
    sort(leafs.begin(), leafs.end(), TreeComparator);

    // Build table
    {
        Nodes::const_iterator first = leafs.begin(), last = leafs.end();
        Node *curNode = *first;
        info.maximum_codelength = curNode->depth;
        Code curCode = CodeMake(curNode->depth, 0);
        table[curNode->element] = curCode;
        ++first;
        while (first != last) {
            assert(curNode->depth >= curCode.codelength);
            curNode = *first;
            // If current codeword and next codeword have equal lengths
            if (curNode->depth == curCode.codelength) {
            // Just increase codeword by 1
                curCode.code += 1;
            }
            // Otherwise
            else {
                // Increase codeword by 1 and _after_ that shift codeword right
                curCode.code = (curCode.code + 1) >> (curNode->depth - curCode.codelength);
            }
            curCode.codelength = curNode->depth;
            table[curNode->element] = curCode;
            ++first;
        }
    }
    
    delete root;

    return info;
}


#endif /* _PHUFFMAN_H_ */
