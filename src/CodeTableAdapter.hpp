#pragma once
#ifndef CODESTABLEADAPTER_H_
#define CODESTABLEADAPTER_H_

#include "POD.h"
#include "DepthCounterNode.hpp"
#include <vector>
#include <map>
#include <iostream>

namespace phuffman {

class CodeTableAdapter {
    CodeTableAdapter();
    CodeTableAdapter(const CodeTableAdapter&);
    CodeTableAdapter& operator=(const CodeTableAdapter&);
public:
    explicit CodeTableAdapter(const unsigned char* data, size_t size);
    explicit CodeTableAdapter(const char* file_data, size_t size);
    ~CodeTableAdapter();

    CodeTableInfo info() const;
    Code operator[](size_t index) const;
    Code at(size_t index) const;
    const CodeTable* c_table() const;

private:
    typedef std::vector<Frequency> Frequencies;
    typedef std::multimap<size_t, DepthCounterNode*> Tree;
    typedef std::vector<DepthCounterNode*> Nodes;
    Frequencies _countFrequencies(const unsigned char* data, size_t size) const;
    DepthCounterNode* _buildTree(Tree *tree) const;
    void _buildTable(const Nodes& leaves);

private:
    CodeTable _adaptee;
};

std::ostream& operator<<(std::ostream& os, const CodeTableAdapter& table);

bool operator==(const CodeTableAdapter& left, const CodeTableAdapter& right);

}

#endif /* CODESTABLEADAPTER_H_ */
