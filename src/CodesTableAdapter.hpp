#pragma once
#ifndef CODESTABLEADAPTER_H_
#define CODESTABLEADAPTER_H_

#include "POD.h"
#include "DepthCounterNode.hpp"
#include <vector>
#include <map>

namespace phuffman {

class CodesTableAdapter {
    CodesTableAdapter();
    CodesTableAdapter(const CodesTableAdapter&);
    CodesTableAdapter& operator=(const CodesTableAdapter&);
public:
    explicit CodesTableAdapter(const unsigned char* data, size_t size);
    explicit CodesTableAdapter(const char* file_data, size_t size);
    ~CodesTableAdapter();

    CodesTableInfo info() const;
    Code operator[](size_t index) const;
    Code at(size_t index) const;
    const CodesTable* c_table() const;

private:
    typedef std::vector<Frequency> Frequencies;
    typedef std::multimap<size_t, DepthCounterNode*> Tree;
    typedef std::vector<DepthCounterNode*> Nodes;
    Frequencies _countFrequencies(const unsigned char* data, size_t size) const;
    DepthCounterNode* _buildTree(Tree *tree) const;
    void _buildTable(const Nodes& leaves);

private:
    CodesTable adaptee;
};

std::ostream& operator<<(std::ostream& os, const CodesTableAdapter& table);

bool operator==(const CodesTableAdapter& left, const CodesTableAdapter& right);

}

#endif /* CODESTABLEADAPTER_H_ */
