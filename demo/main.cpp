#include "phuffman.hpp"
#include <iostream>
#include <sstream>


using namespace std;
using namespace phuffman;

//extern "C" Row* runEncode(unsigned char* a_data, size_t len, Row a_table[256]);

void PrintCodesTable(const CodeTableAdapter& table) {
    for (size_t i=0; i<ALPHABET_SIZE; ++i) {
        if (table[i].codelength > 0) {
            cout << (char)i << '\t' << (int)table[i].code << ' ' << (int)table[i].codelength << endl;
        }
    }
}

void TestCodeTableBuilder() {
    unsigned char test[ALPHABET_SIZE] = {0};
    for (size_t i=0; i<ALPHABET_SIZE; ++i) {
    	test[i] = i;
    }
    cout << test << endl;
    cout << "Build codes table from string" << endl;
    CodeTableAdapter codes(test, sizeof(test)/sizeof(unsigned char)-1);
    PrintCodesTable(codes);
    stringstream ss;
    ss << codes;
    ss.seekg(0, ios_base::beg);

    char* file_data = new char[ALPHABET_SIZE];
    ss.read(file_data, ALPHABET_SIZE);
    cout << "Build codes table from file data" << endl;
    CodeTableAdapter cc(file_data, ALPHABET_SIZE);
    PrintCodesTable(cc);
    cout << "Codes Tables are equal: " << (codes == cc) << endl;
}

int main() {
    static const size_t test_size = ALPHABET_SIZE;
    unsigned char* test = new unsigned char[test_size];
    for (size_t i=0; i<test_size; ++i) {
        test[i] = i;
    }
    try {
        unsigned char* result = NULL;
        size_t result_size = 0;
        opencl::Encode(test, test_size, &result, &result_size);
        delete[] result;
    }
    catch(cl::Error err) {
        cerr << err.err() << endl;
    }
    delete[] test;
    return 0; 
}
