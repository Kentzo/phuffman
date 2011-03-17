#include "phuffman.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <cuda_runtime.h>
#include <cmath>
#include <climits>

using namespace std;
using namespace phuffman;

void PrintCodesTable(const CodesTableAdapter& table) {
    for (size_t i=0; i<ALPHABET_SIZE; ++i) {
        if (table[i].codelength > 0) {
            cout << (char)i << '\t' << (int)table[i].code << ' ' << (int)table[i].codelength << endl;
        }
    }
}

int main() { 
    /*
      unsigned char test[] = {'b', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
      'a', 'b', 'a', 'a', 'a', 'a', 'a', 'a',
      'a', 'a', 'b', 'a', 'a', 'a', 'a', 'a',
      'a', 'a', 'a', 'b', 'a', 'a', 'a', 'a',
      'a', 'a', 'a', 'a', 'b', 'a', 'a', 'a',
      'a', 'a', 'a', 'a', 'a', 'b', 'a', 'a',
      'a', 'a', 'a', 'a', 'a', 'a', 'b', 'a',
      'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b',
      'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'};
    size_t test_length = sizeof(test);
    */
    static const size_t test_length = 1024 * 1024 * 100;
    unsigned char* test = new unsigned char[test_length];
    ifstream urandom("/dev/urandom");
    urandom.read((char *)test, test_length);
    cout << "End generating" << endl;
    CodesTableAdapter codes(test, test_length);

    unsigned int* result = NULL;
    size_t result_length = 0;
    size_t result_length_bit = 0;
    try {
        CUDA::Encode((unsigned char*)test, (size_t)test_length, (CodesTable)*codes.c_table(), (unsigned int**)&result, (size_t*)&result_length, (size_t*)&result_length_bit);
    }
    catch(cudaError_t error) {
        cerr << cudaGetErrorString(error) << endl;
        throw error;
    }
    cout << "Result length: " << result_length << endl;
    cout << "Result length bit: " << result_length_bit << endl;
    //    cout << "Result array: ";
    //    for (size_t i=0; i<result_length; ++i) {
    //        cout << (size_t)result[i] << " ";
    //    }
    cout << endl << endl;
    //    PrintCodesTable(codes);
    delete[] test;
}
