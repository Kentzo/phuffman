#include "phuffman.hpp"
#include <iostream>
#include <sstream>


using namespace std;
using namespace phuffman;

extern "C" void runEncode(unsigned char* a_data, size_t len, CodesTable a_table, unsigned char* result, size_t* result_len);

void PrintCodesTable(const CodesTableAdapter& table) {
    for (size_t i=0; i<ALPHABET_SIZE; ++i) {
        if (table[i].codelength > 0) {
            cout << (char)i << '\t' << (int)table[i].code << ' ' << (int)table[i].codelength << endl;
        }
    }
}

int main() {
  const  size_t dlen = ALPHABET_SIZE*1024*1;
  unsigned char* test = new unsigned char[dlen];//{0};
  for (size_t i=0; i<dlen; ++i) {
    	test[i] = i;
    }

    CodesTableAdapter codes(test, dlen);
    unsigned char* result = new unsigned char[codes.c_table()->info.maximum_codelength * dlen];
    size_t resLen = 0;
    runEncode(test, dlen, *codes.c_table(), result, &resLen);

    cout << "Table CPU:" << endl;
    for (size_t i = 0; i < dlen; ++i) {
      //            cout << (int)test[i] << " " << (int)codes[test[i]].codelength << endl;
    }
    
    delete [] result;
    delete [] test;

    //cout << test << endl;
    //cout << "Build codes table from string" << endl;
    
    //    PrintCodesTable(codes);
    stringstream ss;
    ss << codes;
    ss.seekg(0, ios_base::beg);

    char* file_data = new char[ALPHABET_SIZE];
    ss.read(file_data, ALPHABET_SIZE);
    //cout << "Build codes table from file data" << endl;
    CodesTableAdapter cc(file_data, ALPHABET_SIZE);
    //PrintCodesTable(cc);
    //cout << "Codes Tables are equal: " << (codes == cc) << endl;
}
