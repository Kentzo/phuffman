#include "phuffman.hpp"
#include <iostream>
#include <sstream>
#include <string.h>


using namespace std;
using namespace phuffman;

extern "C" void runEncode(unsigned char* a_data, size_t len, CodesTable a_table, unsigned int* result, size_t* result_len);

void PrintCodesTable(const CodesTableAdapter& table) {
    for (size_t i=0; i<ALPHABET_SIZE; ++i) {
        if (table[i].codelength > 0) {
            cout << (char)i << '\t' << (int)table[i].code << ' ' << (int)table[i].codelength << endl;
        }
    }
}

int main() {
  
  unsigned char test[] = {'a', 'a', 'a', 'c', 'c', 'c', 'a', 'c', 'a', 'b', 'b', 'b', 'b', 'a', 'a', 'a', 'c', 'b', 'b', 'b', 'c', 'a', 'a', 'c', 'b', 'b', 'c', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'a', 'a', 'c', 'b', 'b', 'b', 'b', 'a', 'c', 'a', 'b', 'b', 'b', 'b', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'a', 'a', 'c', 'b', 'b', 'b'};//(unsigned char*)"aaabbb";//new unsigned char[dlen];//{0};
  const  size_t dlen = sizeof(test)/sizeof(char);//ALPHABET_SIZE*1024*1;

    CodesTableAdapter codes(test, dlen);

    unsigned int result[100] = {0};
    size_t resLen = 16;
    runEncode(test, dlen, *codes.c_table(), result, &resLen);

    cout << "Result: ";
    for (size_t i = 0; i < resLen; ++i) {
        cout << result[i] << " ";
      //            cout << (int)test[i] << " " << (int)codes[test[i]].codelength << endl;
    }
    cout << endl;

//    cout << "manual: ";
//    for (int i=0; i<dlen; ++i) {
//        cout << (int)codes[test[i]].code << "(" << (int)codes[test[i]].codelength << ")" << " ";
//    }
//    cout << endl;
    
//    delete [] test;

    //cout << test << endl;
    //cout << "Build codes table from string" << endl;
    
    PrintCodesTable(codes);
   // stringstream ss;
    //ss << codes;
   // ss.seekg(0, ios_base::beg);

   // char* file_data = new char[ALPHABET_SIZE];
  //  ss.read(file_data, ALPHABET_SIZE);
    //cout << "Build codes table from file data" << endl;
   // CodesTableAdapter cc(file_data, ALPHABET_SIZE);
    //PrintCodesTable(cc);
    //cout << "Codes Tables are equal: " << (codes == cc) << endl;
}
