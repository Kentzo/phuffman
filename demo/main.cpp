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

int main() {
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
    /*
  ifstream file("../test");
  file.seekg (0, ios::end);
  unsigned int len = file.tellg();
  file.seekg (0, ios::beg);
  char* buffer = new char [len];
  file.read(buffer, len);

  HuffmanTable h((const unsigned char*)buffer, len);

  Row rows[256];
  for (size_t i = 0; i < HuffmanTable::ALPHABET_SIZE; ++i) {
    rows[i] = h[i];
  }

  Row* cpu_encoded = new Row[len];

  // work on GPU
  Row* encoded = runEncode((unsigned char*)buffer, len, rows);

  //work on CPU
  for (size_t i = 0; i < len; ++i) {
    cpu_encoded[i] = rows[(unsigned char)buffer[i]];      
  }

  // comparison
  for (size_t i = 0; i < len; ++i) {
    if (encoded[i].code != cpu_encoded[i].code || encoded[i].codelength != cpu_encoded[i].codelength) {
      cout << "Comparison result: " << i << " " << (int)encoded[i].code << "\t" << (int)cpu_encoded[i].code << endl;
      break;
    }
  }

  // CPU decoding
  char *decoded = new char[len];
  for (unsigned int i=0; i<len; ++i) {
      Row curRow = cpu_encoded[i];
      unsigned char c = 0;
      for (unsigned char j=0; j<HuffmanTable::ALPHABET_SIZE; ++j) {
          if (rows[j].code == curRow.code && rows[j].codelength == curRow.codelength) {
              c = j;
              break;
          }
      }
      decoded[i] = c;
  }

  // comparison with initial string
  for (size_t i = 0; i < len; ++i) {
    if (decoded[i] != buffer[i]) {
      cout << "Decoding comparison result: " << i << " " << (int)decoded[i] << "\t" << (int)buffer[i] << endl;
      break;
    }
  }
  cout << "Compare decoded and initial strings: " << memcmp(decoded, buffer, len) << endl;

  delete[] decoded;
  delete[] cpu_encoded;
  free(encoded);
  delete[] buffer;*/
  return 0; 
}
