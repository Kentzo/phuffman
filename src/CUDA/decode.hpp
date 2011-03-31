#pragma once

#include "../CodesTable.h"

namespace phuffman {
    namespace CUDA {
        void Decode(unsigned int* encoded_data, size_t encoded_data_length, CodesTable codes_table, unsigned int block_size, unsigned char* block_offsets, unsigned int* block_sizes, size_t block_length, unsigned char** data, size_t* data_length);
    }
}
