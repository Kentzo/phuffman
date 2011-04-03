#pragma once

#include "CodesTable.h"

namespace phuffman {
    namespace CUDA {
        void Encode(unsigned char* data, size_t data_length, CodesTable codes_table, unsigned int** result, size_t* result_length, size_t* result_length_bit,
                    unsigned int block_size = 0, unsigned char** block_offsets = NULL, unsigned int** block_sizes = NULL, size_t* block_length = NULL);
    }
}
