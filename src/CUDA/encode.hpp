#pragma once

#include "../CodesTable.h"
#include <cuda_runtime.h>

namespace phuffman {
    namespace CUDA {
        void Encode(unsigned char* data, size_t length, CodesTable code_table, unsigned int** result, size_t* result_length, size_t* result_length_bit,
                           unsigned int block_size = 0, unsigned char** block_offsets = NULL, size_t* block_offsets_length = NULL);
    }
}
