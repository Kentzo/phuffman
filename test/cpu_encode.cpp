#include <cstdlib>
#include "cpu_encode.hpp"
#include "phuffman_limits.cuh"

namespace phuffman {
    namespace CPU {
        void Encode(unsigned char* data,
                    size_t data_length,
                    CodesTable codes_table,
                    unsigned int** encoded_data,
                    size_t* encoded_data_length,
                    unsigned char* encoded_data_trail_zeroes,
                    size_t block_bit_size/* = 0*/,
                    unsigned char** block_bit_offsets/* = NULL*/,
                    unsigned short** block_sym_sizes/* = NULL*/,
                    size_t* block_count/* = NULL*/)
        {
            *encoded_data_length = 0;
            unsigned char bit_len = 0;
            for (size_t i=0; i<data_length; ++i) {
                unsigned char sym = data[i];
                Code code = codes_table.codes[sym];
                bit_len += code.codelength;
                if (bit_len >= UINT_BIT) {
                    *encoded_data_length += bit_len / UINT_BIT;
                    bit_len %= UINT_BIT;
                }
            }
            *encoded_data_trail_zeroes = (UINT_BIT - bit_len) % UINT_BIT;
            if (*encoded_data_trail_zeroes > 0) {
                *encoded_data_length += 1;
            }

            *encoded_data = static_cast<unsigned int*>(calloc(*encoded_data_length, sizeof(unsigned int)));
            size_t int_idx = 0;
            int written_bits = 0;
            for (size_t i=0; i<data_length; ++i) {
                 unsigned char sym = data[i];
                 Code code = codes_table.codes[sym];
                 // only 8 less sign bits may be non-nil
                 while (code.codelength > UINT_BIT) {
                     ++int_idx;
                     code.codelength -= UINT_BIT;
                 }
                 if ((UINT_BIT - written_bits) > code.codelength) {
                     (*encoded_data)[int_idx] |= static_cast<unsigned int>(code.code) << (UINT_BIT - written_bits - static_cast<int>(code.codelength));
                     written_bits += code.codelength;
                 }
                 else if ((UINT_BIT - written_bits) < code.codelength) {
                     (*encoded_data)[int_idx] |= static_cast<unsigned int>(code.code) >> (static_cast<int>(code.codelength) - UINT_BIT + written_bits);
                     ++int_idx;
                     (*encoded_data)[int_idx] |= static_cast<unsigned int>(code.code) << (UINT_BIT - static_cast<int>(code.codelength) + UINT_BIT - written_bits);
                     written_bits = written_bits + static_cast<int>(code.codelength) - UINT_BIT;
                 }
                 else {
                     (*encoded_data)[int_idx] |= static_cast<unsigned int>(code.code) >> (static_cast<int>(code.codelength) - UINT_BIT + written_bits);
                     ++int_idx;
                     written_bits = 0;
                }
            }
        }
    }
}
