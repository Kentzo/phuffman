#include <cstdlib>
#include <cmath>
#include "cpu_decode.hpp"
#include "constants.h"
#include "phuffman_limits.cuh"

namespace phuffman {
    namespace CPU {
        namespace detail {
            void DecodeBlock(size_t block_int_size, unsigned int* encoded_data_begin, unsigned char bit_lead_offset, unsigned char bit_trail_offset, unsigned char* data_begin, Code codes[ALPHABET_SIZE])
            {
                unsigned char cur_codelength = 0;
                unsigned char cur_value = 0;

                size_t int_idx = 0;
                size_t bit_idx = bit_lead_offset;
                size_t sym_idx = 0;
                size_t read_bits = 0;

                while (read_bits < (block_int_size * UINT_BIT + bit_trail_offset)) {
                    cur_value <<= 1;
                    ++cur_codelength;
                    cur_value |= (encoded_data_begin[int_idx] >> (UINT_BIT - bit_idx - 1)) & 1;
                    ++bit_idx;
                    ++read_bits;
                    for (size_t i=0; i<ALPHABET_SIZE; ++i) {
                        Code code = codes[i];
                        if (code.code == cur_value && code.codelength == cur_codelength) {
                            data_begin[sym_idx] = static_cast<unsigned char>(i);
                            ++sym_idx;
                            cur_value = 0;
                            cur_codelength = 0;
                            break;
                        }
                    }
                    if (bit_idx == UINT_BIT) {
                        ++int_idx;
                        bit_idx = 0;
                    }
                }
            }

            void DecodeLastBlock(unsigned int* encoded_data_begin, unsigned char bit_lead_offset, size_t bit_length, unsigned char* data_begin, Code codes[ALPHABET_SIZE]) {
                unsigned char cur_codelength = 0;
                unsigned char cur_value = 0;

                size_t int_idx = 0;
                size_t bit_idx = bit_lead_offset;
                size_t sym_idx = 0;
                size_t read_bits = 0;

                while (read_bits < bit_length) {
                    cur_value <<= 1;
                    ++cur_codelength;
                    cur_value |= (encoded_data_begin[int_idx] >> (UINT_BIT - bit_idx - 1)) & 1;
                    ++bit_idx;
                    ++read_bits;
                    for (size_t i=0; i<ALPHABET_SIZE; ++i) {
                        Code code = codes[i];
                        if (code.code == cur_value && code.codelength == cur_codelength) {
                            data_begin[sym_idx] = static_cast<unsigned char>(i);
                            ++sym_idx;
                            cur_value = 0;
                            cur_codelength = 0;
                            break;
                        }
                    }
                    if (bit_idx == UINT_BIT) {
                        ++int_idx;
                        bit_idx = 0;
                    }
                }
            }
        }

        void Decode(unsigned char** data,
                    size_t* data_length,
                    CodesTable codes_table,
                    unsigned int* encoded_data,
                    size_t encoded_data_length,
                    unsigned char encoded_data_trail_zeroes,
                    size_t block_int_size,
                    unsigned char* block_bit_offsets,
                    unsigned int* block_sym_sizes,
                    size_t block_count)
        {
            size_t* prefix_sum = new size_t[block_count];
            prefix_sum[0] = 0;
            for (size_t i=1; i<block_count; ++i) {
                prefix_sum[i] = prefix_sum[i-1] + block_sym_sizes[i-1];
            }
            *data_length = prefix_sum[block_count-1] + block_sym_sizes[block_count-1];
            *data = static_cast<unsigned char*>(calloc(*data_length, sizeof(unsigned char)));

            detail::DecodeBlock(block_int_size, encoded_data, 0, block_bit_offsets[0], *data, codes_table.codes);
            // The last block will be decoded separately.
            for(size_t i=1; i<block_count-1; ++i) {
                detail::DecodeBlock(block_int_size, encoded_data + (i * block_int_size), block_bit_offsets[i-1], block_bit_offsets[i], *data + prefix_sum[i], codes_table.codes);
            }
            size_t bit_length = (encoded_data_length % block_int_size) * UINT_BIT;
            if (bit_length == 0) {
                bit_length = block_int_size * UINT_BIT;
            }
            bit_length -= encoded_data_trail_zeroes;
            detail::DecodeLastBlock(encoded_data + ((block_count - 1) * block_int_size),
                            block_bit_offsets[block_count-2],
                            bit_length,
                            *data + prefix_sum[block_count-1],
                            codes_table.codes);

        }
    }
}
