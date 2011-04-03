#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/for_each.h>
#include <thrust/remove.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <cmath>
#include <iostream>
#include "phuffman_math.cu"
#include "phuffman_limits.cuh"
#include "encode.hpp"
#include "constants.h"
#include "Code.h"

namespace phuffman {
    namespace CUDA {
        using thrust::transform;
        using thrust::device_vector;
        using thrust::exclusive_scan;
        using thrust::for_each;
        using thrust::remove_if;
        using thrust::zip_iterator;
        using thrust::make_zip_iterator;
        using thrust::tuple;
        using thrust::make_tuple;
        using thrust::get;
        using std::cerr;
        using std::endl;
        typedef device_vector<unsigned char> DevData;
        typedef device_vector<size_t> DevPrefixSum;
        typedef zip_iterator<tuple<DevData::iterator, DevPrefixSum::iterator> > CharPosIterator;
        typedef tuple<unsigned char, unsigned int> CharPos;

        __constant__ Code _CODES[ALPHABET_SIZE];

        struct _LengthEncode {
            /*!
              @abstract Returns number of bits for code of a given symbol.
              @discussion The result MUST lie within (0, MAXIMUM_CODELENGTH].
            */
            __device__ DevPrefixSum::value_type operator()(DevData::value_type symbol) {
                return _CODES[symbol].codelength;
            }
        };

        struct _NaiveMerge2 {
           unsigned int* _result;

            /*!
              @param result A pointer to the global memory. Memory MUST BE aligned by sizeof(uint2).
            */
            _NaiveMerge2(unsigned int* result) : _result(result) {}

            /*!
              @abstract Merges code of a given symbol into the device global memory at a given position.
              @param tuple A tuple that represents current symbol and position of its code at the result bit array.
            */
            __device__ void operator()(const CharPos& tuple) {
                Code code = _CODES[get<0>(tuple)];
                uint2 code_aligned = make_uint2(0, code.code) << (UINT2_BIT - code.codelength - (get<1>(tuple) % UINT_BIT));
                unsigned int* code_address = _result + get<1>(tuple) / UINT_BIT;
                atomicOr(code_address, code_aligned.x);
                atomicOr(code_address + 1, code_aligned.y);
            }
        };

        struct _CalcBlockData {
            size_t _block_size_bit; // Number of bits in a block
            unsigned char* _block_offsets; // Offset for each block
            unsigned int* _block_sizes; // Number of symblos in each block

            _CalcBlockData(size_t block_int_size, unsigned char* block_bit_offsets, unsigned int* block_sym_sizes) : _block_size_bit(bytes_to_bits(block_int_size * sizeof(unsigned int))), _block_offsets(block_bit_offsets), _block_sizes(block_sym_sizes) {}

            /*!
              @abstract We are calculating indexes of blocks where currect code starts and ends. If the whole code lies within one block then it's non-conflict.
                        Otherwise it's conflict and we use it to calculate offset for a block where "offset" means the number of bits in the next block.
            */
            __device__ void operator()(const CharPos& tuple) {
                unsigned int code_start_pos = get<1>(tuple),
                    code_end_pos = code_start_pos + _CODES[get<0>(tuple)].codelength;
                unsigned int block_start_idx = code_start_pos / _block_size_bit,
                    block_end_idx = code_end_pos / _block_size_bit;
                atomicAdd(_block_sizes + block_start_idx, 1);
                if (block_start_idx != block_end_idx) {
                    _block_offsets[block_start_idx] = code_end_pos % _block_size_bit;
                }
            }
        };

        void Encode(unsigned char* data,
                    size_t data_length,
                    CodesTable codes_table,
                    unsigned int** encoded_data,
                    size_t* encoded_data_length,
                    unsigned char* encoded_data_trail_zeroes,
                    size_t block_int_size/* = 0*/,
                    unsigned char** block_bit_offsets/* = NULL*/,
                    unsigned int** block_sym_sizes/* = NULL*/,
                    size_t* block_count/* = NULL*/)
        {
            cudaError_t error = cudaSuccess;

            if ((error = cudaMemcpyToSymbol(_CODES, codes_table.codes, ALPHABET_SIZE * sizeof(Code))) != cudaSuccess) {
                cerr << "Cannot copy codes into the device memory" << endl;
                throw error;
            }

            unsigned int* dev_result = NULL;
            unsigned char* dev_block_bit_offsets = NULL;
            unsigned int* dev_block_sym_sizes = NULL;
            try {
// Calculate Exclusive Prefix Sum
                DevData dev_data(data, data + data_length);
                DevPrefixSum dev_prefix_sum(data_length, 0);
                transform(dev_data.begin(), dev_data.end(), dev_prefix_sum.begin(), _LengthEncode());
                exclusive_scan(dev_prefix_sum.begin(), dev_prefix_sum.end(), dev_prefix_sum.begin());
                size_t bit_length = dev_prefix_sum.back() + codes_table.codes[dev_data.back()].codelength;
                *encoded_data_trail_zeroes = UINT_BIT - (bit_length % UINT_BIT);
                *encoded_data_length = bit_length / UINT_BIT;
                if (*encoded_data_trail_zeroes > 0) {
                    *encoded_data_length += 1;
                }

// Encode Data
                size_t dev_result_length = ceilf(static_cast<float>(*encoded_data_length) / (sizeof(uint2) / sizeof(unsigned int))); // uint2
                dev_result_length *= sizeof(uint2); // bytes

                if ((error = cudaMalloc(&dev_result, dev_result_length)) != cudaSuccess) {
                    cerr << "Cannot allocate " << dev_result_length << " bytes on the device" << endl;
                    throw error;
                }
                if ((error = cudaMemset(dev_result, 0, dev_result_length)) != cudaSuccess) {
                    cerr << "Cannot nullify " << dev_result_length << " bytes at " << dev_result << endl;
                    throw error;
                }
                CharPosIterator charpos_begin = make_tuple(dev_data.begin(), dev_prefix_sum.begin()),
                                charpos_end = make_tuple(dev_data.end(), dev_prefix_sum.end());
                thrust::for_each(charpos_begin, charpos_end, _NaiveMerge2(dev_result));
                // Copy Encoded data to host
                *encoded_data = static_cast<unsigned int*>(calloc(dev_result_length, sizeof(unsigned char)));
                if ((error = cudaMemcpy(*encoded_data, dev_result, dev_result_length, cudaMemcpyDeviceToHost)) != cudaSuccess) {
                    cerr << "Cannot copy encoded data from device to host" << endl;
                    throw error;
                }
                cudaFree(dev_result);

// Calculate Block Offsets
                if (block_int_size != 0) {
                    *block_count = ceilf(static_cast<float>(*encoded_data_length) / block_int_size);
                    size_t blocks_byte_length = *block_count * block_int_size * sizeof(unsigned int);
                    if ((error = cudaMalloc(&dev_block_bit_offsets, blocks_byte_length)) != cudaSuccess) {
                        cerr << "Cannot allocate " << blocks_byte_length << " bytes on the device" << endl;
                        throw error;
                    }
                    if ((error = cudaMemset(dev_block_bit_offsets, 0, blocks_byte_length)) != cudaSuccess) {
                        cerr << "Cannot nullify " << blocks_byte_length << " bytes at " << dev_block_bit_offsets << endl;
                        throw error;
                    }
                    if ((error = cudaMalloc(&dev_block_sym_sizes, blocks_byte_length)) != cudaSuccess) {
                        cerr << "Cannot allocate " << blocks_byte_length << " bytes on the device" << endl;
                        throw error;
                    }
                    if ((error = cudaMemset(dev_block_sym_sizes, 0, blocks_byte_length)) != cudaSuccess) {
                        cerr << "Cannot nullify " << blocks_byte_length << " bytes at " << dev_block_sym_sizes << endl;
                        throw error;
                    }
                    thrust::for_each(charpos_begin, charpos_end, _CalcBlockData(block_int_size, dev_block_bit_offsets, dev_block_sym_sizes));
                    
                    *block_bit_offsets = static_cast<unsigned char*>(calloc(blocks_byte_length, sizeof(unsigned char)));
                    if ((error = cudaMemcpy(*block_bit_offsets, dev_block_bit_offsets, blocks_byte_length, cudaMemcpyDeviceToHost)) != cudaSuccess) {
                        cerr << "Cannot copy block offsets from device to host" << endl;
                        throw error;
                    }

                    *block_sym_sizes = static_cast<unsigned int*>(calloc(blocks_byte_length, sizeof(unsigned char)));
                    if ((error = cudaMemcpy(*block_sym_sizes, dev_block_sym_sizes, blocks_byte_length, cudaMemcpyDeviceToHost)) != cudaSuccess) {
                        cerr << "Cannot copy block sizes from device to host" << endl;
                        throw error;
                    }

                    cudaFree(dev_block_bit_offsets);
                    cudaFree(dev_block_sym_sizes);
                }

            }
            catch(...) {
                if (dev_result != NULL) {
                    cudaFree(dev_result);
                }
                if (*encoded_data != NULL) {
                    free(*encoded_data);
                    *encoded_data = NULL;
                }
                *encoded_data_length = 0;
                *encoded_data_trail_zeroes = 0;
//
//                if (*block_offsets != NULL) {
//                    free(*block_offsets);
//                    *block_offsets = NULL;
//                }
//                if (dev_block_sizes != NULL) {
//                    cudaFree(dev_block_sizes);
//                }
//                if (*block_sizes != NULL) {
//                    free(*block_sizes);
//                    *block_sizes = NULL;
//                }
//                *block_length = 0;
                throw;
            }
        }
    }
}
