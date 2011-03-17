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
#include "../constants.h"
#include "../Code.h"

namespace phuffman {
    namespace CUDA {
        using thrust::device_ptr;
        using thrust::transform;
        using thrust::device_vector;
        using thrust::exclusive_scan;
        using thrust::for_each;
        using thrust::make_tuple;
        using thrust::get;
        using thrust::remove_if;
        using thrust::make_zip_iterator;
        using thrust::tuple;
        using std::cerr;
        using std::endl;
        typedef thrust::zip_iterator<tuple<device_vector<unsigned char>::iterator, device_vector<unsigned int>::iterator> > CharPosIterator;
        typedef thrust::tuple<unsigned char, unsigned int> CharPos;
        typedef thrust::device_vector<unsigned char> DevData;
        typedef thrust::device_vector<unsigned int> DevPrefixSum;

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
              @param tuple A tuple that represents current symbol and position of its code at the device global memory.
            */
            __device__ void operator()(const CharPos& tuple) {
                Code code = _CODES[get<0>(tuple)];
                uint2 code_aligned = make_uint2(0, code.code) << (UINT2_BIT - code.codelength - (get<1>(tuple) % UINT_BIT));
                unsigned int* code_address = _result + get<1>(tuple) / UINT_BIT;
                atomicOr(code_address, code_aligned.x);
                atomicOr(code_address + 1, code_aligned.y);
            }
        };

        struct _IsConflictBlock {
            unsigned int _block_size_bit;

            _IsConflictBlock(unsigned int block_size) : _block_size_bit(bytes_to_bits(block_size)) {}
            __device__ bool operator()(const CharPos& tuple) {
                unsigned int start_block_idx = get<1>(tuple)/_block_size_bit;
                unsigned int code_address_end = get<1>(tuple) + _CODES[get<0>(tuple)].codelength;
                unsigned int end_block_idx = code_address_end / _block_size_bit;
                return start_block_idx != end_block_idx || (code_address_end % _block_size_bit) == 0;
            }
        };

        struct _CalcOffset {
            unsigned int _block_size_bit;

            _CalcOffset(unsigned int block_size) : _block_size_bit(bytes_to_bits(block_size)) {}
            __device__ unsigned char operator()(CharPos tuple) {
                return (get<1>(tuple) + _CODES[get<0>(tuple)].codelength) % _block_size_bit;
            }
        };

        void Encode(unsigned char* data, size_t length, CodesTable codes_table, unsigned int** result, size_t* result_length, size_t* result_length_bit,
                           unsigned int block_size /*= 0*/, unsigned char** block_offsets /*= NULL*/, size_t* block_offsets_length /*= NULL*/)
        {
            cudaError_t error = cudaSuccess;

            if ((error = cudaMemcpyToSymbol(_CODES, codes_table.codes, ALPHABET_SIZE * sizeof(Code))) != cudaSuccess) {
                cerr << "Cannot copy codes into the device memory" << endl;
                throw error;
            }

            // Calculate Exclusive Prefix Sum
            DevData dev_data(data, data + length);
            DevPrefixSum dev_prefix_sum(length, 0);
            transform(dev_data.begin(), dev_data.end(), dev_prefix_sum.begin(), _LengthEncode());
            exclusive_scan(dev_prefix_sum.begin(), dev_prefix_sum.end(), dev_prefix_sum.begin());

            unsigned int* dev_result = NULL;
            try {
                // Encode Data
                // As we use exclusive prefix sum, we need to add the length of the last element manually
                size_t dev_result_length = dev_prefix_sum.back() + codes_table.codes[*(data + length - 1)].codelength; // bits
                dev_result_length = ceilf(static_cast<float>(dev_result_length) / UINT2_BIT) + 1; // uint2
                dev_result_length *= sizeof(uint2); // bytes

                if ((error = cudaMalloc(&dev_result, dev_result_length)) != cudaSuccess) {
                    cerr << "Cannot allocate " << dev_result_length << "bytes on the device" << endl;
                    throw error;
                }
                if ((error = cudaMemset(dev_result, 0, dev_result_length)) != cudaSuccess) {
                    cerr << "Cannot nullify " << dev_result_length << " bytes of memory at " << dev_result << endl;
                    throw error;
                }
                CharPosIterator charpos_begin(make_tuple(dev_data.begin(), dev_prefix_sum.begin())), charpos_end(make_tuple(dev_data.end(), dev_prefix_sum.end()));
                thrust::for_each(charpos_begin, charpos_end, _NaiveMerge2(dev_result));

                // Copy Data To Host
                *result = static_cast<unsigned int*>(calloc(dev_result_length, sizeof(unsigned char)));
                if ((error = cudaMemcpy(*result, dev_result, dev_result_length, cudaMemcpyDeviceToHost)) != cudaSuccess) {
                    cerr << "Cannot copy data from device to host" << endl;
                    throw error;
                }

                // Get The Size Of The Result In Bits
                *result_length_bit = dev_prefix_sum.back() + codes_table.codes[*(data + length - 1)].codelength;
                *result_length = ceilf(static_cast<float>(*result_length_bit) / UINT_BIT);;

                // Calculate Block Offsets

                //            DevPrefixSum::iterator new_end = remove_if(dev_prefix_sum.begin(), dev_prefix_sum.end(), charpos_begin, _IsConflictBlock(block_size));
                //            *block_offsets = static_cast<unsigned char*>(calloc(new_end - dev_prefix_sum.begin(), sizeof(DevPrefixSum::value_type)));
                //            charpos_end = make_zip_iterator();

                //            transform(dev_prefix_sum.begin(), new_end, *block_offsets, _CalcOffset(block_size));
            }
            catch(...) {
                if (dev_result != NULL) {
                    cudaFree(dev_result);
                }
                if (*result != NULL) {
                    free(*result);
                    *result = NULL;
                }
                *result_length_bit = 0;
                *result_length = 0;
                throw;
            }

            cudaFree(dev_result);
        }
    }
}
