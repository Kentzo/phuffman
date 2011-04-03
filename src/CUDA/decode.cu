#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include "phuffman_math.cu"
#include "decode.hpp"
#include "CodesTable.h"

namespace phuffman {
    namespace CUDA {
        using thrust::exclusive_scan;
        using thrust::for_each;
        using thrust::device_vector;
        using thrust::tuple;
        using thrust::make_tuple;
        using thrust::counting_iterator;
        using thrust::copy;
        using thrust::get;
        using thrust::zip_iterator;
        typedef device_vector<unsigned char> DevBlockOffsets;
        typedef device_vector<unsigned int> DevBlockSizes;
        typedef counting_iterator<unsigned int> IndexIterator;
        typedef zip_iterator<tuple<typename DevBlockOffsets::iterator, typename DevBlockOffsets::iterator, typename DevBlockSizes::iterator, IndexIterator> > BlockDataIterator;
        typedef tuple<typename DevBlockOffsets::value_type, typename DevBlockOffsets::value_type, typename DevBlockSizes::value_type, typename IndexIterator::value_type> BlockData;

        __constant__ Code _CODES[ALPHABET_SIZE];

        namespace detail {
            class DecodeBlock {
            public:
                __host__
                DecodeBlock(unsigned int* dev_encoded_data, unsigned int block_size, unsigned char* dev_data) : _dev_encoded_data(dev_encoded_data), _block_size(block_size), _dev_data(dev_data) {}

                __device__
                void operator()(const BlockData& block) {
                    unsigned char offset = get<0>(block);
                    unsigned char skipped_bits = get<1>(block) % bytes_to_bits(sizeof(unsigned int));
                    unsigned int data_pos = get<2>(block);
                    unsigned int index = get<3>(block);
                    unsigned int encoded_data_pos = (_block_size * index) / sizeof(unsigned int) + get<1>(block) / sizeof(unsigned int);
                    unsigned int block_bit_size = bytes_to_bits(_block_size) + offset - skipped_bits;

                    unsigned char code_val = 0;
                    unsigned char code_length = 0;
                    size_t bit_idx = skipped_bits;
                    while (bit_idx < block_bit_size) {
                        unsigned char bit = (_dev_encoded_data[encoded_data_pos] >> (bytes_to_bits(sizeof(unsigned int)) - (bit_idx % CHAR_BIT) - 1)) & 1;
                        code_val = (code_val << 1) | bit;
                        ++code_length;
                        for (unsigned short i=0; i<ALPHABET_SIZE; ++i) {
                            Code code = _CODES[i];
                            if (code.codelength == code_length && code.code == code_val) {
                                _dev_data[data_pos] = (unsigned char)i;
                                code_val = 0;
                                code_length = 0;
                                break;
                            }
                        }
                        ++bit_idx;
                        encoded_data_pos += bit_idx / bytes_to_bits(sizeof(unsigned int));
                    }
                }
                
            private:
                unsigned int* _dev_encoded_data;
                unsigned int _block_size;
                unsigned char* _dev_data;
            };
        }
        
        void Decode(unsigned int* encoded_data, size_t encoded_data_length, CodesTable codes_table, unsigned int block_size, unsigned char* block_offsets, unsigned int* block_sizes, size_t block_length, unsigned char** data, size_t* data_length)
        {
            
            cudaMemcpyToSymbol(_CODES, codes_table.codes, ALPHABET_SIZE * sizeof(Code));
            DevBlockOffsets dev_offsets(block_length + 1, 0);
            copy(block_offsets, block_offsets + block_length, dev_offsets.begin() + 1);
            DevBlockSizes dev_sizes(block_sizes, block_sizes + block_length);
            unsigned int last_block_size = block_sizes[block_length - 1];
            exclusive_scan(dev_sizes.begin(), dev_sizes.end(), dev_sizes.begin());

            unsigned int* dev_encoded_data = NULL;
            cudaMalloc(&dev_encoded_data, encoded_data_length * sizeof(unsigned int));
            cudaMemcpy(dev_encoded_data, encoded_data, encoded_data_length * sizeof(unsigned int), cudaMemcpyHostToDevice);

            unsigned char* dev_data = NULL;
            *data_length = dev_sizes.back() + last_block_size;
            cudaMalloc(&dev_data, dev_sizes.back() + last_block_size);
            cudaMemset(dev_data, 0, dev_sizes.back() + last_block_size);

            IndexIterator index(0);
            BlockDataIterator blocks_begin = make_tuple(dev_offsets.begin() + 1, dev_offsets.begin(), dev_sizes.begin(), index),
                blocks_end = make_tuple(dev_offsets.end(), dev_offsets.end() - 1, dev_sizes.end(), index + block_length);
            thrust::for_each(blocks_begin, blocks_end, detail::DecodeBlock(dev_encoded_data, block_size, dev_data));

            *data = (unsigned char*)calloc(*data_length, sizeof(unsigned char));
            cudaMemcpy(data, dev_data, *data_length, cudaMemcpyDeviceToHost);
            cudaFree(dev_encoded_data);
            cudaFree(dev_data);
        }
    }
}
