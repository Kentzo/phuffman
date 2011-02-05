#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include "Code.h"
#include "constants.h"
#include "CodesTable.h"
#include <stdio.h>
#include <math.h>
#include <stddef.h>

#include "thrust/scan.h"
#include "cuPrintf.cu"

__constant__ Code codes[ALPHABET_SIZE];

__device__ int4 left_shift_int4(int4 a_value, unsigned int a_num)
{
    int w_l = a_value.w << (a_num % 32);
    int w_h = a_value.w >> (32 - (a_num % 32));

    int z_l = a_value.z << (a_num % 32);
    int z_h = a_value.z >> (32 - (a_num % 32));

    int y_l = a_value.y << (a_num % 32);
    int y_h = a_value.y >> (32 - (a_num % 32));

    int x_l = a_value.x << (a_num % 32);

    if (a_num <= 32) {
        a_value = make_int4(x_l | y_h, y_l | z_h, z_l | w_h, w_l);
    }
    else if (a_num > 32 && a_num <= 64) {
        a_value = make_int4(y_l | z_h, z_l | w_h, w_l, 0);
    }
    else if (a_num > 64 && a_num <= 96) {
        a_value = make_int4(z_l | w_h, w_l, 0, 0);
    }
    else {
        a_value = make_int4(w_l, 0, 0, 0);
    }

    return a_value;
}

__device__ int4 right_shift_int4(int4 a_value, unsigned int a_num)
{
    int w_h = a_value.w >> (a_num % 32);

    int z_h = a_value.z >> (a_num % 32);
    int z_l = a_value.z << (32 - (a_num % 32));

    int y_h = a_value.y >> (a_num % 32);
    int y_l = a_value.y << (32 - (a_num % 32));

    int x_h = a_value.x >> (a_num % 32);
    int x_l = a_value.x << (32 - (a_num %32));

    if (a_num <= 32) {
        a_value = make_int4(x_h, y_h | x_l, z_h | y_l, w_h | z_l);
    }
    else if (a_num > 32 && a_num <= 64) {
        a_value = make_int4(0, x_h, y_h | x_l, z_h | y_l);
    }
    else if (a_num > 64 && a_num <= 96) {
        a_value = make_int4(0, 0, x_h, y_h | x_l);
    }
    else {
        a_value = make_int4(0, 0, 0, x_h);
    }

    return a_value;
}

__device__ int4 or_int4(int4 a_value1, int4 a_value2)
{
    return make_int4(a_value1.x | a_value2.x, a_value1.y | a_value2.y, a_value1.z | a_value2.z, a_value1.w | a_value2.w);
}

__device__ void merge_int4(int4 a_value1, int a_value1_zeroes, int4 a_value2, int a_value2_zeroes, int4 (*result)[2]) {
    int4 a_value2_h = right_shift_int4(a_value2, sizeof(int4) - a_value1_zeroes);
    int4 a_value2_l = left_shift_int4(a_value2, a_value1_zeroes);
    *result[0] = or_int4(a_value1, a_value2_h);
    *result[1] = a_value2_l;
}



__global__ void length_encode(unsigned char* a_data, unsigned int a_data_len, unsigned int* a_result, unsigned int num_to_process)
{
  unsigned int bx = threadIdx.x;
  unsigned int b_idx = min(bx * num_to_process, a_data_len);
  unsigned int e_idx = min(b_idx + num_to_process, a_data_len);
  //  cuPrintf("[%i %i)\n", idx, size);
  for(; b_idx < e_idx; ++b_idx) {
    unsigned char c = a_data[b_idx];
    a_result[b_idx] = codes[c].codelength;
    //cuPrintf("[%i %i %i] %i\n", );
  }
}

__global__ void encode(unsigned char* a_data, unsigned int a_data_len, unsigned char* a_prefix, unsigned int num_to_process)
{
    unsigned int bx = threadIdx.x;
    const int NUM_OF_SYM = 5;
    for (int block_idx = 0; block_idx < num_to_process; block_idx += NUM_OF_SYM) {
        unsigned int b_idx = min(bx * num_to_process + block_idx, a_data_len);
        unsigned int e_idx = min(b_idx + NUM_OF_SYM, a_data_len);

        // Encode
        int4 buf[2];
        int buf_idx = 0; // buffer index
        int x = 0;
        int y = 0;
        for (int j = b_idx; j < e_idx; ++j) {
            unsigned char c = a_data[j];
            x = sizeof(int4) - codes[c].codelength - y;
            if (x >= 0) {
                x = 0;
                buf[buf_idx] = or_int4(buf[buf_idx],
                        left_shift_int4(make_int4(0, 0, 0, codes[c].code), sizeof(int4) - codes[c].codelength - y));
                y += codes[c].codelength;
            }
            else {
                x = -x;
                int4 code = left_shift_int4(make_int4(0, 0, 0, codes[c].code), sizeof(int4) - codes[c].codelength);
                merge_int4(buf[0], sizeof(int4) - y, code, sizeof(int4) - codes[c].codelength, &buf);
                /*int2 high_bits = make_int2(0, codes[c].code) >> x;
                int2 low_bits = make_int2(0, code[c].code) << sizeof(int2) - x;
                buf[buf_idx] |= high_bits;
                buf[buf_idx+1] |= low_bits;*/
                ++buf_idx;
            }
        }
    }
}

extern "C" {

  void runEncode(unsigned char* a_data, size_t len, CodesTable a_table, unsigned char* result, size_t* result_len)
  {
    cudaPrintfInit();
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
      return;
    } 

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceCount - 1);
    float totalGlobal = deviceProp.totalGlobalMem;    
    float maxCodeLen = ceilf((float)a_table.info.maximum_codelength / 8);
    float maxDataLen = floorf((totalGlobal - sizeof(codes))/(5 + maxCodeLen)) ;
    printf("max codelength: %f\n", maxCodeLen);
    size_t num = max(32.0, ceilf(len/512.0));
    CUdevice device;
    dim3 grid_range(1,1,1);
    dim3 block_range(512,1,1);

    cutilSafeCall(cudaMemcpyToSymbol(codes, a_table.codes, ALPHABET_SIZE * sizeof(Code)));
    void* devData = NULL;
    void* devResultAddr = NULL;
    void* devResult = NULL;
    while (len != 0) {
      long long dataLen = min((long long)len, (long long)maxDataLen); // count of elements
      int bytesCopy = 0;

      cutilSafeCall(cudaMalloc(&devData, dataLen));
      cutilSafeCall(cudaMalloc(&devResultAddr, dataLen*sizeof(unsigned int)));
      cutilSafeCall(cudaMalloc(&devResult, dataLen*maxCodeLen));
      cutilSafeCall(cudaMemcpy(devData, a_data, dataLen, cudaMemcpyHostToDevice));

      length_encode<<<grid_range, block_range>>>((unsigned char*)devData, dataLen, (unsigned int*)devResultAddr, num);
      
      thrust::device_ptr<unsigned int> devPtr((unsigned int*)devResultAddr);
      thrust::inclusive_scan(devPtr, devPtr + dataLen, devPtr);



      cudaPrintfDisplay(stdout, true); 
      cudaPrintfEnd();

      cutilSafeCall(cudaMemcpy(&bytesCopy, (unsigned int*)devResultAddr + dataLen - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
      unsigned int* devResultAddrHost = (unsigned int*)calloc(dataLen, sizeof(unsigned int));
      cutilSafeCall(cudaMemcpy(devResultAddrHost, devResultAddr, dataLen*sizeof(unsigned int), cudaMemcpyDeviceToHost));
      for (int i = 0; i < dataLen; ++i) {
	//				printf("%i %i\n ", i, devResultAddrHost[i]);
      }
      free(devResultAddrHost);
      printf("%i %f\n", bytesCopy, maxCodeLen);
      //cutilSafeCall(cudaMemcpy(result, devResult, ceilf((float)bytesCopy / 8), cudaMemcpyDeviceToHost));

      cudaFree(devData);
      cudaFree(devResultAddr);
      cudaFree(devResult);

      len -= dataLen;
    }	 
  }
}

