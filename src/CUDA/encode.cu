#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include "Row.h"
#include <stdio.h>

__constant__ Row table[256];

__global__ void encode(unsigned char* a_data, Row* a_result, size_t len, size_t num)
{  
  unsigned int tx = threadIdx.x;
  unsigned int bx = blockIdx.x;
  unsigned int idx = blockDim.x*bx + tx;

  for (int i = 0; i < num; ++i) {
     int cidx = idx + i;
     unsigned char c = a_data[cidx];
     Row row = table[c];     
     a_result[cidx] = row;
  }
}

extern "C" {
  Row* runEncode(unsigned char* a_data, size_t len, Row a_table[256])
  {
    cutilSafeCall(cudaMemcpyToSymbol(table, a_table, sizeof(Row) * 256));

    void* devData = NULL;
    void* result = NULL;
    cutilSafeCall(cudaMalloc(&devData, len));
    cutilSafeCall(cudaMemcpy(devData, a_data, len, cudaMemcpyHostToDevice));
  
    cutilSafeCall(cudaMalloc(&result, len * sizeof(Row)));

    dim3 grid(44050, 1, 1);
    dim3 block(512, 1, 1);
    encode<<<grid, block>>>((unsigned char*)devData, (Row*)result, len, 1);
		  
    Row* res = (Row*)calloc(len, sizeof(Row));
    cutilSafeCall(cudaMemcpy(res, result, len * sizeof(Row), cudaMemcpyDeviceToHost));

    cudaFree(devData);
    cudaFree(result);  
    return res;
  }
}

