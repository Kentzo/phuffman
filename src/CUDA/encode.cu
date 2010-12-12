#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include "Code.h"
#include "constants.h"
#include "CodesTable.h"
#include <stdio.h>
#include <math.h>
#include <stddef.h>

__constant__ CodesTable table;

__global__ void length_encode(unsigned char* a_data, unsigned int* a_result, size_t num, size_t totalSize)
{  
  unsigned int tx = threadIdx.x;
  unsigned int bx = blockIdx.x;
  unsigned int idx = blockDim.x*bx + tx;
  if (totalSize < idx)
    return;
  num = min((unsigned long long)num, (unsigned long long)totalSize - idx);
  for (int i = 0; i < num; ++i) {
     int cidx = idx + i;
     unsigned char c = a_data[cidx];
     Code row = table.codes[c];     
     a_result[cidx] = (unsigned int)row.codelength;
  }
}

extern "C" {
  void runEncode(unsigned char* a_data, size_t len, CodesTable a_table, unsigned char* result, size_t* result_len)
  {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
      return;
    }
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceCount - 1);
    float totalGlobal = deviceProp.totalGlobalMem;    
    float maxCodeLen = ceilf((float)a_table.info.maximum_codelength / 8);
    float maxDataLen = floorf((totalGlobal - sizeof(table))/(5 + maxCodeLen)) ;

    static const size_t num = 32;
    static const size_t bsize = 512;

    cutilSafeCall(cudaMemcpyToSymbol(table, &a_table, sizeof(CodesTable)));
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

      float gsize = (float(dataLen) / (num * bsize)); 

      gsize = ceilf(gsize);
      printf("gsize = %f\n", gsize);
      
      dim3 grid(gsize, 1, 1);
      dim3 block(bsize, 1, 1);
      length_encode<<<grid, block>>>((unsigned char*)devData, (unsigned int*)devResultAddr, num, dataLen);
      

      cutilSafeCall(cudaMemcpy(&bytesCopy, (unsigned int*)devResultAddr + dataLen - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
      unsigned int* devResultAddrHost = (unsigned int*)calloc(dataLen, sizeof(unsigned int));
      cutilSafeCall(cudaMemcpy(devResultAddrHost, devResultAddr, dataLen*sizeof(unsigned int), cudaMemcpyDeviceToHost));
      for (int i = 0; i < dataLen; ++i) {
	printf("%i %i\n ", i, devResultAddrHost[i]);
      }
      free(devResultAddrHost);
      
      cutilSafeCall(cudaMemcpy(result, devResult, bytesCopy, cudaMemcpyDeviceToHost));

      cudaFree(devData);
      cudaFree(devResultAddr);
      cudaFree(devResult);

      len -= dataLen;
    }    
  }
}

