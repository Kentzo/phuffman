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

__global__ void length_encode(unsigned char* a_data, unsigned int* a_result, unsigned int num, unsigned int totalSize)
{  
  unsigned int bx = threadIdx.x;
  unsigned int by = threadIdx.y;		       
  unsigned int bz = threadIdx.z;
  unsigned int by_size = blockDim.y;	       
  unsigned int bz_size = blockDim.z;  
  unsigned int idx = min((bx*by_size*bz_size + by*bz_size + bz) * num, totalSize);
  unsigned int size = min(idx + num, totalSize);
  //  cuPrintf("[%i %i)\n", idx, size);
  for(; idx<size; ++idx) {		       
    unsigned char c = a_data[idx];
    a_result[idx] = codes[c].codelength;
    //cuPrintf("[%i %i %i] %i\n", );
  }
}

extern "C" {

  void _Sizes(size_t data_size, CUdevice device, size_t* items_per_thread, dim3* grid_range, dim3* block_range) {
    
    CUdevprop properties;
    cuDeviceGetProperties(&properties, device);
    // Retrieve maximum values for calculated parameters
    int max_grid_size = properties.maxGridSize[0];
    
    int max_block_dimensions = 3;

    int* max_block_sizes = properties.maxThreadsDim;

    // Begin calculating
    int threads_num = size_t(ceilf(float(data_size) / *items_per_thread));

    if (threads_num > 0) {
      // Calculate work group size and appropriate value for items_per_thread
      int max_blocks_per_grid = 1;
      for (int i=0; i<max_block_dimensions; ++i) {
	max_blocks_per_grid *= max_block_sizes[i];
      }         
      int grid_size = size_t(ceilf(float(threads_num) / max_blocks_per_grid));
      // Calculate appropriate value for items_per_thread
      while (grid_size > max_grid_size) {
	(*items_per_thread) <<= 1;
	threads_num = size_t(ceilf(float(data_size) / *items_per_thread));
	grid_size = size_t(ceilf(float(threads_num) / max_blocks_per_grid));
      }
      // Calculate work item sizes
      threads_num = size_t(ceilf(float(threads_num) / grid_size));
      int block_sizes[3] = {1, 1, 1};
      for (int i=0; i<max_block_dimensions && threads_num>0; ++i) {
	int size = min(max_block_sizes[i], threads_num);
	printf("dim %i %i\n", i, size);
	block_sizes[i] = size;
	threads_num = ceilf((float)threads_num / size);
	// x * y * z >= threads_num
      }

      *grid_range = dim3(grid_size, 1, 1);
      *block_range = dim3(block_sizes[0], block_sizes[1], block_sizes[2]);
    } else {
      *grid_range = dim3(0, 0, 0);
      *block_range = dim3(0, 0, 0);
    }
  }

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
    //cuDeviceGet(&device, deviceCount - 1);
    //_Sizes(len, device, &num, &grid_range, &block_range);

    printf("Grid range: %i %i %i\n", grid_range.x, grid_range.y, grid_range.z);
    printf("Block range: %i %i %i\n", block_range.x, block_range.y, block_range.z);
    printf("Items per thread: %i\n", num);

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

      length_encode<<<grid_range, block_range>>>((unsigned char*)devData, (unsigned int*)devResultAddr, num, dataLen);
      
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

