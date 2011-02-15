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
//    cuPrintf("left_shift_int4([%i, %i, %i, %i], %i)\n", a_value.x, a_value.y, a_value.z, a_value.w, a_num);
//    cuPrintf("\t%i\n", 1 << 28);
    int w_l = a_value.w << (a_num % 32);
    int w_h = a_value.w >> (32 - (a_num % 32));


    int z_l = a_value.z << (a_num % 32);
    int z_h = a_value.z >> (32 - (a_num % 32));

    int y_l = a_value.y << (a_num % 32);
    int y_h = a_value.y >> (32 - (a_num % 32));

    int x_l = a_value.x << (a_num % 32);

//    cuPrintf("\tx_l: %i\n", x_l);
//    cuPrintf("\ty_l: %i\n", y_l);
//    cuPrintf("\ty_h: %i\n", y_h);
//    cuPrintf("\tz_l: %i\n", z_l);
//    cuPrintf("\tz_h: %i\n", z_h);
//    cuPrintf("\tw_l: %i\n", w_l);
//    cuPrintf("\tw_h: %i\n", w_h);

    if (a_num < 32) {
        a_value = make_int4(x_l | y_h, y_l | z_h, z_l | w_h, w_l);
    }
    else if (a_num >= 32 && a_num < 64) {
        a_value = make_int4(y_l | z_h, z_l | w_h, w_l, 0);
    }
    else if (a_num >= 64 && a_num < 96) {
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

    if (a_num < 32) {
        a_value = make_int4(x_h, y_h | x_l, z_h | y_l, w_h | z_l);
    }
    else if (a_num >= 32 && a_num < 64) {
        a_value = make_int4(0, x_h, y_h | x_l, z_h | y_l);
    }
    else if (a_num >= 64 && a_num < 96) {
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
    int4 a_value2_h = right_shift_int4(a_value2, sizeof(int4) * 8 - a_value1_zeroes);
    int4 a_value2_l = left_shift_int4(a_value2, a_value1_zeroes);
    *result[0] = or_int4(a_value1, a_value2_h);
    *result[1] = a_value2_l;
}


// TODO: проверка на выход за границы массива результата
__device__ void merge_atomic(int4 a_value1[2], unsigned int *result) {
    cuPrintf("merge_atomic\n");
    cuPrintf("%p\n", result);
    cuPrintf("\ta_avlue[0]: %u %u %u %u\n", a_value1[0].x, a_value1[0].y, a_value1[0].z, a_value1[0].w);
    cuPrintf("\ta_value[1]: %u %u %u %u\n", a_value1[1].x, a_value1[1].y, a_value1[1].z, a_value1[1].w);

    cuPrintf("\tbefore: result[0]: %u\n", *result);
    atomicOr(result, a_value1[0].x);
    cuPrintf("\tafter: result[0]: %u\n", *result);

    cuPrintf("\tbefore:result[1]: %u\n", *(result + 1));
    atomicOr(result + 1, a_value1[0].y);
    cuPrintf("\tafter: result[1]: %u\n", *(result + 1));

    cuPrintf("\tbefore:result[2]: %u\n", *(result + 2));
    atomicOr(result + 2, a_value1[0].z);
    cuPrintf("\tafter: result[2]: %u\n", *(result + 2));

    cuPrintf("\tbefore:result[3]: %u\n", *(result + 3));
    atomicOr(result + 3, a_value1[0].w);
    cuPrintf("\tafter: result[3]: %u\n", *(result + 3));

    cuPrintf("\tbefore:result[4]: %u\n", *(result + 4));
    atomicOr(result + 4, a_value1[1].x);
    cuPrintf("\tafter: result[4]: %u\n", *(result + 4));

    cuPrintf("\tbefore:result[5]: %u\n", *(result + 5));
    atomicOr(result + 5, a_value1[1].y);
    cuPrintf("\tafter: result[5]: %u\n", *(result + 5));

    cuPrintf("\tbefore:result[6]: %u\n", *(result + 6));
    atomicOr(result + 6, a_value1[1].z);
    cuPrintf("\tafter: result[6]: %u\n", *(result + 6));

    cuPrintf("\tbefore:result[7]: %u\n", *(result + 7));
    atomicOr(result + 7, a_value1[1].w);
    cuPrintf("\tafter: result[7]: %u\n", *(result + 7));
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

__global__ void encode(unsigned char* a_data, unsigned int a_data_len, unsigned int* a_prefix, unsigned int num_to_process,
                        unsigned int* a_result)
{
    unsigned int bx = threadIdx.x;
    const int NUM_OF_SYM = 4;

    //cuPrintf("%s\n", "encode");
    for (int block_idx = 0; block_idx < num_to_process; block_idx += NUM_OF_SYM) {
        unsigned int b_idx = min(bx * num_to_process + block_idx, a_data_len);
        unsigned int e_idx = min(b_idx + NUM_OF_SYM, a_data_len);

        if (b_idx == a_data_len)
            return;
        
        cuPrintf("block: %i\n", block_idx);

        // Encode
        int4 buf[2] = {make_int4(0, 0, 0, 0), make_int4(0, 0, 0, 0)}; // Буфер должен вмещать NUM_OF_SYM кодов по MAXIMUM_CODELENGTH битов
        int buf_idx = 0;
        int towrite = 0;
        // Пропускаем result_start_bit % sizeof(int) битов, т.к. они будет использованы другим потоком
        unsigned int result_start_bit = a_prefix[b_idx];
        int written = result_start_bit % (sizeof(int) * 8);
        cuPrintf("written: %i\n", written);
        for (int j = b_idx; j < e_idx; ++j) {
            cuPrintf("\tj: %i\n", j);
            unsigned char c = a_data[j];
            towrite = sizeof(int4) * 8 - codes[c].codelength - written;
            // Влазим в текущий буфер
            if (towrite >= 0) {
                //00001100000000000000000000000000
                //00010000000000000000000000000000
                // Сдвигаем код так, чтобы в результате OR он добавился к остальным без пропусков
                //int4 tmp = make_int4(0, 0, 0, codes[c].code);
                //int4 tmp_shift = left_shift_int4(tmp, towrite);
                //cuPrintf("tmp: [%i, %i, %i, %i], towrite: %i, tmp_shift: [%i, %i, %i, %i]\n", tmp.x, tmp.y, tmp.z, tmp.w, towrite, tmp_shift.x, tmp_shift.y, tmp_shift.z, tmp_shift.w);
                buf[buf_idx] = or_int4(buf[buf_idx],
                        left_shift_int4(make_int4(0, 0, 0, codes[c].code), towrite));
                written += codes[c].codelength;
                towrite = 0;
            }
            // Переносим towrite кодов в следущий буфер
            else {
                cuPrintf("FIND_ME\n");
                towrite = -towrite;
                // Разделяем код на 2 части: одну для текущего буфера, другую для следующего
                // Нужно сохранить старшие нули
                int4 code = left_shift_int4(make_int4(0, 0, 0, codes[c].code), sizeof(int4) * 8 - codes[c].codelength);
                merge_int4(buf[0], sizeof(int4) * 8 - written, code, (sizeof(int4) * 8) - codes[c].codelength, &buf);
                ++buf_idx;
            }
        }
        cuPrintf("buf[0]: %i %i %i %i\n", buf[0].x, buf[0].y, buf[0].z, buf[0].w);
        //cuPrintf("buf[1]: %i %i %i %i\n", buf[1].x, buf[1].y, buf[1].z, buf[1].w);
        //cuPrintf("[%i %i %i %i] [%i %i %i %i]\n", buf[0].x, buf[0].y, buf[0].z, buf[0].w,
        //        buf[1].x, buf[1].y, buf[1].z, buf[1].w);
        unsigned int result_start_int = (result_start_bit - (result_start_bit % (sizeof(int) * 8))) / (sizeof(int) * 8);
        merge_atomic(buf, a_result + result_start_int);
    }
}

extern "C" {

  void runEncode(unsigned char* a_data, size_t len, CodesTable a_table, unsigned int* result, size_t* result_len)
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

      int res_size = dataLen*maxCodeLen + sizeof(int) - (dataLen*(int)maxCodeLen % sizeof(int));
      cutilSafeCall(cudaMalloc(&devData, dataLen));
      cutilSafeCall(cudaMalloc(&devResultAddr, dataLen*sizeof(unsigned int)));
      cutilSafeCall(cudaMalloc(&devResult, res_size));
      cutilSafeCall(cudaMemset(devResult, 0, res_size));

      printf("start: %p, end: %p\n", (unsigned char *)devResult, (unsigned char *)devResult + res_size);

//      unsigned int* test = (unsigned int*)calloc(res_size, 1);
//      cutilSafeCall(cudaMemcpy(test, devResult, res_size, cudaMemcpyDeviceToHost));
//      for (int i = 0; i < res_size/sizeof(int); ++i) {
//          printf("%i\n", test[i]);
//      }
//      free(test);

      cutilSafeCall(cudaMemcpy(devData, a_data, dataLen, cudaMemcpyHostToDevice));

      length_encode<<<grid_range, block_range>>>((unsigned char*)devData, dataLen, (unsigned int*)devResultAddr, num);
      
      thrust::device_ptr<unsigned int> devPtr((unsigned int*)devResultAddr);
      thrust::exclusive_scan(devPtr, devPtr + dataLen, devPtr);

      encode<<<grid_range, block_range>>>((unsigned char*)devData, dataLen, (unsigned int*)devResultAddr, num, (unsigned int*)devResult);

      cudaPrintfDisplay(stdout, true); 
      cudaPrintfEnd();
      
      *result_len = res_size / sizeof(int) ;
      cutilSafeCall(cudaMemcpy(result, devResult, res_size, cudaMemcpyDeviceToHost));
      //cuPrintf("[%i %i %i] %i\n", );

      printf("Prefix sum:\n");
      unsigned int* devResultAddrHost = (unsigned int*)calloc(dataLen, sizeof(unsigned int));
      cutilSafeCall(cudaMemcpy(devResultAddrHost, devResultAddr, dataLen*sizeof(unsigned int), cudaMemcpyDeviceToHost));
      for (int i = 0; i < dataLen; ++i) {
          printf("%i) %i\n ", i, devResultAddrHost[i]);
      }
      free(devResultAddrHost);

      cudaFree(devData);
      cudaFree(devResultAddr);
      cudaFree(devResult);

      len -= dataLen;
    }	 
  }
}

