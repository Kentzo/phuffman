#ifndef __CUDACC__
#include "phuffman_math.cuh"

namespace phuffman {
    namespace CUDA {
        __host__ __device__ int4 operator<<(int4 value, unsigned int positions) {
            int x_l = value.x << (positions % bytes_to_bits(sizeof(int)));
            
            int y_l = value.y << (positions % sizeof(int));
            int y_h = value.y >> (sizeof(int) - (positions % bytes_to_bits(sizeof(int))));
            
            int z_l = value.z << (positions % bytes_to_bits(sizeof(int)));
            int z_h = value.z >> (bytes_to_bits(sizeof(int)) - (positions % bytes_to_bits(sizeof(int))));
            
            int w_l = value.w << (positions % bytes_to_bits(sizeof(int)));
            int w_h = value.w >> (bytes_to_bits(sizeof(int)) - (positions % bytes_to_bits(sizeof(int))));
            
            int tmp[8] = {x_l | y_h, y_l | z_h, z_l | w_h, w_l};
            unsigned int offset = positions / bytes_to_bits(sizeof(int));
            return make_int4(tmp[0 + offset], tmp[1 + offset], tmp[2 + offset], tmp[3 + offset]);
        }
        
        __host__ __device__ int2 operator<<(int2 value, unsigned int positions) {
            int x_l = value.x << (positions % bytes_to_bits(sizeof(int)));
            
            int y_l = value.y << (positions % bytes_to_bits(sizeof(int)));
            int y_h = value.y >> (bytes_to_bits(sizeof(int)) - (positions % bytes_to_bits(sizeof(int))));
            
            int tmp[4] = {x_l | y_h, y_l};
            unsigned int offset = positions / bytes_to_bits(sizeof(int));
            return make_int2(tmp[0 + offset], tmp[1 + offset]);
        }
        
        __host__ __device__ int4 operator>>(int4 value, unsigned int positions) {
            int x_h = value.x >> (positions % bytes_to_bits(sizeof(int)));
            int x_l = value.x << (bytes_to_bits(sizeof(int)) - (positions % bytes_to_bits(sizeof(int))));
        
            int y_h = value.y >> (positions % bytes_to_bits(sizeof(int)));
            int y_l = value.y << (bytes_to_bits(sizeof(int)) - (positions % bytes_to_bits(sizeof(int))));
        
            int z_h = value.z >> (positions % bytes_to_bits(sizeof(int)));
            int z_l = value.z << (bytes_to_bits(sizeof(int)) - (positions % bytes_to_bits(sizeof(int))));
            
            int w_h = value.w >> (positions % bytes_to_bits(sizeof(int)));
            
            int tmp[8] = {[4] = x_h, x_l | y_h, y_l | z_h, z_l | w_h};
            unsigned int offset = positions / bytes_to_bits(sizeof(int));
            return make_int4(tmp[4 - offset], tmp[5 - offset], tmp[6 - offset], tmp[7 - offset]);
        }
        
        __host__ __device__ int2 operator>>(int2 value, unsigned int positions) {
            int x_h = value.x >> (positions % bytes_to_bits(sizeof(int)));
            int x_l = value.x << (bytes_to_bits(sizeof(int)) - (positions % bytes_to_bits(sizeof(int))));
            
            int y_h = value.y >> (positions % bytes_to_bits(sizeof(int)));
            
            int tmp[4] = {[2] = x_h, x_l | y_h};
            unsigned int offset = positions / bytes_to_bits(sizeof(int));
            return make_int2(tmp[2 - offset], tmp[3 - offset]);
        }

        __host__ __device__ void operator<<=(int4 &value, unsigned int positions) {
            value = value << positions;
        }
        
        __host__ __device__ void operator<<=(int2 &value, unsigned int positions) {
            value = value << positions;
        }        
        
        __host__ __device__ void operator>>=(int4 &value, unsigned int positions) {
            value = value >> positions;
        }
        
        __host__ __device__ void operator>>=(int2 &value, unsigned int positions) {
            value = value >> positions;
        }
        
        __host__ __device__ int4 merge_int2(int2 a, unsigned int a_effective_size, int2 b) {
            int2 b_h = b >> a_effective_size;
            int2 b_l = b << bytes_to_bits(sizeof(int2)) - a_effective_size;
            return make_int4(a | b_h, b_l);
        }
    }
}

#endif // __CUDACC__
