#ifndef __CUDACC__
#include "phuffman_math.cuh"

namespace phuffman {
    namespace CUDA {
        __host__ __device__ uint4 operator<<(uint4 value, unsigned int positions) {
            uint x_l = value.x << (positions % bytes_to_bits(sizeof(uint)));
            
            uint y_l = value.y << (positions % sizeof(uint));
            uint y_h = value.y >> (sizeof(uint) - (positions % bytes_to_bits(sizeof(uint))));
            
            uint z_l = value.z << (positions % bytes_to_bits(sizeof(uint)));
            uint z_h = value.z >> (bytes_to_bits(sizeof(uint)) - (positions % bytes_to_bits(sizeof(uint))));
            
            uint w_l = value.w << (positions % bytes_to_bits(sizeof(uint)));
            uint w_h = value.w >> (bytes_to_bits(sizeof(uint)) - (positions % bytes_to_bits(sizeof(uint))));
            
            uint tmp[8] = {x_l | y_h, y_l | z_h, z_l | w_h, w_l};
            unsigned int offset = positions / bytes_to_bits(sizeof(uint));
            return make_uint4(tmp[0 + offset], tmp[1 + offset], tmp[2 + offset], tmp[3 + offset]);
        }
        
        __host__ __device__ uint2 operator<<(uint2 value, unsigned int positions) {
            uint x_l = value.x << (positions % bytes_to_bits(sizeof(uint)));
            
            uint y_l = value.y << (positions % bytes_to_bits(sizeof(uint)));
            uint y_h = value.y >> (bytes_to_bits(sizeof(uint)) - (positions % bytes_to_bits(sizeof(uint))));
            
            uint tmp[4] = {x_l | y_h, y_l};
            unsigned int offset = positions / bytes_to_bits(sizeof(uint));
            return make_uint2(tmp[0 + offset], tmp[1 + offset]);
        }
        
        __host__ __device__ uint4 operator>>(uint4 value, unsigned int positions) {
            uint x_h = value.x >> (positions % bytes_to_bits(sizeof(uint)));
            uint x_l = value.x << (bytes_to_bits(sizeof(uint)) - (positions % bytes_to_bits(sizeof(uint))));
        
            uint y_h = value.y >> (positions % bytes_to_bits(sizeof(uint)));
            uint y_l = value.y << (bytes_to_bits(sizeof(uint)) - (positions % bytes_to_bits(sizeof(uint))));
        
            uint z_h = value.z >> (positions % bytes_to_bits(sizeof(uint)));
            uint z_l = value.z << (bytes_to_bits(sizeof(uint)) - (positions % bytes_to_bits(sizeof(uint))));
            
            uint w_h = value.w >> (positions % bytes_to_bits(sizeof(uint)));
            
            int tmp[8] = {[4] = x_h, x_l | y_h, y_l | z_h, z_l | w_h};
            unsigned int offset = positions / bytes_to_bits(sizeof(uint));
            return make_uint4(tmp[4 - offset], tmp[5 - offset], tmp[6 - offset], tmp[7 - offset]);
        }
        
        __host__ __device__ uint2 operator>>(uint2 value, unsigned int positions) {
            uint x_h = value.x >> (positions % bytes_to_bits(sizeof(uint)));
            uint x_l = value.x << (bytes_to_bits(sizeof(uint)) - (positions % bytes_to_bits(sizeof(uint))));
            
            uint y_h = value.y >> (positions % bytes_to_bits(sizeof(uint)));
            
            uint tmp[4] = {[2] = x_h, x_l | y_h};
            unsigned int offset = positions / bytes_to_bits(sizeof(uint));
            return make_uint2(tmp[2 - offset], tmp[3 - offset]);
        }

        __host__ __device__ void operator<<=(uint4 &value, unsigned int positions) {
            value = value << positions;
        }
        
        __host__ __device__ void operator<<=(uint2 &value, unsigned int positions) {
            value = value << positions;
        }        
        
        __host__ __device__ void operator>>=(uint4 &value, unsigned int positions) {
            value = value >> positions;
        }
        
        __host__ __device__ void operator>>=(uint2 &value, unsigned int positions) {
            value = value >> positions;
        }
        
        __host__ __device__ uint4 merge_uint2(uint2 a, unsigned int a_effective_size, uint2 b) {
            uint2 b_h = b >> a_effective_size;
            uint2 b_l = b << bytes_to_bits(sizeof(uint2)) - a_effective_size;
            return make_uint4(a | b_h, b_l);
        }
    }
}

#endif // __CUDACC__
