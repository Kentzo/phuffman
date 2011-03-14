#pragma once

#ifndef __CUDACC__
#include <climits>

namespace phuffman {
    namespace CUDA {
        /*!
          @abstract Returns the number of bits in a given number of bytes.
          @param bytes A number of bytes.
          @result The number of bits in a given number of bytes.
        */
        inline __host__ __device__ unsigned int bytes_to_bits(unsigned int bytes) {
            return bytes * CHAR_BIT;
        }

        /*!
          @abstract Creates int4 from two int2 objects.
        */
        inline __host__ __device__ int4 make_int4(int2 a, int2 b) {
            return make_int4(a.x, a.y, b.x, b.y);
        }

        /*!
          @abstract Shifts positions bits of a given value to the left and returns the result.
          @param value A value to be shifted to the left.
          @param positions A number of bits to shift. MUST NOT be greater than bytes_to_bits(sizeof(int4)).
          @result The shifted value.
        */
        __host__ __device__ int4 operator<<(int4 value, unsigned int positions);

        /*!
          @abstract Shifts positions bits of a given value to the left and returns the result.
          @param value A value to be shifted to the left.
          @param positions A number of bits to shift. MUST NOT be greater than bytes_to_bits(sizeof(int2)).
          @result The shifted value.
        */
        __host__ __device__ int2 operator<<(int2 value, unsigned int positions);

        /*!
          @abstract Shifts positions bits of a given value to the right and returns the result.
          @param value A value to be shifted to the right.
          @param positions A number of bits to shift. MUST NOT be greater than bytes_to_bits(sizeof(int4)).
          @result The shifted value.
        */        
        __host__ __device__ int4 operator>>(int4 value, unsigned int positions);

        /*!
          @abstract Shifts positions bits of a given value to the right and returns the result.
          @param value A value to be shifted to the right.
          @param positions A number of bits to shift. MUST NOT be greater than bytes_to_bits(sizeof(int2)).
          @result The shifted value.
        */                
        __host__ __device__ int2 operator>>(int2 value, unsigned int positions);

        /*!
          @abstract Performs left-shift in place.
        */
        __host__ __device__ void operator<<=(int4 &value, unsigned int positions);
        
        /*!
          @abstract Performs left-shift in place.
        */
        __host__ __device__ void operator<<=(int2 &value, unsigned int positions);

        /*!
          @abstract Performs right-shift in place.
        */        
        __host__ __device__ void operator>>=(int4 &value, unsigned int positions);
        
        /*!
          @abstract Performs right-shift in place.
        */
        __host__ __device__ void operator>>=(int2 &value, unsigned int positions);
        
        /*!
          @abstracts ORs given values.
        */
        inline __host__ __device__ int4 operator|(int4 a, int4 b) {
            return make_int4(a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w);
        }

        /*!
          @abstracts ORs given values.
        */
        inline __host__ __device__ int2 operator|(int2 a, int2 b) {
            return make_int2(a.x | b.x, a.y | b.y);
        }

        /*!
          @abstract Merges two int2 values into one int4 overriding unnecessary bits.
          @param a This value will be left in the result.
          @param a_effective_size Number of leftmost bits that make sense. MUST NOT be greater than bytes_to_bits(sizeof(int2)).
          @param b This valie will be right in the result.
        */
        __host__ __device__ int4 merge_int2(int2 a, unsigned int a_effective_size, int2 b);
    }
}
#endif // __CUDACC__
