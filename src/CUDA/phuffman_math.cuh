#pragma once

#include <cuda_runtime.h>
#include <vector_functions.h>
#include <climits>

namespace phuffman {
    namespace CUDA {
        /*!
          @abstract Returns the number of bits in a given number of bytes.
          @param bytes A number of bytes.
          @result The number of bits in a given number of bytes.
        */
        __host__ __device__ unsigned int bytes_to_bits(unsigned int bytes);

        /*!
          @abstract Creates uint4 from two uint2 objects.
        */

        inline __host__ __device__ uint4 make_uint4(uint2 a, uint2 b) {
            return ::make_uint4(a.x, a.y, b.x, b.y);
        }


        /*!
          @abstract Shifts positions bits of a given value to the left and returns the result.
          @param value A value to be shifted to the left.
          @param positions A number of bits to shift. MUST NOT be greater than bytes_to_bits(sizeof(uint4)).
          @result The shifted value.
        */
        __host__ __device__ uint4 operator<<(uint4 value, unsigned int positions);

        /*!
          @abstract Shifts positions bits of a given value to the left and returns the result.
          @param value A value to be shifted to the left.
          @param positions A number of bits to shift. MUST NOT be greater than bytes_to_bits(sizeof(uint2)).
          @result The shifted value.
        */
        __host__ __device__ uint2 operator<<(uint2 value, unsigned int positions);

        /*!
          @abstract Shifts positions bits of a given value to the right and returns the result.
          @param value A value to be shifted to the right.
          @param positions A number of bits to shift. MUST NOT be greater than bytes_to_bits(sizeof(uint4)).
          @result The shifted value.
        */
        __host__ __device__ uint4 operator>>(uint4 value, unsigned int positions);

        /*!
          @abstract Shifts positions bits of a given value to the right and returns the result.
          @param value A value to be shifted to the right.
          @param positions A number of bits to shift. MUST NOT be greater than bytes_to_bits(sizeof(uint2)).
          @result The shifted value.
        */
        __host__ __device__ uint2 operator>>(uint2 value, unsigned int positions);

        /*!
          @abstract Performs left-shift in place.
        */
        __host__ __device__ void operator<<=(uint4 &value, unsigned int positions);

        /*!
          @abstract Performs left-shift in place.
        */
        __host__ __device__ void operator<<=(uint2 &value, unsigned int positions);

        /*!
          @abstract Performs right-shift in place.
        */
        __host__ __device__ void operator>>=(uint4 &value, unsigned int positions);

        /*!
          @abstract Performs right-shift in place.
        */
        __host__ __device__ void operator>>=(uint2 &value, unsigned int positions);

        /*!
          @abstracts ORs given values.
        */
        __host__ __device__ uint4 operator|(uint4 a, uint4 b);

        /*!
          @abstracts ORs given values.
        */
        __host__ __device__ uint2 operator|(uint2 a, uint2 b);

        /*!
          @abstract Merges two uint2 values into one uint4 overriding unnecessary bits.
          @param a This value will be left in the result.
          @param a_effective_size Number of leftmost bits that make sense. MUST NOT be greater than bytes_to_bits(sizeof(uint2)).
          @param b This valie will be right in the result.
        */
        __host__ __device__ uint4 merge_uint2(uint2 a, unsigned int a_effective_size, uint2 b);
    }
}
