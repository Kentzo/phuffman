#include <climits>
#include "gtest/gtest.h"
#include "phuffman_math.cu"
#include "phuffman_limits.cuh"
#include "cpu_encode.hpp"
#include "phuffman.hpp"
#include "CodesTableAdapter.hpp"
#include <iostream>

#ifdef __CUDACC__
TEST(MathTest, BytesToBitsConverter) {
    using namespace phuffman::CUDA;
    EXPECT_EQ(0, bytes_to_bits(0));
    EXPECT_EQ(CHAR_BIT, bytes_to_bits(1));
    EXPECT_EQ(512, bytes_to_bits(64));
}

TEST(MathTest, LeftShift2) {
    using namespace phuffman::CUDA;
    unsigned int x = 0xAAAAAAAA;
    unsigned int y = 0xBBBBBBBB;
    unsigned int xy = 0xAAAABBBB;
    uint2 test = make_uint2(x, y);
    EXPECT_EQ(0, (test << UINT2_BIT).x);
    EXPECT_EQ(0, (test << UINT2_BIT).y);

    EXPECT_EQ(y, (test << UINT_BIT).x);
    EXPECT_EQ(0, (test << UINT_BIT).y);

    EXPECT_EQ(xy, (test << UINT_BIT/2).x);
    EXPECT_EQ(xy << UINT_BIT/2, (test << UINT_BIT/2).y);
}

TEST(MathTest, RightShift2) {
    using namespace phuffman::CUDA;
    unsigned int x = 0xAAAAAAAA;
    unsigned int y = 0xBBBBBBBB;
    unsigned int xy = 0xAAAABBBB;
    uint2 test = make_uint2(x, y);
    EXPECT_EQ(0, (test >> UINT2_BIT).x);
    EXPECT_EQ(0, (test >> UINT2_BIT).y);

    EXPECT_EQ(0, (test >> UINT_BIT).x);
    EXPECT_EQ(x, (test >> UINT_BIT).y);

    EXPECT_EQ(x >> UINT_BIT/2, (test >> UINT_BIT/2).x);
    EXPECT_EQ(xy, (test >> UINT_BIT/2).y);
}

TEST(EncoderTest, DataEncoding) {
    using namespace std;
    using namespace phuffman;
    unsigned char test[] = {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c'};
    CodesTableAdapter codes(test, sizeof(test));
    unsigned int* result = NULL;
    size_t result_length = 0;
    unsigned char trailing_zeroes = 0;
    CPU::Encode(test, sizeof(test), *codes.c_table(), &result, &result_length, &trailing_zeroes);
    EXPECT_EQ(2, result_length);
    EXPECT_EQ(30, trailing_zeroes);
    EXPECT_EQ(4278190421, result[0]);
    EXPECT_EQ(1073741824, result[1]);
    free(result);
}
#endif // __CUDACC__

int main(int argc, char **argv) {
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
}
