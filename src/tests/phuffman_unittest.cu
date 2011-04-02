#include "gtest/gtest.h"
#include "phuffman_math.cu"
#include "phuffman_limits.cuh"
#include "phuffman_common.cu"
#include <climits>

#ifdef __CUDACC__
using namespace phuffman::CUDA;

TEST(MathTest, BytesToBitsConverter) {
    EXPECT_EQ(0, bytes_to_bits(0));
    EXPECT_EQ(CHAR_BIT, bytes_to_bits(1));
    EXPECT_EQ(512, bytes_to_bits(64));
}

TEST(MathTest, LeftShift2) {
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

TEST(CommonTest, StepIterator) {
    step_iterator<size_t> it(1, 0);
    EXPECT_EQ(0, it + 1);
    // EXPECT_EQ(1, it[1]);
    // EXPECT_EQ(100, it[100]);

    // step_iterator<size_t> it2(5, 10);
    // EXPECT_EQ(50, it2[0]);
    // EXPECT_EQ(55, it2[1]);
    // EXPECT_EQ(550, it2[100]);
}
#endif // __CUDACC__

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
