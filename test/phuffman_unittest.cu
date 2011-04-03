#include <climits>
#include <iostream>
#include <fstream>
#include "gtest/gtest.h"
#include "phuffman_math.cu"
#include "phuffman_limits.cuh"
#include "cpu_encode.hpp"
#include "phuffman.hpp"
#include "CodesTableAdapter.hpp"
#include <unistd.h>

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

TEST(EncoderTest, NaiveDataEncoding) {
    using namespace std;
    using namespace phuffman;
    unsigned char test[] = {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c'};
    CodesTableAdapter codes(test, sizeof(test));
    unsigned int* result = NULL;
    size_t result_length = 0;
    unsigned char trailing_zeroes = 0;
    unsigned char* block_bit_offsets = NULL;
    unsigned int* block_sym_sizes = NULL;
    size_t block_count = 0;
    CPU::Encode(test, sizeof(test), *codes.c_table(), &result, &result_length, &trailing_zeroes, 1, &block_bit_offsets, &block_sym_sizes, &block_count);
    ASSERT_EQ(2, result_length);
    EXPECT_EQ(29, trailing_zeroes);
    EXPECT_EQ(4286578858, result[0]);
    EXPECT_EQ(2684354560, result[1]);
    ASSERT_EQ(2, block_count);
    EXPECT_EQ(1, block_bit_offsets[0]);
    EXPECT_EQ(0, block_bit_offsets[1]);
    EXPECT_EQ(21, block_sym_sizes[0]);
    EXPECT_EQ(1, block_sym_sizes[1]);
    free(block_bit_offsets);
    free(block_sym_sizes);
    free(result);

    block_bit_offsets = NULL;
    block_sym_sizes = NULL;
    block_count = 0;

    CUDA::Encode(test, sizeof(test), *codes.c_table(), &result, &result_length, &trailing_zeroes, 1, &block_bit_offsets, &block_sym_sizes, &block_count);
    ASSERT_EQ(2, result_length);
    EXPECT_EQ(29, trailing_zeroes);
    EXPECT_EQ(4286578858, result[0]);
    EXPECT_EQ(2684354560, result[1]);
    ASSERT_EQ(2, block_count);
    EXPECT_EQ(1, block_bit_offsets[0]);
    EXPECT_EQ(0, block_bit_offsets[1]);
    EXPECT_EQ(21, block_sym_sizes[0]);
    EXPECT_EQ(1, block_sym_sizes[1]);
    free(block_bit_offsets);
    free(block_sym_sizes);
    free(result);
}

TEST(EncoderTest, NaiveDataEncoding1) {
    using namespace std;
    using namespace phuffman;
    unsigned char test[] = {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c'};
    CodesTableAdapter codes(test, sizeof(test));
    unsigned int* result = NULL;
    size_t result_length = 0;
    unsigned char trailing_zeroes = 0;
    unsigned char* block_bit_offsets = NULL;
    unsigned int* block_sym_sizes = NULL;
    size_t block_count = 0;
    CPU::Encode(test, sizeof(test), *codes.c_table(), &result, &result_length, &trailing_zeroes, 1, &block_bit_offsets, &block_sym_sizes, &block_count);
    ASSERT_EQ(2, result_length);
    EXPECT_EQ(30, trailing_zeroes);
    EXPECT_EQ(4278190421, result[0]);
    EXPECT_EQ(1073741824, result[1]);
    ASSERT_EQ(2, block_count);
    EXPECT_EQ(0, block_bit_offsets[0]);
    EXPECT_EQ(0, block_bit_offsets[1]);
    EXPECT_EQ(20, block_sym_sizes[0]);
    EXPECT_EQ(1, block_sym_sizes[1]);
    free(block_bit_offsets);
    free(block_sym_sizes);
    free(result);

    block_bit_offsets = NULL;
    block_sym_sizes = NULL;
    block_count = 0;

    CUDA::Encode(test, sizeof(test), *codes.c_table(), &result, &result_length, &trailing_zeroes, 1, &block_bit_offsets, &block_sym_sizes, &block_count);
    ASSERT_EQ(2, result_length);
    EXPECT_EQ(30, trailing_zeroes);
    EXPECT_EQ(4278190421, result[0]);
    EXPECT_EQ(1073741824, result[1]);
    ASSERT_EQ(2, block_count);
    EXPECT_EQ(0, block_bit_offsets[0]);
    EXPECT_EQ(0, block_bit_offsets[1]);
    EXPECT_EQ(20, block_sym_sizes[0]);
    EXPECT_EQ(1, block_sym_sizes[1]);
    free(block_bit_offsets);
    free(block_sym_sizes);
    free(result);
}


void ReadFile(unsigned char** buffer, size_t* buffer_length, const char* path) {
    using namespace std;

    ifstream file(path);
    if (file.is_open()) {
        file.seekg (0, ios::end);
        *buffer_length = file.tellg();
        file.seekg(0, ios::beg);
        *buffer = new unsigned char[*buffer_length];
        file.read((char*)*buffer, *buffer_length);
        file.close();
    }
}

void CompareEncoders(const char* path, size_t a_block_int_size = 8) {
    using namespace phuffman;
    unsigned char* file_data = NULL;
    size_t file_data_length = 0;
    ReadFile(&file_data, &file_data_length, path);

    CodesTableAdapter codes(file_data, file_data_length);
    unsigned int* cpu_result = NULL;
    size_t cpu_result_length = 0;
    unsigned char cpu_trailing_zeroes = 0;
    unsigned char* cpu_block_bit_offsets = NULL;
    unsigned int* cpu_block_sym_sizes = NULL;
    size_t cpu_block_count = 0;
    CPU::Encode(file_data, file_data_length, *codes.c_table(), &cpu_result, &cpu_result_length, &cpu_trailing_zeroes, a_block_int_size, &cpu_block_bit_offsets, &cpu_block_sym_sizes, &cpu_block_count);

    unsigned int* gpu_result = NULL;
    size_t gpu_result_length = 0;
    unsigned char gpu_trailing_zeroes = 0;
    unsigned char* gpu_block_bit_offsets = NULL;
    unsigned int* gpu_block_sym_sizes = NULL;
    size_t gpu_block_count = 0;
    CUDA::Encode(file_data, file_data_length, *codes.c_table(), &gpu_result, &gpu_result_length, &gpu_trailing_zeroes, a_block_int_size, &gpu_block_bit_offsets, &gpu_block_sym_sizes, &gpu_block_count);

    ASSERT_EQ(cpu_result_length, gpu_result_length);
    ASSERT_EQ(cpu_trailing_zeroes, gpu_trailing_zeroes);

    for (size_t i = 0; i < cpu_result_length; ++i) {
        ASSERT_EQ(cpu_result[i], gpu_result[i]);
    }

    ASSERT_EQ(cpu_block_count, gpu_block_count);

    for (size_t i = 0; i < cpu_block_count; ++i) {
        ASSERT_EQ(cpu_block_bit_offsets[i], gpu_block_bit_offsets[i]);
        ASSERT_EQ(cpu_block_sym_sizes[i], gpu_block_sym_sizes[i]);
    }

    delete[] file_data;
    free(cpu_result);
    free(gpu_result);
    free(cpu_block_bit_offsets);
    free(gpu_block_bit_offsets);
    free(cpu_block_sym_sizes);
    free(gpu_block_sym_sizes);
    
    std::cout << "Compression ratio: " << (static_cast<float>(file_data_length) / (cpu_result_length * sizeof(unsigned int))) << std::endl;
}

TEST(EncodeTest, CalgrayBib) {
    for (size_t i=8; i<=64; i+= 8) {
        CompareEncoders("./test/calgray/bib", i-1);
        CompareEncoders("./test/calgray/bib", i);
        CompareEncoders("./test/calgray/bib", i+1);
    }
}

TEST(EncodeTest, CalgrayBook1) {
    for (size_t i=8; i<=64; i+= 8) {
        CompareEncoders("./test/calgray/book1", i-1);
        CompareEncoders("./test/calgray/book1", i);
        CompareEncoders("./test/calgray/book1", i+1);
    }
}

TEST(EncodeTest, CalgrayBook2) {
    for (size_t i=8; i<=64; i+= 8) {
        CompareEncoders("./test/calgray/book2", i-1);
        CompareEncoders("./test/calgray/book2", i);
        CompareEncoders("./test/calgray/book2", i+1);
    }
}

TEST(EncodeTest, CalgrayProgp) {
    CompareEncoders("./test/calgray/progp");
}

TEST(EncodeTest, CalgrayTrans) {
    CompareEncoders("./test/calgray/trans");
}

TEST(EncodeTest, CalgrayGeo) {
    CompareEncoders("./test/calgray/geo");
}

TEST(EncodeTest, CalgrayNews) {
    CompareEncoders("./test/calgray/news");
}

TEST(EncodeTest, CalgrayObj1) {
    CompareEncoders("./test/calgray/obj1");
}

TEST(EncodeTest, CalgrayObj2) {
    CompareEncoders("./test/calgray/obj2");
}

TEST(EncodeTest, CalgrayPaper1) {
    CompareEncoders("./test/calgray/paper1");
}

TEST(EncodeTest, CalgrayPaper2) {
    CompareEncoders("./test/calgray/paper2");
}

TEST(EncodeTest, CalgrayPaper3) {
    CompareEncoders("./test/calgray/paper3");
}

TEST(EncodeTest, CalgrayPaper4) {
    CompareEncoders("./test/calgray/paper4");
}

TEST(EncodeTest, CalgrayPaper5) {
    CompareEncoders("./test/calgray/paper5");
}

TEST(EncodeTest, CalgrayPaper6) {
    CompareEncoders("./test/calgray/paper6");
}

TEST(EncodeTest, CalgrayPic) {
    CompareEncoders("./test/calgray/pic");
}

TEST(EncodeTest, CalgrayProgc) {
    CompareEncoders("./test/calgray/progc");
}

TEST(EncodeTest, CalgrayProgl) {
    CompareEncoders("./test/calgray/progl");
}

#endif // __CUDACC__

int main(int argc, char **argv) {    
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
}
