#include <climits>
#include <iostream>
#include <fstream>
#include "gtest/gtest.h"
#include "phuffman_math.cu"
#include "phuffman_limits.cuh"
#include "cpu_encode.hpp"
#include "cpu_decode.hpp"
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

// --------------------------------------------------------------------------------

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

void TestCodesTable(const char* path) {
    using namespace phuffman;
    unsigned char* file_data = NULL;
    size_t file_data_length = 0;
    ReadFile(&file_data, &file_data_length, path);

    CodesTableAdapter codes(file_data, file_data_length);
    for (size_t i=0; i<ALPHABET_SIZE; ++i) {
        ASSERT_LT(codes[i].code, 1 << (CHAR_BIT + 1));
    }
    delete[] file_data;
}

TEST(CodesTableTest, CalgaryBib) {
    TestCodesTable("./test/calgary/bib");
}

TEST(CodesTableTest, CalgaryBook1) {
    TestCodesTable("./test/calgary/book1");
}

TEST(CodesTableTest, CalgaryBook2) {
    TestCodesTable("./test/calgary/book2");
}

TEST(CodesTableTest, CalgaryProgp) {
    TestCodesTable("./test/calgary/progp");
}

TEST(CodesTableTest, CalgaryTrans) {
    TestCodesTable("./test/calgary/trans");
}

TEST(CodesTableTest, CalgaryGeo) {
    TestCodesTable("./test/calgary/geo");
}

TEST(CodesTableTest, CalgaryNews) {
    TestCodesTable("./test/calgary/news");
}

TEST(CodesTableTest, CalgaryObj1) {
    TestCodesTable("./test/calgary/obj1");
}

TEST(CodesTableTest, CalgaryObj2) {
    TestCodesTable("./test/calgary/obj2");
}

TEST(CodesTableTest, CalgaryPaper1) {
    TestCodesTable("./test/calgary/paper1");
}

TEST(CodesTableTest, CalgaryPaper2) {
    TestCodesTable("./test/calgary/paper2");
}

TEST(CodesTableTest, CalgaryPaper3) {
    TestCodesTable("./test/calgary/paper3");
}

TEST(CodesTableTest, CalgaryPaper4) {
    TestCodesTable("./test/calgary/paper4");
}

TEST(CodesTableTest, CalgaryPaper5) {
    TestCodesTable("./test/calgary/paper5");
}

TEST(CodesTableTest, CalgaryPaper6) {
    TestCodesTable("./test/calgary/paper6");
}

TEST(CodesTableTest, CalgaryPic) {
    TestCodesTable("./test/calgary/pic");
}

TEST(CodesTableTest, CalgaryProgc) {
    TestCodesTable("./test/calgary/progc");
}

TEST(CodesTableTest, CalgaryProgl) {
    TestCodesTable("./test/calgary/progl");
}

// --------------------------------------------------------------------------------

TEST(EncoderTest, NonAlignedNaiveEncoding) {
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

TEST(EncoderTest, AlignedNaiveEncoding) {
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
}

TEST(EncodeTest, CalgaryBib) {
    CompareEncoders("./test/calgary/bib");
}

TEST(EncodeTest, CalgaryBook1) {
    CompareEncoders("./test/calgary/book1");
}

TEST(EncodeTest, CalgaryBook2) {
    CompareEncoders("./test/calgary/book2");
}

TEST(EncodeTest, CalgaryProgp) {
    CompareEncoders("./test/calgary/progp");
}

TEST(EncodeTest, CalgaryTrans) {
    CompareEncoders("./test/calgary/trans");
}

TEST(EncodeTest, CalgaryGeo) {
    CompareEncoders("./test/calgary/geo");
}

TEST(EncodeTest, CalgaryNews) {
    CompareEncoders("./test/calgary/news");
}

TEST(EncodeTest, CalgaryObj1) {
    CompareEncoders("./test/calgary/obj1");
}

TEST(EncodeTest, CalgaryObj2) {
    CompareEncoders("./test/calgary/obj2");
}

TEST(EncodeTest, CalgaryPaper1) {
    CompareEncoders("./test/calgary/paper1");
}

TEST(EncodeTest, CalgaryPaper2) {
    CompareEncoders("./test/calgary/paper2");
}

TEST(EncodeTest, CalgaryPaper3) {
    CompareEncoders("./test/calgary/paper3");
}

TEST(EncodeTest, CalgaryPaper4) {
    CompareEncoders("./test/calgary/paper4");
}

TEST(EncodeTest, CalgaryPaper5) {
    CompareEncoders("./test/calgary/paper5");
}

TEST(EncodeTest, CalgaryPaper6) {
    CompareEncoders("./test/calgary/paper6");
}

TEST(EncodeTest, CalgaryPic) {
    CompareEncoders("./test/calgary/pic");
}

TEST(EncodeTest, CalgaryProgc) {
    CompareEncoders("./test/calgary/progc");
}

TEST(EncodeTest, CalgaryProgl) {
    CompareEncoders("./test/calgary/progl");
}

// --------------------------------------------------------------------------------

void TestCPUDecoder(const char* path, size_t a_block_int_size = 8) {
    using namespace phuffman;
    unsigned char* file_data = NULL;
    size_t file_data_length = 0;
    ReadFile(&file_data, &file_data_length, path);

    CodesTableAdapter codes(file_data, file_data_length);
    unsigned int* cpu_encoded = NULL;
    size_t cpu_encoded_length = 0;
    unsigned char cpu_trailing_zeroes = 0;
    unsigned char* cpu_block_bit_offsets = NULL;
    unsigned int* cpu_block_sym_sizes = NULL;
    size_t cpu_block_count = 0;
    CPU::Encode(file_data, file_data_length, *codes.c_table(), &cpu_encoded, &cpu_encoded_length, &cpu_trailing_zeroes, a_block_int_size, &cpu_block_bit_offsets, &cpu_block_sym_sizes, &cpu_block_count);

    unsigned char* decoded_data = NULL;
    size_t decoded_data_length = 0;
    CPU::Decode(&decoded_data, &decoded_data_length, *codes.c_table(), cpu_encoded, cpu_encoded_length, cpu_trailing_zeroes, a_block_int_size, cpu_block_bit_offsets, cpu_block_sym_sizes, cpu_block_count);

    ASSERT_EQ(file_data_length, decoded_data_length);

    for (size_t i=0; i<decoded_data_length; ++i) {
        ASSERT_EQ(file_data[i], decoded_data[i]);
    }

    delete[] file_data;
    free(decoded_data);
    free(cpu_encoded);
    free(cpu_block_bit_offsets);
    free(cpu_block_sym_sizes);
}

TEST(CPUDecoderTest, CalgaryBib) {
    TestCPUDecoder("./test/calgary/bib");
}

TEST(CPUDecoderTest, CalgaryBook1) {
    TestCPUDecoder("./test/calgary/book1");
}

TEST(CPUDecoderTest, CalgaryBook2) {
    TestCPUDecoder("./test/calgary/book2");
}

TEST(CPUDecoderTest, CalgaryProgp) {
    TestCPUDecoder("./test/calgary/progp");
}

TEST(CPUDecoderTest, CalgaryTrans) {
    TestCPUDecoder("./test/calgary/trans");
}

TEST(CPUDecoderTest, CalgaryGeo) {
    TestCPUDecoder("./test/calgary/geo");
}

TEST(CPUDecoderTest, CalgaryNews) {
    TestCPUDecoder("./test/calgary/news");
}

TEST(CPUDecoderTest, CalgaryObj1) {
    TestCPUDecoder("./test/calgary/obj1");
}

TEST(CPUDecoderTest, CalgaryObj2) {
    TestCPUDecoder("./test/calgary/obj2");
}

TEST(CPUDecoderTest, CalgaryPaper1) {
    TestCPUDecoder("./test/calgary/paper1");
}

TEST(CPUDecoderTest, CalgaryPaper2) {
    TestCPUDecoder("./test/calgary/paper2");
}

TEST(CPUDecoderTest, CalgaryPaper3) {
    TestCPUDecoder("./test/calgary/paper3");
}

TEST(CPUDecoderTest, CalgaryPaper4) {
    TestCPUDecoder("./test/calgary/paper4");
}

TEST(CPUDecoderTest, CalgaryPaper5) {
    TestCPUDecoder("./test/calgary/paper5");
}

TEST(CPUDecoderTest, CalgaryPaper6) {
    TestCPUDecoder("./test/calgary/paper6");
}

TEST(CPUDecoderTest, CalgaryPic) {
    TestCPUDecoder("./test/calgary/pic");
}

TEST(CPUDecoderTest, CalgaryProgc) {
    TestCPUDecoder("./test/calgary/progc");
}

TEST(CPUDecoderTest, CalgaryProgl) {
    TestCPUDecoder("./test/calgary/progl");
}
#endif // __CUDACC__

int main(int argc, char **argv) {
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
}
