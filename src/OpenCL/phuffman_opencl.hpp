#define __CL_ENABLE_EXCEPTIONS
#define __NO_STD_VECTOR
#include "cl.hpp"

namespace phuffman {
    namespace opencl {
        void Encode(const unsigned char* data, size_t data_size, unsigned char** result, size_t* result_size);
        void Encode(const unsigned char* data, size_t data_size, unsigned char** result, size_t* result_size, const CodeTableAdapter& codes_table);
    }
}

#undef __CL_ENABLE_EXCEPTIONS
#undef __NO_STD_VECTOR
