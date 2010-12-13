#include "cl.hpp"

#ifdef __VERBOSE
#include <iostream>

namespace phuffman {
    namespace opencl {
        std::ostream& operator<<(std::ostream& out, const cl::NDRange& range);
    }
    namespace cuda {
    }
}
#else // __VERBOSE
#define LOG(...) {}
#endif // __VERBOSE
