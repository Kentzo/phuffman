#include "cl.hpp"
#include <iostream>

namespace phuffman {
    namespace opencl {
        std::ostream& operator<<(std::ostream& out, const cl::NDRange& range);
    }
    namespace cuda {
    }
}
