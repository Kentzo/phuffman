#include "verbose.hpp"
#include <iostream>

using namespace std;

namespace phuffman {
    namespace opencl {
        ostream& operator<<(ostream& out, const cl::NDRange& range) {
            const size_t* st_range = (const size_t*)range;
            for (size_t i=0; i<range.dimensions(); ++i) {
                out << "[" << st_range[i] << "]";
            }
            return out;
        }
    }
    namespace cuda {
    }
}
