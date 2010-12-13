#define __VERBOSE

#include "phuffman.hpp"
#ifdef __VERBOSE
#include "verbose.hpp"
#endif // __VERBOSE

#include <iostream>
#include <fstream>
#include <utility>
#include <cassert>
#include <cmath>

using namespace std;

namespace phuffman {
    namespace opencl {
        using namespace cl; // size_t is redefined as a structure...
        using std::size_t;

        /**
           @abstract Calculates work group and work item ranges for a given device and data size
           @discussion If items_per_thread is enough to assign all data given value will be used. Otherwise it will be set to appropriate value.
         */
        void _Sizes(size_t data_size, const Device& device, size_t* items_per_thread, NDRange* work_group_range, NDRange* work_item_range) {
            // Retrieve maximum values for calculated parameters
            size_t max_work_group_size = 0;
            device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_work_group_size);
            assert(max_work_group_size > 0); // Provided device doesn't support OpenCL

            size_t max_work_item_dimensions = 0;
            device.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &max_work_item_dimensions);
            assert(max_work_item_dimensions > 0); // Provided device doesn't support OpenCL
            assert(max_work_item_dimensions <= 3); // OpenCL 1.0 and 1.1 supports maximum 3 dimensions

            size_t max_work_item_sizes[3] = {1};
            device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &max_work_item_sizes);
            // It is enough to check only first value, cause OpenCL guarantee the minimum value is (1, 1, 1)
            assert(max_work_item_sizes[0] > 0); // Provided device doesn't support OpenCL

            // Begin calculating
            size_t threads_num = size_t(ceilf(float(data_size) / *items_per_thread));

            if (threads_num > 0) {
                // Calculate work group size and appropriate value for items_per_thread
                size_t max_work_items_per_group = 1;
                for (size_t i=0; i<max_work_item_dimensions; ++i) {
                    max_work_items_per_group *= max_work_item_sizes[i];
                }         
                size_t work_group_size = size_t(ceilf(float(threads_num) / max_work_items_per_group));
                // Calculate appropriate value for items_per_thread
                while (work_group_size > max_work_group_size) {
                    (*items_per_thread) <<= 1;
                    threads_num = size_t(ceilf(float(data_size) / *items_per_thread));
                    work_group_size = size_t(ceilf(float(threads_num) / max_work_items_per_group));
                }
                // Calculate work item sizes
                threads_num = size_t(ceilf(float(threads_num) / work_group_size));
                size_t work_item_sizes[3] = {1};
                size_t dimensions = max_work_item_dimensions;
                for (size_t i=0; i<max_work_item_dimensions && threads_num>0; ++i) {
                    size_t size = min(max_work_item_sizes[i], threads_num);
                    work_item_sizes[i] = size;
                    threads_num -= size;
                    dimensions = i + 1;
                }

                *work_group_range = NDRange(work_group_size);
                switch (dimensions) {
                case 1:
                    *work_item_range = NDRange(work_item_sizes[0]);
                    break;
                case 2:
                    *work_item_range = NDRange(work_item_sizes[0], work_item_sizes[1]);
                    break;
                case 3:
                    *work_item_range = NDRange(work_item_sizes[0], work_item_sizes[1], work_item_sizes[2]);
                    break;
                }
            }
            else {
                *work_group_range = NDRange(0);
                *work_item_range = NDRange(0);
            }
        }

        bool _BuildKernels(Context* context, Program* program, CommandQueue* queue) {
        	static const char* ENCODING_SOURCE_PATH = "encoding.cl";

            char* source = NULL;
            size_t source_size = 0;
            {
                ifstream source_file(ENCODING_SOURCE_PATH);
                if (source_file.is_open()) {
                    source_file.seekg(0, ios_base::end);
                    size_t size = source_file.tellg();
                    source_file.seekg(0, ios_base::beg);
                    source = new char[size];
                    source_file.read(source, size);
                    source_size = size;
                }
                else {
                    // Kernel source cannot be read
                    cerr << "Kernel file ('" << ENCODING_SOURCE_PATH << "') doesn't exist" << endl;
                    return false;
                }
            }

            // Setup OpenCL
            *context = Context(CL_DEVICE_TYPE_CPU);
			VECTOR_CLASS<Device> devices = context->getInfo<CL_CONTEXT_DEVICES>();
			Program::Sources sources(1, std::make_pair(source, source_size));
			*program = Program(*context, sources);
			*queue = CommandQueue(*context, devices[0]);
			program->build(devices, "-I /Users/kent90/Documents/Projects/My/CourseWork/phuffman/build/");
        }

        void _EncodeWithCodelengths(const CommandQueue& queue, const Program& program, const Buffer& device_data, const Buffer& device_codes, const Buffer& device_addresses) 
        {
            static const size_t MINIMUM_ITEMS_PER_THREAD = 32;
            static const char* ENCODE_WITH_CODELENGTHS_FUNCTION_NAME = "EncodeWithCodelengths";

            Device device;
            queue.getInfo(CL_QUEUE_DEVICE, &device);
            size_t data_size = 0;
            device_data.getInfo(CL_MEM_SIZE, &data_size);

#ifdef __VERBOSE
            cout << endl << "Start encoding codelengths" << endl;
#endif // __VERBOSE
            NDRange work_group_range, work_item_range;
            size_t items_per_thread = MINIMUM_ITEMS_PER_THREAD;
            _Sizes(data_size, device, &items_per_thread, &work_group_range, &work_item_range);
#ifdef __VERBOSE
            cout << "Work group range: " << work_group_range << endl;
            cout << "Work item range: " << work_item_range << endl;
            cout << "Items per thread: " << items_per_thread << endl;
#endif // __VERBOSE
            Kernel kernel(program, ENCODE_WITH_CODELENGTHS_FUNCTION_NAME);
            KernelFunctor EncodeWithCodelengths = kernel.bind(queue, work_group_range, work_item_range);
            
            EncodeWithCodelengths(device_data, device_addresses, device_codes, items_per_thread, data_size).wait();

#ifdef __VERBOSE
            cout << "End encoding codelengths" << endl << endl;
#endif // __VERBOSE
        }

        void Encode(const unsigned char* data, size_t data_size, unsigned char** result, size_t* result_size, const CodeTableAdapter& codes_table) {
#ifdef __VERBOSE
            cout << "Data size: " << data_size << endl;
#endif // __VERBOSE

            Context context;
            Program program;
            CommandQueue queue;
            _BuildKernels(&context, &program, &queue);

            // Setup memory objects
            Buffer device_data(context, CL_MEM_READ_ONLY, data_size);
            queue.enqueueWriteBuffer(device_data, CL_TRUE, 0, data_size, data);
            Buffer device_codes(context, CL_MEM_READ_ONLY, ALPHABET_SIZE * sizeof(Code));
            queue.enqueueWriteBuffer(device_codes, CL_TRUE, 0, ALPHABET_SIZE * sizeof(Code), codes_table.c_table()->codes);
            Buffer device_addresses(context, CL_MEM_READ_WRITE, data_size * sizeof(unsigned int));
            
            // Start encoding
            _EncodeWithCodelengths(queue, program, device_data, device_codes, device_addresses);

            *result_size = data_size * sizeof(unsigned int);
            *result = new unsigned char[*result_size];
            queue.enqueueReadBuffer(device_addresses, CL_TRUE, 0, *result_size, *result);
        }

        void Encode(const unsigned char* data, size_t data_size, unsigned char** result, size_t* result_size) {
        	CodeTableAdapter codes_table(data, data_size);
        	Encode(data, data_size, result, result_size, codes_table);
        }
    }
}
