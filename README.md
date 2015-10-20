![Travis-CI](https://travis-ci.org/cudarrays/cudarrays.svg?branch=master)
<a href="https://scan.coverity.com/projects/4910">
<img alt="Coverity Scan Build Status"
src="https://scan.coverity.com/projects/4910/badge.svg"/>
</a>

![CUDArrays](https://raw.githubusercontent.com/wiki/cudarrays/cudarrays/images/cudarrays_logo_300x75.png)

Programming framework for C++ and multi-GPU CUDA applications.

CUDArrays is based on multi-dimensional array types that are commonly found in
scientific HPC applications. Thanks to CUDArrays, C++ programs can now use both
statically- and dynamically-sized multi-dimensional arrays, a feature which has
available if Fortran for decades. Thus, programmers are relieved from the task
of flattening multi-dimensional structures into memory buffers and
computing dimension offsets by hand.

When used in CUDA programs, arrays are transparently available in the GPU
memory and the do not need to be explicitly copied between CPU and host
memories. Moreover, arrays can be partitioned or replicated across GPU memories
and CUDA kernels' grids are also decomposed so that the computation can be
performed on all the GPUs in parallel.

CUDArrays also offers an wide range of features such as user-defined array
dimension layout, user-defined memory alignment and iterators that enable
compatibility with the algorithms in the STL.

## Example: multi-GPU vector addition using dynamically-sized arrays and CUDA
```Cuda
#include <iostream>
#include <cudarrays/types.hpp>
#include <cudarrays/launch.hpp>

using namespace cudarrays;

__global__ void
vecadd_kernel( vector_view<float> C,
              vector_cview<float> A,
              vector_cview<float> B)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    C(idx) = A(idx) + B(idx);
}

int main()
{
    init_lib();

    static const array_size_t ELEMS = 1024;
    // Declare vectors
    auto A = make_vector<float>({ELEMS});
    auto B = make_vector<float>({ELEMS});
    auto C = make_vector<float>({ELEMS});
    // Initialize input vectors
    for (unsigned i = 0; i < ELEMS; ++i) {
        A(i)      = float(i);
        B(i)      = float(i + 1.f);
    }

    cuda_conf conf{ELEMS / 512, 512};
    // Launch vecadd kernel. The kernel is executed on all GPUs.
    // The computation grid is decomposed on its X dimension.
    bool status = launch(vecadd_kernel, conf, compute_conf<1>{compute::x})(C, A, B);

    for (unsigned i = 0; i < ELEMS; ++i) {
        std::cout << C(i) << " ";
    }
    std::cout << "\n";

    return 0;
}

```
