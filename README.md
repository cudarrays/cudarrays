# CUDArrays
C++ programming framework for multi-GPU CUDA applications.

## Example: vector addition
```Cuda
#include <iostream>
#include <cudarrays/types.hpp>
#include <cudarrays/launch.hpp>

using namespace cudarrays;

__global__ void
vecadd_kernel( vector_ref<float> C,
              vector_cref<float> A,
              vector_cref<float> B)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    C(idx) = A(idx) + B(idx);
}

int main()
{
    static const array_size_t ELEMS = 1024;
    // Declare vectors
    vector<float> A{{ELEMS}};
    vector<float> B{{ELEMS}};
    vector<float> C{{ELEMS}};
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
