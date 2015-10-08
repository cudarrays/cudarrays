/*
 * CUDArrays is a library for easy multi-GPU program development.
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2013-2015 Barcelona Supercomputing Center and
 *                         University of Illinois
 *
 *  Developed by: Javier Cabezas <javier.cabezas@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE. */

#include <iostream>
#include <cudarrays/types.hpp>
#include <cudarrays/launch.hpp>
#include <cudarrays/gpu.cuh>

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

    cuda_conf conf{ELEMS / 256, 256};
    // Launch vecadd kernel. The kernel is executed on all GPUs.
    // The computation grid is decomposed on its X dimension.
    bool status = launch(vecadd_kernel, conf, compute_conf<1>{compute::x})(C, A, B);
    if (!status) {
        fprintf(stderr, "Error launching kernel 'vecadd_kernel'\n");
        abort();
    }

    for (unsigned i = 0; i < ELEMS; ++i) {
        std::cout << C(i) << " ";
    }
    std::cout << "\n";

    return 0;
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
