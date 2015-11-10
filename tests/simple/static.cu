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

#include <cstdio>

#include <cudarrays/static_array.hpp>

using namespace cudarrays;

template <typename T>
__host__ __device__
void test_single(T &A)
{
    for (unsigned i = 0; i < A.template dim<0>(); ++i) {
        for (unsigned j = 0; j < A.template dim<1>(); ++j) {
            A(i, j) = 1;
        }
    }

    A(1, 1) = 3;

    for (unsigned i = 0; i < A.template dim<0>(); ++i) {
        for (unsigned j = 0; j < A.template dim<1>(); ++j) {
            printf("%d ", A(i, j));
        }
        printf("\n");
    }
}

__global__
void test_kernel_single()
{
    static_array<int [3][3]> A;

    test_single(A);
}

static constexpr size_t BLOCK_X = 3;
static constexpr size_t BLOCK_Y = 3;

__global__
void test_kernel_shared()
{
    static_array<int [BLOCK_Y][BLOCK_X], memory_space::shared, layout::rmo, align<4>> A;

    bool first = threadIdx.y == 0 && threadIdx.x == 0;
    bool central = threadIdx.y == 1 && threadIdx.x == 1;

    int value = central? 3: 1;

    A(threadIdx.y, threadIdx.x) = value;

    __syncthreads();

    if (first)
        for (unsigned i = 0; i < A.dim<0>(); ++i) {
            for (unsigned j = 0; j < A.dim<1>(); ++j) {
                printf("%d ", A(i, j));
            }
            printf("\n");
        }
}

int main()
{
    static_array<int [3][3], memory_space::local, layout::rmo, align<1024, 2>> A;

    printf("Host\n");
    test_single(A);

    printf("Device\n");
    test_kernel_single<<<1, 1>>>();
    cudaDeviceSynchronize();

    printf("Device __shared__\n");
    test_kernel_shared<<<1, dim3(BLOCK_X, BLOCK_Y)>>>();
    cudaDeviceSynchronize();

    return 0;
}
