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

#include <cudarrays/common.hpp>

#include <cudarrays/types.hpp>
#include <cudarrays/launch.hpp>

#include "inc_kernel.cuh"

using namespace cudarrays;

unsigned INPUTSET = 0;
bool TEST = true;

array_size_t INC_ELEMS[1] = { 1024 * 1024 };

using mapping1D = std::array<int, 1>;

template <typename Storage>
bool
launch_test_inc(compute_conf<1> gpus, mapping1D infoB,
                                      mapping1D infoA)
{
    if (gpus.procs > system::gpu_count()) return false;

    static const auto ELEMS = INC_ELEMS[INPUTSET];
    static const auto ITER = 10;

    using array1D_inc = float [ELEMS];

    auto A = make_vector<float, Storage>({ELEMS});
    auto B = make_vector<float, Storage>({ELEMS});

    auto A_ptr = A;

    A.template distribute<1>({gpus, infoA});
    B.template distribute<1>({gpus, infoB});

    float c = 1.f;

    array1D_inc &A_host = *(array1D_inc *) new float[ELEMS];
    array1D_inc &B_host = *(array1D_inc *) new float[ELEMS];

    {
        for (auto i = 0u; i < ELEMS; ++i) {
            A(i)      = float(i);
            A_host[i] = float(i);
        }
    }

    {
        cuda_conf conf(ELEMS / 512, 512);

        for (auto i = 0u; i < ITER; ++i) {
            bool status = launch(inc_kernel<Storage, Storage>, conf, gpus)(B, A, c);
            if (!status) {
                fprintf(stderr, "Error launching kernel 'vecadd_kernel'\n");
                abort();
            }
        }
    }

    if (TEST)
    {

        for (auto i = 0u; i < ELEMS; ++i) {
            B_host[i] = A_host[i] + c * 2 * ITER;
        }
    }

    if (TEST)
    {
        for (auto i = 0u; i < ELEMS; ++i) {
            if (B_host[i] != B(i)) {
                std::cout << "B: Position " << i << " "
                                            << B_host[i]
                                            << " vs "
                                            << B(i) << std::endl;
                abort();
            }
        }
    }

    delete [] &A_host;
    delete [] &B_host;

    return true;
}

int main()
{
    bool ok = false;

    ok = launch_test_inc<replicate::x>({ compute::x, 1 },
                                         {{0}}, {{0}});
    printf("DISTRIBUTED_REPLICATED 1: %d\n", ok);
    ok = launch_test_inc<replicate::x>({compute::x, 2},
                                         {{0}}, {{0}});
    printf("DISTRIBUTED_REPLICATED 2: %d\n", ok);
    ok = launch_test_inc<replicate::x>({compute::x, 4},
                                         {{0}}, {{0}});
    printf("DISTRIBUTED_REPLICATED 4: %d\n", ok);
    ok = launch_test_inc<vm::x>({ compute::x, 1 },
                                     {{0}}, {{0}});
    printf("DISTRIBUTED_VM 1: %d\n", ok);
    ok = launch_test_inc<vm::x>({compute::x, 2},
                                     {{0}}, {{0}});
    printf("DISTRIBUTED_VM 2: %d\n", ok);
    ok = launch_test_inc<vm::x>({compute::x, 4},
                                     {{0}}, {{0}});
    printf("DISTRIBUTED_VM 4: %d\n", ok);
    ok = launch_test_inc<reshape::x>({compute::x, 1},
                                          {{0}}, {{0}});
    printf("RESHAPE_BLOCK   1: %d\n", ok);
    ok = launch_test_inc<reshape::x>({compute::x, 2},
                                          {{0}}, {{0}});
    printf("RESHAPE_BLOCK_X 2: %d\n", ok);
    ok = launch_test_inc<reshape::x>({compute::x, 4},
                                          {{0}}, {{0}});
    printf("RESHAPE_BLOCK_X 4: %d\n", ok);

    return 0;
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
