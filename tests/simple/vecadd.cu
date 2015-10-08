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

#include <array>
#include <iostream>

#include <cudarrays/common.hpp>

#include <cudarrays/dynarray.hpp>
#include <cudarrays/launch.hpp>

#include "vecadd_kernel.cuh"

using namespace cudarrays;

unsigned INPUTSET = 0;
bool TEST = true;

array_size_t VECADD_ELEMS[1] = { 1024L * 1024L };

using mapping1D = std::array<int, 1>;

template <typename Storage>
bool
launch_test_vecadd(compute_conf<1> gpus, mapping1D infoC,
                                         mapping1D infoA,
                                         mapping1D infoB)
{
    if (gpus.procs > system::gpu_count()) return false;

    static const array_size_t ELEMS = VECADD_ELEMS[INPUTSET];

    using array1D_vecadd = float [ELEMS];

    auto A = make_vector<float, Storage>({ELEMS});
    auto B = make_vector<float, Storage>({ELEMS});
    auto C = make_vector<float, Storage>({ELEMS});

    A.template distribute<1>({gpus, infoA});
    B.template distribute<1>({gpus, infoB});
    C.template distribute<1>({gpus, infoC});

    array1D_vecadd &A_host = *(array1D_vecadd *) new float[ELEMS];
    array1D_vecadd &B_host = *(array1D_vecadd *) new float[ELEMS];
    array1D_vecadd &C_host = *(array1D_vecadd *) new float[ELEMS];

    {
        for (unsigned i = 0; i < ELEMS; ++i) {
            A(i)      = float(i);
            A_host[i] = float(i);

            B(i)      = float(i + 1.f);
            B_host[i] = float(i + 1.f);

            C(i)      = 0.f;
            C_host[i] = 0.f;
        }
    }

    {
        cuda_conf conf{ELEMS / 512, 512};

        bool status = launch(vecadd_kernel<Storage, Storage>, conf, gpus)(C, A, B);
        if (!status) {
            fprintf(stderr, "Error launching kernel 'vecadd_kernel'\n");
            abort();
        }
    }

    if (TEST)
    {
        for (unsigned i = 0; i < ELEMS; ++i) {
            C_host[i] = A_host[i] + B_host[i];
        }
    }

    if (TEST)
    {
        for (unsigned i = 0; i < ELEMS; ++i) {
            if (C_host[i] != C(i)) {
                std::cout << "C: Position " << i << " "
                                            << C_host[i]
                                            << " vs "
                                            << C(i) << std::endl;
                abort();
                return false;
            }
        }
    }

    delete [] &A_host;
    delete [] &B_host;
    delete [] &C_host;

    return true;
}

void dead_code()
{
    launch_test_vecadd<reshape_cyclic::x>({compute::x, 1}, {{0}}, {{0}}, {{0}});
}


int main()
{
    bool ok = false;

    ok = launch_test_vecadd<replicate::x>({ compute::x, 1 },
                                          {{0}}, {{0}}, {{0}});
    printf("REPLICATED 1: %d\n", ok);
    ok = launch_test_vecadd<replicate::x>({ compute::x, 2 },
                                          {{0}}, {{0}}, {{0}});
    printf("REPLICATED 2: %d\n", ok);
    ok = launch_test_vecadd<replicate::x>({ compute::x, 4 },
                                          {{0}}, {{0}}, {{0}});
    printf("REPLICATED 4: %d\n", ok);
    ok = launch_test_vecadd<vm::x>({ compute::x, 1 },
                                   {{0}}, {{0}}, {{0}});
    printf("DISTRIBUTED_VM 1: %d\n", ok);
    ok = launch_test_vecadd<vm::x>({compute::x, 2},
                                   {{0}}, {{0}}, {{0}});
    printf("DISTRIBUTED_VM 2: %d\n", ok);
    ok = launch_test_vecadd<vm::x>({compute::x, 4},
                                   {{0}}, {{0}}, {{0}});
    printf("DISTRIBUTED_VM 4: %d\n", ok);
    ok = launch_test_vecadd<reshape::x>({compute::x, 1},
                                        {{0}}, {{0}}, {{0}});
    printf("RESHAPE_BLOCK   1: %d\n", ok);
    ok = launch_test_vecadd<reshape::x>({compute::x, 2},
                                        {{0}}, {{0}}, {{0}});
    printf("RESHAPE_BLOCK_X 2: %d\n", ok);
    ok = launch_test_vecadd<reshape::x>({compute::x, 4},
                                        {{0}}, {{0}}, {{0}});
    printf("RESHAPE_BLOCK_X 4: %d\n", ok);

    return 0;
}

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
