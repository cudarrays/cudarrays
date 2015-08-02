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

#include <cudarrays/dynarray.hpp>
#include <cudarrays/launch.hpp>

namespace cudarrays {
void init_lib();
}

#include "stencil_kernel.cuh"

using namespace cudarrays;

bool TEST = true;

array_size_t STENCIL_ELEMS[2] = { 1024, 1024};

using mapping2D = std::array<int, 2>;

template <typename StorageB, typename StorageA = StorageB>
bool
launch_test_stencil(compute_conf<2> gpus, mapping2D infoB,
                                          mapping2D infoA)
{
    if (gpus.procs > config::MAX_GPUS) return false;

    static const array_size_t ELEMS_Y = STENCIL_ELEMS[0];
    static const array_size_t ELEMS_X = STENCIL_ELEMS[1];

    static const array_size_t TOTAL_ELEMS_Y = ELEMS_Y + 2 * STENCIL;
    static const array_size_t TOTAL_ELEMS_X = ELEMS_X + 2 * STENCIL;

    using array2D_stencil = float [TOTAL_ELEMS_Y][TOTAL_ELEMS_X];
    using my_arrayA = matrix<float, layout::rmo, StorageA>;
    using my_arrayB = matrix<float, layout::rmo, StorageB>;

    my_arrayA A{{TOTAL_ELEMS_Y, TOTAL_ELEMS_X}};
    my_arrayB B{{TOTAL_ELEMS_Y, TOTAL_ELEMS_X}};

    A.template distribute<2>({gpus, infoA});
    B.template distribute<2>({gpus, infoB});

    array2D_stencil &A_host = *(array2D_stencil *) new float[TOTAL_ELEMS_Y * TOTAL_ELEMS_X];
    array2D_stencil &B_host = *(array2D_stencil *) new float[TOTAL_ELEMS_Y * TOTAL_ELEMS_X];

    {
        std::fill(A.begin(), A.end(), 0.f);
        std::fill(B.begin(), B.end(), 0.f);

        std::fill(&A_host[0][0], &A_host[0][0] + TOTAL_ELEMS_Y * TOTAL_ELEMS_X, 0.f);
        std::fill(&B_host[0][0], &B_host[0][0] + TOTAL_ELEMS_Y * TOTAL_ELEMS_X, 0.f);

        for (unsigned i = STENCIL; i < ELEMS_Y + STENCIL; ++i) {
            for (unsigned j = STENCIL; j < ELEMS_X + STENCIL; ++j) {
                A(i, j)      = float(i * ELEMS_X + j);
                A_host[i][j] = float(i * ELEMS_X + j);
            }
        }
    }

    {
        cuda_conf conf{dim3(ELEMS_X / STENCIL_BLOCK_X,
                            ELEMS_Y / STENCIL_BLOCK_Y),
                       dim3(STENCIL_BLOCK_X,
                            STENCIL_BLOCK_Y)};

        bool status = launch(stencil_kernel<StorageB, StorageA>, conf, gpus)(B, A);
        if (!status) {
            fprintf(stderr, "Error launching kernel 'stencil_kernel'\n");
            abort();
        }
    }

    if (TEST)
    {
        #pragma omp parallel for
        for (unsigned i = STENCIL; i < ELEMS_Y + STENCIL; ++i) {
            for (unsigned j = STENCIL; j < ELEMS_X + STENCIL; ++j) {
                float value = A_host[i][j];
                for (int k = 1; k <= STENCIL; ++k) {
                    value += 3.f * (A_host[i][j - k] + A_host[i][j + k]) +
                             2.f * (A_host[i - k][j] + A_host[i + k][j]);
                }
                B_host[i][j] = value;
            }
        }
    }

    bool ok = true;
    if (TEST)
    {
        for (unsigned i = STENCIL; i < ELEMS_Y + STENCIL; ++i) {
            for (unsigned j = STENCIL; j < ELEMS_X + STENCIL; ++j) {
                if (B_host[i][j] != B(i, j)) {
                    std::cout << "C: Position {" << i << ", " << j << "} "
                                                 << B_host[i][j]
                                                 << " vs "
                                                 << B(i, j) << std::endl;
                    ok = false;
                    abort();
                }
            }
        }
    }

    delete [] &A_host;
    delete [] &B_host;

    return ok;
}

static const int NIL = -1;

template <typename Impl>
void test_conf(unsigned gpus)
{
    bool ok = false;

    ok = launch_test_stencil<typename Impl::x>({compute::x, gpus},
                                               {NIL, 0},
                                               {NIL, 0});
    printf("%s X %u: %d\n", Impl::name, gpus, ok);
    ok = launch_test_stencil<typename Impl::y>({compute::y, gpus},
                                               {1, NIL},
                                               {1, NIL});
    printf("%s Y %u: %d\n", Impl::name, gpus, ok);

    if (gpus == 1 || gpus >= 4) {
        ok = launch_test_stencil<typename Impl::xy>({compute::xy, gpus},
                                                    {1, 0},
                                                    {1, 0});
        printf("%s XY %u: %d\n", Impl::name, gpus, ok);
        ok = launch_test_stencil<typename Impl::xy>({compute::xy, gpus},
                                                    {0, 1},
                                                    {0, 1});
        printf("%s YX %u: %d\n", Impl::name, gpus, ok);
    }
}

int main()
{
    init_lib();
    bool ok;
    ok = launch_test_stencil<replicate::none>({compute::none, 1},
                                              {NIL, NIL},
                                              {NIL, NIL});
    printf("%s %u: %d\n", replicate::name, 1, ok);

    for (auto gpus : {1, 2, 4}) {
        test_conf<vm>(gpus);
        test_conf<reshape>(gpus);
#if 0
        test_conf<reshape_cyclic<2>>(gpus);
        test_conf<reshape_block_cyclic<2>>(gpus);
#endif
    }

    return 0;
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
