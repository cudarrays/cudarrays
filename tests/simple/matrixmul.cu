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

#include "matrixmul_kernel.cuh"

using namespace cudarrays;

unsigned INPUTSET = 0;
bool TEST = true;

array_size_t matrixmul_ELEMS[1] = { 512 };

template <typename StorageC, typename StorageA = StorageC, typename StorageB = StorageC>
bool
launch_test_matrixmul(compute_conf<2> gpus, std::array<int, 2> infoC,
                                            std::array<int, 2> infoA,
                                            std::array<int, 2> infoB)
{
    if (gpus.procs > cudarrays::MAX_GPUS) return false;

    static const array_size_t ELEMS = matrixmul_ELEMS[INPUTSET];

    using array2D_matrixmul = float [ELEMS][ELEMS];
    using my_arrayA = matrix<float, layout::cmo, StorageA>;
    using my_arrayB = matrix<float, layout::rmo, StorageB>;
    using my_arrayC = matrix<float, layout::cmo, StorageC>;

    my_arrayA A{{ELEMS, ELEMS}};
    my_arrayB B{{ELEMS, ELEMS}};
    my_arrayC C{{ELEMS, ELEMS}};

    A.template distribute<2>({gpus, infoA});
    B.template distribute<2>({gpus, infoB});
    C.template distribute<2>({gpus, infoC});

    array2D_matrixmul &A_host = *(array2D_matrixmul *) new float[ELEMS * ELEMS];
    array2D_matrixmul &B_host = *(array2D_matrixmul *) new float[ELEMS * ELEMS];
    array2D_matrixmul &C_host = *(array2D_matrixmul *) new float[ELEMS * ELEMS];

    {
        for (unsigned i = 0; i < ELEMS; ++i) {
            for (unsigned j = 0; j < ELEMS; ++j) {
                A(i, j)      = float(i);
                A_host[i][j] = float(i);

                B(i, j)      = float(i + 1.f);
                B_host[i][j] = float(i + 1.f);

                C(i, j)      = 0.f;
                C_host[i][j] = 0.f;
            }
        }
    }

    {
        cuda_conf conf{dim3(ELEMS / (MATRIXMUL_TILE_N * MATRIXMUL_TILE_TB_HEIGHT),
                            ELEMS / MATRIXMUL_TILE_N),
                       dim3(MATRIXMUL_TILE_N,
                            MATRIXMUL_TILE_TB_HEIGHT), 0, 0};

        auto times_exec = launch(matrixmul_kernel<StorageC, StorageA, StorageB>, conf, gpus)(C, A, B);
    }

    if (TEST)
    {
        #pragma omp parallel for
        for (unsigned i = 0; i < ELEMS; ++i) {
            for (unsigned j = 0; j < ELEMS; ++j) {
                float tmp = 0.f;
                for (int k = 0; k < ELEMS; ++k) {
                    tmp += A_host[i][k] * B_host[k][j];
                }
                C_host[i][j] = tmp;
            }
        }
    }

    if (TEST)
    {
        for (unsigned i = 0; i < ELEMS; ++i) {
            for (unsigned j = 0; j < ELEMS; ++j) {
                if (C_host[i][j] != C(i, j)) {
                    std::cout << "C: Position {" << i << ", " << j << "} "
                                                << C_host[i][j]
                                                << " vs "
                                                << C(i, j) << std::endl;
                    abort();
                    return false;
                }
            }
        }
    }

    delete [] &A_host;
    delete [] &B_host;
    delete [] &C_host;

    return true;
}

int main(int argc, char *argv[])
{
    init_lib();

    static const int NONE = -1;

    bool ok = false;

    ok = launch_test_matrixmul<replicate::none>({compute::none, 1},
                                                {NONE, NONE},
                                                {NONE, NONE},
                                                {NONE, NONE});
    printf("COHERENT 1: %d\n", ok);

    ok = launch_test_matrixmul<vm::none>({compute::none, 1},
                                         {NONE, NONE},
                                         {NONE, NONE},
                                         {NONE, NONE});
    printf("VM 1: %d\n", ok);

    ok = launch_test_matrixmul<vm::none>({compute::y, 2},
                                         {0, NONE},
                                         {0, NONE},
                                         {0, NONE});
    printf("VM 2: %d\n", ok);

    ok = launch_test_matrixmul<vm::none>({compute::y, 2},
                                         {0, NONE},
                                         {0, NONE},
                                         {0, NONE});
    printf("VM 4: %d\n", ok);

    // TODO: implement proper replication
#if 0
    ok = launch_test_matrixmul<reshape::none>({1, compute::none},
                                               {NONE, NONE},
                                               {NONE, NONE},
                                               {NONE, NONE});
    printf("RESHAPE_BLOCK   1: %d\n", ok);
#endif

    ok = launch_test_matrixmul<reshape::x, reshape::x, replicate::none>
                                           ({compute::x, 2},
                                            {NONE, 0},
                                            {NONE, 0},
                                            {NONE, NONE});
    printf("RESHAPE_BLOCK_X 2: %d\n", ok);

    ok = launch_test_matrixmul<reshape::y, replicate::none, reshape::x>
                                           ({compute::y, 2},
                                            {1, NONE},
                                            {NONE, NONE},
                                            {NONE, 1});
    printf("RESHAPE_BLOCK_Y 2: %d\n", ok);

    ok = launch_test_matrixmul<reshape::x, reshape::x, replicate::none>
                                           ({compute::x, 4},
                                            {NONE, 0},
                                            {NONE, 0},
                                            {NONE, NONE});
    printf("RESHAPE_BLOCK_X 4, 4: %d\n", ok);

    ok = launch_test_matrixmul<reshape::y, replicate::none, reshape::x>
                                           ({compute::y, 4},
                                            {1, NONE},
                                            {NONE, NONE},
                                            {NONE, 1});
    printf("RESHAPE_BLOCK_Y 4, 4: %d\n", ok);

    ok = launch_test_matrixmul<reshape::xy, replicate::x, replicate::x>
                                            ({compute::xy, 4},
                                             {1, 0},
                                             {NONE, 0},
                                             {NONE, 1});
    printf("RESHAPE_BLOCK_XY 4, 4: %d\n", ok);
#if 0
    ok = launch_test_matrixmul<storage_tag<COMPUTE_ELEMS> >(1, 1);
    printf("COMPUTE_ELEMS 1: %d\n", ok);
    ok = launch_test_matrixmul<storage_tag<COMPUTE_ELEMS> >(2, 2);
    printf("COMPUTE_ELEMS 2: %d\n", ok);
#endif

    return 0;
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
