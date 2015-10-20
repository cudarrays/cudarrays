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

#include <chrono>

#include <cudarrays/common.hpp>

#include <cudarrays/types.hpp>
#include <cudarrays/launch.hpp>

using namespace cudarrays;

const unsigned INPUTSET = 2;
bool TEST = true;

constexpr const array_size_t VOLADD_ELEMS_X[3] = { 4, 128, 768 };
constexpr const array_size_t VOLADD_ELEMS_Y[3] = { 5,  32, 512 };
constexpr const array_size_t VOLADD_ELEMS_Z[3] = { 6,  32, 384 };

bool
launch_test_vecadd_base()
{
    static constexpr const array_size_t ELEMS_X = VOLADD_ELEMS_X[INPUTSET];
    static constexpr const array_size_t ELEMS_Y = VOLADD_ELEMS_Y[INPUTSET];
    static constexpr const array_size_t ELEMS_Z = VOLADD_ELEMS_Z[INPUTSET];

    float *A = new float[ELEMS_Z * ELEMS_Y * ELEMS_X];
    float *B = new float[ELEMS_Z * ELEMS_Y * ELEMS_X];
    float *C = new float[ELEMS_Z * ELEMS_Y * ELEMS_X];

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                A[i * ELEMS_Y * ELEMS_X + j * ELEMS_X + k] = 1.f;
            }
        }
    }
    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                B[i * ELEMS_Y * ELEMS_X + j * ELEMS_X + k] = 2.f;
            }
        }
    }
    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                C[i * ELEMS_Y * ELEMS_X + j * ELEMS_X + k] = 0.f;
            }
        }
    }

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                C[i * ELEMS_Y * ELEMS_X + j * ELEMS_X + k] =
                    A[i * ELEMS_Y * ELEMS_X + j * ELEMS_X + k] +
                    B[i * ELEMS_Y * ELEMS_X + j * ELEMS_X + k];
            }
        }
    }

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                assert(C[i * ELEMS_Y * ELEMS_X + j * ELEMS_X + k] == 3.f);
            }
        }
    }

    delete [] A;
    delete [] B;
    delete [] C;

    return true;
}


bool
launch_test_vecadd_base_static()
{
    static constexpr const array_size_t ELEMS_X = VOLADD_ELEMS_X[INPUTSET];
    static constexpr const array_size_t ELEMS_Y = VOLADD_ELEMS_Y[INPUTSET];
    static constexpr const array_size_t ELEMS_Z = VOLADD_ELEMS_Z[INPUTSET];

    using array_type = float[ELEMS_Z][ELEMS_Y][ELEMS_X];

    array_type &A = *(array_type *) new float[ELEMS_Z * ELEMS_Y * ELEMS_X];
    array_type &B = *(array_type *) new float[ELEMS_Z * ELEMS_Y * ELEMS_X];
    array_type &C = *(array_type *) new float[ELEMS_Z * ELEMS_Y * ELEMS_X];

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                A[i][j][k] = 1.f;
            }
        }
    }
    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                B[i][j][k] = 2.f;
            }
        }
    }
    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                C[i][j][k] = 0.f;
            }
        }
    }

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                C[i][j][k] = A[i][j][k] + B[i][j][k];
            }
        }
    }

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                assert(C[i][j][k] == 3.f);
            }
        }
    }

    delete [] &A;
    delete [] &B;
    delete [] &C;

    return true;
}


bool
launch_test_vecadd()
{
    static constexpr const array_size_t ELEMS_X = VOLADD_ELEMS_X[INPUTSET];
    static constexpr const array_size_t ELEMS_Y = VOLADD_ELEMS_Y[INPUTSET];
    static constexpr const array_size_t ELEMS_Z = VOLADD_ELEMS_Z[INPUTSET];

    auto A = make_array<float ***>({{ELEMS_Z, ELEMS_Y, ELEMS_X}});
    auto B = make_array<float ***>({{ELEMS_Z, ELEMS_Y, ELEMS_X}});
    auto C = make_array<float ***>({{ELEMS_Z, ELEMS_Y, ELEMS_X}});

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                A(i, j, k) = 1.f;
            }
        }
    }

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                B(i, j, k) = 2.f;
            }
        }
    }
    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                C(i, j, k) = 0.f;
            }
        }
    }

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                C(i, j, k) = A(i, j, k) + B(i, j, k);
            }
        }
    }

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                assert(C(i, j, k) == 3.f);
            }
        }
    }

    return true;
}

bool
launch_test_vecadd_static()
{
    static constexpr const array_size_t ELEMS_X = VOLADD_ELEMS_X[INPUTSET];
    static constexpr const array_size_t ELEMS_Y = VOLADD_ELEMS_Y[INPUTSET];
    static constexpr const array_size_t ELEMS_Z = VOLADD_ELEMS_Z[INPUTSET];

    auto A = make_array<float [ELEMS_Z][ELEMS_Y][ELEMS_X]>();
    auto B = make_array<float [ELEMS_Z][ELEMS_Y][ELEMS_X]>();
    auto C = make_array<float [ELEMS_Z][ELEMS_Y][ELEMS_X]>();

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                A(i, j, k) = 1.f;
            }
        }
    }
    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                B(i, j, k) = 2.f;
            }
        }
    }
    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                C(i, j, k) = 0.f;
            }
        }
    }

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                C(i, j, k) = A(i, j, k) + B(i, j, k);
            }
        }
    }

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                assert(C(i, j, k) == 3.f);
            }
        }
    }

    return true;
}

bool
launch_test_vecadd_for_each()
{
    static constexpr const array_size_t ELEMS_X = VOLADD_ELEMS_X[INPUTSET];
    static constexpr const array_size_t ELEMS_Y = VOLADD_ELEMS_Y[INPUTSET];
    static constexpr const array_size_t ELEMS_Z = VOLADD_ELEMS_Z[INPUTSET];

    auto A = make_array<float ***>({{ELEMS_Z, ELEMS_Y, ELEMS_X}});
    auto B = make_array<float ***>({{ELEMS_Z, ELEMS_Y, ELEMS_X}});
    auto C = make_array<float ***>({{ELEMS_Z, ELEMS_Y, ELEMS_X}});

    for_each_element(A, [](unsigned, unsigned, unsigned)
                        { return 1.f; });

    for_each_element(B, [](unsigned, unsigned, unsigned)
                        { return 2.f; });

    for_each_element(C, [](unsigned, unsigned, unsigned)
                        { return 0.f; });

    for_each_element(C, [&](unsigned i, unsigned j, unsigned k)
                        { return A(i, j, k) + B(i, j, k); });

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                assert(C(i, j, k) == 3.f);
            }
        }
    }

    return true;
}

bool
launch_test_vecadd_for_each_static()
{
    static constexpr const array_size_t ELEMS_X = VOLADD_ELEMS_X[INPUTSET];
    static constexpr const array_size_t ELEMS_Y = VOLADD_ELEMS_Y[INPUTSET];
    static constexpr const array_size_t ELEMS_Z = VOLADD_ELEMS_Z[INPUTSET];

    auto A = make_array<float [ELEMS_Z][ELEMS_Y][ELEMS_X]>();
    auto B = make_array<float [ELEMS_Z][ELEMS_Y][ELEMS_X]>();
    auto C = make_array<float [ELEMS_Z][ELEMS_Y][ELEMS_X]>();

    for_each_element(A, [](unsigned, unsigned, unsigned)
                        { return 1.f; });

    for_each_element(B, [](unsigned, unsigned, unsigned)
                        { return 2.f; });

    for_each_element(C, [](unsigned, unsigned, unsigned)
                        { return 0.f; });

    for_each_element(C, [&](unsigned i, unsigned j, unsigned k)
                        { return A(i, j, k) + B(i, j, k); });

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                assert(C(i, j, k) == 3.f);
            }
        }
    }

    return true;
}


bool
launch_test_vecadd_iterator()
{
    static constexpr const array_size_t ELEMS_X = VOLADD_ELEMS_X[INPUTSET];
    static constexpr const array_size_t ELEMS_Y = VOLADD_ELEMS_Y[INPUTSET];
    static constexpr const array_size_t ELEMS_Z = VOLADD_ELEMS_Z[INPUTSET];

    auto A = make_array<float ***>({{ELEMS_Z, ELEMS_Y, ELEMS_X}});
    auto B = make_array<float ***>({{ELEMS_Z, ELEMS_Y, ELEMS_X}});
    auto C = make_array<float ***>({{ELEMS_Z, ELEMS_Y, ELEMS_X}});

    auto valuesA = A.value_iterator();
    std::fill(valuesA.begin(), valuesA.end(), 1.f);

    auto valuesB = B.value_iterator();
    std::fill(valuesB.begin(), valuesB.end(), 2.f);

    auto valuesC = C.value_iterator();
    std::fill(valuesC.begin(), valuesC.end(), 0.f);

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                C(i, j, k) = A(i, j, k) + B(i, j, k);
            }
        }
    }

    bool ok = std::all_of(valuesC.begin(), valuesC.end(),
                          [](float v) { return v == 3.f; });
    assert(ok);

    return true;
}

bool
launch_test_vecadd_iterator_static()
{
    static constexpr const array_size_t ELEMS_X = VOLADD_ELEMS_X[INPUTSET];
    static constexpr const array_size_t ELEMS_Y = VOLADD_ELEMS_Y[INPUTSET];
    static constexpr const array_size_t ELEMS_Z = VOLADD_ELEMS_Z[INPUTSET];

    auto A = make_array<float [ELEMS_Z][ELEMS_Y][ELEMS_X]>();
    auto B = make_array<float [ELEMS_Z][ELEMS_Y][ELEMS_X]>();
    auto C = make_array<float [ELEMS_Z][ELEMS_Y][ELEMS_X]>();

    auto valuesA = A.value_iterator();
    std::fill(valuesA.begin(), valuesA.end(), 1.f);

    auto valuesB = B.value_iterator();
    std::fill(valuesB.begin(), valuesB.end(), 2.f);

    auto valuesC = C.value_iterator();
    std::fill(valuesC.begin(), valuesC.end(), 0.f);

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                C(i, j, k) = A(i, j, k) + B(i, j, k);
            }
        }
    }

    bool ok = std::all_of(valuesC.begin(), valuesC.end(),
                          [](float v) { return v == 3.f; });
    assert(ok);


    return true;
}


bool
launch_test_vecadd_dim_iterator()
{
    static constexpr const array_size_t ELEMS_X = VOLADD_ELEMS_X[INPUTSET];
    static constexpr const array_size_t ELEMS_Y = VOLADD_ELEMS_Y[INPUTSET];
    static constexpr const array_size_t ELEMS_Z = VOLADD_ELEMS_Z[INPUTSET];

    auto A = make_array<float ***>({{ELEMS_Z, ELEMS_Y, ELEMS_X}});
    auto B = make_array<float ***>({{ELEMS_Z, ELEMS_Y, ELEMS_X}});
    auto C = make_array<float ***>({{ELEMS_Z, ELEMS_Y, ELEMS_X}});

    for (auto plane : A.dim_iterator()) {
        for (auto row : plane) {
            for (auto &col : row) {
                col = 1.f;
            }
        }
    }

    for (auto plane : B.dim_iterator()) {
        for (auto row : plane) {
            for (auto &col : row) {
                col = 2.f;
            }
        }
    }

    for (auto plane : C.dim_iterator()) {
        for (auto row : plane) {
            for (auto &col : row) {
                col = 0.f;
            }
        }
    }

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                C(i, j, k) = A(i, j, k) + B(i, j, k);
            }
        }
    }

    for (auto plane : C.dim_iterator()) {
        for (auto row : plane) {
            for (auto col : row) {
                assert(col == 3.f);
            }
        }
    }


    return true;
}

bool
launch_test_vecadd_dim_iterator_static()
{
    static constexpr const array_size_t ELEMS_X = VOLADD_ELEMS_X[INPUTSET];
    static constexpr const array_size_t ELEMS_Y = VOLADD_ELEMS_Y[INPUTSET];
    static constexpr const array_size_t ELEMS_Z = VOLADD_ELEMS_Z[INPUTSET];

    auto A = make_array<float [ELEMS_Z][ELEMS_Y][ELEMS_X]>();
    auto B = make_array<float [ELEMS_Z][ELEMS_Y][ELEMS_X]>();
    auto C = make_array<float [ELEMS_Z][ELEMS_Y][ELEMS_X]>();

    for (auto plane : A.dim_iterator()) {
        for (auto row : plane) {
            for (auto &col : row) {
                col = 1.f;
            }
        }
    }

    for (auto plane : B.dim_iterator()) {
        for (auto row : plane) {
            for (auto &col : row) {
                col = 2.f;
            }
        }
    }

    for (auto plane : C.dim_iterator()) {
        for (auto row : plane) {
            for (auto &col : row) {
                col = 0.f;
            }
        }
    }

    for (unsigned i = 0; i < ELEMS_Z; ++i) {
        for (unsigned j = 0; j < ELEMS_Y; ++j) {
            for (unsigned k = 0; k < ELEMS_X; ++k) {
                C(i, j, k) = A(i, j, k) + B(i, j, k);
            }
        }
    }

    for (auto plane : C.dim_iterator()) {
        for (auto row : plane) {
            for (auto col : row) {
                assert(col == 3.f);
            }
        }
    }

    return true;
}

int main()
{
    launch_test_vecadd_base();
    launch_test_vecadd_base_static();
    launch_test_vecadd();
    launch_test_vecadd_static();
    launch_test_vecadd_for_each();
    launch_test_vecadd_for_each_static();
    launch_test_vecadd_iterator();
    launch_test_vecadd_iterator_static();
    launch_test_vecadd_dim_iterator();
    launch_test_vecadd_dim_iterator_static();

    auto start1 = std::chrono::high_resolution_clock::now();
    launch_test_vecadd_base();
    auto end1 = std::chrono::high_resolution_clock::now();
    auto dur1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1);

    auto start2 = std::chrono::high_resolution_clock::now();
    launch_test_vecadd_base_static();
    auto end2 = std::chrono::high_resolution_clock::now();
    auto dur2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2);

    auto start3 = std::chrono::high_resolution_clock::now();
    launch_test_vecadd();
    auto end3 = std::chrono::high_resolution_clock::now();
    auto dur3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3);

    auto start4 = std::chrono::high_resolution_clock::now();
    launch_test_vecadd_static();
    auto end4 = std::chrono::high_resolution_clock::now();
    auto dur4 = std::chrono::duration_cast<std::chrono::nanoseconds>(end4 - start4);

    auto start5 = std::chrono::high_resolution_clock::now();
    launch_test_vecadd_for_each();
    auto end5 = std::chrono::high_resolution_clock::now();
    auto dur5 = std::chrono::duration_cast<std::chrono::nanoseconds>(end5 - start5);

    auto start6 = std::chrono::high_resolution_clock::now();
    launch_test_vecadd_for_each_static();
    auto end6 = std::chrono::high_resolution_clock::now();
    auto dur6 = std::chrono::duration_cast<std::chrono::nanoseconds>(end6 - start6);

    auto start7 = std::chrono::high_resolution_clock::now();
    launch_test_vecadd_iterator();
    auto end7 = std::chrono::high_resolution_clock::now();
    auto dur7 = std::chrono::duration_cast<std::chrono::nanoseconds>(end7 - start7);

    auto start8 = std::chrono::high_resolution_clock::now();
    launch_test_vecadd_iterator_static();
    auto end8 = std::chrono::high_resolution_clock::now();
    auto dur8 = std::chrono::duration_cast<std::chrono::nanoseconds>(end8 - start8);

    auto start9 = std::chrono::high_resolution_clock::now();
    launch_test_vecadd_dim_iterator();
    auto end9 = std::chrono::high_resolution_clock::now();
    auto dur9 = std::chrono::duration_cast<std::chrono::nanoseconds>(end9 - start9);

    auto start10 = std::chrono::high_resolution_clock::now();
    launch_test_vecadd_dim_iterator_static();
    auto end10 = std::chrono::high_resolution_clock::now();
    auto dur10 = std::chrono::duration_cast<std::chrono::nanoseconds>(end10 - start10);

    std::cout << "BASE:                  " << dur1.count() << "\n";
    std::cout << "BASE STATIC:           " << dur2.count() << "\n";
    std::cout << "INDEX:                 " << dur3.count() << "\n";
    std::cout << "INDEX STATIC:          " << dur4.count() << "\n";
    std::cout << "INDEX FOR_EACH:        " << dur5.count() << "\n";
    std::cout << "INDEX FOR_EACH STATIC: " << dur6.count() << "\n";
    std::cout << "ITERATOR:              " << dur7.count() << "\n";
    std::cout << "ITERATOR STATIC:       " << dur8.count() << "\n";
    std::cout << "DIM_ITERATOR:          " << dur9.count() << "\n";
    std::cout << "DIM_ITERATOR STATIC:   " << dur10.count() << "\n";

    return 0;
}

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
