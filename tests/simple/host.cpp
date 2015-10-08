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

#include <cudarrays/types.hpp>
#include <cudarrays/launch.hpp>

using namespace cudarrays;

const unsigned INPUTSET = 1;
bool TEST = true;

constexpr const array_size_t VOLADD_ELEMS_X[2] = { 4, 128 };
constexpr const array_size_t VOLADD_ELEMS_Y[2] = { 5,  32 };
constexpr const array_size_t VOLADD_ELEMS_Z[2] = { 6,  32 };

bool
launch_test_vecadd_iterator()
{
    static constexpr const array_size_t ELEMS_X = VOLADD_ELEMS_X[INPUTSET];
    static constexpr const array_size_t ELEMS_Y = VOLADD_ELEMS_Y[INPUTSET];
    static constexpr const array_size_t ELEMS_Z = VOLADD_ELEMS_Z[INPUTSET];

    auto A = make_array<float ***>({ELEMS_Z, ELEMS_Y, ELEMS_X});
    auto B = make_array<float ***>({ELEMS_Z, ELEMS_Y, ELEMS_X});
    auto C = make_array<float ***>({ELEMS_Z, ELEMS_Y, ELEMS_X});

    {
        for (auto &plane : A.dim_iterator()) {
            for (auto &row : plane) {
                for (auto &col : row) {
                    col = 1.f;
                }
            }
        }

        for (auto &plane : B.dim_iterator()) {
            for (auto &row : plane) {
                for (auto &col : row) {
                    col = 2.f;
                }
            }
        }

        for (auto &plane : C.dim_iterator()) {
            for (auto &row : plane) {
                for (auto &col : row) {
                    col = 0.f;
                }
            }
        }
    }

    {
        for (unsigned i = 0; i < ELEMS_Z; ++i) {
            for (unsigned j = 0; j < ELEMS_Y; ++j) {
                for (unsigned k = 0; k < ELEMS_X; ++k) {
                    C(i, j, k) = A(i, j, k) + B(i, j, k);
                }
            }
        }
    }

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

    {
        for (auto &plane : A.dim_iterator()) {
            for (auto &row : plane) {
                for (auto &col : row) {
                    col = 1.f;
                }
            }
        }

        for (auto &plane : B.dim_iterator()) {
            for (auto &row : plane) {
                for (auto &col : row) {
                    col = 2.f;
                }
            }
        }

        for (auto &plane : C.dim_iterator()) {
            for (auto &row : plane) {
                for (auto &col : row) {
                    col = 0.f;
                }
            }
        }
    }

    {
        for (unsigned i = 0; i < ELEMS_Z; ++i) {
            for (unsigned j = 0; j < ELEMS_Y; ++j) {
                for (unsigned k = 0; k < ELEMS_X; ++k) {
                    C(i, j, k) = A(i, j, k) + B(i, j, k);
                }
            }
        }
    }

    return true;
}

bool
launch_test_vecadd()
{
    static constexpr const array_size_t ELEMS_X = VOLADD_ELEMS_X[INPUTSET];
    static constexpr const array_size_t ELEMS_Y = VOLADD_ELEMS_Y[INPUTSET];
    static constexpr const array_size_t ELEMS_Z = VOLADD_ELEMS_Z[INPUTSET];

    auto A = make_array<float ***>({ELEMS_Z, ELEMS_Y, ELEMS_X});
    auto B = make_array<float ***>({ELEMS_Z, ELEMS_Y, ELEMS_X});
    auto C = make_array<float ***>({ELEMS_Z, ELEMS_Y, ELEMS_X});

    {
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
    }

    {
        for (unsigned i = 0; i < ELEMS_Z; ++i) {
            for (unsigned j = 0; j < ELEMS_Y; ++j) {
                for (unsigned k = 0; k < ELEMS_X; ++k) {
                    C(i, j, k) = A(i, j, k) + B(i, j, k);
                }
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

    {
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
    }

    {
        for (unsigned i = 0; i < ELEMS_Z; ++i) {
            for (unsigned j = 0; j < ELEMS_Y; ++j) {
                for (unsigned k = 0; k < ELEMS_X; ++k) {
                    C(i, j, k) = A(i, j, k) + B(i, j, k);
                }
            }
        }
    }

    return true;
}


int main()
{
    launch_test_vecadd();
    launch_test_vecadd_static();
    launch_test_vecadd_iterator();

    return 0;
}

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
