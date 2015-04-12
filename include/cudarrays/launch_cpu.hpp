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

#pragma once
#ifndef CUDARRAYS_LAUNCH_CPU_HPP_
#define CUDARRAYS_LAUNCH_CPU_HPP_

#include <vector>
#include <cxxabi.h>

#include "common.hpp"
#include "dynarray.hpp"

extern "C"
int cuda_grid_get_size_x();
extern "C"
int cuda_grid_get_size_y();
extern "C"
int cuda_grid_get_size_z();

extern "C"
int cuda_block_get_size_x();
extern "C"
int cuda_block_get_size_y();
extern "C"
int cuda_block_get_size_z();

extern "C"
int cuda_block_get_idx_x();
extern "C"
int cuda_block_get_idx_y();
extern "C"
int cuda_block_get_idx_z();

extern "C"
int cuda_thread_get_idx_x();
extern "C"
int cuda_thread_get_idx_y();
extern "C"
int cuda_thread_get_idx_z();

void cuda_block_synchronize();

namespace cudarrays {

struct cuda_conf :
    protected std::tuple<dim3, dim3, size_t, cudaStream_t> {
    using parent_tuple = std::tuple<dim3, dim3, size_t, cudaStream_t>;

    cuda_conf(dim3 grid, dim3 block) :
        parent_tuple(grid, block, 0, 0)
    {
    }

    dim3 get_grid() const
    {
        return std::get<0>(*this);
    }

    dim3 get_block() const
    {
        return std::get<1>(*this);
    }
};

template <typename... Args>
struct argument_manager {
    template <unsigned Idx, typename T>
    static void
    set_args(T &arg)
    {
        using Type = typename std::tuple_element<Idx,
                                                 std::tuple<Args...>>::type;

        //std::cout << abi::__cxa_demangle(typeid(Type).name(), 0, 0, 0) << std::endl;
        static_assert(std::is_cuda_convertible<T, Type>::value == true,
                      "Types are not compatible");
    }

    template <unsigned Idx, typename T, typename... Args2>
    static void
    set_args(T &arg, Args2&... args2)
    {
        using Type = typename std::tuple_element<Idx,
                                                 std::tuple<Args...>>::type;

        //std::cout << abi::__cxa_demangle(typeid(Type).name(), 0, 0, 0) << std::endl;
        static_assert(std::is_cuda_convertible<T, Type>::value == true,
                      "Types are not compatible");
    }

    template <typename... Args2>
    static void
    set_args(Args2&... args2)
    {
        set_args<0, Args2...>(args2...);
    }
};

void
init_grid(dim3 grid, dim3 block);

template <typename R, typename... Args>
class launcher_cpu {
    R(&f_)(Args...);
    const char *funName_;
    cuda_conf conf_;

public:
    launcher_cpu(R(&f)(Args...), const cuda_conf &conf) :
        f_(f),
        funName_(typeid(f).name()),
        conf_(conf)
    {
    }

    template <typename... Args2>
    std::vector<float>
    operator()(Args2 &...args2)
    {
        size_t dimSize = 0;
        std::vector<float> times;

        dim3 grid  = conf_.get_grid();
        dim3 block = conf_.get_block();

        DEBUG("Launch> orig: %zd %zd %zd", grid.x, grid.y, grid.z);

        using my_arguments = argument_manager<Args..., dim3, dim3>;
        my_arguments::set_args(args2...);

        for (int bz = 0; bz < grid.z; ++bz) {
            for (int by = 0; by < grid.y; ++by) {
                for (int bx = 0; bx < grid.x; ++bx) {
                    for (int tz = 0; tz < block.z; ++tz) {
                        for (int ty = 0; ty < block.y; ++ty) {
                            for (int tx = 0; tx < block.x; ++tx) {
                                f_(args2...);
                            }
                        }
                    }
                }
            }
        }

        return times;
    }
};

template <typename R, typename... Args>
launcher_cpu<R, Args...>
launch_cpu(R(&f)(Args...), const cuda_conf &conf, unsigned gpus = 1, int dim = -1, bool transposeXY = false)
{
    // int status;
    // char *realname = abi::__cxa_demangle(typeid(f).name(), 0, 0, &status);
    // printf("Func: %s\n", typeid(f).name());
    return launcher_cpu<R, Args...>(f, conf);
}

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
