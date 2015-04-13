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
#ifndef CUDARRAYS_DETAIL_UTILS_LOG_HPP_
#define CUDARRAYS_DETAIL_UTILS_LOG_HPP_

#include <cstdio>
#include <string>
#include <utility>

#include "../../common.hpp"

namespace detail {
namespace cudarrays {
    template <typename... Args>
    static void
    print(FILE *out, std::string msg, Args &&...args)
    {
        std::string msg_nl = msg + "\n";
        fprintf(out, msg_nl.c_str(), args...);
    }

    template <typename... Args>
    static void
    fatal(std::string msg, Args &&...args)
    {
        print(stderr, msg, args...);

        abort();
    }

    static void
    fatal(std::string _msg)
    {
        fatal(_msg, "");
    }
}
}

namespace cudarrays {

template <typename... Args>
static void
DEBUG(std::string msg, Args &&...args)
{
    if (!config::OPTION_DEBUG) return;

    detail::cudarrays::print(stdout, msg, std::forward<Args>(args)...);
}

static void
DEBUG(std::string _msg)
{
    DEBUG(_msg, "");
}

#define FATAL(...) do {                             \
    detail::cudarrays::print(stderr, "%s:%d\n", __FILE__, __LINE__); \
    detail::cudarrays::fatal(__VA_ARGS__);       \
    } while (0)


#define ASSERT(c) do {                                           \
        if (!(c)) {                                              \
            fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);      \
            detail::cudarrays::fatal("Condition '"#c"' failed"); \
        }                                                        \
    } while (0)
}

#endif