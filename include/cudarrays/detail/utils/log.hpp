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

namespace cudarrays {
namespace detail {
    static constexpr size_t TmpBufferSize_ = 4096;
    thread_local static char TmpBuffer_[TmpBufferSize_];

    template <typename... Args>
    static void
    print(FILE *out, std::string msg, Args &&...args)
    {
        std::string msg_nl = msg + "\n";
        fprintf(out, msg_nl.c_str(), args...);
    }

    template <typename... Args>
    static void
    print(FILE *out, std::string msg)
    {
        std::string msg_nl = msg + "\n";
        fprintf(out, msg_nl.c_str(), nullptr);
    }

    template <typename... Args>
    static void
    print(FILE * /*out*/)
    {
    }

    template <typename... Args>
    static std::string
    format_str(std::string msg, Args &&...args)
    {
        std::string msg_nl = msg + "\n";
        snprintf(TmpBuffer_, TmpBufferSize_, msg_nl.c_str(), args...);

        return std::string{TmpBuffer_};
    }

    template <typename... Args>
    static std::string
    format_str(std::string msg)
    {
        std::string msg_nl = msg + "\n";
        snprintf(TmpBuffer_, TmpBufferSize_, msg_nl.c_str(), nullptr);

        return std::string{TmpBuffer_};
    }
}

template <typename... Args>
static void
DEBUG(std::string msg, Args &&...args)
{
    if (!config::OPTION_DEBUG) return;

    detail::print(stdout, msg, std::forward<Args>(args)...);
}

class DEBUG_SCOPE {
    std::string msg_;

public:
    template <typename... Args>
    DEBUG_SCOPE(std::string msg, Args &&...args) :
        msg_(detail::format_str(msg, std::forward<Args>(args)...))
    {
        if (!config::OPTION_DEBUG) return;

        std::string line = std::string(msg_.size() + 6, '=');

        detail::print(stdout, line + "BEGIN");
        detail::print(stdout, msg_);
    }

    ~DEBUG_SCOPE()
    {
        if (!config::OPTION_DEBUG) return;

        std::string line = std::string(msg_.size() + 4, '=');

        detail::print(stdout, msg_ + "END");
        detail::print(stdout, line);
    }
};

#define TRACE_FUNCTION() DEBUG_SCOPE(__PRETTY_FUNCTION__)

#define FATAL(...) do {                                                  \
        cudarrays::detail::print(stderr, "%s:%d\n", __FILE__, __LINE__); \
        cudarrays::detail::print(stderr, __VA_ARGS__);                   \
        abort();                                                         \
    } while (0)


#define ASSERT(c,...) do {                                                   \
        if (!(c)) {                                                          \
            cudarrays::detail::print(stderr,##__VA_ARGS__);                  \
            cudarrays::detail::print(stderr, "%s:%d\n", __FILE__, __LINE__); \
            cudarrays::detail::print(stderr, "Condition '"#c"' not met");    \
            abort();                                                         \
        }                                                                    \
    } while (0)
}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
