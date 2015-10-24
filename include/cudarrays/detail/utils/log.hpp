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
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <utility>

#include <sstream>

#include "../../common.hpp"
#include "../../coherence.hpp"

#include "base.hpp"
#include "misc.hpp"
#include "option.hpp"

namespace cudarrays {
    extern utils::option<bool> LOG_DEBUG;
    extern utils::option<bool> LOG_TRACE;
    extern utils::option<bool> LOG_VERBOSE;
    extern utils::option<bool> LOG_SHOW_PATH;
    extern utils::option<bool> LOG_SHORT_PATH;
    extern utils::option<bool> LOG_SHOW_SYMBOL;
    extern utils::option<bool> LOG_STRIP_NAMESPACE;

    struct function_name {
        std::string prefix;
        std::string func;
    };

    template <typename T>
    std::string
    array_to_string(T *array, unsigned dims)
    {
        std::stringstream str_tmp;

        for (unsigned i = 0; i < dims; ++i) {
            str_tmp << array[i];
            if (i < dims - 1) str_tmp << ", ";
        }

        return str_tmp.str();
    }

    static constexpr size_t TmpBufferSize_ = 4096;
    thread_local static char TmpBuffer_[TmpBufferSize_];

    template <typename T>
    static inline
    const T &replace_id(const T &arg)
    {
        return arg;
    }

    static inline
    std::string replace_id(const coherent &arg)
    {
#if CUDARRAYS_DEBUG == 1
        return std::to_string(arg.get_id());
#else
        return std::to_string(reinterpret_cast<std::ptrdiff_t>(arg.host_addr()));
#endif
    }

    template <typename T>
    static inline
    const T &replace_string(const T &arg)
    {
        return arg;
    }

    static inline
    const char *replace_string(const char *arg)
    {
        return arg;
    }

    static inline
    const char *replace_string(const std::string &arg)
    {
        return arg.c_str();
    }

    template <typename T>
    static inline
    const T &replace_array(const T &arg)
    {
        return arg;
    }

    template <typename T, size_t dims>
    static std::string replace_array(const T (&array)[dims])
    {
        return array_to_string(array, dims);
    }

    template <typename T>
    static std::string replace_array(const T (&array)[0])
    {
        return array_to_string(array, 0);
    }

    template <size_t dims>
    static const char *replace_array(const char (&array)[dims])
    {
        return array;
    }

    static inline const char *replace_array(const char (&array)[0])
    {
        return array;
    }

    template <typename T, size_t Dims>
    static std::string replace_array(const std::array<T, Dims> &array)
    {
        return array_to_string(array.data(), Dims);
    }

    template <typename... Args>
    static std::string
    format_str(const std::string &msg, const Args &...args)
    {
        snprintf(TmpBuffer_, TmpBufferSize_, msg.c_str(),
                 replace_string(replace_array(replace_id(args)))...);

        return std::string{TmpBuffer_};
    }

    template <typename... Args>
    static std::string
    format_str(const std::string &msg)
    {
        snprintf(TmpBuffer_, TmpBufferSize_, msg.c_str(), nullptr);

        return std::string{TmpBuffer_};
    }

    static std::string
    format_header(std::string file, unsigned line, const std::string &tag, const function_name &fun)
    {
        std::string ret = "[" + tag + "]";

        if (LOG_SHOW_PATH) {
            if (LOG_SHORT_PATH) {
                static size_t dir_root_len    = std::numeric_limits<size_t>::max();
                static size_t dir_install_len = std::numeric_limits<size_t>::max();

                if (dir_root_len == std::numeric_limits<size_t>::max())
                    dir_root_len = strlen(CUDARRAYS_ROOT_DIR);
                if (dir_install_len == std::numeric_limits<size_t>::max())
                    dir_install_len = strlen(CUDARRAYS_INSTALL_DIR);

                auto pos = file.find(CUDARRAYS_ROOT_DIR);
                if (pos != std::string::npos)
                    file = file.substr(pos + dir_root_len + 1);
                else if ((pos = file.find(CUDARRAYS_INSTALL_DIR)) != std::string::npos)
                    file = file.substr(pos + dir_install_len + 1);
            }
            ret += " " + file + ":" + std::to_string(line);
        }
        if (LOG_SHOW_SYMBOL) {
            ret += " " + fun.func;
        }

        return ret;
    }

    static inline function_name
    format_function_name(const char *name)
    {
        function_name ret;
        std::string nameStr(name);

        auto pos = nameStr.find(" [with");
        if (pos != std::string::npos)
            nameStr = nameStr.substr(0, pos);
        if (LOG_STRIP_NAMESPACE)
            nameStr = utils::string_replace_all(nameStr, "cudarrays::", "");

        ret.func = nameStr;

        return ret;
    }

    template <typename... Args>
    static void
    print(FILE *out,
          const std::string &file,
          unsigned line,
          const std::string &tag,
          const function_name &fun,
          const std::string &msg, const Args &...args)
    {
        std::string header = format_header(file, line, tag, fun);
        std::string body   = format_str(msg, args...);

        fprintf(out, "%s %s\n", header.c_str(), body.c_str());
    }

    template <typename... Args>
    static void
    print(FILE *out,
          const std::string &file,
          unsigned line,
          const std::string &tag,
          const function_name &fun)
    {
        std::string header = format_header(file, line, tag, fun);

        fprintf(out, "%s ", header.c_str());
    }

    class trace_scope {
        std::string file_;
        unsigned line_;
        std::string tag_;
        function_name fun_;
        std::string msg_;

    public:
        template <typename... Args>
        trace_scope(bool enable,
                    const std::string &file,
                    unsigned line,
                    const std::string &tag,
                    const function_name &fun,
                    const std::string &msg, const Args &...args) :
            file_(file),
            line_(line),
            tag_(tag),
            fun_(fun),
            msg_(format_str(msg, args...))
        {
            if (!enable) return;

            print(stdout, file_, line_, tag_, fun_, msg_);
        }

        trace_scope(bool enable,
                    const std::string &file,
                    unsigned line,
                    const std::string &tag,
                    const function_name &fun) :
            trace_scope(enable, file, line, tag, fun, "")
        {}
    };

#define TRACE_FUNCTION()                                               \
    cudarrays::trace_scope                                             \
        tracer__{LOG_TRACE, __FILE__, __LINE__, "TRACE",               \
                 cudarrays::format_function_name(__PRETTY_FUNCTION__)}

#define DEBUG(...)                                                                            \
    do {                                                                                      \
        if (!LOG_DEBUG) break;                                                                \
        cudarrays::print(stdout, __FILE__, __LINE__, "DEBUG",                                 \
                         cudarrays::format_function_name(__PRETTY_FUNCTION__),##__VA_ARGS__); \
    } while (0)

#define INFO(...)                                                                             \
    do {                                                                                      \
        if (!LOG_VERBOSE) break;                                                              \
        cudarrays::print(stdout, __FILE__, __LINE__, "INFO",                                  \
                         cudarrays::format_function_name(__PRETTY_FUNCTION__),##__VA_ARGS__); \
    } while (0)

#define FATAL(...) do {                                                                       \
        cudarrays::print(stderr, __FILE__, __LINE__, "FATAL",                                 \
                         cudarrays::format_function_name(__PRETTY_FUNCTION__),##__VA_ARGS__); \
        abort();                                                                              \
    } while (0)

#define ASSERT(c,...) do {                                                                        \
        if (!(c)) {                                                                               \
            cudarrays::print(stderr, __FILE__, __LINE__, "ASSERT",                                \
                             cudarrays::format_function_name(__PRETTY_FUNCTION__),                \
                             "Condition '"#c"' not met");                                         \
            cudarrays::print(stderr, __FILE__, __LINE__, "ASSERT",                                \
                             cudarrays::format_function_name(__PRETTY_FUNCTION__),##__VA_ARGS__); \
            abort();                                                                              \
        }                                                                                         \
    } while (0)

} // namespace cudarrays

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
