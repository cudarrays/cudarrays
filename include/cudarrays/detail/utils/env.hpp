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
#ifndef CUDARRAYS_DETAIL_UTILS_ENV_HPP_
#define CUDARRAYS_DETAIL_UTILS_ENV_HPP_

#include <cstdlib>
#include <string>

#include <algorithm>
#include <iostream>

namespace detail {

namespace utils {

template <typename T>
class string_to {
public:
    static bool convert(std::string name, T &val)
    {
        val = T(name);

        return true;
    }
};

template <>
class string_to<bool> {
public:
    static bool convert(std::string name, bool &val)
    {
        std::string tmp = name;
        std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);

        if (tmp == "y" || tmp == "yes") {
            val = true;
            return true;
        }
        if (tmp == "n" || tmp == "no") {
            val= false;
            return true;
        }

        long int _val = std::strtol(name.c_str(), NULL, 10);
        val = _val != 0;

        return true;
    }
};

template <>
class string_to<int> {
public:
    static bool convert(std::string name, int &val)
    {
        val = std::strtol(name.c_str(), NULL, 10);

        return true;
    }
};

template <>
class string_to<unsigned> {
public:
    static bool convert(std::string name, unsigned &val)
    {
        val = std::strtol(name.c_str(), NULL, 10);

        return true;
    }
};

template <>
class string_to<long int> {
public:
    static bool convert(std::string name, long int &val)
    {
        val = std::strtol(name.c_str(), NULL, 10);

        return true;
    }
};

template <>
class string_to<long unsigned> {
public:
    static bool convert(std::string name, long unsigned &val)
    {
        val = std::strtoul(name.c_str(), NULL, 10);

        return true;
    }
};

template <>
class string_to<float> {
public:
    static bool convert(std::string name, float &val)
    {
        val = std::strtof(name.c_str(), NULL);

        return true;
    }
};

}
}

namespace utils {

template <typename T1>
static T1 getenv(const std::string &name, T1 default_value)
{
    char *val = ::getenv(name.c_str());

    if (val == NULL) {
        return default_value;
    }

    T1 out;

    bool ok = ::detail::utils::string_to<T1>::convert(val, out);
    if (ok) return out;
    else return T1(0);
}

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
