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
#ifndef CUDARRAYS_COMPUTE_HPP_
#define CUDARRAYS_COMPUTE_HPP_

#include <array>

#include "utils.hpp"

namespace cudarrays {

template <unsigned Dims>
struct compute_part_helper;

template <>
struct compute_part_helper<1> {
    static std::array<bool, 1>
    make_array(partition c)
    {
        std::array<bool, 1> ret;
        if (c == partition::NONE)   ret = { false };
        else if (c == partition::X) ret = { true  };
        else abort();
        return ret;
    }
};

template <>
struct compute_part_helper<2> {
    static std::array<bool, 2>
    make_array(partition c)
    {
        std::array<bool, 2> ret;
        if (c == partition::NONE)    ret = { false, false };
        else if (c == partition::X)  ret = { false, true  };
        else if (c == partition::Y)  ret = { true, false  };
        else if (c == partition::XY) ret = { true, true   };
        else abort();
        return ret;
    }
};

template <>
struct compute_part_helper<3> {
    static std::array<bool, 3>
    make_array(partition c)
    {
        std::array<bool, 3> ret;
        if (c == partition::NONE)     ret = { false, false, false };
        else if (c == partition::X)   ret = { false, false, true  };
        else if (c == partition::Y)   ret = { false, true,  false };
        else if (c == partition::Z)   ret = { true,  false, false };
        else if (c == partition::XY)  ret = { false, true,  true  };
        else if (c == partition::XZ)  ret = { true,  false, true  };
        else if (c == partition::YZ)  ret = { true,  true,  false };
        else if (c == partition::XYZ) ret = { true,  true,  true  };
        else abort();
        return ret;
    }
};

template <unsigned Dims>
struct compute_conf {
    std::array<bool, Dims> info;
    unsigned procs;

    compute_conf(partition c, unsigned _procs = 0) :
        procs(_procs)
    {
        info = compute_part_helper<Dims>::make_array(c);
    }

    constexpr bool is_dim_part(unsigned dim) const
    {
        return info[dim];
    }

    unsigned get_part_dims() const
    {
        return utils::count(info, true);
    }
};

}

#endif
