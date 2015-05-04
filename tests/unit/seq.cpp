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
#include <type_traits>

#include "cudarrays/utils.hpp"

#include "gtest/gtest.h"

class seq_test :
    public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

template <unsigned...> struct var;

TEST_F(seq_test, create)
{
    ASSERT_EQ(SEQ_SIZE(SEQ_WITH_TYPE(unsigned)), 0u);
    ASSERT_EQ(SEQ_SIZE(SEQ(1)),       1u);
    ASSERT_EQ(SEQ_SIZE(SEQ(1, 2, 3)), 3u);

    using t = var<1, 2, 3>;

    ASSERT_EQ(SEQ_SIZE(SEQ_WRAP(unsigned, t)), 3u);
    bool result = std::is_same<t,
                                SEQ_UNWRAP(SEQ_WRAP(unsigned, t), var<>)>::value;
    ASSERT_EQ(result, true);
}
