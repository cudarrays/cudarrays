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

#ifndef CUDARRRAYS_TESTS_UNIT_HPP_
#define CUDARRRAYS_TESTS_UNIT_HPP_

#define CUDARRAYS_TEST(C,T) class C##_##T##_Test;

CUDARRAYS_TEST(storage_test, dim_manager)
CUDARRAYS_TEST(storage_test, dim_manager_offset)
CUDARRAYS_TEST(storage_test, dim_manager_impl_offset)
CUDARRAYS_TEST(storage_test, dim_manager_get_dim)
CUDARRAYS_TEST(storage_test, host_storage)
CUDARRAYS_TEST(storage_test, permute_indices1)
CUDARRAYS_TEST(storage_test, permute_indices2)
CUDARRAYS_TEST(storage_test, reorder)
CUDARRAYS_TEST(storage_test, vm_page_allocator1)
CUDARRAYS_TEST(storage_test, vm_page_allocator2)

CUDARRAYS_TEST(iterator_test, iterator1d)
CUDARRAYS_TEST(iterator_test, iterator2d)

CUDARRAYS_TEST(lib_storage_test, host_replicated)
CUDARRAYS_TEST(lib_storage_test, host_reshape_block)
CUDARRAYS_TEST(lib_storage_test, host_reshape_block_cyclic)
CUDARRAYS_TEST(lib_storage_test, host_reshape_cyclic)
CUDARRAYS_TEST(lib_storage_test, host_vm)

#endif
