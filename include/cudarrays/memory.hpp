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
#ifndef CUDARRAYS_MEMORY_HPP_
#define CUDARRAYS_MEMORY_HPP_

#include <functional>

namespace cudarrays {

using handler_fn = std::function<bool (bool)>;

void register_range(void *addr, size_t count);
void unregister_range(void *addr);

void protect_range(void *addr, size_t count, const handler_fn &fn, bool modified);
void unprotect_range(void *addr);

void handler_sigsegv_overload();
void handler_sigsegv_restore();

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
