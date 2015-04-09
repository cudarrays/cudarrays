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

#ifndef CUDARRAYS_ARRAY_
#define CUDARRAYS_ARRAY_

#if 0

#include <cassert>
#include <cstdlib>
#include <cstring>

#ifndef __CUDACC__
#include <array>
#endif

#include <memory>
#include <utility>

#include "compiler.hpp"
#include "common.hpp"
#include "storage.hpp"
#include "dist_storage.hpp"

namespace cudarrays {

template <typename T, size_t Size, bool Const = false, storage_type Storage = DISTRIBUTED_ALLOC>
class array;

template <typename T, size_t Size, bool Const, storage_type Storage>
class array :
    public array_storage<T[Size], Storage> {
public:
#ifndef __CUDACC__
    static const storage_type storage = Storage;

    typedef extents<Size>                   extents_type;
    typedef array_storage<T[Size], Storage> parent_storage;

    array(const extents_type &extents,
          const extents_type &alignment = fill<Size, ssize_t, 0>()) :
        parent_storage(extents, alignment)
    {
    }

    array(array &&a) :
        parent_storage(std::forward<parent_storage>(a))
    {
    }

    template <bool Const2>
    __host__ __device__
    __forceinline__
    array(const typename enable_if<Const && !Const2,
                                   array<T[Size], Const2, Storage> >::type &a) :
        parent_storage(a)
    {
        abort();
    }
#endif

    __host__ __device__
    __forceinline__
    virtual ~array()
    {
    }
};

template <typename T, size_t Size, bool Const>
class array<T[Size], Const, DISTRIBUTED> :
    public array_storage<T[Size], DISTRIBUTED> {
public:
#ifndef __CUDACC__
    static const storage_type storage = DISTRIBUTED;

    typedef extents<Size>                       extents_type;
    typedef array_storage<T[Size], DISTRIBUTED> parent_storage;

    array(const extents_type &extents, size_t chunks,
          const extents_type &alignment = fill<Size, ssize_t, 0>()) :
        parent_storage(extents, chunks, alignment)
    {
    }

    array(array &&a) :
        parent_storage(std::move(a))
    {
    }

    template <bool Const2>
    __host__ __device__
    __forceinline__
    array(const typename enable_if<Const && !Const2,
                                   array<T[Size], Const2, DISTRIBUTED> >::type &a) :
        parent_storage(a)
    {
        abort();
    }
#endif

    __host__ __device__
    __forceinline__
    virtual ~array()
    {
    }
};

template <typename T, size_t Size, bool Const>
class array<T[Size], Const, DISTRIBUTED_INDEXDIM> :
    public array_storage<T[Size], DISTRIBUTED_INDEXDIM> {
public:
#ifndef __CUDACC__
    static const storage_type storage = DISTRIBUTED_INDEXDIM;

    typedef extents<Size>                                extents_type;
    typedef array_storage<T[Size], DISTRIBUTED_INDEXDIM> parent_storage;

    array(const extents_type &extents, size_t chunks, const extents_type &alignment = fill<Size, ssize_t, 0>()) :
        parent_storage(extents, chunks, alignment)
    {
    }

    __host__ __device__
    __forceinline__
    array(array &&a) :
        parent_storage(std::move(a))
    {
    }

    template <bool Const2>
    __host__ __device__
    __forceinline__
    array(const typename enable_if<Const && !Const2,
                                   array<T[Size], Const2, DISTRIBUTED_INDEXDIM> >::type &a) :
        parent_storage(a)
    {
        abort();
    }
#endif

    __host__ __device__
    __forceinline__
    virtual ~array()
    {
    }
};

template <typename T, size_t Size, bool Const>
class array<T[Size], Const, DISTRIBUTED_ALLOC> :
    public array_storage<T[Size], DISTRIBUTED_ALLOC> {
public:
#ifndef __CUDACC__
    static const storage_type storage = DISTRIBUTED_ALLOC;

    typedef extents<Size>                             extents_type;
    typedef array_storage<T[Size], DISTRIBUTED_ALLOC> parent_storage;

    array(const extents_type &extents, size_t chunks,
             const extents_type &alignment = fill<Size, ssize_t, 0>()) :
        parent_storage(extents, chunks, alignment)
    {
    }

    array(array &&a) :
        parent_storage(std::move(a))
    {
    }

    template <bool Const2>
    __host__ __device__
    __forceinline__
    array(const typename enable_if<Const && !Const2,
                                   array<T[Size], Const2, DISTRIBUTED_ALLOC> >::type &a) :
        parent_storage(a)
    {
        abort();
    }
#endif

    __host__ __device__
    __forceinline__
    virtual ~array()
    {
    }
};

template <typename T, size_t Size, bool Const>
class array<T[Size], Const, DISTRIBUTED_ALLOCB> :
    public array_storage<T[Size], DISTRIBUTED_ALLOCB> {
public:
#ifndef __CUDACC__
    static const storage_type storage = DISTRIBUTED_ALLOCB;

    typedef extents<Size>                              extents_type;
    typedef array_storage<T[Size], DISTRIBUTED_ALLOCB> parent_storage;

#ifdef GENERIC_INDEX
    array(const extents_type &extents, size_t chunks,
             const extents_type &alignment = fill<Size, ssize_t, 0>(),
             const std::array<size_t, Size> &order = generate<Size, size_t, seq::dec>()) :
        parent_storage(extents, chunks, alignment, order)
#else
    array(const extents_type &extents, size_t chunks,
             const extents_type &alignment = fill<Size, ssize_t, 0>()) :
        parent_storage(extents, chunks, alignment)
#endif
    {
    }

    array(array &&a) :
        parent_storage(std::move(a))
    {
    }

    template <bool Const2>
    __host__ __device__
    __forceinline__
    array(const typename enable_if<Const && !Const2,
                                      array<T[Size], Const2, DISTRIBUTED_ALLOCB> >::type &a) :
        parent_storage(a)
    {
    }
#endif

    __host__ __device__
    __forceinline__
    virtual ~array()
    {
    }
};

template <typename T, size_t Size, bool Const>
class array<T[Size], Const, DISTRIBUTED_ALLOCC> :
    public array_storage<T[Size], DISTRIBUTED_ALLOCC> {
public:
#ifndef __CUDACC__
    typedef extents<Size>                              extents_type;
    typedef array_storage<T[Size], DISTRIBUTED_ALLOCC> parent_storage;

#ifdef GENERIC_INDEX
    array(const extents_type &extents, size_t chunks,
             const extents_type &alignment = fill<Size, ssize_t, 0>(),
             const std::array<size_t, Size> &order = generate<Size, size_t, seq::dec>(),
             size_t dim = Size - 1) :
        parent_storage(extents, chunks, alignment, order, dim)
#else
    array(const extents_type &extents, size_t chunks,
             const extents_type &alignment = fill<Size, ssize_t, 0>(),
             size_t dim = Size - 1) :
        parent_storage(extents, chunks, alignment, dim)
#endif
    {
    }

    array(array &&a) :
        parent_storage(std::move(a))
    {
    }

    template <bool Const2>
    __host__ __device__
    __forceinline__
    array(const typename enable_if<Const && !Const2,
                                      array<T[Size], Const2, DISTRIBUTED_ALLOCC> >::type &a) :
        parent_storage(a)
    {
    }
#endif

    __host__ __device__
    __forceinline__
    virtual ~array()
    {
    }
};


template <typename T, size_t Size, bool Const>
class array<T[Size], Const, REPLICATED> :
    public array_storage<T[Size], REPLICATED> {
public:
#ifndef __CUDACC__
    typedef extents<Size>                      extents_type;
    typedef array_storage<T[Size], REPLICATED> parent_storage;

#ifdef GENERIC_INDEX
    array(const extents_type &extents, size_t chunks,
             const extents_type &alignment = fill<Size, ssize_t, 0>(),
             const std::array<size_t, Size> order = generate<Size, size_t, seq::dec>()) :
        parent_storage(extents, chunks, alignment, order)
#else
    array(const extents_type &extents, size_t gpus,
             const extents_type &alignment = fill<Size, ssize_t, 0>()) :
        parent_storage(extents, gpus, alignment)
#endif
    {
    }

    array(array &&a) :
        parent_storage(std::move(a))
    {
    }

    template <bool Const2>
    __host__ __device__
    __forceinline__
    array(const typename enable_if<Const && !Const2,
                                      array<T[Size], Const2, REPLICATED> >::type &a) :
        parent_storage(a)
    {
        abort();
    }
#endif

    __host__ __device__
    __forceinline__
    virtual ~array()
    {
    }

};

}

namespace std {

#ifndef __CUDACC__
template <typename T, size_t Size, bool Const, storage_type Storage>
struct is_const< cudarrays::array<T[Size], Const, Storage> > {
    static const bool value = Const;

    typedef bool                                value_type;
    typedef std::integral_constant<bool, value>       type;

    operator bool()
    {
        return value;
    }
};

template <typename T1, typename T2>
struct is_cuda_convertible {
    static const bool value = (sizeof(T1) == sizeof(T2)) && is_convertible<T1, T2>::value;

    typedef bool                                value_type;
    typedef std::integral_constant<bool, value>       type;

    operator bool()
    {
        return value;
    }
};

template <typename T, size_t Size, bool Const, bool Const2, storage_type Storage>
struct is_cuda_convertible<cudarrays::array<T[Size], Const,  Storage>,
                           cudarrays::array<T[Size], Const2, Storage> > {
    static const bool value = (Const == Const2) || (!Const && Const2);

    typedef bool                                value_type;
    typedef std::integral_constant<bool, value>       type;

    operator bool()
    {
        return value;
    }
};
#endif

}

#endif

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
