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
#ifndef CUDARRAYS_STATIC_ARRAY_HPP_
#define CUDARRAYS_STATIC_ARRAY_HPP_

#include "array_traits.hpp"
#include "storage.hpp"

#include "detail/dynarray/indexing.hpp"

namespace cudarrays {

enum memory_space {
    local     = 1,
    shared    = 2,
};

template <typename T, size_t Elems, typename Align, memory_space Space>
class static_array_storage;

template <typename T, size_t Elems, typename Align>
class static_array_storage<T, Elems, Align, memory_space::local> {
public:
    __host__ __device__
    inline
    T *get_data()
    {
        return data_ + Align::get_offset();
    }

public:
    T data_[Elems] __attribute__((aligned(Align::alignment * sizeof(T))));
};

template <typename T, size_t Elems, typename Align>
class static_array_storage<T, Elems, Align, memory_space::shared> {
public:
    __host__ __device__
    inline
    static T *get_data()
    {
#ifdef __CUDA_ARCH__
        __shared__ T data[Elems] __align__(Align::alignment * sizeof(T));
        return data + Align::get_offset();
#else
        return nullptr;
#endif
    }
};

template <typename T,
          memory_space Space   = memory_space::local,
          typename StorageType = layout::rmo,
          typename Align       = noalign>
class static_array {
public:
    using          array_type = T;
    using      alignment_type = Align;
    using   array_traits_type = array_traits<array_type>;
    using storage_traits_type = storage_traits<array_type, StorageType, alignment_type>;

    using       permuter_type = typename storage_traits_type::permuter_type;

    using     difference_type = array_index_t;
    using          value_type = typename array_traits_type::value_type;
    using        indexer_type = linearizer_hybrid<typename storage_traits_type::offsets_seq>;

    static_assert(array_traits_type::dynamic_dimensions == 0, "Dynamic dimensions are not allowed in static_array");

    static constexpr bool has_alignment = alignment_type::alignment > 1;

    static constexpr auto   elements = SEQ_PROD(typename storage_traits_type::extents_align_seq);
    static constexpr auto dimensions = array_traits_type::dimensions;

    __host__ __device__
    static_array() {};
    static_array(const static_array &a) = delete;

    template <unsigned Orig>
    __host__ __device__
    array_size_t dim() const
    {
        static_assert(Orig < dimensions, "Wrong dimension id");

        constexpr auto new_dim = permuter_type::template dim_index<Orig>();
        return SEQ_AT(typename storage_traits_type::extents_noalign_seq, new_dim);
    }

    template <unsigned Orig>
    __host__ __device__
    array_size_t dim_align() const
    {
        static_assert(Orig < dimensions, "Wrong dimension id");

        constexpr auto new_dim = permuter_type::template dim_index<Orig>();
        return SEQ_AT(typename storage_traits_type::extents_seq, new_dim);
    }


    //
    // Common operations
    //
    template <typename ...Idxs>
    __array_index__
    value_type &at(Idxs &&...idxs)
    {
        static_assert(sizeof...(Idxs) == dimensions, "Wrong number of indexes");

        return access_element_helper<SEQ_GEN_INC(dimensions)>::at(this->storage_.get_data(),
                                                                  array_index_t(std::forward<Idxs>(idxs))...);
    }

    template <typename ...Idxs>
    __array_index__
    const value_type &at(Idxs &&...idxs) const
    {
        static_assert(sizeof...(Idxs) == dimensions, "Wrong number of indexes");

        return access_element_helper<SEQ_GEN_INC(dimensions)>::at_const(this->storage_.get_data(),
                                                                        array_index_t(std::forward<Idxs>(idxs))...);
    }

    template <typename ...Idxs>
    __array_index__
    value_type &operator()(Idxs &&...idxs)
    {
        return at(std::forward<Idxs>(idxs)...);
    }

    template <typename ...Idxs>
    __array_index__
    const value_type &operator()(Idxs &&...idxs) const
    {
        return at(std::forward<Idxs>(idxs)...);
    }

private:
    template <typename Selector>
    struct access_element_helper;

    template <unsigned ...Vals>
    struct access_element_helper<SEQ_WITH_TYPE(unsigned, Vals...)> {
        template <typename... Idxs>
        __array_index__
        static
        value_type &at(value_type *data, Idxs &&...idxs)
        {
            static_assert(sizeof...(Idxs) == sizeof...(Vals), "Wrong number of indexes");
            auto idx = indexer_type::access_pos(nullptr,
                                                permuter_type::template select<Vals>(std::forward<Idxs>(idxs)...)...);
            return data[idx];
        }

        template <typename... Idxs>
        __array_index__
        static
        const value_type &at_const(const value_type *data, Idxs &&...idxs)
        {
            static_assert(sizeof...(Idxs) == sizeof...(Vals), "Wrong number of indexes");
            auto idx = indexer_type::access_pos(nullptr,
                                                permuter_type::template select<Vals>(std::forward<Idxs>(idxs)...)...);
            return data[idx];
        }

    };

public:
    static_array_storage<value_type, elements, alignment_type, Space> storage_;
};

}

#endif
