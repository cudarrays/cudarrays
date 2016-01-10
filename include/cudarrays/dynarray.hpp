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
#ifndef CUDARRAYS_DYNARRAY_HPP_
#define CUDARRAYS_DYNARRAY_HPP_

#include <cstdlib>
#include <cstring>

#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include "compiler.hpp"
#include "common.hpp"
#include "memory.hpp"
#include "storage.hpp"
#include "storage_impl.hpp"

#include "gpu.cuh"

#include "detail/dynarray/iterator.hpp"
#include "detail/dynarray/dim_iterator.hpp"
#include "detail/coherence/default.hpp"

namespace cudarrays {

template <typename T>
class dynarray_view_common;
template <typename T>
class dynarray_view;
template <typename T>
class dynarray_cview;

template <typename T,
          typename StorageType     = layout::rmo,
          typename Align           = noalign,
          typename PartConf        = automatic::none,
          typename CoherencePolicy = default_coherence>
class dynarray :
    public coherent {
    friend dynarray_view_common<dynarray>;
    friend dynarray_view<dynarray>;
    friend dynarray_cview<dynarray>;

public:
    using            array_type = T;
    using        alignment_type = Align;
    using     array_traits_type = array_traits<array_type>;
    using   storage_traits_type = dist_storage_traits<array_type, StorageType, alignment_type, PartConf>;

    static constexpr bool has_alignment = alignment_type::alignment > 1;

    using         permuter_type = typename storage_traits_type::permuter_type;

    using       difference_type = array_index_t;
    using            value_type = typename array_traits_type::value_type;
    using coherence_policy_type = CoherencePolicy;
    using          indexer_type = linearizer_hybrid<typename storage_traits_type::offsets_seq>;

    using device_storage_type = dynarray_storage<PartConf::impl,
                                                 storage_traits_type>;

    static constexpr auto dimensions = array_traits_type::dimensions;

    using       value_iterator_type = array_iterator_facade<dynarray, false>;
    using const_value_iterator_type = array_iterator_facade<dynarray, true>;

    using       dim_iterator_type = array_dim_iterator_facade<dynarray, false>;
    using const_dim_iterator_type = array_dim_iterator_facade<dynarray, true>;

    static dynarray *make(const extents<array_traits_type::dynamic_dimensions> &ext,
                          coherence_policy_type coherence)
    {
        return new dynarray(ext, coherence);
    }

private:
    __host__
    explicit dynarray(const extents<array_traits_type::dynamic_dimensions> &extents,
                      coherence_policy_type coherence) :
        coherencePolicy_{coherence},
        device_(permuter_type::reorder(array_traits_type::make_extents(extents)))
    {
        // LIBRARY ENTRY POINT
        cudarrays_entry_point();

        // Alloc host memory
        host_.alloc(device_.get_dim_manager().get_elems_align() * sizeof(value_type));
        coherencePolicy_.bind(*this);
    }

    __host__
    explicit dynarray(const dynarray &a) :
        device_(a.device_),
        coherencePolicy_(a.coherencePolicy_)
    {
    }

public:
    __host__
    virtual ~dynarray()
    {
        coherencePolicy_.unbind();
    }

    template <unsigned DimsComp>
    __host__ bool
    distribute(const compute_mapping<DimsComp, dimensions> &mapping)
    {
        auto mapping2 = mapping;
        mapping2.info = permuter_type::reorder(mapping2.info);

        return device_.template distribute<DimsComp>(mapping2);
    }

    __host__ bool
    distribute(const std::vector<unsigned> &gpus) override final
    {
        return device_.distribute(gpus);
    }

    __host__ bool
    is_distributed() const override final
    {
        return device_.is_distributed();
    }

    template <unsigned Orig>
    __array_bounds__
    array_size_t dim() const
    {
        auto new_dim = permuter_type::template dim_index<Orig>();
        return device_.get_dim_manager().dim(new_dim);
    }

    __array_bounds__
    array_size_t dim(unsigned dim) const
    {
        auto new_dim = permuter_type::dim_index(dim);
        return device_.get_dim_manager().dim(new_dim);
    }

    void
    set_current_gpu(unsigned idx) override final
    {
        device_.set_current_gpu(idx);
    }

    coherence_policy_type &get_coherence_policy() noexcept override final
    {
        return coherencePolicy_;
    }

    inline
    void *host_addr() noexcept override final
    {
        return host_.addr();
    }

    inline
    const void *host_addr() const noexcept override final
    {
        return host_.addr();
    }

    size_t size() const noexcept override final
    {
        return host_.size();
    }

    void to_device() override final
    {
        device_.to_device(host_);
    }

    void to_host() override final
    {
        device_.to_host(host_);
    }

    inline
    const dim_manager<value_type, alignment_type, dimensions> &
    get_dim_manager() const
    {
        return device_.get_dim_manager();
    }

    //
    // Common operations
    //
    template <typename ...Idxs>
    __array_index__
    value_type &at(Idxs &&...idxs)
    {
        static_assert(sizeof...(Idxs) == dimensions, "Wrong number of indexes");

        return access_element_helper<SEQ_GEN_INC(dimensions)>::at(device_, host_, array_index_t(std::forward<Idxs>(idxs))...);
    }

    template <typename ...Idxs>
    __array_index__
    const value_type &at(Idxs &&...idxs) const
    {
        static_assert(sizeof...(Idxs) == dimensions, "Wrong number of indexes");

        return access_element_helper<SEQ_GEN_INC(dimensions)>::at_const(device_, host_, array_index_t(std::forward<Idxs>(idxs))...);
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

    //
    // Iterator interface
    //
    value_iterator_type value_iterator()
    {
        return value_iterator_type{*this};
    }

    const_value_iterator_type value_iterator() const
    {
        return const_value_iterator_type{*this};
    }

    dim_iterator_type dim_iterator()
    {
        return dim_iterator_type{*this};
    }

    const_dim_iterator_type dim_iterator() const
    {
        return const_dim_iterator_type{*this};
    }

    friend value_iterator_type;
    friend const_value_iterator_type;
    friend dim_iterator_type;
    friend const_dim_iterator_type;

private:
    template <typename Selector>
    struct access_element_helper;

    template <unsigned ...Vals>
    struct access_element_helper<SEQ_WITH_TYPE(unsigned, Vals...)> {
        template <typename... Idxs>
        __array_index__
        static
        value_type &at(device_storage_type &device,
                       host_storage<storage_traits_type> &host,
                       Idxs &&...idxs)
        {
            static_assert(sizeof...(Idxs) == sizeof...(Vals), "Wrong number of indexes");
#ifdef __CUDA_ARCH__
            return device.access_pos(permuter_type::template select<Vals>(std::forward<Idxs>(idxs)...)...);
#else
            auto idx = indexer_type::access_pos(device.get_dim_manager().get_strides(),
                                                permuter_type::template select<Vals>(std::forward<Idxs>(idxs)...)...);
            return host.addr()[idx];
#endif
        }

        template <typename... Idxs>
        __array_index__
        static
        const value_type &at_const(const device_storage_type &device,
                                   const host_storage<storage_traits_type> &host,
                                   Idxs &&...idxs)
        {
            static_assert(sizeof...(Idxs) == sizeof...(Vals), "Wrong number of indexes");
#ifdef __CUDA_ARCH__
            return device.access_pos(permuter_type::template select<Vals>(std::forward<Idxs>(idxs)...)...);
#else
            auto idx = indexer_type::access_pos(device.get_dim_manager().get_strides(),
                                                permuter_type::template select<Vals>(std::forward<Idxs>(idxs)...)...);
            return host.addr()[idx];
#endif
        }

    };

    coherence_policy_type coherencePolicy_;
    device_storage_type   device_;
    host_storage<storage_traits_type> host_;
};

}

#endif // CUDARRAYS_DYNARRAY_HPP_

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
