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

#ifndef CUDARRAYS_ARRAY_HPP_
#define CUDARRAYS_ARRAY_HPP_

#include <cstdlib>
#include <cstring>

#include <limits>
#include <type_traits>

#include <utility>

#include "compiler.hpp"
#include "utils.hpp"
#include "common.hpp"
#include "memory.hpp"
#include "storage.hpp"
#include "storage_impl.hpp"

#include "gpu.cuh"

#include "detail/dynarray/iterator.hpp"
#include "detail/coherence/default.hpp"

#include "traits.hpp"

namespace cudarrays {

template <typename T,
          typename StorageType = layout::rmo,
          typename PartConf = automatic::none,
          template <typename> class CoherencePolicy = default_coherence>
class array :
    public coherent,
    public array_traits<T> {
public:
    using traits = array_traits<T>;

    using dim_order_type = typename make_dim_order<traits::dimensions, StorageType>::type;
    using  permuter_type = utils::permuter<traits::dimensions, dim_order_type>;

    using   host_storage_type = host_storage<typename traits::value_type>;
    using device_storage_type =
        dynarray_storage<typename traits::value_type, traits::dimensions, PartConf::final_impl,
                         typename reorder_gather_static< // User-provided dimension ordering
                             traits::dimensions,
                             bool,
                             typename PartConf::template part_type<traits::dimensions>,
                             dim_order_type
                         >::type>;

    using       difference_type = array_index_t;
    using coherence_policy_type = CoherencePolicy<array>;
    using          indexer_type = linearizer_static<T>;

    __host__
    explicit array(const align_t &align = align_t{0, 0},
                   coherence_policy_type coherence = coherence_policy_type()) :
        device_(permuter_type::reorder(traits::extents_type::get()), align),
        coherencePolicy_(coherence)
    {
        coherencePolicy_.bind(this);

        // Alloc host memory
        host_.alloc(device_.get_dim_manager().get_elems_align(), device_.get_dim_manager().offset());
        // TODO: Move this to a better place
        register_range(this->get_host_storage().base_addr(),
                       this->get_host_storage().size());
    }

    __host__
    explicit array(const array &a) :
        device_(a.device_),
        coherencePolicy_(a.coherencePolicy_)
    {
    }

    __host__
    array &operator=(const array &a)
    {
        if (&a != this) {
            device_          = a.device_;
            coherencePolicy_ = a.coherencePolicy_;
        }

        return *this;
    }

    __host__
    virtual ~array()
    {
        unregister_range(this->get_host_storage().base_addr());
    }

    __array_index__
    typename traits::value_type &operator()(array_index_t idx)
    {
#ifdef __CUDA_ARCH__
        return device_.access_pos(0, 0, idx);
#else
        return host_.addr()[idx];
#endif
    }

    __array_index__
    const typename traits::value_type &operator()(array_index_t idx) const
    {
#ifdef __CUDA_ARCH__
        return device_.access_pos(0, 0, idx);
#else
        return host_.addr()[idx];
#endif
    }

    __array_index__
    typename traits::value_type &operator()(array_index_t idx1, array_index_t idx2)
    {
        array_index_t i1 = permuter_type::template select<0>(idx1, idx2);
        array_index_t i2 = permuter_type::template select<1>(idx1, idx2);

#ifdef __CUDA_ARCH__
        return device_.access_pos(0, i1, i2);
#else
        auto idx = indexer_type::access_pos(i1, i2);
        return host_.addr()[idx];
#endif
    }

    __array_index__
    const typename traits::value_type &operator()(array_index_t idx1, array_index_t idx2) const
    {
        array_index_t i1 = permuter_type::template select<0>(idx1, idx2);
        array_index_t i2 = permuter_type::template select<1>(idx1, idx2);

#ifdef __CUDA_ARCH__
        return device_.access_pos(0, i1, i2);
#else
        auto idx = indexer_type::access_pos(i1, i2);
        return host_.addr()[idx];
#endif
    }

    __array_index__
    typename traits::value_type &operator()(array_index_t idx1, array_index_t idx2, array_index_t idx3)
    {
        array_index_t i1 = permuter_type::template select<0>(idx1, idx2, idx3);
        array_index_t i2 = permuter_type::template select<1>(idx1, idx2, idx3);
        array_index_t i3 = permuter_type::template select<2>(idx1, idx2, idx3);

#ifdef __CUDA_ARCH__
        return device_.access_pos(i1, i2, i3);
#else
        auto idx = indexer_type::access_pos(i1, i2, i3);
        return host_.addr()[idx];
#endif
    }

    __array_index__
    const typename traits::value_type &operator()(array_index_t idx1, array_index_t idx2, array_index_t idx3) const
    {
        array_index_t i1 = permuter_type::template select<0>(idx1, idx2, idx3);
        array_index_t i2 = permuter_type::template select<1>(idx1, idx2, idx3);
        array_index_t i3 = permuter_type::template select<2>(idx1, idx2, idx3);

#ifdef __CUDA_ARCH__
        return device_.access_pos(i1, i2, i3);
#else
        auto idx = indexer_type::access_pos(i1, i2, i3);
        return host_.addr()[idx];
#endif
    }

    template <unsigned DimsComp>
    __host__ bool
    distribute(compute_mapping<DimsComp, traits::dimensions> mapping)
    {
        auto mapping2 = mapping;
        mapping2.info = permuter_type::reorder(mapping2.info);

        return device_.template distribute<DimsComp>(mapping2);
    }

    __host__ bool
    distribute(const std::vector<unsigned> &gpus)
    {
        return device_.distribute(gpus);
    }

    __host__ bool
    is_distributed()
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
    set_current_gpu(unsigned idx)
    {
        device_.set_current_gpu(idx);
    }

    coherence_policy &get_coherence_policy()
    {
        return coherencePolicy_;
    }

    host_storage_type &get_host_storage()
    {
        return host_;
    }

    void to_device()
    {
        device_.to_device(host_);
    }

    void to_host()
    {
        device_.to_host(host_);
    }

    //
    // Common operations
    //

    //
    // Iterator interface
    //
    using       iterator = myiterator<array, false>;
    using const_iterator = myiterator<array, true>;

    using       reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    iterator begin()
    {
        array_index_t dims[traits::dimensions];
        std::fill(dims, dims + traits::dimensions, 0);
        return iterator(*this, dims);
    }

    const_iterator begin() const
    {
        return cbegin();
    }

    const_iterator cbegin() const
    {
        array_index_t dims[traits::dimensions];
        std::fill(dims, dims + traits::dimensions, 0);
        return const_iterator(*this, dims);
    }

    reverse_iterator rbegin()
    {
        array_index_t dims[traits::dimensions];
        for (unsigned i = 0; i < traits::dimensions; ++i) {
            dims[i] = this->dim(i) - 1;
        }
        return reverse_iterator(iterator(*this, dims));
    }

    iterator end()
    {
        array_index_t dims[traits::dimensions];
        dims[0] = this->dim(0);
        if (traits::dimensions > 1) {
            std::fill(dims + 1, dims + traits::dimensions, 0);
        }
        return iterator(*this, dims);
    }

    const_iterator end() const
    {
        return cend();
    }

    const_iterator cend() const
    {
        array_index_t dims[traits::dimensions];
        dims[0] = this->dim(0);
        if (traits::dimensions > 1) {
            std::fill(dims + 1, dims + traits::dimensions, 0);
        }
        return const_iterator(*this, dims);
    }

    reverse_iterator rend()
    {
        array_index_t dims[traits::dimensions];
        dims[0] = -1;
        for (unsigned i = 0; i < traits::dimensions; ++i) {
            dims[i] = this->dim(i) - 1;
        }
        return reverse_iterator(iterator(*this, dims));
    }

    const_reverse_iterator rend() const
    {
        return crend();
    }

    const_reverse_iterator crend() const
    {
        array_index_t dims[traits::dimensions];
        dims[0] = -1;
        for (unsigned i = 0; i < traits::dimensions; ++i) {
            dims[i] = this->dim(i) - 1;
        }
        return const_reverse_iterator(const_iterator(*this, dims));
    }

    friend myiterator<array, false>;
    friend myiterator<array, true>;

private:
    coherence_policy_type coherencePolicy_;
    host_storage_type     host_;
    device_storage_type   device_;
};

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
