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

#include <cassert>
#include <cstdlib>
#include <cstring>

#include <limits>
#include <type_traits>

#include <utility>

#include "coherence.hpp"
#include "compiler.hpp"
#include "utils.hpp"
#include "common.hpp"
#include "memory.hpp"
#include "storage.hpp"
#include "storage_impl.hpp"
#include "iterator.hpp"

namespace cudarrays {

template <typename T, unsigned Dims,
          typename StorageType = layout::rmo,
          typename PartConf = tag_auto::none,
          template <typename> class CoherencePolicy = default_coherence>
class dynarray :
    public coherent {
public:
    using reorder_conf_type = typename make_reorder<Dims, StorageType>::type;
    using     permuter_type = utils::permuter<Dims, reorder_conf_type>;

    using        storage_type =
        array_storage<T, Dims, PartConf::final_impl,
                      typename reorder_gather_static< // User-provided reordering
                          Dims,
                          bool,
                          typename PartConf::template part_type<Dims>,
                          reorder_conf_type
                      >::type>;

    using host_storage_type = typename storage_type::host_storage_type;

    using      value_type = T;
    using difference_type = array_index_t;

    using coherence_policy_type = CoherencePolicy<dynarray>;

    static constexpr unsigned dimensions = Dims;

    using extents_type = std::array<array_size_t, Dims>;

    __host__
    explicit dynarray(const extents_type &extents,
                      const align_t &align = align_t{0, 0},
                      coherence_policy_type coherence = coherence_policy_type()) :
        storage_(permuter_type::reorder(extents), align),
        coherencePolicy_(coherence)
    {
        coherencePolicy_.bind(this);
        // TODO: Move this to a better place
        register_range(this->get_host_storage().get_base_addr(),
                       this->get_host_storage().size());
    }

    __host__
    explicit dynarray(const dynarray &a) :
        storage_(a.storage_),
        coherencePolicy_(a.coherencePolicy_)
    {
    }

    __host__
    dynarray &operator=(const dynarray &a)
    {
        if (&a != this) {
            storage_         = a.storage_;
            coherencePolicy_ = a.coherencePolicy_;
        }

        return *this;
    }

    __host__
    virtual ~dynarray()
    {
        unregister_range(this->get_host_storage().get_base_addr());
    }

    __array_index__
    T &operator()(array_index_t idx)
    {
        return storage_.access_pos(0, 0, idx);
    }

    __array_index__
    const T &operator()(array_index_t idx) const
    {
        return storage_.access_pos(0, 0, idx);
    }

    __array_index__
    T &operator()(array_index_t idx1, array_index_t idx2)
    {
        array_index_t i1 = permuter_type::template select<0>(idx1, idx2);
        array_index_t i2 = permuter_type::template select<1>(idx1, idx2);

        return storage_.access_pos(0, i1, i2);
    }

    __array_index__
    const T &operator()(array_index_t idx1, array_index_t idx2) const
    {
        array_index_t i1 = permuter_type::template select<0>(idx1, idx2);
        array_index_t i2 = permuter_type::template select<1>(idx1, idx2);

        return storage_.access_pos(0, i1, i2);
    }

    __array_index__
    T &operator()(array_index_t idx1, array_index_t idx2, array_index_t idx3)
    {
        array_index_t i1 = permuter_type::template select<0>(idx1, idx2, idx3);
        array_index_t i2 = permuter_type::template select<1>(idx1, idx2, idx3);
        array_index_t i3 = permuter_type::template select<2>(idx1, idx2, idx3);

        return storage_.access_pos(i1, i2, i3);
    }

    __array_index__
    const T &operator()(array_index_t idx1, array_index_t idx2, array_index_t idx3) const
    {
        array_index_t i1 = permuter_type::template select<0>(idx1, idx2, idx3);
        array_index_t i2 = permuter_type::template select<1>(idx1, idx2, idx3);
        array_index_t i3 = permuter_type::template select<2>(idx1, idx2, idx3);

        return storage_.access_pos(i1, i2, i3);
    }

    template <unsigned DimsComp>
    __host__ bool
    distribute(compute_mapping<DimsComp, Dims> mapping)
    {
        auto mapping2 = mapping;
        mapping2.info = permuter_type::reorder(mapping2.info);

        return storage_.template distribute<DimsComp>(mapping2);
    }

    inline __host__ __device__
    array_size_t get_elems() const
    {
        return storage_.get_dim_manager().get_elems();
    }

    inline __host__ __device__
    array_size_t get_elems_align() const
    {
        return storage_.get_dim_manager().get_elems_align();
    }

    template <unsigned Orig>
    __array_bounds__
    array_size_t get_dim() const
    {
        auto new_dim = permuter_type::template dim_index<Orig>();
        return storage_.get_dim_manager().get_dim(new_dim);
    }

    template <unsigned Orig>
    __array_bounds__
    array_size_t get_dim_align() const
    {
        auto new_dim = permuter_type::template dim_index<Orig>();
        return storage_.get_dim_manager().get_dim_align(new_dim);
    }


    __array_bounds__
    array_size_t get_dim(unsigned dim) const
    {
        auto new_dim = permuter_type::dim_index(dim);
        return storage_.get_dim_manager().get_dim(new_dim);
    }

    __array_bounds__
    array_size_t get_dim_align(unsigned dim) const
    {
        auto new_dim = permuter_type::dim_index(dim);
        return storage_.get_dim_manager().get_dim_align(new_dim);
    }

    void
    set_current_gpu(unsigned idx)
    {
        storage_.set_current_gpu(idx);
    }

    coherence_policy &get_coherence_policy()
    {
        return coherencePolicy_;
    }

    host_storage_type &get_host_storage()
    {
        return storage_.get_host_storage();
    }

    void to_device()
    {
        storage_.to_device();
    }

    void to_host()
    {
        storage_.to_host();
    }

    //
    // Common operations
    //

    //
    // Iterator interface
    //
    using       iterator = myiterator<dynarray, false>;
    using const_iterator = myiterator<dynarray, true>;

    using       reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    iterator begin()
    {
        array_index_t dims[Dims];
        std::fill(dims, dims + Dims, 0);
        return iterator(*this, dims);
    }

    const_iterator begin() const
    {
        return cbegin();
    }

    const_iterator cbegin() const
    {
        array_index_t dims[Dims];
        std::fill(dims, dims + Dims, 0);
        return const_iterator(*this, dims);
    }

    reverse_iterator rbegin()
    {
        array_index_t dims[Dims];
        for (unsigned i = 0; i < Dims; ++i) {
            dims[i] = this->get_dim(i) - 1;
        }
        return reverse_iterator(iterator(*this, dims));
    }

    iterator end()
    {
        array_index_t dims[Dims];
        dims[0] = this->get_dim(0);
        if (Dims > 1) {
            std::fill(dims + 1, dims + Dims, 0);
        }
        return iterator(*this, dims);
    }

    const_iterator end() const
    {
        return cend();
    }

    const_iterator cend() const
    {
        array_index_t dims[Dims];
        dims[0] = this->get_dim(0);
        if (Dims > 1) {
            std::fill(dims + 1, dims + Dims, 0);
        }
        return const_iterator(*this, dims);
    }

    reverse_iterator rend()
    {
        array_index_t dims[Dims];
        dims[0] = -1;
        for (unsigned i = 0; i < Dims; ++i) {
            dims[i] = this->get_dim(i) - 1;
        }
        return reverse_iterator(iterator(*this, dims));
    }

    const_reverse_iterator rend() const
    {
        return crend();
    }

    const_reverse_iterator crend() const
    {
        array_index_t dims[Dims];
        dims[0] = -1;
        for (unsigned i = 0; i < Dims; ++i) {
            dims[i] = this->get_dim(i) - 1;
        }
        return const_reverse_iterator(const_iterator(*this, dims));
    }

    friend myiterator<dynarray, false>;
    friend myiterator<dynarray, true>;

private:
    storage_type          storage_;
    coherence_policy_type coherencePolicy_;
};

template <typename Array>
class dynarray_ref {
    using dynarray_type = Array;
    dynarray_ref() = delete;
#ifdef __CUDA_ARCH__
    // Use the whole object to avoid extra indirection on the GPU. Kernel launch performs the conversion
    dynarray_type array_;
#else
public:
    dynarray_ref(dynarray_type &a) :
        array_(a)
    {}

private:
    dynarray_type &array_;
#endif
public:
    using host_storage_type = typename dynarray_type::host_storage_type;

    using      value_type = typename dynarray_type::value_type;
    using difference_type = typename dynarray_type::difference_type;

    using coherence_policy_type = typename dynarray_type::coherence_policy_type;

    static constexpr unsigned dimensions = dynarray_type::dimensions;

    // Forward calls to the parent array
    __array_index__
    value_type &operator()(array_index_t idx)
    {
        return array_(idx);
    }

    __array_index__
    const value_type &operator()(array_index_t idx) const
    {
        return array_(idx);
    }

    __array_index__
    value_type &operator()(array_index_t idx1, array_index_t idx2)
    {
        return array_(idx1, idx2);
    }

    __array_index__
    const value_type &operator()(array_index_t idx1, array_index_t idx2) const
    {
        return array_(idx1, idx2);
    }

    __array_index__
    value_type &operator()(array_index_t idx1, array_index_t idx2, array_index_t idx3)
    {
        return array_(idx1, idx2, idx3);
    }

    __array_index__
    const value_type &operator()(array_index_t idx1, array_index_t idx2, array_index_t idx3) const
    {
        return array_(idx1, idx2, idx3);
    }

    inline __host__ __device__
    array_size_t get_elems() const
    {
        return array_.get_elems();
    }

    inline __host__ __device__
    array_size_t get_elems_align() const
    {
        return array_.get_elems_align();
    }

    template <unsigned Orig>
    __array_bounds__
    array_size_t get_dim() const
    {
        return array_.get_dim<Orig>();
    }

    template <unsigned Orig>
    __array_bounds__
    array_size_t get_dim_align() const
    {
        return array_.get_dim_align<Orig>();
    }

    __array_bounds__
    array_size_t get_dim(unsigned dim) const
    {
        return array_.get_dim(dim);
    }

    __array_bounds__
    array_size_t get_dim_align(unsigned dim) const
    {
        return array_.get_dim_align(dim);
    }

    void
    set_current_gpu(unsigned idx)
    {
        array_.set_current_gpu(idx);
    }

    coherence_policy_type &get_coherence_policy()
    {
        return array_.get_coherence_policy();
    }

    host_storage_type &get_host_storage()
    {
        return array_.get_host_storage();
    }
};

template <typename Array>
class dynarray_cref {
    using dynarray_type = Array;
    dynarray_cref() = delete;
#ifdef __CUDA_ARCH__
    // Use the whole object to avoid extra indirection on the GPU. Kernel launch performs the conversion
    dynarray_type array_;
#else
public:
    dynarray_cref(const dynarray_type &a) :
        array_(a)
    {}

private:
    const dynarray_type &array_;
#endif
public:
    using host_storage_type = typename dynarray_type::host_storage_type;

    using      value_type = typename dynarray_type::value_type;
    using difference_type = typename dynarray_type::difference_type;

    using coherence_policy_type = typename dynarray_type::coherence_policy_type;

    static constexpr unsigned dimensions = dynarray_type::dimensions;

    // Forward calls to constant methods only
    __array_index__
    const value_type &operator()(array_index_t idx) const
    {
        return array_(idx);
    }

    __array_index__
    const value_type &operator()(array_index_t idx1, array_index_t idx2) const
    {
        return array_(idx1, idx2);
    }

    __array_index__
    const value_type &operator()(array_index_t idx1, array_index_t idx2, array_index_t idx3) const
    {
        return array_(idx1, idx2, idx3);
    }

    inline __host__ __device__
    array_size_t get_elems() const
    {
        return array_.get_elems();
    }

    inline __host__ __device__
    array_size_t get_elems_align() const
    {
        return array_.get_elems_align();
    }

    template <unsigned Orig>
    __array_bounds__
    array_size_t get_dim() const
    {
        return array_.get_dim<Orig>();
    }

    template <unsigned Orig>
    __array_bounds__
    array_size_t get_dim_align() const
    {
        return array_.get_dim_align<Orig>();
    }

    __array_bounds__
    array_size_t get_dim(unsigned dim) const
    {
        return array_.get_dim(dim);
    }

    __array_bounds__
    array_size_t get_dim_align(unsigned dim) const
    {
        return array_.get_dim_align(dim);
    }
};

}

namespace std {
template <typename T, unsigned Dims, typename StorageType, class PartConf, template <typename> class CoherencePolicy>
struct is_convertible<cudarrays::dynarray<T, Dims, StorageType, PartConf, CoherencePolicy>,
                      cudarrays::dynarray_ref<cudarrays::dynarray<T, Dims, StorageType, PartConf, CoherencePolicy>>> {
    static constexpr bool value = true;

    using value_type = bool;
    using       type = std::integral_constant<bool, value>;

    operator bool()
    {
        return value;
    }
};

template <typename T, unsigned Dims, typename StorageType, class PartConf, template <typename> class CoherencePolicy>
struct is_convertible<cudarrays::dynarray<T, Dims, StorageType, PartConf, CoherencePolicy>,
                      cudarrays::dynarray_cref<cudarrays::dynarray<T, Dims, StorageType, PartConf, CoherencePolicy>>> {
    static constexpr bool value = true;

    using value_type = bool;
    using       type = std::integral_constant<bool, value>;

    operator bool()
    {
        return value;
    }
};

template <typename Array>
struct
is_const<cudarrays::dynarray_ref<Array>> {
    static constexpr bool value = false;

    using value_type = bool;
    using       type = std::integral_constant<bool, value>;

    operator bool()
    {
        return value;
    }
};

template <typename Array>
struct
is_const<cudarrays::dynarray_cref<Array>> {
    static constexpr bool value = true;

    using value_type = bool;
    using       type = std::integral_constant<bool, value>;

    operator bool()
    {
        return value;
    }
};

}

#endif // CUDARRAYS_DYNARRAY_HPP_

/* im:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
