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

namespace cudarrays {

template <typename T,
          typename StorageType = layout::rmo,
          typename PartConf = automatic::none,
          template <typename> class CoherencePolicy = default_coherence>
class dynarray :
    public coherent {
public:
    using           array_type = T;
    using    array_traits_type = array_traits<array_type>;
    using  storage_traits_type = storage_traits<array_type, StorageType, PartConf>;

    using         permuter_type = typename storage_traits_type::permuter_type;

    using       difference_type = array_index_t;
    using            value_type = typename array_traits_type::value_type;
    using coherence_policy_type = CoherencePolicy<dynarray>;
    using          indexer_type = linearizer_hybrid<typename storage_traits_type::offsets_seq>;

    using     host_storage_type = host_storage<value_type>;

    using device_storage_type =
        dynarray_storage<value_type,
                         PartConf::final_impl,
                         storage_traits_type>;

    static constexpr auto dimensions = array_traits_type::dimensions;

    __host__
    explicit dynarray(const extents<array_traits_type::dynamic_dimensions> &extents,
                      const align_t &align = align_t{0, 0},
                      coherence_policy_type coherence = coherence_policy_type()) :
        coherencePolicy_(coherence),
        device_(permuter_type::reorder(array_traits_type::make_extents(extents)), align)
    {
        coherencePolicy_.bind(this);

        // Alloc host memory
        host_.alloc(device_.get_dim_manager().get_elems_align(), device_.get_dim_manager().offset());
        // TODO: Move this to a better place
        register_range(this->get_host_storage().base_addr(),
                       this->get_host_storage().size());
    }

    __host__
    explicit dynarray(const dynarray &a) :
        device_(a.device_),
        coherencePolicy_(a.coherencePolicy_)
    {
    }

    __host__
    dynarray &operator=(const dynarray &a)
    {
        if (&a != this) {
            device_          = a.device_;
            coherencePolicy_ = a.coherencePolicy_;
        }

        return *this;
    }

    __host__
    virtual ~dynarray()
    {
        unregister_range(this->get_host_storage().base_addr());
    }

    template <typename ...Idxs>
    __array_index__
    value_type &operator()(Idxs ...idxs)
    {
        return access_element_helper<SEQ_GEN_INC(unsigned(sizeof...(Idxs)))>::at(device_, host_, array_index_t(idxs)...);
    }

    template <typename ...Idxs>
    __array_index__
    const value_type &operator()(Idxs ...idxs) const
    {
        return access_element_helper<SEQ_GEN_INC(unsigned(sizeof...(Idxs)))>::at_const(device_, host_, array_index_t(idxs)...);
    }

    template <unsigned DimsComp>
    __host__ bool
    distribute(compute_mapping<DimsComp, array_traits_type::dimensions> mapping)
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
    using       iterator = myiterator<dynarray, false>;
    using const_iterator = myiterator<dynarray, true>;

    using       reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    iterator begin()
    {
        array_index_t dims[array_traits_type::dimensions];
        std::fill(dims, dims + array_traits_type::dimensions, 0);
        return iterator(*this, dims);
    }

    const_iterator begin() const
    {
        return cbegin();
    }

    const_iterator cbegin() const
    {
        array_index_t dims[array_traits_type::dimensions];
        std::fill(dims, dims + array_traits_type::dimensions, 0);
        return const_iterator(*this, dims);
    }

    reverse_iterator rbegin()
    {
        array_index_t dims[array_traits_type::dimensions];
        for (unsigned i = 0; i < array_traits_type::dimensions; ++i) {
            dims[i] = this->dim(i) - 1;
        }
        return reverse_iterator(iterator(*this, dims));
    }

    iterator end()
    {
        array_index_t dims[array_traits_type::dimensions];
        dims[0] = this->dim(0);
        if (array_traits_type::dimensions > 1) {
            std::fill(dims + 1, dims + array_traits_type::dimensions, 0);
        }
        return iterator(*this, dims);
    }

    const_iterator end() const
    {
        return cend();
    }

    const_iterator cend() const
    {
        array_index_t dims[array_traits_type::dimensions];
        dims[0] = this->dim(0);
        if (array_traits_type::dimensions > 1) {
            std::fill(dims + 1, dims + array_traits_type::dimensions, 0);
        }
        return const_iterator(*this, dims);
    }

    reverse_iterator rend()
    {
        array_index_t dims[array_traits_type::dimensions];
        dims[0] = -1;
        for (unsigned i = 0; i < array_traits_type::dimensions; ++i) {
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
        array_index_t dims[array_traits_type::dimensions];
        dims[0] = -1;
        for (unsigned i = 0; i < array_traits_type::dimensions; ++i) {
            dims[i] = this->dim(i) - 1;
        }
        return const_reverse_iterator(const_iterator(*this, dims));
    }

    friend myiterator<dynarray, false>;
    friend myiterator<dynarray, true>;

private:
    template <typename Selector>
    struct access_element_helper;

    template <unsigned ...Vals>
    struct access_element_helper<SEQ_WITH_TYPE(unsigned, Vals...)> {
        template <typename... Idxs>
        __array_index__
        static
        value_type &at(device_storage_type &device,
                       host_storage_type   &host,
                       Idxs ...idxs)
        {
#ifdef __CUDA_ARCH__
            return device.access_pos(permuter_type::template select<Vals>(idxs...)...);
#else
            auto idx = indexer_type::access_pos(device.get_dim_manager().get_offs_align(),
                                                permuter_type::template select<Vals>(idxs...)...);
            return host.addr()[idx];
#endif
        }

        template <typename... Idxs>
        __array_index__
        static
        const value_type &at_const(const device_storage_type &device,
                                   const host_storage_type   &host,
                                   Idxs ...idxs)
        {
#ifdef __CUDA_ARCH__
            return device.access_pos(permuter_type::template select<Vals>(idxs...)...);
#else
            auto idx = indexer_type::access_pos(device.get_dim_manager().get_offs_align(),
                                                permuter_type::template select<Vals>(idxs...)...);
            return host.addr()[idx];
#endif
        }

    };

    coherence_policy_type coherencePolicy_;
    device_storage_type   device_;
    host_storage_type     host_;
};

template <typename Array>
class dynarray_ref {
    using dynarray_type = Array;
    dynarray_ref() = delete;
private:
#ifdef __CUDA_ARCH__
    // Use the whole object to avoid extra indirection on the GPU. Kernel launch performs the conversion
    // TODO: check if extra indirection introduces overhead
    dynarray_type array_;
#else
    dynarray_type &array_;
#endif
public:
    __host__
    dynarray_ref(dynarray_type &a) :
        array_(a)
    {}

    using     host_storage_type = typename dynarray_type::host_storage_type;

    using            value_type = typename dynarray_type::value_type;
    using       difference_type = typename dynarray_type::difference_type;

    using coherence_policy_type = typename dynarray_type::coherence_policy_type;

    // Forward calls to the parent array
    template <typename... T>
    __array_index__
    value_type &operator()(T &&... indices)
    {
        return array_(std::forward<T>(indices)...);
    }

    template <typename... T>
    __array_index__
    const value_type &operator()(T &&... indices) const
    {
        return array_(std::forward<T>(indices)...);
    }

    template <unsigned Dim>
    __array_bounds__
    array_size_t dim() const
    {
        return array_.dim<Dim>();
    }

    __array_bounds__
    array_size_t dim(unsigned dim) const
    {
        return array_.dim(dim);
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
private:
#ifdef __CUDA_ARCH__
    // Use the whole object to avoid extra indirection on the GPU. Kernel launch performs the conversion
    // TODO: check if extra indirection introduces overhead
    const dynarray_type array_;
#else
    const dynarray_type &array_;
#endif
public:
    __host__
    dynarray_cref(const dynarray_type &a) :
        array_(a)
    {}

    using host_storage_type = typename dynarray_type::host_storage_type;

    using      value_type = typename dynarray_type::value_type;
    using difference_type = typename dynarray_type::difference_type;

    using coherence_policy_type = typename dynarray_type::coherence_policy_type;

    // Forward calls to constant methods only
    template <typename... T>
    __array_index__
    const value_type &operator()(T &&... indices) const
    {
        return array_(std::forward<T>(indices)...);
    }

    template <unsigned Dim>
    __array_bounds__
    array_size_t dim() const
    {
        return array_.dim<Dim>();
    }

    __array_bounds__
    array_size_t dim(unsigned dim) const
    {
        return array_.dim(dim);
    }
};

template <typename Array>
__host__
dynarray_ref<Array> make_ref(Array &a)
{
    return dynarray_ref<Array>(a);
}

template <typename Array>
__host__
dynarray_cref<Array> make_cref(const Array &a)
{
    return dynarray_cref<Array>(a);
}


}

namespace std {
template <typename T, typename StorageType, class PartConf, template <typename> class CoherencePolicy>
struct is_convertible<cudarrays::dynarray<T, StorageType, PartConf, CoherencePolicy>,
                      cudarrays::dynarray_ref<cudarrays::dynarray<T, StorageType, PartConf, CoherencePolicy>>> {
    static constexpr bool value = true;

    using value_type = bool;
    using       type = std::integral_constant<bool, value>;

    operator bool()
    {
        return value;
    }
};

template <typename T, typename StorageType, class PartConf, template <typename> class CoherencePolicy>
struct is_convertible<cudarrays::dynarray<T, StorageType, PartConf, CoherencePolicy>,
                      cudarrays::dynarray_cref<cudarrays::dynarray<T, StorageType, PartConf, CoherencePolicy>>> {
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

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
