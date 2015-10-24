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
#include "utils.hpp"
#include "common.hpp"
#include "memory.hpp"
#include "storage.hpp"
#include "storage_impl.hpp"

#include "gpu.cuh"

#include "detail/dynarray/iterator.hpp"
#include "detail/dynarray/dim_iterator.hpp"
#include "detail/coherence/default.hpp"

namespace cudarrays {

namespace detail {
template <typename T>
struct fake_shared_ptr {
    char bytes[sizeof(std::shared_ptr<T>)];

    __host__ __device__
    fake_shared_ptr()
    {
    }

    __host__
    fake_shared_ptr(T *)
    {
    }

    T *get()
    {
        return nullptr;
    }

    template <typename Deleter>
    void reset(T *, Deleter )
    {
    }

    operator bool() const
    {
        return false;
    }

    T *operator->()
    {
        return nullptr;
    }

    const T *operator->() const
    {
        return nullptr;
    }

};
}

template <typename Array>
class dynarray_view_common :
    public coherent {
protected:
    using dynarray_type = Array;
private:
    // Instantiate a fake shared_ptr to avoid calling host code from the GPU
    template <typename T>
#ifdef __CUDA_ARCH__
    using shared_ptr_type = detail::fake_shared_ptr<T>;
#else
    using shared_ptr_type = std::shared_ptr<T>;
#endif

    shared_ptr_type<dynarray_type> array_;
    dynarray_type *array_gpu_;
    shared_ptr_type<dynarray_type *> arrays_gpu_;

public:
    __host__ __device__
    inline
    dynarray_type &get_array()
    {
        // Select the proper pointer for host and device code
#ifdef __CUDA_ARCH__
        return *array_gpu_;
#else
        return *array_.get();
#endif
    }

    __host__ __device__
    inline
    const dynarray_type &get_array() const
    {
        // Select the proper pointer for host and device code
#ifdef __CUDA_ARCH__
        return *array_gpu_;
#else
        return *array_.get();
#endif
    }

    void create_gpu_handles()
    {
        auto ptr = new dynarray_type*[system::gpu_count()];
        auto deleter = [](dynarray_type **obj)
                       {
                           for (auto gpu : utils::make_range(system::gpu_count()))
                               CUDA_CALL(cudaFree(obj[gpu]));
                           delete [] obj;
                       };
        arrays_gpu_.reset(ptr, deleter);
        for (auto gpu : utils::make_range(system::gpu_count())) {
            dynarray_type *tmp;
            CUDA_CALL(cudaSetDevice(gpu));
            CUDA_CALL(cudaMalloc((void **)&tmp, sizeof(dynarray_type)));

            array_->set_current_gpu(gpu);

            CUDA_CALL(cudaMemcpy(tmp, array_.get(), sizeof(dynarray_type), cudaMemcpyDefault));
            CUDA_CALL(cudaSetDevice(gpu));

            ptr[gpu] = tmp;
        }
    }

    dynarray_view_common(dynarray_type *a) :
        array_{a},
        array_gpu_{nullptr}
    {}

    __host__ __device__
    dynarray_view_common(const dynarray_view_common &a) :
#ifdef __CUDA_ARCH__
        array_gpu_{a.array_gpu_}
#else
        array_{a.array_},
        arrays_gpu_{a.arrays_gpu_}
#endif
    {
    }

public:
    using            value_type = typename dynarray_type::value_type;
    using       difference_type = typename dynarray_type::difference_type;

    using coherence_policy_type = typename dynarray_type::coherence_policy_type;

    static constexpr auto dimensions = dynarray_type::dimensions;

    coherence_policy_type &get_coherence_policy() override final
    {
        return get_array().get_coherence_policy();
    }

    void
    set_current_gpu(unsigned idx) override final
    {
        array_gpu_ = arrays_gpu_.get()[idx];
        get_array().set_current_gpu(idx);
    }

    bool is_distributed() const override final
    {
        return get_array().is_distributed();
    }

    template <unsigned DimsComp>
    bool distribute(const compute_mapping<DimsComp, dynarray_type::dimensions> &mapping)
    {
        auto ret = get_array().distribute(mapping);
        // Create GPU array copies lazily
        if (!arrays_gpu_) {
            create_gpu_handles();
        }
        return ret;
    }

    bool distribute(const std::vector<unsigned> &gpus) override final
    {
        auto ret = get_array().distribute(gpus);
        // Create GPU array copies lazily
        if (!arrays_gpu_) {
            create_gpu_handles();
        }
        return ret;
    }

    void to_device() override final
    {
        get_array().to_device();
    }

    void to_host() override final
    {
        get_array().to_host();
    }

    void *host_addr() override final
    {
        return get_array().host_addr();
    }

    const void *host_addr() const override final
    {
        return get_array().host_addr();
    }

    size_t size() const override final
    {
        return get_array().size();
    }

    //
    // Iterator interface
    //
    typename dynarray_type::const_value_iterator_type value_iterator() const
    {
        return get_array().value_iterator();
    }

    typename dynarray_type::const_dim_iterator_type dim_iterator() const
    {
        return get_array().dim_iterator();
    }
};

template <typename Array>
class dynarray_cview;

template <typename Array>
class dynarray_view :
    public dynarray_view_common<Array> {
    using dynarray_view_common_type = dynarray_view_common<Array>;
    using dynarray_type = typename dynarray_view_common_type::dynarray_type;

    using dynarray_cview_type = dynarray_cview<Array>;

    dynarray_view() = delete;
public:
    __host__
    explicit dynarray_view(dynarray_type *a) :
        dynarray_view_common<Array>{a}
    {}

    __host__ __device__
    dynarray_view(const dynarray_cview_type &a) = delete;

    using            value_type = typename dynarray_view_common_type::value_type;
    using       difference_type = typename dynarray_view_common_type::difference_type;

    using coherence_policy_type = typename dynarray_view_common_type::coherence_policy_type;

    // Forward calls to the parent array
    template <typename... T>
    __array_index__
    value_type &operator()(T &&... indices)
    {
        return this->get_array()(std::forward<T>(indices)...);
    }

    template <typename... T>
    __array_index__
    const value_type &operator()(T &&... indices) const
    {
        return this->get_array()(std::forward<T>(indices)...);
    }

    template <unsigned Dim>
    __array_bounds__
    array_size_t dim() const
    {
        return this->get_array().template dim<Dim>();
    }

    __array_bounds__
    array_size_t dim(unsigned dim) const
    {
        return this->get_array().dim(dim);
    }

    //
    // Iterator interface
    //
    typename dynarray_type::value_iterator_type value_iterator()
    {
        return this->get_array().value_iterator();
    }

    typename dynarray_type::dim_iterator_type dim_iterator()
    {
        return this->get_array().dim_iterator();
    }
};

template <typename Array>
class dynarray_cview :
    public dynarray_view_common<Array> {
    using dynarray_view_common_type = dynarray_view_common<Array>;
    using dynarray_type = typename dynarray_view_common_type::dynarray_type;

    using dynarray_view_type = dynarray_view<Array>;

    dynarray_cview() = delete;
public:
    __host__
    explicit dynarray_cview(dynarray_type *a) :
        dynarray_view_common<Array>{a}
    {}

    __host__ __device__
    dynarray_cview(const dynarray_cview &a) :
        dynarray_view_common_type{a}
    {
    }

    __host__ __device__
    dynarray_cview(const dynarray_view_type &a) :
        dynarray_view_common_type{a}
    {
    }

    using            value_type = typename dynarray_view_common_type::value_type;
    using       difference_type = typename dynarray_view_common_type::difference_type;

    using coherence_policy_type = typename dynarray_view_common_type::coherence_policy_type;

    // Forward calls to the parent array
    template <typename... T>
    __array_index__
    const value_type &operator()(T &&... indices) const
    {
        return this->get_array()(std::forward<T>(indices)...);
    }

    template <unsigned Dim>
    __array_bounds__
    array_size_t dim() const
    {
        return this->get_array().template dim<Dim>();
    }

    __array_bounds__
    array_size_t dim(unsigned dim) const
    {
        return this->get_array().dim(dim);
    }

};

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

    coherence_policy_type &get_coherence_policy() override final
    {
        return coherencePolicy_;
    }

    inline
    void *host_addr() override final
    {
        return host_.addr();
    }

    inline
    const void *host_addr() const override final
    {
        return host_.addr();
    }

    size_t size() const override final
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

template <typename T,
          typename StorageType     = layout::rmo,
          typename Align           = noalign,
          typename PartConf        = automatic::none,
          typename CoherencePolicy = default_coherence,
          unsigned Dims            = array_traits<T>::dimensions,
          unsigned DynDims         = array_traits<T>::dynamic_dimensions>
inline
utils::enable_if_t<
    utils::is_greater(Dims, 1u) &&
    utils::is_greater(DynDims, 0u),
    dynarray_view<dynarray<T, StorageType, Align, PartConf, CoherencePolicy>>
>
make_array(const extents<DynDims> &ext,
           const CoherencePolicy &coherence = CoherencePolicy{})
{
    using dynarray_type = dynarray<T, StorageType, Align, PartConf, CoherencePolicy>;
    auto *ret = dynarray_type::make(ext, coherence);
    return dynarray_view<dynarray_type>{ret};
}

template <typename T,
          typename StorageType     = layout::rmo,
          typename Align           = noalign,
          typename PartConf        = automatic::none,
          typename CoherencePolicy = default_coherence,
          unsigned Dims            = array_traits<T>::dimensions,
          unsigned DynDims         = array_traits<T>::dynamic_dimensions>
inline
utils::enable_if_t<
    utils::is_greater(Dims, 1u) &&
    utils::is_equal(DynDims, 0u),
    dynarray_view<dynarray<T, StorageType, Align, PartConf, CoherencePolicy>>
>
make_array(const CoherencePolicy &coherence = CoherencePolicy{})
{
    using dynarray_type = dynarray<T, StorageType, Align, PartConf, CoherencePolicy>;
    extents<0> ext;
    auto *ret = dynarray_type::make(ext, coherence);
    return dynarray_view<dynarray_type>{ret};
}

template <typename T,
          typename Align           = noalign,
          typename PartConf        = automatic::none,
          typename CoherencePolicy = default_coherence,
          unsigned Dims            = array_traits<T>::dimensions,
          unsigned DynDims         = array_traits<T>::dynamic_dimensions>
inline
utils::enable_if_t<
    utils::is_equal(Dims, 1u) &&
    utils::is_greater(DynDims, 0u),
    dynarray_view<dynarray<T, layout::rmo, Align, PartConf, CoherencePolicy>>
>
make_array(const extents<DynDims> &ext,
           const CoherencePolicy &coherence = CoherencePolicy{})
{
    using dynarray_type = dynarray<T, layout::rmo, Align, PartConf, CoherencePolicy>;
    auto *ret = dynarray_type::make(ext, coherence);
    return dynarray_view<dynarray_type>{ret};
}

template <typename T,
          typename Align           = noalign,
          typename PartConf        = automatic::none,
          typename CoherencePolicy = default_coherence,
          unsigned Dims            = array_traits<T>::dimensions,
          unsigned DynDims         = array_traits<T>::dynamic_dimensions>
inline
utils::enable_if_t<
    utils::is_equal(Dims, 1u) &&
    utils::is_equal(DynDims, 0u),
    dynarray_view<dynarray<T, layout::rmo, Align, PartConf, CoherencePolicy>>
>
make_array(const CoherencePolicy &coherence = CoherencePolicy{})
{
    using dynarray_type = dynarray<T, layout::rmo, Align, PartConf, CoherencePolicy>;
    extents<0> ext;
    auto *ret = dynarray_type::make(ext, coherence);
    return dynarray_view<dynarray_type>{ret};
}

}

namespace std {

template <typename Array>
struct
is_const<cudarrays::dynarray_view<Array>> {
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
is_const<cudarrays::dynarray_cview<Array>> {
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
