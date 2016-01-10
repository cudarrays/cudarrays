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
#ifndef CUDARRAYS_DYNARRAY_VIEW_HPP_
#define CUDARRAYS_DYNARRAY_VIEW_HPP_

#include <cstdlib>
#include <cstring>

#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include "dynarray.hpp"

namespace cudarrays {

namespace detail {
template <typename T>
struct fake_shared_ptr {
    char bytes[sizeof(std::shared_ptr<T>)];

    __host__ __device__
    fake_shared_ptr() noexcept
    {
    }

    __host__
    fake_shared_ptr(T *) noexcept
    {
    }

    T *get() noexcept
    {
        return nullptr;
    }

    template <typename Deleter>
    void reset(T *, Deleter ) noexcept
    {
    }

    operator bool() const noexcept
    {
        return false;
    }

    T *operator->() noexcept
    {
        return nullptr;
    }

    const T *operator->() const noexcept
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
    dynarray_type &get_array() noexcept
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
    const dynarray_type &get_array() const noexcept
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

    dynarray_view_common(dynarray_type *a) noexcept :
        array_{a},
        array_gpu_{nullptr}
    {}

    __host__ __device__
    dynarray_view_common(const dynarray_view_common &a) noexcept :
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

    coherence_policy_type &get_coherence_policy() noexcept override final
    {
        return get_array().get_coherence_policy();
    }

    void set_current_gpu(unsigned idx) noexcept override final
    {
        array_gpu_ = arrays_gpu_.get()[idx];
        get_array().set_current_gpu(idx);
    }

    bool is_distributed() const noexcept override final
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

    void *host_addr() noexcept override final
    {
        return get_array().host_addr();
    }

    const void *host_addr() const noexcept override final
    {
        return get_array().host_addr();
    }

    size_t size() const noexcept override final
    {
        return get_array().size();
    }

    template <unsigned Dim>
    __array_bounds__
    array_size_t dim() const noexcept
    {
        return get_array().template dim<Dim>();
    }

    __array_bounds__
    array_size_t dim(unsigned dim) const
    {
        return get_array().dim(dim);
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
    explicit dynarray_view(dynarray_type *a) noexcept :
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
    explicit dynarray_cview(dynarray_type *a) noexcept :
        dynarray_view_common<Array>{a}
    {}

    __host__ __device__
    dynarray_cview(const dynarray_cview &a) noexcept :
        dynarray_view_common_type{a}
    {
    }

    __host__ __device__
    dynarray_cview(const dynarray_view_type &a) noexcept :
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

};

template <typename T,
          typename StorageType     = layout::rmo,
          typename Align           = noalign,
          typename PartConf        = automatic::none,
          typename CoherencePolicy = default_coherence,
          unsigned Dims            = array_traits<T>::dimensions,
          unsigned DynDims         = array_traits<T>::dynamic_dimensions>
inline
utils::enable_if_t<utils::is_greater(Dims, 1u) &&
                   utils::is_greater(DynDims, 0u),
                   dynarray_view<dynarray<T,
                                          StorageType, Align, PartConf, CoherencePolicy>>>
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
utils::enable_if_t<utils::is_greater(Dims, 1u) &&
                   utils::is_equal(DynDims, 0u),
                   dynarray_view<dynarray<T, StorageType, Align, PartConf, CoherencePolicy>>>
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
utils::enable_if_t<utils::is_equal(Dims, 1u) &&
                   utils::is_greater(DynDims, 0u),
                   dynarray_view<dynarray<T, layout::rmo, Align, PartConf, CoherencePolicy>>>
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
utils::enable_if_t<utils::is_equal(Dims, 1u) &&
                   utils::is_equal(DynDims, 0u),
                   dynarray_view<dynarray<T, layout::rmo, Align, PartConf, CoherencePolicy>>>
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

    operator bool() noexcept
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

    operator bool() noexcept
    {
        return value;
    }
};


}

#endif // CUDARRAYS_DYNARRAY_HPP_

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
