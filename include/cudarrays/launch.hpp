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
#ifndef CUDARRAYS_LAUNCH_HPP_
#define CUDARRAYS_LAUNCH_HPP_

#include <fstream>
#include <future>
#include <sstream>
#include <vector>

#include <cxxabi.h>

#include "common.hpp"
// #include "trace.hpp"
#include "dynarray.hpp"

namespace cudarrays {

void update_gpu_global_grid(dim3);
void update_gpu_offset(dim3);

template <typename T>
static inline void *mycudaAddressOf(T &val)
{
    auto ret = (void *)&(const_cast<char &>(reinterpret_cast<const volatile char &>(val)));
    return ret;
}

#if 0
// #ifdef CUDARRAYS_TRACE
std::ofstream &operator<<(std::ofstream &os, const tracer<WARP> &t)
{
    cudaError_t err;
    err = cudaMemcpy(t.traces_,
                     t.tracesStartDev_, t.ntraces() * sizeof(long long), cudaMemcpyDeviceToHost);
    ASSERT(err == cudaSuccess);
    err = cudaMemcpy(t.traces_ + t.ntraces(),
                     t.tracesEndDev_, t.ntraces() * sizeof(long long), cudaMemcpyDeviceToHost);
    ASSERT(err == cudaSuccess);
    unsigned long ntraces = 0;
    // Count # of valid traces
    for (unsigned long long i = 0; i < t.ntraces(); ++i) {
        if (t.traces_[i] != 0 || t.traces_[i + t.ntraces()] != 0)
            ++ntraces;
    }
    os << ntraces << "\n";
    for (unsigned long long i = 0; i < t.ntraces(); ++i) {
        if (t.traces_[i] != 0 || t.traces_[i + t.ntraces()] != 0)
            os << t.traces_[i] << "," << t.traces_[i + t.ntraces()] << "\n";
    }

    return os;
}

std::ofstream &operator<<(std::ofstream &os, const tracer<BLOCK> &t)
{
    os << t.ntraces() << "\n";

    for (unsigned long long i = 0; i < t.ntraces(); ++i) {
        os << t.traces_[i] << "," << t.traces_[i] << "\n";
    }

    return os;
}

unsigned
get_dim3_ndims(dim3 d)
{
    unsigned ndims = 0;
    if (d.x > 1)
        ++ndims;
    if (d.y > 1)
        ++ndims;
    if (d.z > 1)
        ++ndims;

    return ndims;
}
#endif

struct cuda_conf {
    dim3 grid;
    dim3 block;
    size_t shared;
    cudaStream_t stream;

    cuda_conf(dim3 _grid, dim3 _block,
              size_t _shared = 0,
              cudaStream_t _stream = 0) :
        grid(_grid),
        block(_block),
        shared(_shared),
        stream(_stream)
    {
    }
};

using coherence_info = std::pair<coherent *, bool>;

template <typename To>
struct check_arg_type {
    template <typename From>
    static constexpr bool against(const From &)
    {
        static_assert(std::is_convertible<From, To>::value, "Incompatible types");
        return true;
    }
};

template <typename To>
struct check_arg_type <To *> {
    template <typename From>
    static constexpr bool against(From * const&)
    {
        static_assert(std::is_convertible<From *, To *>::value, "Incompatible types");
        return true;
    }
};

template <typename... Args>
struct argument_manager {
    static thread_local void *CurrentKernel;
    static thread_local void *Params[sizeof...(Args)];
    static thread_local std::vector<coherence_info> CoherentParams;

    template <typename T>
    static void
    set_coherent_arg(T &&/*arg*/, bool /*Const*/)
    {
        // Not coherent. Do nothing
    }

    template <typename Array>
    static void
    set_coherent_arg(dynarray_view<Array> &arg, bool Const)
    {
        // Store dynarray arguments in the vector of coherent objects
        CoherentParams.push_back(coherence_info{&arg, Const});
    }

    template <typename Array>
    static void
    set_coherent_arg(dynarray_cview<Array> &arg, bool Const)
    {
        // Store dynarray arguments in the vector of coherent objects
        CoherentParams.push_back(coherence_info{&arg, Const});
    }

    template <unsigned Idx, typename T>
    static void
    parse_args(T &arg)
    {
        using Type = typename std::tuple_element<Idx,
                                                 std::tuple<Args...>>::type;

        bool Const = std::is_const<Type>::value;

        check_arg_type<Type>::against(arg);
        set_coherent_arg(arg, Const);

        Params[Idx] = mycudaAddressOf(arg);
    }

    template <unsigned Idx, typename T, typename... ArgsPassed>
    static void
    parse_args(T &arg, ArgsPassed&... args2)
    {
        parse_args<Idx, T>(arg); // Process current argument

        parse_args<Idx + 1, ArgsPassed...>(args2...); // Process next arguments
    }

    template <typename... ArgsPassed>
    static void
    parse_args(ArgsPassed&... args2)
    {
        CoherentParams.clear();

        parse_args<0, ArgsPassed...>(args2...);
    }

#if 0 // TODO: implement
    template <typename T>
    static void
    partition_array(dim3 grid, T &arg, const compiler_array_info &info)
    {
        // TODO: Implement
        unsigned ndims = get_dim3_ndims(grid);

        //fprintf(stderr, "Dims: %u\n", info.ndims_);
        unsigned i = 0;
        for (auto &dimInfo : info.dimsInfo_) {
            //fprintf(stderr, "\t%u: ", i++);

            for (auto &dim : dimInfo.gridDims_) {
                //fprintf(stderr, "%u, ", dim);
            }

            //fprintf(stderr, "\n");
        }
    }

    template <unsigned Idx, typename T>
    static void
    compiler_layout_args(dim3 grid, T &arg)
    {
        const auto *info = compiler_get_array_info(CurrentKernel, Idx);
        if (info)
            partition_array(grid, arg, *info);
    }

    template <unsigned Idx, typename T, typename... ArgsPassed>
    static void
    compiler_layout_args(dim3 grid, T &arg, ArgsPassed&... args2)
    {
        const auto *info = compiler_get_array_info(CurrentKernel, Idx);
        if (info)
            partition_array(grid, arg, *info);

        compiler_layout_args<Idx + 1, ArgsPassed...>(grid, args2...);
    }

    template <typename... ArgsPassed>
    static void
    compiler_layout_args(dim3 grid, ArgsPassed&... args2)
    {
        compiler_layout_args<0, ArgsPassed...>(grid, args2...);
    }
#endif
};

template <typename... Args>
thread_local
void *argument_manager<Args...>::CurrentKernel;
template <typename... Args>
thread_local
void *argument_manager<Args...>::Params[sizeof...(Args)];
template <typename... Args>
thread_local
std::vector<coherence_info> argument_manager<Args...>::CoherentParams;

static void init_streams(unsigned gpus)
{
    if (EventsBegin.size() < gpus) {
        // Only create the streams/events for the GPUs not yet allocated
        for (unsigned i = EventsBegin.size(); i < gpus; ++i) {
            // Preallocate events for kernel execution
            cudaEvent_t begin, end;
            cudaError_t err;

            err = cudaSetDevice(i);
            ASSERT(err == cudaSuccess);

            err = cudaEventCreate(&begin);
            ASSERT(err == cudaSuccess);
            err = cudaEventCreate(&end);
            ASSERT(err == cudaSuccess);

            EventsBegin.push_back(begin);
            EventsEnd.push_back(end);
        }
    }
}

template <unsigned Dims, typename R, typename... Args>
class launcher_common {
    static constexpr unsigned DimIdxX = 2 - (3 - Dims);
    static constexpr unsigned DimIdxY = 1 - (3 - Dims);
    static constexpr unsigned DimIdxZ = 0 - (3 - Dims);

    R(&f_)(Args...);
    const char *funName_;
    cuda_conf conf_;
    unsigned gpus_;
    bool transposeXY_;
    std::vector<unsigned> gpuGrid_;

protected:
    std::vector<coherence_info> coherentParams_;
    std::vector<unsigned> activeGPUs_;

private:
    static void
    set_args_current_gpu(const std::vector<coherence_info> &objects, unsigned gpu)
    {
        // Set current GPU in all coherent objects
        for (auto object : objects) {
            object.first->set_current_gpu(gpu);
        }
    }

    static void
    release_args(const std::vector<coherence_info> &objects, const std::vector<unsigned> &gpus)
    {
        for (auto object : objects) {
            object.first->get_coherence_policy().release(gpus, object.second);
        }
    }

    static void
    acquire_args(const std::vector<coherence_info> &objects)
    {
        for (auto object : objects) {
            object.first->get_coherence_policy().acquire();
        }
    }

protected:
    using my_arguments = argument_manager<Args...>;

    launcher_common(R(&f)(Args...), const char *funName, const cuda_conf &conf, compute_conf<Dims> gpuConf, bool transposeXY) :
        f_(f),
        funName_(funName),
        conf_(conf),
        transposeXY_(transposeXY)
    {
        if (gpuConf.procs > system::gpu_count()) {
            printf("WARNING: # requested GPUs > # installed GPUs\n");
        }
        // TODO: use peer GPUS for those array implementations that require remote access
        if (gpuConf.procs == 0) {
            gpus_ = system::gpu_count();
        } else {
            gpus_ = std::min(gpuConf.procs, system::gpu_count());
        }

        init_streams(gpus_);

        unsigned partDims = utils::count(gpuConf.info, true);

        auto factorsGPUs = utils::get_factors(gpus_);
        if (gpus_ == 1) {
            factorsGPUs.push_back(1);
        }
        utils::sort(factorsGPUs, std::greater<unsigned>());

        unsigned j = 0;
        for (unsigned i : utils::make_range(Dims)) {
            unsigned partition = 1;
            if (gpuConf.info[i]) {
                auto pos = factorsGPUs.begin() + j;
                size_t inc = (j == 0)? factorsGPUs.size() - partDims + 1: 1;

                partition = std::accumulate(pos, pos + inc, 1, std::multiplies<unsigned>());
                j += inc;
            }

            gpuGrid_.push_back(partition);
        }
    }

    std::tuple<dim3, dim3> compute_tiles()
    {
        dim3 step{0, 0, 0};
        dim3 grid{1, 1, 1};

        dim3 total_grid = conf_.grid;

        if (Dims > 2) {
            grid.z = utils::div_ceil(total_grid.z, gpuGrid_[DimIdxZ]);
            if (gpuGrid_[DimIdxZ] > 1) {
                step.z = grid.z;
            }
        }
        if (Dims > 1) {
            grid.y = utils::div_ceil(total_grid.y, gpuGrid_[DimIdxY]);
            if (gpuGrid_[DimIdxY] > 1) {
                step.y = grid.y;
            }
        }

        grid.x = utils::div_ceil(total_grid.x, gpuGrid_[DimIdxX]);
        if (gpuGrid_[DimIdxX] > 1) {
            step.x = grid.x;
        }

        return std::make_tuple(step, grid);
    }

    template <typename... ArgsPassed>
    bool
    execute(ArgsPassed &&...args2)
    {
        static_assert(sizeof...(Args) == sizeof...(ArgsPassed),
                      "Wrong number of passed arguments to the kernel");
        dim3 total_grid = conf_.grid;

        DEBUG("GPUS: %u", gpus_);
        DEBUG("orig: %zd %zd %zd", total_grid.z, total_grid.y, total_grid.x);

        dim3 step;
        dim3 grid;

        std::tie(step, grid) = compute_tiles();

        DEBUG("step: %zd %zd %zd", step.z, step.y, step.x);

        if (CUDARRAYS_COMPILER_INFO) {
            // TODO: implement
            // my_arguments::CurrentKernel = (void *) &f_;
            // my_arguments::compiler_layout_args(total_grid, VALID_VAL_MAX(arg));
        }

        my_arguments::parse_args(std::forward<ArgsPassed>(args2)...);

        coherentParams_ = my_arguments::CoherentParams;

        activeGPUs_.clear();

        for (unsigned i : utils::make_range(gpus_)) {
            cudaError_t err = cudaSetDevice(i);
            ASSERT(err == cudaSuccess);
            err = cudaDeviceSynchronize();
            ASSERT(err == cudaSuccess);
        }

        unsigned gpu = 0;
        dim3 off{0, 0, 0};
        for (unsigned i : utils::make_range(Dims > 2? gpuGrid_[DimIdxZ]: 1)) {
            off.y = 0;
            for (unsigned j : utils::make_range(Dims > 1? gpuGrid_[DimIdxY]: 1)) {
                off.x = 0;
                for (unsigned k : utils::make_range(gpuGrid_[DimIdxX])) {
                    dim3 local = grid;
                    DEBUG("local: %zd %zd %zd", local.z, local.y, local.x);
                    DEBUG("off: %zd %zd %zd", off.z, off.y, off.x);

                    if (off.z + step.z > total_grid.z) {
                        if (off.z <= total_grid.z)
                            local.z = total_grid.z - off.z;
                        else
                            local.z = 0;
                    } else if (off.y + step.y > total_grid.y) {
                        if (off.y <= total_grid.y)
                            local.y = total_grid.y - off.y;
                        else
                            local.y = 0;
                    } else if (off.x + step.x > total_grid.x) {
                        if (off.x <= total_grid.x)
                            local.x = total_grid.x - off.x;
                        else
                            local.x = 0;
                    }

                    if (local.z > 0 && local.y > 0 && local.x > 0) {
                        cudaError_t err = cudaSetDevice(gpu);
                        ASSERT(err == cudaSuccess);

                        DEBUG("gpu %zd: %zd %zd %zd", gpu, i, j, k);
                        DEBUG("grid: %zd %zd %zd", local.z, local.y, local.x);

                        update_gpu_global_grid(total_grid);
                        update_gpu_offset(off);
                    }

                    activeGPUs_.push_back(gpu);

                    off.x += step.x;
                    ++gpu;
                }
                off.y += step.y;
            }
            off.z += step.z;
        }

        release_args(coherentParams_, activeGPUs_);

        gpu = 0;
        off.z = 0;
        for (unsigned i : utils::make_range(Dims > 2? gpuGrid_[DimIdxZ]: 1)) {
            off.y = 0;
            for (unsigned j : utils::make_range(Dims > 1? gpuGrid_[DimIdxY]: 1)) {
                off.x = 0;
                for (unsigned k : utils::make_range(gpuGrid_[DimIdxX])) {
                    dim3 local = grid;
                    DEBUG("local: %zd %zd %zd", local.z, local.y, local.x);
                    DEBUG("off: %zd %zd %zd", off.z, off.y, off.x);

                    if (off.z + step.z > total_grid.z) {
                        if (off.z <= total_grid.z)
                            local.z = total_grid.z - off.z;
                        else
                            local.z = 0;
                    } else if (off.y + step.y > total_grid.y) {
                        if (off.y <= total_grid.y)
                            local.y = total_grid.y - off.y;
                        else
                            local.y = 0;
                    } else if (off.x + step.x > total_grid.x) {
                        if (off.x <= total_grid.x)
                            local.x = total_grid.x - off.x;
                        else
                            local.x = 0;
                    }

                    if (local.z > 0 && local.y > 0 && local.x > 0) {
                        cudaError_t err = cudaSetDevice(gpu);
                        ASSERT(err == cudaSuccess);
                        DEBUG("gpu %zd: %zd %zd %zd", gpu, i, j, k);
                        DEBUG("grid: %zd %zd %zd", local.z, local.y, local.x);

                        if (transposeXY_) {
                            std::swap(local.x, local.y);
                        }

                        // Set arguments
                        set_args_current_gpu(coherentParams_, gpu);

                        err = cudaEventRecord(EventsBegin[gpu], StreamsIn[gpu]);
                        ASSERT(err == cudaSuccess);

#if 0
// #ifdef CUDARRAYS_TRACE
                        if (funName_.size() == 0) {
                            FATAL("You must provide a function name");
                            abort();
                        }
                        static int kernelCount = 0;
                        std::stringstream path;
                        path << "trace-";
                        path << funName_ << "-";
                        path << kernelCount++;

                        std::ofstream out(path.str());
                        TRACER_DRIVER driver(local, block);
#endif
                        err = cudaLaunchKernel((const char *)(const void *) f_,
                                               local, conf_.block,
                                               my_arguments::Params,
                                               conf_.shared,
                                               StreamsIn[gpu]);
                        ASSERT(err == cudaSuccess);

                        err = cudaEventRecord(EventsEnd[gpu], StreamsIn[gpu]);
                        ASSERT(err == cudaSuccess);
                    }
                    off.x += step.x;
                    ++gpu;
                }
                off.y += step.y;
            }
            off.z += step.z;
        }

        return true;
    }

public:
    static bool wait(const std::vector<unsigned> &activeGPUs, const std::vector<coherence_info> &coherentParams)
    {
        for (unsigned i : activeGPUs) {
            cudaError_t err = cudaEventSynchronize(EventsEnd[i]);
            ASSERT(err == cudaSuccess);
        }

        acquire_args(coherentParams);

        return true;
    }
};

template <unsigned Dims, typename R, typename... Args>
class launcher :
    public launcher_common<Dims, R, Args...> {
    using common_parent = launcher_common<Dims, R, Args...>;
public:
    template <typename... Args2>
    launcher(R(&f)(Args...), const char *funName, const cuda_conf &conf, compute_conf<Dims> gpuConf, bool transposeXY) :
        common_parent(f, funName, conf, gpuConf, transposeXY)
    {
    }

    template <typename... ArgsPassed>
    bool
    operator()(ArgsPassed &&...args2)
    {
        auto ret = this->execute(std::forward<ArgsPassed>(args2)...);
        if (ret) {
            auto gpus = this->activeGPUs_;
            auto params = this->coherentParams_;
            ret = common_parent::wait(gpus, params);
        }

        return ret;
    }
};

template <unsigned Dims, typename R, typename... Args>
class launcher_async :
    public launcher_common<Dims, R, Args...> {
    using common_parent = launcher_common<Dims, R, Args...>;
public:
    launcher_async(R(&f)(Args...), const char *funName, const cuda_conf &conf, compute_conf<Dims> gpuConf, bool transposeXY) :
        common_parent(f, funName, conf, gpuConf, transposeXY)
    {
    }

    template <typename... ArgsPassed>
    std::future<bool>
    operator()(ArgsPassed &&...args2)
    {
        auto ret = this->execute(std::forward<ArgsPassed>(args2)...);
        if (ret) {
            auto gpus = this->activeGPUs_;
            auto params = this->coherentParams_;
            return std::async(std::launch::async, [=]() { return common_parent::wait(gpus, params); });
        } else {
            auto task = std::packaged_task<bool()>([]() { return false; });
            task();
            return task.get_future();
        }
    }
};

template <unsigned DimsComp, typename R, typename... Args>
launcher<DimsComp, R, Args...>
launch_(const char *name,
        R(&f)(Args...),
        const cuda_conf &conf,
        compute_conf<DimsComp> gpuConf = {compute::none, 1},
        bool transposeXY = false)
{
    // LIBRARY ENTRY POINT
    cudarrays_entry_point();

    return launcher<DimsComp, R, Args...>(f, name, conf, gpuConf, transposeXY);
}

template <unsigned DimsComp, typename R, typename... Args>
launcher_async<DimsComp, R, Args...>
launch_async_(const char *name,
              R(&f)(Args...),
              const cuda_conf &conf,
              compute_conf<DimsComp> gpuConf = {compute::none, 1},
              bool transposeXY = false)
{
    // LIBRARY ENTRY POINT
    cudarrays_entry_point();

    return launcher_async<DimsComp, R, Args...>(f, name, conf, gpuConf, transposeXY);
}

#define launch(f,...)       launch_(#f, f,##__VA_ARGS__)
#define launch_async(f,...) launch_async_(#f, f,##__VA_ARGS__)

template <typename Array, typename F, unsigned Dim = Array::dimensions>
struct unwrap_extents {
    template <typename... Idx>
    inline static
    void call(Array &a, const F &f, Idx && ... idx)
    {
        constexpr unsigned current_dim = Array::dimensions - Dim;

        for (array_size_t i = 0; i < a.template dim<current_dim>(); ++i) {
            unwrap_extents<Array, F, Dim - 1>::call(a, f,
                                                    std::forward<Idx>(idx)..., i);
        }
    }
};

template <typename Array, typename F>
struct unwrap_extents<Array, F, 0> {
    template <typename... Idx>
    inline static
    void call(Array &a, const F &f, Idx && ... idx)
    {
        a(std::forward<Idx>(idx)...) = f(std::forward<Idx>(idx)...);
    }
};

template <typename Array, typename F>
inline static
void
for_each_element(Array & a, const F &f)
{
    unwrap_extents<Array, F>::call(a, f);
}

}

#endif // CUDARRAYS_LAUNCH_HPP_

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
