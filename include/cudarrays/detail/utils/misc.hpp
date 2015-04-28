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
 * to use, copy, modify, seq_merge, publish, distribute, sublicense, and/or sell
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
#ifndef CUDARRAYS_DETAIL_UTILS_MISC_HPP_
#define CUDARRAYS_DETAIL_UTILS_MISC_HPP_

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <vector>

#include "../../config.hpp"

namespace utils {

namespace mpl {

template <typename T, T... Values>
struct sequence {
    using type = T;

    static constexpr size_t size = sizeof...(Values);

    static constexpr std::array<T, sizeof...(Values)> as_array()
    {
        return {Values...};
    }
};

template <typename T>
static T
deduce(T...) {}

#define SEQ(v,...) utils::mpl::sequence<decltype(utils::mpl::deduce(v)),v,##__VA_ARGS__>

#define SEQ_EMPTY(t) utils::mpl::sequence<t>

namespace detail {

template <typename T, class S>
struct seq_size_detail;

template <typename T, T... Vals>
struct seq_size_detail<T, sequence<T, Vals...>> {
    static constexpr size_t value = sizeof...(Vals);
};

template <typename S>
struct seq_size {
    static constexpr auto value = seq_size_detail<typename S::type, S>::value;
};

template <typename S>
struct seq_type {
    using type = typename S::type;
};

template <typename T, class O>
struct seq_wrap;

template <typename T, T... Values, template <T...> class O>
struct seq_wrap<T, O<Values...>> {
    using type = sequence<T, Values...>;
};

template <typename T, typename S, typename O>
struct seq_unwrap_detail;

template <typename T, T... Values, template <T...> class O>
struct seq_unwrap_detail<T, sequence<T, Values...>, O<>> {
    using type = O<Values...>;
};

template <typename S,
          typename O>
struct seq_unwrap {
    using type = typename seq_unwrap_detail<typename seq_type<S>::type, S, O>::type;
};

template <typename T, typename S1, typename S2>
struct seq_merge_detail;

template <typename T, T... Values1, T... Values2>
struct seq_merge_detail<T, sequence<T, Values1...>, sequence<T, Values2...>> {
    using type = sequence<T, Values1..., Values2...>;
};

template <class V1, class V2>
struct seq_merge {
    static_assert(std::is_same<typename seq_type<V1>::type,
                               typename seq_type<V2>::type>::value == true,
                  "Sequence types must match");
    using type = typename seq_merge_detail<typename seq_type<V1>::type, V1, V2>::type;
};

template <typename T, unsigned Size, T FillValue, T... Generated>
struct seq_gen_fill_detail {
    using type = typename seq_gen_fill_detail<T, Size - 1, FillValue, FillValue, Generated...>::type;
};

template <typename T, T FillValue, T... Generated>
struct seq_gen_fill_detail<T, 0, FillValue, Generated...> {
    using type = sequence<T, Generated...>;
};

template <typename T, T FillValue, unsigned Size>
struct sec_gen_fill {
    using type = typename seq_gen_fill_detail<T, Size, FillValue>::type;
};

template <typename S, typename seq_type<S>::type... FillValues>
struct seq_prepend {
    using append_values_type = sequence<typename seq_type<S>::type, FillValues...>;
    using type = typename seq_merge<append_values_type, S>::type;
};

template <typename S, typename seq_type<S>::type... FillValues>
struct seq_append {
    using append_values_type = sequence<typename seq_type<S>::type, FillValues...>;
    using type = typename seq_merge<S, append_values_type>::type;
};

template <typename T, typename S, typename D>
struct seq_reverse_detail;

template <typename T, T Current, T... Vals, T... Vals2>
struct seq_reverse_detail<T, sequence<T, Current, Vals...>, sequence<T, Vals2...>> {
    using type = typename seq_reverse_detail<T, sequence<T, Vals...>, sequence<T, Current, Vals2...>>::type;
};

template <typename T, T Current, T... Vals2>
struct seq_reverse_detail<T, sequence<T, Current>, sequence<T, Vals2...>> {
    using type = sequence<T, Current, Vals2...>;
};

template <typename S>
struct seq_reverse {
    using my_seq_type = typename seq_type<S>::type;
    using type = typename seq_reverse_detail<my_seq_type, S, sequence<my_seq_type>>::type;
};

template <typename T, unsigned Size, T Current, T... Generated>
struct seq_gen_inc_detail {
    using type = typename seq_gen_inc_detail<T, Size - 1, Current + 1, Generated..., Current>::type;
};

template <typename T, T Current, T... Generated>
struct seq_gen_inc_detail<T, 0, Current, Generated...> {
    using type = sequence<T, Generated...>;
};

template <typename T, T Size>
struct seq_gen_inc {
    using type = typename seq_gen_inc_detail<T, Size, 0>::type;
};

template <typename T, unsigned Size, T Current, T... Generated>
struct seq_gen_dec_detail {
    using type = typename seq_gen_dec_detail<T, Size - 1, Current - 1, Current, Generated...>::type;
};

template <typename T, T Current, T... Generated>
struct seq_gen_dec_detail<T, 1, Current, Generated...> {
    using type = sequence<T, Generated...>;
};

template <typename T, unsigned Size>
struct seq_gen_dec {
    using type = typename seq_gen_dec_detail<T, Size, Size - 1>::type;
};

template <size_t N, typename T, typename S>
struct seq_at_detail;

template <size_t N, typename T, T Current, T... Values>
struct seq_at_detail<N, T, sequence<T, Current, Values...>> {
    static constexpr T value = seq_at_detail<N - 1, T, sequence<T, Values...>>::value;
};

template <typename T, T Current, T... Values>
struct seq_at_detail<0, T, sequence<T, Current, Values...>> {
    static constexpr T value = Current;
};

template <typename S, size_t N>
struct seq_at {
    static_assert(N < seq_size<S>::value, "Out of bounds");
    static constexpr auto value = seq_at_detail<N, typename seq_type<S>::type, S>::value;
};

template <bool ValidIndex, size_t N, typename T, typename S, T Value>
struct seq_at_or_detail {
    static constexpr auto value = seq_at<S, N>::value;
};

template <size_t N, typename T, typename S, T Value>
struct seq_at_or_detail<false, N, T, S, Value> {
    static constexpr T value = Value;
};

template <typename S, size_t N, typename seq_type<S>::type Value>
struct seq_at_or {
    static constexpr auto value = seq_at_or_detail<N >= 0 && N < seq_size<S>::value, N,
                                                   typename seq_type<S>::type, S, Value>::value;
};

template <typename T, T Value, typename S>
struct seq_count_detail;

template <typename T, T Value, T Current, T... Values>
struct seq_count_detail<T, Value, sequence<T, Current, Values...>> {
    static constexpr T value = seq_count_detail<T, Value, sequence<T, Values...>>::value;
};

template <typename T, T Value, T... Values>
struct seq_count_detail<T, Value, sequence<T, Value, Values...>> {
    static constexpr T value = seq_count_detail<T, Value, sequence<T, Values...>>::value + 1;
};

template <typename T, T Value, T Current>
struct seq_count_detail<T, Value, sequence<T, Current>> {
    static constexpr T value = 0;
};

template <typename T, T Value>
struct seq_count_detail<T, Value, sequence<T, Value>> {
    static constexpr T value = 1;
};

template <typename S,
          typename seq_type<S>::type Value>
struct seq_count {
    static constexpr auto value = seq_count_detail<typename seq_type<S>::type, Value, S>::value;
};

template <typename T, ssize_t Idx, T Value, typename S>
struct seq_find_first_detail;

template <typename T, ssize_t Idx, T Value, T Current, T... Values>
struct seq_find_first_detail<T, Idx, Value, sequence<T, Current, Values...>> {
    static constexpr auto value = seq_find_first_detail<T, Idx + 1, Value, sequence<T, Values...>>::value;
};

template <typename T, size_t Idx, T Value, T... Values>
struct seq_find_first_detail<T, Idx, Value, sequence<T, Value, Values...>> {
    static constexpr ssize_t value = Idx;
};

template <typename T, size_t Idx, T Value, T Current>
struct seq_find_first_detail<T, Idx, Value, sequence<T, Current>> {
    static constexpr ssize_t value = -1;
};

template <typename T, size_t Idx, T Value>
struct seq_find_first_detail<T, Idx, Value, sequence<T, Value>> {
    static constexpr ssize_t value = Idx;
};

template <typename S,
          typename seq_type<S>::type Value>
struct seq_find_first {
    static constexpr auto value = seq_find_first_detail<typename seq_type<S>::type, 0, Value, S>::value;
};

template <typename S,
          typename seq_type<S>::type Value>
struct seq_find_last {
    static constexpr auto value = seq_find_first_detail<typename seq_type<S>::type, 0, Value, typename seq_reverse<S>::type>::value;
};

}

#define SEQ_T_OP(o,...) typename utils::mpl::detail::seq_##o<__VA_ARGS__>::type
#define SEQ_V_OP(o,...) (utils::mpl::detail::seq_##o<__VA_ARGS__>::value)

#define SEQ_APPEND(...)   SEQ_T_OP(append,##__VA_ARGS__)
#define SEQ_PREPEND(...)  SEQ_T_OP(prepend,##__VA_ARGS__)
#define SEQ_REVERSE(...)  SEQ_T_OP(reverse,##__VA_ARGS__)

#define SEQ_GEN_FILL(...) SEQ_T_OP(fill,##__VA_ARGS__)
#define SEQ_GEN_DEC(...)  SEQ_T_OP(gen_dec,##__VA_ARGS__)
#define SEQ_GEN_INC(...)  SEQ_T_OP(gen_inc,##__VA_ARGS__)

#define SEQ_WRAP(...)    SEQ_T_OP(wrap,##__VA_ARGS__)
#define SEQ_UNWRAP(...)  SEQ_T_OP(unwrap,##__VA_ARGS__)

#define SEQ_TYPE(...)    SEQ_T_OP(type,##__VA_ARGS__)

#define SEQ_AT(...)         (SEQ_V_OP(at,##__VA_ARGS__))
#define SEQ_AT_OR(...)      (SEQ_V_OP(at_or,##__VA_ARGS__))
#define SEQ_COUNT(...)      (SEQ_V_OP(count,##__VA_ARGS__))
#define SEQ_HAS(...)        (SEQ_V_OP(count,##__VA_ARGS__) > 0)
#define SEQ_SIZE(...)       (SEQ_V_OP(size,##__VA_ARGS__))
#define SEQ_FIND_FIRST(...) (SEQ_V_OP(find_first,##__VA_ARGS__))
#define SEQ_FIND_LAST(...)  (SEQ_V_OP(find_last,##__VA_ARGS__))

template <typename T, typename T2, class U, class M>
struct reorder_gather_static_detail;

template <typename T, typename T2, T... Vals, T2... Idxs>
struct reorder_gather_static_detail<T, T2, mpl::sequence<T, Vals...>, mpl::sequence<T2, Idxs...>> {
    using       type = SEQ(SEQ_AT(SEQ(Vals...), Idxs)...);
};

template <class U, class M>
struct reorder_gather_static {
    static_assert(SEQ_SIZE(U) == SEQ_SIZE(M), "Wrong reorder configuration");
    using type = typename reorder_gather_static_detail<SEQ_TYPE(U), SEQ_TYPE(M), U, M>::type;
};

#define SEQ_REORDER(...) typename utils::mpl::reorder_gather_static<__VA_ARGS__>::type

}

template <typename T>
void
_get_factors(T n, std::vector<T> &factors)
{
    if (n <= 1) return;

    unsigned z = 2;
    while (z <= n) {
        if (n % z == 0) {
            break;
        }
        ++z;
    }
    factors.push_back(z);
    _get_factors(n/z, factors);
    return;
}

template <typename T>
std::vector<T>
get_factors(T n)
{
    std::vector<T> ret;

    _get_factors(n, ret);

    return ret;
}

template <class T, class U>
T div_ceil(T a, U b)
{
    static_assert(std::is_integral<T>::value &&
                  std::is_integral<U>::value, "div_ceil works on integral types only");
    ASSERT(b > U(0));
    T res = a / b;
    if (a % b != 0) ++res;
    return res;
}

template <class T, class U>
typename std::common_type<T, U>::type
round_next(T val, U step)
{
    static_assert(std::is_integral<T>::value &&
                  std::is_integral<U>::value, "round_next works on integral types only");
    if (val % step != 0)
        val = ((val / step) + 1) * step;
    return val;
}

#ifdef CUDARRAYS_UNITTEST
#define CUDARRAYS_TESTED(C,T) friend ::C##_##T##_Test;
#else
#define CUDARRAYS_TESTED(C,T)
#endif

#define CUDA_CALL(x)                                       \
do {                                                       \
    cudaError_t err__ = (x);                               \
    if (err__ != cudaSuccess) {                            \
        fprintf(stderr,                                    \
                "Error calling CUDA: %d. Message: '%s'\n", \
                err__,                                     \
                cudaGetErrorString(err__));                \
        abort();                                           \
    }                                                      \
} while (0)

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
