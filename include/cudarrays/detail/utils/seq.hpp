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
#ifndef CUDARRAYS_DETAIL_UTILS_SEQ_HPP_
#define CUDARRAYS_DETAIL_UTILS_SEQ_HPP_

#include <array>
#include <tuple>

namespace utils {

namespace mpl {

template <typename T, T... Values>
struct sequence {
    using type = T;

    static constexpr size_t size = sizeof...(Values);

    static constexpr std::array<T, sizeof...(Values)> as_array()
    {
        return {{Values...}};
    }
};

template <typename ...T>
auto deduce(T...) -> typename std::tuple_element<0, std::tuple<T...>>::type
{
    return std::get<0, std::tuple<T...>>();
}

#define SEQ(...)             utils::mpl::sequence<decltype(utils::mpl::deduce(__VA_ARGS__)),##__VA_ARGS__>
#define SEQ_WITH_TYPE(t,...) utils::mpl::sequence<t,##__VA_ARGS__>

#define SEQ_T_OP(o,...) typename utils::mpl::seq_##o<__VA_ARGS__>::type
#define SEQ_V_OP(o,...) (utils::mpl::seq_##o<__VA_ARGS__>::value)

namespace detail __attribute__ ((visibility ("hidden"))) {

    template <typename T, class S>
    struct seq_size;

    template <typename T, T... Vals>
    struct seq_size<T, SEQ_WITH_TYPE(T, Vals...)> {
        static constexpr size_t value = sizeof...(Vals);
    };

}

template <typename S>
struct seq_size {
    static constexpr size_t value = detail::seq_size<typename S::type, S>::value;
};

template <typename S>
constexpr size_t seq_size<S>::value;

#define SEQ_SIZE(...) (SEQ_V_OP(size,##__VA_ARGS__))

template <typename S>
struct seq_type {
    using type = typename S::type;
};

#define SEQ_TYPE(...) SEQ_T_OP(type,##__VA_ARGS__)

template <typename T, class O>
struct seq_wrap;

template <typename T, T... Values, template <T...> class O>
struct seq_wrap<T, O<Values...>> {
    using type = SEQ_WITH_TYPE(T, Values...);
};

#define SEQ_WRAP(...) SEQ_T_OP(wrap,##__VA_ARGS__)

namespace detail __attribute__ ((visibility ("hidden"))) {

    template <typename T, typename S, typename O>
    struct seq_unwrap;

    template <typename T, T... Values, template <T...> class O>
    struct seq_unwrap<T, SEQ_WITH_TYPE(T, Values...), O<>> {
        using type = O<Values...>;
    };

}

template <typename S,
          typename O>
struct seq_unwrap {
    using type = typename detail::seq_unwrap<typename seq_type<S>::type, S, O>::type;
};

#define SEQ_UNWRAP(...) SEQ_T_OP(unwrap,##__VA_ARGS__)

namespace detail __attribute__ ((visibility ("hidden"))) {

    template <typename T, typename S1, typename S2>
    struct seq_merge;

    template <typename T, T... Values1, T... Values2>
    struct seq_merge<T, SEQ_WITH_TYPE(T, Values1...), SEQ_WITH_TYPE(T, Values2...)> {
        using type = SEQ_WITH_TYPE(T, Values1..., Values2...);
    };

}

template <class S1, class S2>
struct seq_merge {
    static_assert(std::is_same<SEQ_TYPE(S1), SEQ_TYPE(S2)>::value == true,
                  "Sequence types must match");
    using type = typename detail::seq_merge<SEQ_TYPE(S1), S1, S2>::type;
};

#define SEQ_MERGE(...) SEQ_T_OP(merge,##__VA_ARGS__)

namespace detail __attribute__ ((visibility ("hidden"))) {

    template <typename T, unsigned Size, T FillValue, T... Generated>
    struct seq_gen_fill {
        using type = typename seq_gen_fill<T, Size - 1, FillValue, FillValue, Generated...>::type;
    };

    template <typename T, T FillValue, T... Generated>
    struct seq_gen_fill<T, 0, FillValue, Generated...> {
        using type = SEQ_WITH_TYPE(T, Generated...);
    };

}

template <typename T, T FillValue, unsigned Size>
struct seq_gen_fill {
    using type = typename detail::seq_gen_fill<T, Size, FillValue>::type;
};

#define SEQ_GEN_FILL(v,n)             SEQ_T_OP(gen_fill,decltype(utils::mpl::deduce(v)),v,n)
#define SEQ_GEN_FILL_WITH_TYPE(t,v,n) SEQ_T_OP(gen_fill,t,v,n)

template <typename S, SEQ_TYPE(S)... FillValues>
struct seq_prepend {
    using s_type = SEQ_TYPE(S);
    using type = typename seq_merge<SEQ_WITH_TYPE(s_type, FillValues...), S>::type;
};

#define SEQ_PREPEND(...) SEQ_T_OP(prepend,##__VA_ARGS__)

template <typename S, SEQ_TYPE(S)... FillValues>
struct seq_append {
    using s_type = SEQ_TYPE(S);
    using type = typename seq_merge<S, SEQ_WITH_TYPE(s_type, FillValues...)>::type;
};

#define SEQ_APPEND(...) SEQ_T_OP(append,##__VA_ARGS__)

namespace detail __attribute__ ((visibility ("hidden"))) {

template <typename T, typename S>
struct seq_pop;

template <typename T, T Value, T... Values>
struct seq_pop<T, SEQ_WITH_TYPE(T, Value, Values...)> {
    using type = SEQ_WITH_TYPE(T, Values...);
};

}

template <typename S>
struct seq_pop {
    using type = typename detail::seq_pop<SEQ_TYPE(S), S>::type;
};

#define SEQ_POP(s) SEQ_T_OP(pop,s)

namespace detail __attribute__ ((visibility ("hidden"))) {

    template <typename T, typename S, T... Generated>
    struct seq_reverse;

    template <typename T, T Current, T... Vals, T... Generated>
    struct seq_reverse<T, SEQ_WITH_TYPE(T, Current, Vals...), Generated...> {
        using type = typename seq_reverse<T, SEQ_WITH_TYPE(T, Vals...), Current, Generated...>::type;
    };

    template <typename T, T... Generated>
    struct seq_reverse<T, SEQ_WITH_TYPE(T), Generated...> {
        using type = SEQ_WITH_TYPE(T, Generated...);
    };

}

template <typename S>
struct seq_reverse {
    using type = typename detail::seq_reverse<SEQ_TYPE(S), S>::type;
};

#define SEQ_REVERSE(...) SEQ_T_OP(reverse,##__VA_ARGS__)

namespace detail __attribute__ ((visibility ("hidden"))) {

    template <typename T, unsigned Size, T Current, T... Generated>
    struct seq_gen_inc {
        using type = typename seq_gen_inc<T, Size - 1, Current + 1, Generated..., Current>::type;
    };

    template <typename T, T Current, T... Generated>
    struct seq_gen_inc<T, 0, Current, Generated...> {
        using type = SEQ_WITH_TYPE(T, Generated...);
    };

}

template <typename T, T Size>
struct seq_gen_inc {
    using type = typename detail::seq_gen_inc<T, Size, 0>::type;
};

#define SEQ_GEN_INC(v)             SEQ_T_OP(gen_inc,decltype(utils::mpl::deduce(v)),v)
#define SEQ_GEN_INC_WITH_TYPE(t,v) SEQ_T_OP(gen_inc,t,v)

namespace detail __attribute__ ((visibility ("hidden"))) {

    template <typename T, unsigned Size, T Current, T... Generated>
    struct seq_gen_dec {
        using type = typename seq_gen_dec<T, Size - 1, Current - 1, Current, Generated...>::type;
    };

    template <typename T, T Current, T... Generated>
    struct seq_gen_dec<T, 1, Current, Generated...> {
        using type = SEQ_WITH_TYPE(T, Generated...);
    };

}

template <typename T, unsigned Size>
struct seq_gen_dec {
    using type = typename detail::seq_gen_dec<T, Size, Size - 1>::type;
};

#define SEQ_GEN_DEC(...) SEQ_T_OP(gen_dec,##__VA_ARGS__)

namespace detail __attribute__ ((visibility ("hidden"))) {

    template <size_t N, typename T, typename S>
    struct seq_at;

    template <size_t N, typename T, T Current, T... Values>
    struct seq_at<N, T, SEQ_WITH_TYPE(T, Current, Values...)> {
        static constexpr T value = seq_at<N - 1, T, SEQ_WITH_TYPE(T, Values...)>::value;
    };

    template <typename T, T Current, T... Values>
    struct seq_at<0, T, SEQ_WITH_TYPE(T, Current, Values...)> {
        static constexpr T value = Current;
    };

}

template <typename S, size_t N>
struct seq_at {
    static_assert(N < seq_size<S>::value, "Out of bounds");
    static constexpr SEQ_TYPE(S) value = detail::seq_at<N, SEQ_TYPE(S), S>::value;
};

template <typename S, size_t N>
constexpr SEQ_TYPE(S) seq_at<S, N>::value;

#define SEQ_AT(...) (SEQ_V_OP(at,##__VA_ARGS__))

namespace detail __attribute__ ((visibility ("hidden"))) {

    template <size_t N, typename T, T Value, typename P, typename R>
    struct seq_set;

    template <size_t N, typename T, T Value, T Current, T... Values, T... Values2>
    struct seq_set<N, T, Value,
                   SEQ_WITH_TYPE(T, Current, Values...),
                   SEQ_WITH_TYPE(T, Values2...)> {
        using type = typename seq_set<N - 1, T, Value,
                                      SEQ_WITH_TYPE(T, Values...),
                                      SEQ_WITH_TYPE(T, Values2..., Current)
                                      >::type;
    };

    template <typename T, T Value, T Current, T... Values, T... Values2>
    struct seq_set<0, T, Value,
                   SEQ_WITH_TYPE(T, Current, Values...),
                   SEQ_WITH_TYPE(T, Values2...)> {
        using type = SEQ_WITH_TYPE(T, Values2..., Value, Values...);
    };

}

template <typename S, size_t N, SEQ_TYPE(S) Value>
struct seq_set {
    static_assert(N < seq_size<S>::value, "Out of bounds");
    using type = typename detail::seq_set<N, SEQ_TYPE(S), Value, S, SEQ_WITH_TYPE(SEQ_TYPE(S))>::type;
};

#define SEQ_SET(...) SEQ_T_OP(set,##__VA_ARGS__)

namespace detail __attribute__ ((visibility ("hidden"))) {

template <bool ValidIndex, size_t N, typename T, typename S, T Value>
struct seq_at_or {
    static constexpr auto value = SEQ_AT(S, N);
};

template <size_t N, typename T, typename S, T Value>
struct seq_at_or<false, N, T, S, Value> {
    static constexpr T value = Value;
};

}

template <typename S, size_t N, SEQ_TYPE(S) Value>
struct seq_at_or {
    static constexpr SEQ_TYPE(S) value = detail::seq_at_or<N >= 0 && N < SEQ_SIZE(S),
                                                           N,
                                                           SEQ_TYPE(S), S, Value>::value;
};

template <typename S, size_t N, SEQ_TYPE(S) Value>
constexpr SEQ_TYPE(S) seq_at_or<S, N, Value>::value;

#define SEQ_AT_OR(...) (SEQ_V_OP(at_or,##__VA_ARGS__))

namespace detail __attribute__ ((visibility ("hidden"))) {

    template <typename T, T Value, typename S>
    struct seq_count;

    template <typename T, T Value, T Current, T... Values>
    struct seq_count<T, Value, SEQ_WITH_TYPE(T, Current, Values...)> {
        static constexpr T value = seq_count<T, Value, SEQ_WITH_TYPE(T, Values...)>::value;
    };

    template <typename T, T Value, T... Values>
    struct seq_count<T, Value, SEQ_WITH_TYPE(T, Value, Values...)> {
        static constexpr T value = seq_count<T, Value, SEQ_WITH_TYPE(T, Values...)>::value + 1;
    };

    template <typename T, T Value>
    struct seq_count<T, Value, SEQ_WITH_TYPE(T)> {
        static constexpr T value = 0;
    };

}

template <typename S, SEQ_TYPE(S) Value>
struct seq_count {
    static constexpr SEQ_TYPE(S) value = detail::seq_count<SEQ_TYPE(S), Value, S>::value;
};

template <typename S, SEQ_TYPE(S) Value>
constexpr SEQ_TYPE(S) seq_count<S, Value>::value;

#define SEQ_COUNT(...) (SEQ_V_OP(count,##__VA_ARGS__))
#define SEQ_HAS(...)   (SEQ_V_OP(count,##__VA_ARGS__) > 0)

namespace detail __attribute__ ((visibility ("hidden"))) {

    template <typename T, size_t Idx, T Value, typename S>
    struct seq_find_first;

    template <typename T, size_t Idx, T Value, T Current, T... Values>
    struct seq_find_first<T, Idx, Value, SEQ_WITH_TYPE(T, Current, Values...)> {
        static constexpr ssize_t value = seq_find_first<T, Idx + 1, Value, SEQ_WITH_TYPE(T, Values...)>::value;
    };

    template <typename T, size_t Idx, T Value, T... Values>
    struct seq_find_first<T, Idx, Value, SEQ_WITH_TYPE(T, Value, Values...)> {
        static constexpr ssize_t value = Idx;
    };

    template <typename T, size_t Idx, T Value>
    struct seq_find_first<T, Idx, Value, SEQ_WITH_TYPE(T)> {
        static constexpr ssize_t value = -1;
    };

}

template <typename S, SEQ_TYPE(S) Value>
struct seq_find_first {
    static constexpr ssize_t value = detail::seq_find_first<SEQ_TYPE(S), 0, Value, S>::value;
};

template <typename S, SEQ_TYPE(S) Value>
constexpr ssize_t seq_find_first<S, Value>::value;

#define SEQ_FIND_FIRST(...) (SEQ_V_OP(find_first,##__VA_ARGS__))

template <typename S, SEQ_TYPE(S) Value>
struct seq_find_last {
    static constexpr ssize_t value = std::conditional<SEQ_COUNT(S, Value) == 0,
                                                      std::integral_constant<ssize_t, -1>,
                                                      std::integral_constant<ssize_t, SEQ_SIZE(S) -
                                                                                      (SEQ_FIND_FIRST(SEQ_REVERSE(S), Value) + 1)>
                                                     >::type::value;
};

#define SEQ_FIND_LAST(...)  (SEQ_V_OP(find_last,##__VA_ARGS__))

namespace detail {

template <typename S, size_t Index, SEQ_TYPE(S) Current = 0>
struct seq_sum  {
    static constexpr SEQ_TYPE(S) value = seq_sum<S, Index - 1, Current + SEQ_AT(S, Index)>::value;
};

template <typename S, SEQ_TYPE(S) Current>
struct seq_sum<S, 0, Current>  {
    static constexpr SEQ_TYPE(S) value = Current + SEQ_AT(S, 0);
};

}

template <typename S>
struct seq_sum {
    using type = SEQ_TYPE(S);
    static constexpr type value = detail::seq_sum<S, SEQ_SIZE(S) - 1>::value;
};

#define SEQ_SUM(s) SEQ_V_OP(sum,s)

namespace detail {

template <typename S, size_t Index, SEQ_TYPE(S) Current = 1>
struct seq_prod  {
    static constexpr SEQ_TYPE(S) value = seq_prod<S, Index - 1, Current * SEQ_AT(S, Index)>::value;
};

template <typename S, SEQ_TYPE(S) Current>
struct seq_prod<S, 0, Current>  {
    static constexpr SEQ_TYPE(S) value = Current * SEQ_AT(S, 0);
};

}

template <typename S>
struct seq_prod {
    using type = SEQ_TYPE(S);
    static constexpr type value = detail::seq_prod<S, SEQ_SIZE(S) - 1>::value;
};

#define SEQ_PROD(s) SEQ_V_OP(prod,s)


namespace detail __attribute__ ((visibility ("hidden"))) {

    template <typename T, typename T2, class U, class M>
    struct seq_reorder_gather;

    template <typename T, typename T2, T... Vals, T2... Idxs>
    struct seq_reorder_gather<T, T2, SEQ_WITH_TYPE(T, Vals...),
                                     SEQ_WITH_TYPE(T2, Idxs...)> {
        using type = SEQ(SEQ_AT(SEQ_WITH_TYPE(T, Vals...), Idxs)...);
    };

}

template <class U, class M>
struct seq_reorder_gather {
    static_assert(seq_size<U>::value == seq_size<M>::value, "Wrong reorder configuration");
    using type = typename detail::seq_reorder_gather<SEQ_TYPE(U), SEQ_TYPE(M), U, M>::type;
};

#define SEQ_REORDER(...) SEQ_T_OP(reorder_gather,##__VA_ARGS__)

} // namespace mpl
} // namespace util

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
