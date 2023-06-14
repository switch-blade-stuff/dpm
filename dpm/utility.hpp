/*
 * Created by switch_blade on 2023-02-10.
 */

#pragma once

#include "detail/config.hpp"

#include <type_traits>
#include <concepts>
#include <utility>
#include <cstddef>
#include <cstdint>

#if defined(__ibmxl__) || defined(__xlC__)
#include <builtins.h>
#endif

#if defined(_MSC_VER)
#define DPM_ASSUME(x) __assume(x)
#elif 0 && defined(__clang__) /* See https://github.com/llvm/llvm-project/issues/55636 and https://github.com/llvm/llvm-project/issues/45902 */
#define DPM_ASSUME(x) __builtin_assume(x)
#elif defined(__GNUC__)
#define DPM_ASSUME(x) if (!(x)) __builtin_unreachable()
#else
#define DPM_ASSUME(x)
#endif

#if defined(__clang__) || defined(__GNUC__)
#define DPM_UNREACHABLE() __builtin_unreachable()
#elif defined(__cpp_lib_unreachable) && __cpp_lib_unreachable >= 202202L
#define DPM_UNREACHABLE() std::unreachable()
#else
#define DPM_UNREACHABLE() DPM_ASSUME(false)
#endif

namespace dpm
{
	namespace detail
	{
		template<typename I, std::size_t N>
		concept integral_of_size = std::integral<I> && sizeof(I) == N;
		template<typename I, std::size_t N>
		concept signed_integral_of_size = integral_of_size<I, N> && std::signed_integral<I>;
		template<typename I, std::size_t N>
		concept unsigned_integral_of_size = integral_of_size<I, N> && std::unsigned_integral<I>;

#pragma region "shuffle utils"
		template<typename, std::size_t...>
		struct reverse_sequence;
		template<std::size_t I, std::size_t... Js>
		struct reverse_sequence<std::index_sequence<Js...>, I> { using type = std::index_sequence<I, Js...>; };
		template<std::size_t I, std::size_t... Is, std::size_t... Js>
		struct reverse_sequence<std::index_sequence<Js...>, I, Is...> : reverse_sequence<std::index_sequence<I, Js...>, Is...> {};
		template<std::size_t... Is>
		using reverse_sequence_t = typename reverse_sequence<std::index_sequence<>, Is...>::type;

		template<typename, typename, std::size_t, std::size_t, std::size_t...>
		struct extract_sequence;
		template<std::size_t P, std::size_t N, std::size_t... Is, std::size_t... Js, std::size_t... Ks> requires(sizeof...(Js) == P && sizeof...(Ks) == N)
		struct extract_sequence<std::index_sequence<Js...>, std::index_sequence<Ks...>, P, N, Is...> { using type = std::index_sequence<Ks...>; };
		template<std::size_t P, std::size_t N, std::size_t I, std::size_t... Is, std::size_t... Js, std::size_t... Ks> requires(sizeof...(Js) == P && sizeof...(Ks) != N)
		struct extract_sequence<std::index_sequence<Js...>, std::index_sequence<Ks...>, P, N, I, Is...> : extract_sequence<std::index_sequence<Js...>, std::index_sequence<Ks..., I>, P, N, Is...> {};
		template<std::size_t P, std::size_t N, std::size_t I, std::size_t... Is, std::size_t... Js, std::size_t... Ks> requires(sizeof...(Js) != P && sizeof...(Ks) != N)
		struct extract_sequence<std::index_sequence<Js...>, std::index_sequence<Ks...>, P, N, I, Is...> : extract_sequence<std::index_sequence<I, Js...>, std::index_sequence<Ks...>, P, N, Is...> {};
		template<std::size_t Pos, std::size_t N, std::size_t... Is>
		using extract_sequence_t = typename extract_sequence<std::index_sequence<>, std::index_sequence<>, Pos, N, Is...>::type;

		template<typename, std::size_t, std::size_t, std::size_t, std::size_t...>
		struct repeat_sequence;
		template<std::size_t N, std::size_t Off, std::size_t I, std::size_t... Js>
		struct repeat_sequence<std::index_sequence<Js...>, N, Off, N, I> { using type = std::index_sequence<Js...>; };
		template<std::size_t N, std::size_t Off, std::size_t I, std::size_t... Is, std::size_t... Js>
		struct repeat_sequence<std::index_sequence<Js...>, N, Off, N, I, Is...> : repeat_sequence<std::index_sequence<Js...>, N, Off, 0, Is...> {};
		template<std::size_t N, std::size_t Off, std::size_t J, std::size_t I, std::size_t... Is, std::size_t... Js>
		struct repeat_sequence<std::index_sequence<Js...>, N, Off, J, I, Is...> : repeat_sequence<std::index_sequence<Js..., I * N + (N - J - 1) * Off>, N, Off, J + 1, I, Is...> {};
		template<std::size_t N, std::size_t Off, std::size_t... Is>
		using repeat_sequence_t = typename repeat_sequence<std::index_sequence<>, N, Off, 0, Is...>::type;

		template<typename, std::size_t, std::size_t, std::size_t...>
		struct pad_sequence;
		template<std::size_t N, std::size_t Inc, std::size_t I, std::size_t... Js> requires(sizeof...(Js) >= N)
		struct pad_sequence<std::index_sequence<Js...>, N, Inc, I> { using type = std::index_sequence<Js...>; };
		template<std::size_t N, std::size_t Inc, std::size_t I, std::size_t... Js>
		struct pad_sequence<std::index_sequence<Js...>, N, Inc, I> : pad_sequence<std::index_sequence<Js..., I>, N, Inc, I + Inc> {};
		template<std::size_t N, std::size_t Inc, std::size_t I, std::size_t... Is, std::size_t... Js>
		struct pad_sequence<std::index_sequence<Js...>, N, Inc, I, Is...> : pad_sequence<std::index_sequence<Js..., I>, N, Inc, Is...> {};
		template<std::size_t N, std::size_t Inc, std::size_t... Is>
		using pad_sequence_t = typename pad_sequence<std::index_sequence<>, N, Inc, Is...>::type;

		template<std::size_t I, std::size_t... Is>
		constexpr void shuffle_elements(std::index_sequence<I, Is...>, auto *dst, const auto *src) noexcept
		{
			*dst = src[I];
			if constexpr (sizeof...(Is) != 0) shuffle_elements(std::index_sequence<Is...>{}, dst + 1, src);
		}
#pragma endregion
	}

	/** Utility used to obtain a platform-specific signed integer who's size is at least \a N bytes. */
	template<std::size_t N>
	struct int_of_size;

	/** Alias for `typename int_of_size<N>::type`. */
	template<std::size_t N>
	using int_of_size_t = typename int_of_size<N>::type;

	/** Utility used to obtain a platform-specific unsigned integer who's size is at least \a N bytes. */
	template<std::size_t N>
	struct uint_of_size;

	/** Alias for `typename uint_of_size<N>::type`. */
	template<std::size_t N>
	using uint_of_size_t = typename uint_of_size<N>::type;

	template<>
	struct int_of_size<1> { using type = std::int8_t; };
	template<>
	struct int_of_size<2> { using type = std::int16_t; };
	template<std::size_t N> requires(N > 2 && N <= 4)
	struct int_of_size<N> { using type = std::int32_t; };
	template<std::size_t N> requires(N > 4 && N <= 8)
	struct int_of_size<N> { using type = std::int64_t; };

	template<>
	struct uint_of_size<1> { using type = std::uint8_t; };
	template<>
	struct uint_of_size<2> { using type = std::uint16_t; };
	template<std::size_t N> requires(N > 2 && N <= 4)
	struct uint_of_size<N> { using type = std::uint32_t; };
	template<std::size_t N> requires(N > 4 && N <= 8)
	struct uint_of_size<N> { using type = std::uint64_t; };

	namespace detail
	{
		template<typename T>
		[[nodiscard]] constexpr bool test_bit(T x, int pos) noexcept { return x & static_cast<T>(1 << pos); }
		template<typename T>
		constexpr void mask_bit(T &x, int pos, bool bit = true) noexcept { x &= ~static_cast<T>(!bit << pos); }
		template<std::size_t N, typename T = uint_of_size_t<(N / 8 + !!(N % 8))>>
		[[nodiscard]] constexpr T fill_bits() noexcept
		{
			T result = 0;
			for (T i = 0; i < N; ++i) result |= static_cast<T>(1ull << i);
			return result;
		}

		template<typename T>
		[[nodiscard]] constexpr T extend_bool(bool b) noexcept { return -static_cast<T>(b); }

		template<typename T, std::size_t N, std::size_t VSize>
		[[nodiscard]] constexpr std::size_t align_data() noexcept
		{
			const auto size_mult = VSize / sizeof(T);
			return N / size_mult + !!(N % size_mult);
		}

		template<template<typename> typename, typename>
		struct is_template_instance : std::false_type {};
		template<template<typename> typename T, typename U>
		struct is_template_instance<T, T<U>> : std::true_type {};
		template<typename U, template<typename> typename T>
		concept template_instance = is_template_instance<T, U>::value;
	}
}
namespace dpm::detail
{
	DPM_API_PUBLIC void assert_err(const char *file, std::uint_least32_t line, const char *func, const char *cnd, const char *msg) noexcept;

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#define DPM_DEBUGTRAP() __debugbreak()
#elif defined(__has_builtin) && !defined(__ibmxl__) && __has_builtin(__builtin_debugtrap)
#define DPM_DEBUGTRAP() __builtin_debugtrap()
#elif defined(__STDC_HOSTED__) && (__STDC_HOSTED__ == 0) && defined(__GNUC__)
#define DPM_DEBUGTRAP() __builtin_trap()
#elif defined(__ARMCC_VERSION)
#define DPM_DEBUGTRAP() __breakpoint(42)
#elif defined(__ibmxl__) || defined(__xlC__)
#define DPM_DEBUGTRAP() __trap(42)
#elif defined(__DMC__) && defined(_M_IX86)
#define DPM_DEBUGTRAP() do { __asm int 3h; } while(false)
#elif defined(__i386__) || defined(__x86_64__)
#define DPM_DEBUGTRAP() do { __asm__ __volatile__("int3"); } while(false)
#else
	[[noreturn]] DPM_API_PUBLIC void assert_trap() noexcept;
#define DPM_DEBUGTRAP() dpm::detail::assert_trap()
#endif
}

#ifdef __cpp_lib_source_location

#include <source_location>

namespace dpm::detail
{
	DPM_FORCEINLINE void assert_err(std::source_location loc, const char *cnd, const char *msg) noexcept { assert_err(loc.file_name(), loc.line(), loc.function_name(), cnd, msg); }
}

#define DPM_SOURCE_LOC_TYPE std::source_location
#define DPM_SOURCE_LOC_CURRENT std::source_location::current()
#else
namespace dpm::detail
{
	/* Compatibility with C++20 source_location. */
	class source_location
	{
	public:
		constexpr source_location() noexcept = default;
		constexpr source_location(const char *func, const char *file, unsigned long line) noexcept : m_func(func), m_file(file), m_line(line) {}

		[[nodiscard]] constexpr const char *function_name() const noexcept { return m_func; }
		[[nodiscard]] constexpr const char *file_name() const noexcept { return m_file; }
		[[nodiscard]] constexpr std::uint_least32_t line() const noexcept { return m_line; }
		[[nodiscard]] constexpr std::uint_least32_t column() const noexcept { return 0; }

	private:
		const char *m_func = nullptr;
		const char *m_file = nullptr;
		std::uint_least32_t m_line = 0;
	};

	DPM_FORCEINLINE void assert_err(source_location loc, const char *cnd, const char *msg) noexcept { assert_err(loc.file, loc.line, loc.func, cnd, msg); }
}

#define DPM_SOURCE_LOC_TYPE dpm::detail::source_location
#define DPM_SOURCE_LOC_CURRENT dpm::detail::source_location{__FILE__, __LINE__, DPM_FUNCNAME}
#endif

#define DPM_ASSERT_MSG_LOC_ALWAYS(cnd, msg, src_loc)        \
    do { if (!(cnd)) [[unlikely]] {                         \
        dpm::detail::assert_err(src_loc, (#cnd), (msg));    \
        DPM_DEBUGTRAP();                                    \
    }} while(false)

#ifndef NDEBUG
#define DPM_ASSERT_MSG_LOC(cnd, msg, src_loc) DPM_ASSERT_MSG_LOC_ALWAYS(cnd, msg, src_loc)
#else
#define DPM_ASSERT_MSG_LOC(cnd, msg, src_loc)
#endif

#define DPM_ASSERT_MSG_ALWAYS(cnd, msg)                                                 \
    do { if (!(cnd)) [[unlikely]] {                                                     \
        dpm::detail::assert_err((__FILE__), (__LINE__), (DPM_FUNCNAME), (#cnd), (msg)); \
        DPM_DEBUGTRAP();                                                                \
    }} while(false)
#define DPM_ASSERT_ALWAYS(cnd) DPM_ASSERT_MSG_ALWAYS(cnd, nullptr)

#ifndef NDEBUG
#define DPM_ASSERT_MSG(cnd, msg) DPM_ASSERT_MSG_ALWAYS(cnd, msg)
#define DPM_ASSERT(cnd) DPM_ASSERT_ALWAYS(cnd)
#else
#define DPM_ASSERT_MSG(cnd, msg) DPM_ASSUME(cnd)
#define DPM_ASSERT(cnd) DPM_ASSUME(cnd)
#endif
