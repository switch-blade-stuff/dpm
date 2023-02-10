/*
 * Created by switch_blade on 2023-02-10.
 */

#pragma once

#include "detail/api.hpp"

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

#if defined(_MSC_VER)
#define DPM_FUNCNAME __FUNCSIG__
#define DPM_FORCEINLINE inline __forceinline
#define DPM_NEVER_INLINE __declspec(noinline)
#define DPM_PURE

/* Windows calling convention will never use vector registers for function arguments unless explicitly required. */
#define DPM_VECTORCALL __vectorcall

#elif defined(__clang__) || defined(__GNUC__)
#define DPM_FUNCNAME __PRETTY_FUNCTION__
#define DPM_FORCEINLINE inline __attribute__((always_inline))
#define DPM_NEVER_INLINE __attribute__((noinline))
#define DPM_PURE __attribute__((pure))
#define DPM_VECTORCALL
#else
#define DPM_FUNCNAME __func__
#define DPM_FORCEINLINE inline
#define DPM_NEVER_INLINE
#define DPM_VECTORCALL
#define DPM_PURE
#endif

namespace dpm::detail
{
#if defined(_MSC_VER)
	[[noreturn]] DPM_FORCEINLINE void assert_trap() noexcept { __debugbreak(); }
#elif defined(__clang__) && defined(__has_builtin) && __has_builtin(__builtin_debugtrap)
	[[noreturn]] DPM_FORCEINLINE void assert_trap() noexcept { __builtin_debugtrap(); }
#elif defined(__GNUC__)
	[[noreturn]] DPM_FORCEINLINE void assert_trap() noexcept { __builtin_trap(); }
#else
	[[noreturn]] DPM_PUBLIC void assert_trap() noexcept;
#endif

	DPM_PUBLIC void assert_err(const char *file, unsigned long line, const char *func, const char *cnd, const char *msg) noexcept;

	DPM_FORCEINLINE void assert(bool cnd, const char *file, unsigned long line, const char *func, const char *cnd_str, const char *msg) noexcept
	{
		if (!cnd) [[unlikely]]
		{
			assert_err(file, line, func, cnd_str, msg);
			assert_trap();
		}
	}
}

#define DPM_ASSERT_MSG_ALWAYS(cnd, msg) do { dpm::detail::assert((cnd), (__FILE__), (__LINE__), (DPM_FUNCNAME), (#cnd), (msg)); DPM_ASSUME(cnd); } while(false)
#define DPM_ASSERT_ALWAYS(cnd) DPM_ASSERT_MSG_ALWAYS(cnd, nullptr)

#ifndef NDEBUG
#define DPM_ASSERT_MSG(cnd, msg) DPM_ASSERT_MSG_ALWAYS(cnd, msg)
#define DPM_ASSERT(cnd) DPM_ASSERT_ALWAYS(cnd)
#else
#define DPM_ASSERT_MSG(cnd, msg) DPM_ASSUME(cnd)
#define DPM_ASSERT(cnd) DPM_ASSUME(cnd)
#endif
