/*
 * Created by switch_blade on 2023-02-10.
 */

#pragma once

#include <version>

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

#if defined(__ibmxl__) || defined(__xlC__)
#include <builtins.h>
#endif

namespace dpm::detail
{
	DPM_PUBLIC void assert_err(const char *file, unsigned long line, const char *func, const char *cnd, const char *msg) noexcept;

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
#define DPM_DEBUGTRAP() (__asm int 3h)
#elif defined(__i386__) || defined(__x86_64__)
#define DPM_DEBUGTRAP() (__asm__ __volatile__("int3"))
#else
	[[noreturn]] DPM_PUBLIC void assert_trap() noexcept;
#define DPM_DEBUGTRAP() dpm::detail::assert_trap()
#endif
}

#ifdef __cpp_lib_source_location
#include <source_location>

namespace dpm::detail
{
	DPM_FORCEINLINE void assert_err(std::source_location loc, const char *cnd, const char *msg) noexcept { assert_err(loc.file_name(), loc.line(), loc.function_name(), cnd, msg); }
}

#define DPM_ASSERT_LOC_TYPE std::source_location
#define DPM_ASSERT_LOC_CURRENT std::source_location::current()
#else
namespace dpm::detail
{
	struct source_location
	{
		const char *func = nullptr;
		const char *file = nullptr;
		unsigned long line = 0;
	};

	DPM_FORCEINLINE void assert_err(source_location loc, const char *cnd, const char *msg) noexcept { assert_err(loc.file, loc.line, loc.func, cnd, msg); }
}

#define DPM_ASSERT_LOC_TYPE dpm::detail::source_location
#define DPM_ASSERT_LOC_CURRENT dpm::detail::source_location{__FILE__, __LINE__, DPM_FUNCNAME}
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
