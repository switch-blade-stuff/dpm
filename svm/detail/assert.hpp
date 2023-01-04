/*
 * Created by switchblade on 2023-01-03.
 */

#pragma once

#include "define.hpp"

#if !defined(NDEBUG) || defined(SVM_DEBUG)

#ifndef SVM_USE_IMPORT

#include <cstdlib>
#include <cstdio>

#endif

#if defined(__has_builtin) && !defined(__ibmxl__)
#if __has_builtin(__builtin_debugtrap)
#define SVM_DEBUG_TRAP() __builtin_debugtrap()
#elif __has_builtin(__debugbreak)
#define SVM_DEBUG_TRAP() __debugbreak()
#endif
#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)
#define SVM_DEBUG_TRAP() __debugbreak()
#elif defined(__ARMCC_VERSION)
#define SVM_DEBUG_TRAP() __breakpoint(42)
#elif defined(__ibmxl__) || defined(__xlC__)
#include <builtins.h>
#define SVM_DEBUG_TRAP() __trap(42)
#elif defined(__DMC__) && defined(_M_IX86)
#define SVM_DEBUG_TRAP() (__asm int 3h)
#elif defined(__i386__) || defined(__x86_64__)
#define SVM_DEBUG_TRAP() (__asm__ __volatile__("int3"))
#elif defined(__STDC_HOSTED__) && (__STDC_HOSTED__ == 0) && defined(__GNUC__)
#define SVM_DEBUG_TRAP() __builtin_trap()
#endif

#ifndef SVM_DEBUG_TRAP
#ifndef SVM_USE_IMPORT

#ifndef SVM_USE_IMPORT

#include <csignal>

#endif

#endif
#if defined(SIGTRAP)
#define SVM_DEBUG_TRAP() raise(SIGTRAP)
#else
#define SVM_DEBUG_TRAP() raise(SIGABRT)
#endif
#endif

#if defined(_MSC_VER) || defined(__CYGWIN__)
#define SVM_PRETTY_FUNC __FUNCSIG__
#elif defined(__clang__) || defined(__GNUC__)
#define SVM_PRETTY_FUNC __PRETTY_FUNCTION__
#endif

namespace svm::detail
{
	[[maybe_unused]] inline void assert_msg(bool cnd, const char *cstr, const char *file, std::size_t line, const char *func, const char *msg) noexcept
	{
		if (!cnd)
		{
			printf("Assertion (%s) failed at '%s:%zu' in '%s'", cstr, file, line, func);
			if (msg) printf("%s", msg);
			SVM_DEBUG_TRAP();
			std::abort();
		}
	}
}

#define SVM_ASSERT(cnd, msg) svm::detail::assert_msg(cnd, (#cnd), (__FILE__), (__LINE__), (SVM_PRETTY_FUNC), (msg))
#else
#define SVM_ASSERT(cnd, msg)
#endif
