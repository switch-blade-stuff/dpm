/*
 * Created by switchblade on 2023-01-03.
 */

#pragma once

#include "define.hpp"

#if !defined(NDEBUG) || defined(DPM_DEBUG)

#include <cstdio>

#if defined(_MSC_VER)
#define DPM_FUNCNAME __FUNCSIG__
#elif defined(__clang__) || defined(__GNUC__)
#define DPM_FUNCNAME __PRETTY_FUNCTION__
#else
#define DPM_FUNCNAME __func__

#include <csignal>

#endif

namespace dpm::detail
{
	DPM_FORCEINLINE void trap_assert(bool cnd, const char *msg, const char *file, std::size_t line, const char *func) noexcept
	{
		if (cnd) [[likely]] return;

		fprintf(stderr, "%s:%zu: %s: Assertion `%s` failed.\n", file, line, func, msg);
#if defined(_MSC_VER)
		__debugbreak();
#elif defined(__clang__) && defined(__has_builtin) && __has_builtin(__builtin_debugtrap)
		__builtin_debugtrap();
#elif defined(__GNUC__)
		__builtin_trap();
#elif defined(SIGTRAP)
		 std::raise(SIGTRAP);
#else
		 std::raise(SIGABRT);
#endif
	}
}

#define DPM_ASSERT(cnd) do { dpm::detail::trap_assert((cnd), (#cnd), (__FILE__), (__LINE__), (DPM_FUNCNAME)); DPM_ASSUME(cnd); } while(false)
#else
#define DPM_ASSERT(cnd) DPM_ASSUME(cnd)
#endif
