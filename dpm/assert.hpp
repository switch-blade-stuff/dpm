/*
 * Created by switch_blade on 2023-02-10.
 */

#pragma once

#include "detail/define.hpp"

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

namespace dpm
{
	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Triggers a runtime debugger trap if \a cnd evaluates to `false`. If neither debug trap intrinsics nor `SIGTRAP` are supported, falls back to `raise(SIGABRT)` */
		DPM_FORCEINLINE void assert_trap(bool cnd, const char *cnd_str, const char *msg, const char *file, std::size_t line, const char *func) noexcept
		{
			if (cnd) [[likely]] return;

			/* Print assertion message to stderr. */
			std::fprintf(stderr, "%s:%zu: %s: Assertion `%s` failed", file, line, func, cnd_str);
			if (msg != nullptr)
				std::fprintf(stderr, " - %s\n", msg);
			else
				std::fputc('.', stderr);

			/* Trigger a debugger trap. */
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
}

#define DPM_ASSERT_MSG(cnd, msg) do { dpm::ext::assert_trap((cnd), (#cnd), (msg), (__FILE__), (__LINE__), (DPM_FUNCNAME)); DPM_ASSUME(cnd); } while(false)
#else
#define DPM_ASSERT_MSG(cnd, msg) DPM_ASSUME(cnd)
#endif

#define DPM_ASSERT(cnd) DPM_ASSERT_MSG(cnd, nullptr)
