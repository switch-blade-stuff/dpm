/*
 * Created by switch_blade on 2023-02-10.
 */

#include "../debug.hpp"

#include <cstdio>

#if !(defined(_MSC_VER) || defined(__clang__) || defined(__GNUC__))

#include <csignal>

[[noreturn]] DPM_PUBLIC void dpm::detail::assert_trap() noexcept
{
#ifdef SIGTRAP
	std::raise(SIGTRAP);
#else
	std::raise(SIGABRT);
#endif
}

#endif

DPM_PUBLIC void dpm::detail::assert_err(const char *file, unsigned long line, const char *func, const char *cnd, const char *msg) noexcept
{
	std::fprintf(stderr, "%s:%lu: %s: Assertion `%s` failed", file, line, func, cnd);
	if (msg != nullptr)
		std::fprintf(stderr, " - %s\n", msg);
	else
		std::fputc('.', stderr);
}