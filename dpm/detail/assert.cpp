/*
 * Created by switch_blade on 2023-02-10.
 */

#include "../debug.hpp"

#include <csignal>
#include <cstdio>

namespace dpm::detail
{
	void assert_err(const char *file, unsigned long line, const char *func, const char *cnd, const char *msg) noexcept
	{
		std::fprintf(stderr, "%s:%lu: %s: Assertion `%s` failed", file, line, func, cnd);
		if (msg != nullptr)
			std::fprintf(stderr, " - %s\n", msg);
		else
			std::fputc('.', stderr);
	}

	[[maybe_unused]] void assert_trap() noexcept
	{
#ifdef SIGTRAP
	std::raise(SIGTRAP);
#else
	std::raise(SIGABRT);
#endif
	}
}