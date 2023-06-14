/*
 * Created by switch_blade on 2023-02-10.
 */

#include "../utility.hpp"

#include <csignal>
#include <cstdio>

namespace dpm::detail
{
	void assert_err(const char *file, std::uint_least32_t line, const char *func, const char *cnd, const char *msg) noexcept
	{
		std::fprintf(stderr, "%s:%u: %s: Assertion `%s` failed", file, line, func, cnd);
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