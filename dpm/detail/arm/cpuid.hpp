/*
 * Created by switchblade on 2023-01-02.
 */

#pragma once

#include "../define.hpp"

#if defined(DPM_ARCH_ARM) && defined(DPM_DYNAMIC_DISPATCH)

#include "../utility.hpp"

namespace dpm::detail
{
	class cpuid
	{
	public:
		/* TODO: Implement ARM CPUID */
		[[nodiscard]] static bool is_aarch64() noexcept;
		[[nodiscard]] static bool has_neon() noexcept;

	private:
		DPM_PUBLIC static const cpuid cpu_info;

		DPM_PUBLIC cpuid() noexcept;
	};
}

#endif