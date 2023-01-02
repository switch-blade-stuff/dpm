/*
 * Created by switchblade on 2023-01-02.
 */

#pragma once

#include "../define.hpp"

#if defined(SVM_ARCH_ARM) && defined(SVM_DYNAMIC_DISPATCH)

#include "../utility.hpp"

namespace svm::detail
{
	class cpuid
	{
	public:
		/* TODO: Implement ARM CPUID */
		[[nodiscard]] static bool is_aarch64() noexcept;
		[[nodiscard]] static bool has_neon() noexcept;

	private:
		SVM_PUBLIC static const cpuid cpu_info;

		SVM_PUBLIC cpuid() noexcept;
	};
}

#endif