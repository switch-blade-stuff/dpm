/*
 * Created by switchblade on 2022-12-31.
 */

#include "cpuid.hpp"

#if defined(SVM_ARCH_X86) && defined(SVM_DYNAMIC_DISPATCH)
#ifndef SVM_USE_IMPORT

#include <cstring>

#endif

#if defined(_MSC_VER) || defined(__CYGWIN__)

#include <intrin.h>

#endif

namespace svm::detail
{
	using cpuid_regs = std::uint32_t[4];

	static SVM_FORCEINLINE void platform_cpuid(std::uint32_t leaf, cpuid_regs &regs)
	{
#if defined(_MSC_VER) || defined(__CYGWIN__)
		__cpuid(reinterpret_cast<int *>(regs), static_cast<int>(leaf));
#elif defined(__GNUC__) || defined(__clang__)
		__asm("xchgq  %%rbx,%q1\n"
		      "\tcpuid\n"
		      "\txchgq  %%rbx,%q1"
				: "=a"(regs[0]), "=r" (regs[1]), "=c"(regs[2]), "=d"(regs[3])
				: "0"(leaf));
#else
#error "Unsupported compiler"
#endif
	}

	cpuid::cpuid() noexcept
	{
		cpuid_regs cpu_info = {0};
		platform_cpuid(0, cpu_info);

		{
			char vendor[13] = {0};
			reinterpret_cast<std::uint32_t &>(vendor[0]) = cpu_info[1];
			reinterpret_cast<std::uint32_t &>(vendor[4]) = cpu_info[3];
			reinterpret_cast<std::uint32_t &>(vendor[8]) = cpu_info[2];

			if (strcmp(vendor, "GenuineIntel") == 0)
				m_is_intel = true;
			else if (strcmp(vendor, "AuthenticAMD") == 0)
				m_is_amd = true;
		}

		/* Initialize flags. */
		const auto max_leaf = cpu_info[0];
		if (max_leaf >= 1) /* flags for leaf = 0x00000001 */
		{
			platform_cpuid(1, cpu_info);
			m_flags_l1_ecx = cpu_info[2];
			m_flags_l1_edx = cpu_info[3];
		}
		if (max_leaf >= 7) /* flags for leaf = 0x00000007 */
		{
			platform_cpuid(7, cpu_info);
			m_flags_l7_ebx = cpu_info[1];
		}

		/* Get extended info. */
		platform_cpuid(0x80000000, cpu_info);
		if (cpu_info[0] >= 0x80000001) /* flags for leaf = 0x80000001 */
		{
			platform_cpuid(0x80000001, cpu_info);
			m_flags_l81_ecx = cpu_info[2];
		}
	}

	const cpuid cpuid::cpu_data;
}

#endif