/*
 * Created by switchblade on 2022-12-31.
 */

#pragma once

#include "../define.hpp"

#if defined(SVM_ARCH_X86) && defined(SVM_DYNAMIC_DISPATCH)

#include "../utility.hpp"

namespace svm::detail
{
	class cpuid
	{
	public:
		[[nodiscard]] static bool is_intel() noexcept { return cpu_data.m_is_intel; }
		[[nodiscard]] static bool is_amd() noexcept { return cpu_data.m_is_amd; }

		[[nodiscard]] static bool has_msr() noexcept { return test_bit(cpu_data.m_flags_l1_edx, 5); }
		[[nodiscard]] static bool has_cx8() noexcept { return test_bit(cpu_data.m_flags_l1_edx, 8); }
		[[nodiscard]] static bool has_sep() noexcept { return test_bit(cpu_data.m_flags_l1_edx, 11); }
		[[nodiscard]] static bool has_cmov() noexcept { return test_bit(cpu_data.m_flags_l1_edx, 15); }
		[[nodiscard]] static bool has_clfsh() noexcept { return test_bit(cpu_data.m_flags_l1_edx, 19); }

		[[nodiscard]] static bool has_f16c() noexcept { return test_bit(cpu_data.m_flags_l1_ecx, 29); }
		[[nodiscard]] static bool has_cmpxchg16b() noexcept { return test_bit(cpu_data.m_flags_l1_ecx, 13); }

		[[nodiscard]] static bool has_sse() noexcept { return test_bit(cpu_data.m_flags_l1_edx, 25); }
		[[nodiscard]] static bool has_sse2() noexcept { return test_bit(cpu_data.m_flags_l1_edx, 26); }
		[[nodiscard]] static bool has_sse3() noexcept { return test_bit(cpu_data.m_flags_l1_ecx, 0); }
		[[nodiscard]] static bool has_ssse3() noexcept { return test_bit(cpu_data.m_flags_l1_ecx, 9); }
		[[nodiscard]] static bool has_sse4_1() noexcept { return test_bit(cpu_data.m_flags_l1_ecx, 19); }
		[[nodiscard]] static bool has_sse4_2() noexcept { return test_bit(cpu_data.m_flags_l1_ecx, 20); }
		[[nodiscard]] static bool has_sse4a() noexcept { return is_amd() && test_bit(cpu_data.m_flags_l81_ecx, 6); }

		[[nodiscard]] static bool has_avx() noexcept { return test_bit(cpu_data.m_flags_l1_ecx, 28); }
		[[nodiscard]] static bool has_avx2() noexcept { return test_bit(cpu_data.m_flags_l7_ebx, 5); }
		[[nodiscard]] static bool has_fma() noexcept { return test_bit(cpu_data.m_flags_l1_ecx, 12); }

		[[nodiscard]] static bool has_avx512f() noexcept { return test_bit(cpu_data.m_flags_l7_ebx, 16); }
		[[nodiscard]] static bool has_avx512pf() noexcept { return test_bit(cpu_data.m_flags_l7_ebx, 26); }
		[[nodiscard]] static bool has_avx512er() noexcept { return test_bit(cpu_data.m_flags_l7_ebx, 27); }
		[[nodiscard]] static bool has_avx512cd() noexcept { return test_bit(cpu_data.m_flags_l7_ebx, 28); }

	private:
		SVM_PUBLIC static const cpuid cpu_data;

		SVM_PUBLIC cpuid() noexcept;

		std::uint32_t m_flags_l1_ecx = 0;
		std::uint32_t m_flags_l1_edx = 0;
		std::uint32_t m_flags_l7_ebx = 0;
		std::uint32_t m_flags_l81_ecx = 0;

		bool m_is_intel = false;
		bool m_is_amd = false;
	};
}

#endif