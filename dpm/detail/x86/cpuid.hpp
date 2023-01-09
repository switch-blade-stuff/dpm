/*
 * Created by switchblade on 2022-12-31.
 */

#pragma once

#include "../define.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_DYNAMIC_DISPATCH)

#include "../utility.hpp"

namespace dpm
{
	DPM_DECLARE_EXT_NAMESPACE
	{
		class DPM_PUBLIC cpuid
		{
			constexpr static int msr_bit = 5;
			constexpr static int cx8_bit = 8;
			constexpr static int sep_bit = 11;
			constexpr static int cmov_bit = 15;
			constexpr static int clfsh_bit = 19;
			constexpr static int xsave_bit = 27;

			constexpr static int f16c_bit = 29;
			constexpr static int cmpxchg16b_bit = 13;

			constexpr static int sse_bit = 25;
			constexpr static int sse2_bit = 26;
			constexpr static int sse3_bit = 0;
			constexpr static int ssse3_bit = 9;
			constexpr static int sse4_1_bit = 19;
			constexpr static int sse4_2_bit = 20;

			constexpr static int avx_bit = 28;
			constexpr static int avx2_bit = 5;
			constexpr static int fma_bit = 12;

			constexpr static int avx512f_bit = 16;
			constexpr static int avx512cd_bit = 28;
			constexpr static int avx512bw_bit = 30;
			constexpr static int avx512dq_bit = 17;
			constexpr static int avx512vl_bit = 31;

		public:
			[[nodiscard]] static bool is_intel() noexcept { return cpu_info.m_is_intel; }
			[[nodiscard]] static bool is_amd() noexcept { return cpu_info.m_is_amd; }

			[[nodiscard]] static bool has_msr() noexcept { return detail::test_bit(cpu_info.m_flags_l1_edx, msr_bit); }
			[[nodiscard]] static bool has_cx8() noexcept { return detail::test_bit(cpu_info.m_flags_l1_edx, cx8_bit); }
			[[nodiscard]] static bool has_sep() noexcept { return detail::test_bit(cpu_info.m_flags_l1_edx, sep_bit); }
			[[nodiscard]] static bool has_cmov() noexcept { return detail::test_bit(cpu_info.m_flags_l1_edx, cmov_bit); }
			[[nodiscard]] static bool has_clfsh() noexcept { return detail::test_bit(cpu_info.m_flags_l1_edx, clfsh_bit); }
			[[nodiscard]] static bool has_xsave() noexcept { return detail::test_bit(cpu_info.m_flags_l1_ecx, xsave_bit); }

			[[nodiscard]] static bool has_f16c() noexcept { return detail::test_bit(cpu_info.m_flags_l1_ecx, f16c_bit); }
			[[nodiscard]] static bool has_cmpxchg16b() noexcept { return detail::test_bit(cpu_info.m_flags_l1_ecx, cmpxchg16b_bit); }

			[[nodiscard]] static bool has_sse() noexcept { return detail::test_bit(cpu_info.m_flags_l1_edx, sse_bit); }
			[[nodiscard]] static bool has_sse2() noexcept { return detail::test_bit(cpu_info.m_flags_l1_edx, sse2_bit); }
			[[nodiscard]] static bool has_sse3() noexcept { return detail::test_bit(cpu_info.m_flags_l1_ecx, sse3_bit); }
			[[nodiscard]] static bool has_ssse3() noexcept { return detail::test_bit(cpu_info.m_flags_l1_ecx, ssse3_bit); }
			[[nodiscard]] static bool has_sse4_1() noexcept { return detail::test_bit(cpu_info.m_flags_l1_ecx, sse4_1_bit); }
			[[nodiscard]] static bool has_sse4_2() noexcept { return detail::test_bit(cpu_info.m_flags_l1_ecx, sse4_2_bit); }

			[[nodiscard]] static bool has_avx() noexcept { return detail::test_bit(cpu_info.m_flags_l1_ecx, avx_bit); }
			[[nodiscard]] static bool has_avx2() noexcept { return detail::test_bit(cpu_info.m_flags_l7_ebx, avx2_bit); }
			[[nodiscard]] static bool has_fma() noexcept { return detail::test_bit(cpu_info.m_flags_l1_ecx, fma_bit); }

			[[nodiscard]] static bool has_avx512f() noexcept { return detail::test_bit(cpu_info.m_flags_l7_ebx, avx512f_bit); }
			[[nodiscard]] static bool has_avx512cd() noexcept { return detail::test_bit(cpu_info.m_flags_l7_ebx, avx512cd_bit); }
			[[nodiscard]] static bool has_avx512bw() noexcept { return detail::test_bit(cpu_info.m_flags_l7_ebx, avx512bw_bit); }
			[[nodiscard]] static bool has_avx512dq() noexcept { return detail::test_bit(cpu_info.m_flags_l7_ebx, avx512dq_bit); }
			[[nodiscard]] static bool has_avx512vl() noexcept { return detail::test_bit(cpu_info.m_flags_l7_ebx, avx512vl_bit); }

		private:
			static const cpuid cpu_info;

			DPM_PRIVATE cpuid() noexcept;

			std::uint32_t m_flags_l1_ecx = 0;
			std::uint32_t m_flags_l1_edx = 0;
			std::uint32_t m_flags_l7_ebx = 0;

			bool m_is_intel = false;
			bool m_is_amd = false;
		};
	}
}

#endif