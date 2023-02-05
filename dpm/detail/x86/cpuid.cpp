/*
 * Created by switchblade on 2022-12-31.
 */

#include "cpuid.hpp"

#ifdef DPM_ARCH_X86

#if defined(__linux__) || defined(__ANDROID__) || defined(__FreeBSD__)

#include <unistd.h>
#include <fcntl.h>

#elif defined(_WIN32) || defined(__CYGWIN__)

#include <windows.h>

#elif defined(__APPLE__)

#include <sys/sysctl.h>

#endif

#if defined(_MSC_VER) || defined(__CYGWIN__)

#include <immintrin.h>
#include <intrin.h>

#endif

#ifndef DPM_USE_IMPORT

#include <string_view>
#include <string>

#endif

namespace dpm
{
	DPM_DECLARE_EXT_NAMESPACE
	{
		using cpuid_regs = std::uint32_t[4];

#if defined(_MSC_VER) || defined(__CYGWIN__)
		static DPM_FORCEINLINE void platform_cpuid(std::uint32_t leaf, cpuid_regs &regs) noexcept
		{
			__cpuid(reinterpret_cast<int *>(regs), static_cast<int>(leaf));
		}
		static DPM_FORCEINLINE std::uint32_t platrofm_xcr0() noexcept
		{
			return static_cast<std::uint32_t>(_xgetbv(0));
		}
#elif defined(__clang__) || defined(__GNUC__)
		static void DPM_FORCEINLINE platform_cpuid(std::uint32_t leaf, cpuid_regs &regs) noexcept
		{
			__asm("xchgq  %%rbx,%q1\n"
			      "\tcpuid\n"
			      "\txchgq  %%rbx,%q1"
					: "=a"(regs[0]), "=r" (regs[1]), "=c"(regs[2]), "=d"(regs[3])
					: "0"(leaf));
		}
		static std::uint32_t DPM_FORCEINLINE platrofm_xcr0() noexcept
		{
			uint32_t eax, edx;
			__asm(".byte 0x0F, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
			return eax;
		}
#else
#error "Unsupported compiler"
#endif

#ifdef __APPLE__
		static bool get_sysctl(std::string_view name) noexcept
		{
			std::size_t enabled_len = sizeof(int);
			int enabled = 0;
			return !sysctlbyname(name, &enabled, &enabled_len, nullptr, 0) && enabled;
		}
#endif

		cpuid::cpuid() noexcept
		{
			using namespace std::literals;
			{
				cpuid_regs regs = {0};
				platform_cpuid(0, regs);

				/* Check the vendor string to see if we are Intel or AMD */
				char vendor[13] = {0};
				reinterpret_cast<std::uint32_t &>(vendor[0]) = regs[1];
				reinterpret_cast<std::uint32_t &>(vendor[4]) = regs[3];
				reinterpret_cast<std::uint32_t &>(vendor[8]) = regs[2];

				if (const auto sv = std::string_view{vendor}; sv == "GenuineIntel"sv)
					m_is_intel = true;
				else if (sv == "AuthenticAMD"sv)
					m_is_amd = true;

				/* Initialize extension flags. */
				const auto max_leaf = regs[0];
				if (max_leaf >= 1) /* flags for leaf = 0x00000001 */
				{
					platform_cpuid(1, regs);
					m_flags_l1_ecx = regs[2];
					m_flags_l1_edx = regs[3];
				}
				if (max_leaf >= 7) /* flags for leaf = 0x00000007 */
				{
					platform_cpuid(7, regs);
					m_flags_l7_ebx = regs[1];
				}
			}

			/* Check for extension OS support. If XSAVE is used, check via XCR0. */
			if (detail::test_bit(m_flags_l1_ecx, xsave_bit))
			{
				const auto xcr0 = platrofm_xcr0();

				const bool sse_mask = xcr0 & 0x2;
				detail::mask_bit(m_flags_l1_edx, sse_bit, sse_mask);
				detail::mask_bit(m_flags_l1_edx, sse2_bit, sse_mask);
				detail::mask_bit(m_flags_l1_ecx, sse3_bit, sse_mask);
				detail::mask_bit(m_flags_l1_ecx, ssse3_bit, sse_mask);
				detail::mask_bit(m_flags_l1_ecx, sse4_1_bit, sse_mask);
				detail::mask_bit(m_flags_l1_ecx, sse4_2_bit, sse_mask);

				const bool avx_mask = (xcr0 & 0x6) == 0x6;
				detail::mask_bit(m_flags_l1_ecx, avx_bit, avx_mask);
				detail::mask_bit(m_flags_l7_ebx, avx2_bit, avx_mask);
				detail::mask_bit(m_flags_l1_ecx, fma_bit, avx_mask);

#ifdef __APPLE__
				const bool avx512_mask = get_sysctl("hw.optional.avx512f"sv);
#else
				const bool avx512_mask = (xcr0 & 0xe6) == 0xe6;
#endif

				detail::mask_bit(m_flags_l7_ebx, avx512f_bit, avx512_mask);
				detail::mask_bit(m_flags_l7_ebx, avx512cd_bit, avx512_mask);
				detail::mask_bit(m_flags_l7_ebx, avx512bw_bit, avx512_mask);
				detail::mask_bit(m_flags_l7_ebx, avx512dq_bit, avx512_mask);
				detail::mask_bit(m_flags_l7_ebx, avx512vl_bit, avx512_mask);
			}
			else
			{
				bool sse_mask = false;
				bool sse2_mask = false;
				bool sse3_mask = false;
				bool ssse3_mask = false;
				bool sse4_1_mask = false;
				bool sse4_2_mask = false;

#if defined(_WIN32) || defined(__CYGWIN__)
				sse_mask = IsProcessorFeaturePresent(PF_XMMI_INSTRUCTIONS_AVAILABLE);
				sse2_mask = IsProcessorFeaturePresent(PF_XMMI64_INSTRUCTIONS_AVAILABLE);
				sse3_mask = IsProcessorFeaturePresent(PF_SSE3_INSTRUCTIONS_AVAILABLE);
				ssse3_mask = IsProcessorFeaturePresent(PF_SSSE3_INSTRUCTIONS_AVAILABLE);
				sse4_1_mask = IsProcessorFeaturePresent(PF_SSE4_1_INSTRUCTIONS_AVAILABLE);
				sse4_2_mask = IsProcessorFeaturePresent(PF_SSE4_2_INSTRUCTIONS_AVAILABLE);
#elif defined(__linux__) || defined(__ANDROID__)
				if (const auto fd = open("/proc/cpuinfo", O_RDONLY); fd >= 0)
				{
					for (auto read_buff = std::string(4096, '\0');;)
					{
						const auto n = read(fd, read_buff.data(), read_buff.size());
						if (n <= 0) break;

						const std::string_view sv = std::string_view{read_buff.data(), static_cast<std::size_t>(n)};
						if (auto pos = sv.find("flags"sv); pos != std::string_view::npos)
						{
							pos = sv.find_first_not_of("\t :"sv, pos + 5);
							if (pos == std::string_view::npos) continue;

							sse_mask |= sv.find("m128"sv, pos) != std::string_view::npos;
							sse2_mask |= sv.find("sse2"sv, pos) != std::string_view::npos;
							sse3_mask |= sv.find("pni"sv, pos) != std::string_view::npos;
							ssse3_mask |= sv.find("ssse3"sv, pos) != std::string_view::npos;
							sse4_1_mask |= sv.find("sse4_1"sv, pos) != std::string_view::npos;
							sse4_2_mask |= sv.find("sse4_2"sv, pos) != std::string_view::npos;
						}
						if (sv.size() < read_buff.size()) break;
					}
					close(fd);
				}
#elif defined(__FreeBSD__)
				if (const auto fd = open("/var/run/dmesg.boot", O_RDONLY); dmsg >= 0)
				{
					for (auto read_buff = std::string(4096, '\0');;)
					{
						const auto n = read(fd, read_buff.data(), read_buff.size());
						if (n <= 0) break;

						const std::string_view sv = std::string_view{read_buff.data(), static_cast<std::size_t>(n)};
						if (auto pos = sv.find("  Features"sv); pos != std::string_view::npos)
						{
							if (const auto i = sv.find('<', pos); i != std::string_view::npos)
								pos = i;

							sse_mask |= sv.find("SSE"sv, pos) != std::string_view::npos;
							sse2_mask |= sv.find("SSE2"sv, pos) != std::string_view::npos;
							sse3_mask |= sv.find("SSE3"sv, pos) != std::string_view::npos;
							ssse3_mask |= sv.find("SSSE3"sv, pos) != std::string_view::npos;
							sse4_1_mask |= sv.find("SSE4.1"sv, pos) != std::string_view::npos;
							sse4_2_mask |= sv.find("SSE4.2"sv, pos) != std::string_view::npos;
						}
						if (sv.size() < read_buff.size()) break;
					}
					close(fd);
				}
#elif defined(__APPLE__)
				sse_mask = get_sysctl("hw.optional.sse");
				sse2_mask = get_sysctl("hw.optional.sse2");
				sse3_mask = get_sysctl("hw.optional.sse3");
				ssse3_mask = get_sysctl("hw.optional.supplementalsse3");
				sse4_1_mask = get_sysctl("hw.optional.sse4_1");
				sse4_2_mask = get_sysctl("hw.optional.sse4_2");
#else
#error "Unsupported OS"
#endif

				detail::mask_bit(m_flags_l1_edx, sse_bit, sse_mask);
				detail::mask_bit(m_flags_l1_edx, sse2_bit, sse2_mask);
				detail::mask_bit(m_flags_l1_ecx, sse3_bit, sse3_mask);
				detail::mask_bit(m_flags_l1_ecx, ssse3_bit, ssse3_mask);
				detail::mask_bit(m_flags_l1_ecx, sse4_1_bit, sse4_1_mask);
				detail::mask_bit(m_flags_l1_ecx, sse4_2_bit, sse4_2_mask);
			}
		}

		const cpuid cpuid::cpu_info;
	}
}

#endif