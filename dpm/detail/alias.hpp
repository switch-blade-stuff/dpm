/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "define.hpp"
#include "utility.hpp"

#ifndef DPM_USE_IMPORT

#include <utility>

#endif

namespace dpm::detail
{
	/* SIMD vector & mask elements are referenced via aliasing-enabled types. If a reference wrapper is used, RVO will bypass deleted
	 * constructors and may cause dangling references. As such, `reinterpret_cast` into a reference of a type that is allowed to alias
	 * others instead. Masks always need this due to the extension of boolean values into an integer mask, vectors only need it on GCC
	 * due to strict aliasing (other compilers care less about aliasing). See the following godbolt example for details, note that GCC
	 * enforces strict aliasing and turns the indented `vpaddd` into `vpxor`, while MSVC and CLang do not:
     * https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIAGz%2BpK4AMngMmAByPgBGmMQgAKyJpAAOqAqETgwe3r4BQemZjgJhEdEscQnJtpj2JQxCBEzEBLk%2BfoG19dlNLQRlUbHxSSkKza3t%2BV3j/YMVVaMAlLaoXsTI7BwmGgCC5gDM4cjeWADUJgdueCws4QTE4QB0CJfYO/tmRwwnXueXbmQ43wgjeH0Ox1OmAuVxaxCYAE8wXsIT8oTC3DFCMj9iivngqFgqGcAPoAWSEbhJADVsAAlcF4g5E8LQ2luAAqAHk6W5diEQqSSQA3TAOEiiWi0cFfFkRM5k3YATRJ/IAkrshIzdgRMCxUgZdQCCAjUoxWNCOTivJkjGdMix0CS6nrGAQYQARM5Wg5WFFMupKbWHOVs7Ccnl8gUy5mYGjyxUq9WaoVMAgPPAxLy6kkkqC5liIkliPBMBS5pZLbW6/WGzDG03mtjenHjYheBwK5WqkIaoT2m5Ol1sUEogDsftxuzOM7ONYNafrVxNZuYzbkONnA8dzvoI/d5n8qDN8KIxEuHogcguZn8h%2BFYi8mCrey3JgnHy3W5YIsf0IvZwPt4S6Tl%2Bs7EJgBDrAwZwAFQEAgeAKJcoGzu%2BHraluVCPK43pnMe8RpiQsEQGgDDjNuQ57m6N7%2BEwpBnGRFEOlRrqCLRMQvtOaEfq%2BYEXHxYEQVBxAwUwTw/kBT5wWcMQSb%2BwEoZ%2BPEYf6uypI8D66iAykzhyZySX%2BSnjhhvrBl8rgEtqTDZqgZzstyvL8oKLBeLQZjFrQpYKBABZcGYAAceBnEwXHvpOOwAJxMe6NlEIBZieqStwkkoBBcM6qR4AcZgQMCIAgFiBAkqI4wAvlID3DlJIEG8EBmJWxm4tFAgUXFdnoGmTAkolAEQfc8QaZBJVlrVVwxZRu5sWNbgVVVHljdgcF1feDVNVFE3tWcnXNMWSX9YIg3CSNZVXCxU37uVBDoAV801W8y0HNgECHqF63%2BpF/DEGceXXQVmQAF6YDVZzBQBGgoaDGJnJIkPWNYeBLNtXXFiYiQWHgaNegBO3dUwaMY1jMm4z1BOY4kpkRRokXCdBIXrWOqm7FtDmRs5BluR5DCoCWZa%2BSSLD%2BUFIVhbxzWbbZCVJQLP5pRlmBZTlv03YVhAnTNc2CNVi31Y1ZkfRL8Uk71BxegdurEENxWlTNE2awQ2uPc9q1Vvr4utbFksk0w%2B2YANlvHTbV0q3d7qwStt5vW7jKfSQP0VYDwPumDptnBDvpQwCMNw5YCNI97ZNEzjKP4%2Bj5NerByO7WYhcU%2B9NOQXT%2BPR4zHArLQnCJLwfgcFopCoJwbjw5Y9prBs/5fDwpAEJobcrAA1kkAVPF8ZjJJI/hjlwAWRZIXAHCkHccJIvAsBIGgaKQPd9wPHC8AoICXzPvdt6QcCwDAiAoKg%2Bp0PE5CUDQL/egCQTiGGAFwRIF9SBYGFHgTY1I8CYAAO5clXJwKeNBaAWwfhAGIs9SBYmYMQBEGDeBEJaAiLkMRtBimflPIB%2B4uQMFoKQl%2BMC9TgPEOw/AEEHB4FFA/dhmBVBimzFsKeA0j59y8jEeEJCPBYAIRmM%2B3BX5UAMMABQiCUFoMYGQmQggRBiHYFIQx8glBqAIboMw%2BhwEoGHjYWRD9IArGPA0IRABaYEF5TC50sGYDQZxPFckSp4wswpVD3xdPw5wEBXBTD8FwYIDB0DzGGAkZJRQsgCESXobJDR0mVBGMkuwdDegTDaJ4DoegymxMaJUopixSmVLyS0uYrIFglJWAoMemwJDt07t3Aht8ziqACv4Tx/hJCMQMHaSBTwNCLJ%2BrgQgcdDhcCWLwZ%2BWhKykEXnvJ4Y5N6SEkIkHKAUuAaESAFAK%2BhOAn1IGfA4BwngvPeR8z5QRr68FvvfR%2B09Z4rHfl/IBWUQEANIj/cFIwtHMFSAoBAqACAcLgQgpBqD0FqL4HQHBlB8HsIoSQgxRKqE0LoQYxhbpmGsIIVgQsRhuF914eUwRBCRFiO0tiqRBDZHyIRIorYfcVFkJWBopgWidGYv0di/gRjJSmOkHKixKh1DsN0MkuZxhHH6EzC4iAbjUgeM4N466vjHGBOCaE4JESondHKXEhJ1T8jJNCJ0jJ%2BSMg5JyM6pJaQvWFPdcUzJ9r6l9EmL62pMSGjhoGEG5pthWmRvaa0Jp3TVjrH6Zs%2B5HAu5XxGZwMZEypkzOAMgZAP0HheAYPPJGEBVlnhvAcTZ2ygUrAQJgJgWAEgGpzY855rzPlDved8gtd9bAAp2a/EFEAkBgr/mQCgULgEjGFMgVIqQRRcEiiSOs4wSSqGmai%2BBmApV6J7pg3F8RcEEr7qSkl4RKHUNoQ4SlP8mEsLYcyzhjKhW8BZfwtlwjRHIHEQYnl7C%2BWUMFcox4qieBis0dojF56DHKuMeIMxyrFCqusSAA4dijAOP8U4vV8BDXGo4AAenyn4qwASglUetVRloyAkC7GpAADWiT0R1qS2kpLSfGkp/rijZAEwU7IaaQ11JjUmvIfrZMVI6eUD1KaqkKdqY04TmSel9NMYM3Nwz2GjMPTMlgCg12AW3U8Pd7p634EbRsrZgKX57KPv2/Dg7h1DvzSZzg/yn5tv2SAQ5xyxynPOYFK5Ny7lHwOMZm%2BAXXO7MM2YRLvzktTr2aKYgmRnCSCAA%3D */

	template<typename T>
	struct select_alias { using type = T; };

	typedef DPM_MAY_ALIAS float alias_float_t;
	typedef DPM_MAY_ALIAS double alias_double_t;

	typedef DPM_MAY_ALIAS std::int8_t alias_int8_t;
	typedef DPM_MAY_ALIAS std::int16_t alias_int16_t;
	typedef DPM_MAY_ALIAS std::int32_t alias_int32_t;
	typedef DPM_MAY_ALIAS std::int64_t alias_int64_t;

	typedef DPM_MAY_ALIAS std::uint8_t alias_uint8_t;
	typedef DPM_MAY_ALIAS std::uint16_t alias_uint16_t;
	typedef DPM_MAY_ALIAS std::uint32_t alias_uint32_t;
	typedef DPM_MAY_ALIAS std::uint64_t alias_uint64_t;

	template<>
	struct select_alias<float> { using type = alias_float_t; };
	template<>
	struct select_alias<double> { using type = alias_double_t; };

	template<>
	struct select_alias<std::int8_t> { using type = alias_int8_t; };
	template<>
	struct select_alias<std::int16_t> { using type = alias_int16_t; };
	template<>
	struct select_alias<std::int32_t> { using type = alias_int32_t; };
	template<>
	struct select_alias<std::int64_t> { using type = alias_int64_t; };

	template<>
	struct select_alias<std::uint8_t> { using type = alias_uint8_t; };
	template<>
	struct select_alias<std::uint16_t> { using type = alias_uint16_t; };
	template<>
	struct select_alias<std::uint32_t> { using type = alias_uint32_t; };
	template<>
	struct select_alias<std::uint64_t> { using type = alias_uint64_t; };

	template<typename T>
	using simd_alias = typename select_alias<T>::type;

	template<typename T>
	class DPM_MAY_ALIAS mask_alias
	{
	public:
		mask_alias() = delete;
		mask_alias(const mask_alias &) = delete;

		constexpr DPM_FORCEINLINE operator bool() const noexcept { return static_cast<bool>(m_value); }

		template<std::convertible_to<bool> U>
		constexpr DPM_FORCEINLINE mask_alias &operator=(U &&value) noexcept
		{
			m_value = extend_bool<T>(value);
			return *this;
		}

		template<std::convertible_to<bool> U>
		constexpr DPM_FORCEINLINE mask_alias &operator|=(U &&value) noexcept
		{
			m_value |= extend_bool<T>(value);
			return *this;
		}
		template<std::convertible_to<bool> U>
		constexpr DPM_FORCEINLINE mask_alias &operator&=(U &&value) noexcept
		{
			m_value &= extend_bool<T>(value);
			return *this;
		}
		template<std::convertible_to<bool> U>
		constexpr DPM_FORCEINLINE mask_alias &operator^=(U &&value) noexcept
		{
			m_value ^= extend_bool<T>(value);
			return *this;
		}

		friend constexpr void swap(mask_alias &a, mask_alias &b) noexcept { std::swap(a.m_value, b.m_value); }
		friend constexpr void swap(bool &a, mask_alias &b) noexcept { a = static_cast<bool>(std::exchange(b.m_value, extend_bool<T>(a))); }
		friend constexpr void swap(mask_alias &a, bool &b) noexcept { b = static_cast<bool>(std::exchange(a.m_value, extend_bool<T>(b))); }

	private:
		T m_value;
	};
}