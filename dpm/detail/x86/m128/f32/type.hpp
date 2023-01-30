/*
 * Created by switchblade on 2023-01-06.
 */

#pragma once

#include "../../type_fwd.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE)

#include "../cvt.hpp"

namespace dpm
{
	DPM_DECLARE_EXT_NAMESPACE
	{
		template<std::size_t N, std::size_t Align> requires detail::x86_overload_128<float, N, Align>
		struct native_data_type<simd<float, detail::avec<N, Align>>> { using type = __m128; };
		template<std::size_t N, std::size_t Align> requires detail::x86_overload_128<float, N, Align>
		struct native_data_size<simd<float, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<float, N, 16>()> {};
	}

	template<std::size_t N, std::size_t A> requires detail::x86_overload_128<float, N, A>
	class const_where_expression<simd_mask<float, detail::avec<N, A>>, simd<float, detail::avec<N, A>>>
	{
		template<typename U, typename Abi, typename K>
		friend inline simd<U, Abi> ext::blend(const simd<U, Abi> &, const const_where_expression<K, simd<U, Abi>> &);
		template<typename U, typename Abi, typename K>
		friend inline simd_mask<U, Abi> ext::blend(const simd_mask<U, Abi> &, const const_where_expression<K, simd_mask<U, Abi>> &);

		template<typename U>
		friend inline U ext::blend(const U &, const const_where_expression<bool, U> &);

	protected:
		using mask_t = simd_mask<float, detail::avec<N, A>>;
		using simd_t = simd<float, detail::avec<N, A>>;
		using value_type = float;

	public:
		const_where_expression(const const_where_expression &) = delete;
		const_where_expression &operator=(const const_where_expression &) = delete;

		const_where_expression(mask_t mask, simd_t &data) noexcept : m_mask(mask), m_data(data) {}
		const_where_expression(mask_t mask, const simd_t &data) noexcept : m_mask(mask), m_data(const_cast<simd_t &>(data)) {}

		[[nodiscard]] DPM_FORCEINLINE simd_t operator-() const && noexcept { return ext::blend(m_data, -m_data, m_mask); }
		[[nodiscard]] DPM_FORCEINLINE simd_t operator+() const && noexcept { return ext::blend(m_data, +m_data, m_mask); }

		/** Copies selected elements to \a mem. */
		template<typename U, typename Flags>
		DPM_FORCEINLINE void copy_to(U *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
#ifdef DPM_HAS_AVX
			if constexpr (detail::aligned_tag<Flags, 16> && sizeof(U) == 4)
			{
				const auto v_mask = ext::to_native_data(m_mask);
				const auto v_data = ext::to_native_data(m_data);
				for (std::size_t i = 0; i < mask_t::size(); i += 4)
				{
					const auto mi = std::bit_cast<__m128i>(detail::maskzero<float>(v_mask[i / 4], mask_t::size() - i));
					auto v = v_data[i / 4];
					if constexpr (detail::unsigned_integral_of_size<std::remove_volatile_t<U>, 4>)
						v = std::bit_cast<__m128>(detail::cvt_f32_u32(v));
					else if constexpr (detail::signed_integral_of_size<std::remove_volatile_t<U>, 4>)
						v = std::bit_cast<__m128>(_mm_cvtps_epi32(v));
					_mm_maskstore_ps(mem + i, mi, v);
				}
			}
			else
#endif
				for (std::size_t i = 0; i < mask_t::size(); ++i) if (m_mask[i]) mem[i] = static_cast<U>(m_data[i]);
		}

	protected:
		mask_t m_mask;
		simd_t &m_data;
	};
	template<std::size_t N, std::size_t A> requires detail::x86_overload_128<float, N, A>
	class where_expression<simd_mask<float, detail::avec<N, A>>, simd<float, detail::avec<N, A>>> : public const_where_expression<simd_mask<float, detail::avec<N, A>>, simd<float, detail::avec<N, A>>>
	{
		using base_expr = const_where_expression<simd_mask<float, detail::avec<N, A>>, simd<float, detail::avec<N, A>>>;
		using value_type = typename base_expr::value_type;
		using simd_t = typename base_expr::simd_t;
		using mask_t = typename base_expr::mask_t;

		using base_expr::m_mask;
		using base_expr::m_data;

	public:
		using base_expr::const_where_expression;

		template<typename U>
		DPM_FORCEINLINE void operator=(U &&value) && noexcept requires std::is_convertible_v<U, value_type> { m_data = ext::blend(m_data, simd_t{std::forward<U>(value)}, m_mask); }

		/** Copies selected elements from \a mem. */
		template<typename U, typename Flags>
		DPM_FORCEINLINE void copy_from(const U *mem, Flags) && noexcept requires is_simd_flag_type_v<Flags>
		{
#ifdef DPM_HAS_AVX
			if constexpr (detail::aligned_tag<Flags, 16> && sizeof(U) == 4)
			{
				const auto v_mask = ext::to_native_data(m_mask);
				const auto v_data = ext::to_native_data(m_data);
				for (std::size_t i = 0; i < mask_t::size(); i += 4)
				{
					const auto mi = std::bit_cast<__m128i>(detail::maskzero<T>(v_mask[i / 4], mask_t::size() - i));
					auto v = _mm_maskload_ps(mem + i, mi);
					if constexpr (detail::unsigned_integral_of_size<std::remove_volatile_t<U>, 4>)
						v_data[i / 4] = detail::cvt_u32_f32(std::bit_cast<__m128i>(v));
					else if constexpr (detail::signed_integral_of_size<std::remove_volatile_t<U>, 4>)
						v_data[i / 4] = _mm_cvtepi32_ps(std::bit_cast<__m128>(v));
					else
						v_data[i / 4] = v;
				}
			}
			else
#endif
				for (std::size_t i = 0; i < mask_t::size(); ++i) if (m_mask[i]) m_data[i] = static_cast<float>(mem[i]);
		}
	};

	namespace detail
	{
		template<std::size_t N, std::size_t A> requires detail::x86_overload_128<float, N, A>
		struct native_access<simd<float, avec<N, A>>>
		{
			using simd_t = simd<float, avec<N, A>>;

			static std::span<__m128, simd_t::data_size> to_native_data(simd_t &x) noexcept { return {x.m_data}; }
			static std::span<const __m128, simd_t::data_size> to_native_data(const simd_t &x) noexcept { return {x.m_data}; }
		};
	}

#pragma region "simd reductions"
	namespace detail
	{
		template<std::size_t N, std::size_t A, typename Op>
		DPM_FORCEINLINE __m128 reduce_lanes_f32(const simd<float, detail::avec<N, A>> &x, __m128 idt, Op op) noexcept
		{
			auto res = _mm_undefined_ps();
			for (std::size_t i = 0; i < x.size(); i += 4)
			{
				if (const auto v = maskblend_f32(x.size() - i, ext::to_native_data(x)[i / 4], idt); i != 0)
					res = op(res, v);
				else
					res = v;
			}
			return res;
		}
		template<std::size_t N, std::size_t A, typename Op>
		DPM_FORCEINLINE float reduce_f32(const simd<float, detail::avec<N, A>> &x, __m128 idt, Op op) noexcept
		{
			const auto a = reduce_lanes_f32(x, idt, op);
#ifndef DPM_HAS_SSE3
			const auto b = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 3, 1, 1));
#else
			const auto b = _mm_movehdup_ps(a);
#endif
			const auto c = op(a, b);
			return _mm_cvtss_f32(op(c, _mm_movehl_ps(b, c)));
		}
	}

	/** Horizontally reduced elements of \a x using operation `Op`. */
	template<std::size_t N, std::size_t A, typename Op = std::plus<>>
	[[nodiscard]] DPM_FORCEINLINE float reduce(const simd<float, detail::avec<N, A>> &x, Op op = {}) noexcept requires detail::x86_overload_128<float, N, A>
	{
		if constexpr (std::same_as<Op, std::plus<>> || std::same_as<Op, std::plus<float>>)
			return detail::reduce_f32(x, _mm_setzero_ps(), [](auto a, auto b) { return _mm_add_ps(a, b); });
		else if constexpr (std::same_as<Op, std::multiplies<>> || std::same_as<Op, std::multiplies<float>>)
			return detail::reduce_f32(x, _mm_set1_ps(1.0), [](auto a, auto b) { return _mm_mul_ps(a, b); });
		else
			return detail::reduce_impl<simd<float, detail::avec<N, A>>::size()>(x, op);
	}

	/** Calculates horizontal minimum of elements of \a x. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE float hmin(const simd<float, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_128<float, N, A>
	{
		const auto max = std::numeric_limits<float>::max();
		return detail::reduce_f32(x, _mm_set1_ps(max), [](auto a, auto b) { return _mm_min_ps(a, b); });
	}
	/** Calculates horizontal maximum of elements of \a x. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE float hmax(const simd<float, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_128<float, N, A>
	{
		const auto min = std::numeric_limits<float>::min();
		return detail::reduce_f32(x, _mm_set1_ps(min), [](auto a, auto b) { return _mm_max_ps(a, b); });
	}
#pragma endregion

#pragma region "simd algorithms"
	/** Returns an SIMD vector of minimum elements of \a a and \a b. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE simd<float, detail::avec<N, A>> min(
			const simd<float, detail::avec<N, A>> &a,
			const simd<float, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_128<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<float, detail::avec<N, A>>>;

		simd<float, detail::avec<N, A>> result = {};
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < data_size; ++i) result_data[i] = _mm_min_ps(a_data[i], b_data[i]);
		return result;
	}
	/** Returns an SIMD vector of maximum elements of \a a and \a b. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE simd<float, detail::avec<N, A>> max(
			const simd<float, detail::avec<N, A>> &a,
			const simd<float, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_128<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<float, detail::avec<N, A>>>;

		simd<float, detail::avec<N, A>> result = {};
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < data_size; ++i) result_data[i] = _mm_max_ps(a_data[i], b_data[i]);
		return result;
	}

	/** Returns a pair of SIMD vectors of minimum and maximum elements of \a a and \a b. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::pair<simd<float, detail::avec<N, A>>, simd<float, detail::avec<N, A>>> minmax(
			const simd<float, detail::avec<N, A>> &a,
			const simd<float, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_128<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<float, detail::avec<N, A>>>;

		std::pair<simd<float, detail::avec<N, A>>, simd<float, detail::avec<N, A>>> result = {};
		auto min_data = ext::to_native_data(result.first), max_data = ext::to_native_data(result.second);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < data_size; ++i)
		{
			min_data[i] = _mm_min_ps(a_data[i], b_data[i]);
			max_data[i] = _mm_max_ps(a_data[i], b_data[i]);
		}
		return result;
	}

	/** Clamps elements \f \a x between corresponding elements of \a ming and \a max. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE simd<float, detail::avec<N, A>> clamp(
			const simd<float, detail::avec<N, A>> &x,
			const simd<float, detail::avec<N, A>> &min,
			const simd<float, detail::avec<N, A>> &max)
	noexcept requires detail::x86_overload_128<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<float, detail::avec<N, A>>>;

		simd<float, detail::avec<N, A>> result = {};
		auto result_data = ext::to_native_data(result);
		const auto min_data = ext::to_native_data(min);
		const auto max_data = ext::to_native_data(max);
		const auto x_data = ext::to_native_data(x);

		for (std::size_t i = 0; i < data_size; ++i) result_data[i] = _mm_min_ps(_mm_max_ps(x_data[i], min_data[i]), max_data[i]);
		return result;
	}
#pragma endregion
}

#endif