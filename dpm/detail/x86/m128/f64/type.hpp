/*
 * Created by switchblade on 2023-01-06.
 */

#pragma once

#include "../../fwd.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2)

#include "utility.hpp"

namespace dpm
{
	namespace detail
	{
		template<std::size_t I, typename OutAbi, typename XAbi, typename... Abis>
		[[nodiscard]] inline auto concat_impl(simd_mask<double, OutAbi> &out, const simd_mask<double, XAbi> &x, const simd_mask<double, Abis> &...rest) noexcept
		{
			auto *data = reinterpret_cast<double *>(ext::to_native_data(out).data());
			if constexpr (I % sizeof(__m128d) != 0)
				x.copy_to(data + I, element_aligned);
			else
				x.copy_to(data + I, vector_aligned);

			if constexpr (sizeof...(Abis) != 0) concat_impl<I + simd_mask<double, XAbi>::size()>(out, rest...);
		}
		template<std::size_t I, typename OutAbi, typename XAbi, typename... Abis>
		[[nodiscard]] inline auto concat_impl(simd<double, OutAbi> &out, const simd<double, XAbi> &x, const simd<double, Abis> &...rest) noexcept
		{
			auto *data = reinterpret_cast<double *>(ext::to_native_data(out).data());
			if constexpr (I % sizeof(__m128d) != 0)
				x.copy_to(data + I, element_aligned);
			else
				x.copy_to(data + I, vector_aligned);

			if constexpr (sizeof...(Abis) != 0) concat_impl<I + simd<double, XAbi>::size()>(out, rest...);
		}

		[[nodiscard]] inline __m128d x86_maskzero_f64(std::size_t n, __m128d v) noexcept
		{
			if (n == 1)
			{
#if defined(DPM_HAS_SSE4_1)
				return _mm_blend_pd(v, _mm_setzero_pd(), 0b10);
#else
				const auto mask = std::bit_cast<double>(0xffff'ffff'ffff'ffff);
				return _mm_and_pd(v, _mm_set_pd(0.0, mask));
#endif
			}
			else
				return v;
		}
		[[nodiscard]] inline __m128d x86_maskone_vector_f64(std::size_t n, __m128d v) noexcept
		{
			if (n == 1)
			{
#if defined(DPM_HAS_SSE4_1)
				const auto mask = std::bit_cast<double>(0xffff'ffff'ffff'ffff);
				return _mm_blend_pd(v, _mm_set1_pd(mask), 0b10);
#else
				const auto mask = std::bit_cast<double>(0xffff'ffff'ffff'ffff);
				return _mm_or_pd(v, _mm_set_pd(mask, 0.0));
#endif
			}
			else
				return v;
		}
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		template<std::size_t N, std::size_t Align> requires detail::x86_overload_m128<double, N, Align>
		struct native_data_type<simd_mask<double, detail::avec<N, Align>>> { using type = __m128d; };
		template<std::size_t N, std::size_t Align> requires detail::x86_overload_m128<double, N, Align>
		struct native_data_size<simd_mask<double, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<double, N, 16>()> {};

		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128d, detail::align_data<double, N, 16>()> to_native_data(simd_mask<double, detail::avec<N, A>> &) noexcept requires detail::x86_overload_m128<double, N, A>;
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128d, detail::align_data<double, N, 16>()> to_native_data(const simd_mask<double, detail::avec<N, A>> &) noexcept requires detail::x86_overload_m128<double, N, A>;
	}

	template<std::size_t N, std::size_t Align> requires detail::x86_overload_m128<double, N, Align>
	class simd_mask<double, detail::avec<N, Align>>
	{
		friend struct detail::native_access<simd_mask>;

		constexpr static auto data_size = ext::native_data_size_v<simd_mask>;
		constexpr static auto alignment = std::max(Align, alignof(__m128d));

		using data_type = __m128d[data_size];
		using value_proxy = detail::mask_element<std::int64_t>;

	public:
		using value_type = bool;
		using reference = detail::mask_element<std::int64_t> &;

		using abi_type = detail::avec<N, Align>;
		using simd_type = simd<double, abi_type>;

		/** Returns width of the SIMD mask. */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	public:
		constexpr simd_mask() noexcept = default;
		constexpr simd_mask(const simd_mask &) noexcept = default;
		constexpr simd_mask &operator=(const simd_mask &) noexcept = default;
		constexpr simd_mask(simd_mask &&) noexcept = default;
		constexpr simd_mask &operator=(simd_mask &&) noexcept = default;

		/** Initializes the SIMD mask object with a native SSE mask vector.
		 * @note This constructor is available for overload resolution only when the SIMD mask contains a single SSE vector. */
		constexpr simd_mask(__m128d native) noexcept requires (data_size == 1) { m_data[0] = native; }
		/** Initializes the SIMD mask object with an array of native SSE mask vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd_mask) / sizeof(__m128d)`. */
		constexpr simd_mask(const __m128d (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

		/** Initializes the underlying elements with \a value. */
		simd_mask(value_type value) noexcept
		{
			const auto v = value ? _mm_set1_pd(std::bit_cast<double>(0xffff'ffff'ffff'ffff)) : _mm_setzero_pd();
			std::fill_n(m_data, data_size, v);
		}
		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		simd_mask(const simd_mask<U, detail::avec<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (std::same_as<U, value_type> && alignof(decltype(other)) >= alignment)
				std::copy_n(reinterpret_cast<const __m128d *>(ext::to_native_data(other).data()), data_size, m_data);
			else
				for (std::size_t i = 0; i < size(); ++i) operator[](i) = other[i];
		}
		/** Initializes the underlying elements from \a mem. */
		template<typename Flags>
		simd_mask(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying elements from \a mem. */
		template<typename Flags>
		void copy_from(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags>
		{
			for (std::size_t i = 0; i < size(); i += 2)
			{
				double f0 = 0.0, f1 = 0.0;
				switch (size() - i)
				{
					default: f1 = std::bit_cast<double>(detail::extend_bool<std::int64_t>(mem[i + 1])); [[fallthrough]];
					case 1: f0 = std::bit_cast<double>(detail::extend_bool<std::int64_t>(mem[i]));
				}
				m_data[i / 2] = _mm_set_pd(f1, f0);
			}
		}
		/** Copies the underlying elements to \a mem. */
		template<typename Flags>
		void copy_to(value_type *mem, Flags) const noexcept requires is_simd_flag_type_v<Flags>
		{
			for (std::size_t i = 0; i < size(); i += 2)
				switch (const auto bits = _mm_movemask_pd(m_data[i / 2]); size() - i)
				{
					default: mem[i + 1] = bits & 0b10; [[fallthrough]];
					case 1: mem[i] = bits & 0b01;
				}
		}

		[[nodiscard]] reference operator[](std::size_t i) noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<value_proxy *>(m_data)[i];
		}
		[[nodiscard]] value_type operator[](std::size_t i) const noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<const std::int64_t *>(m_data)[i];
		}

		[[nodiscard]] simd_mask operator!() const noexcept
		{
			simd_mask result;
			const auto mask = _mm_set1_pd(std::bit_cast<double>(0xffff'ffff'ffff'ffff));
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_xor_pd(m_data[i], mask);
			return result;
		}

		[[nodiscard]] friend simd_mask operator&(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_and_pd(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend simd_mask operator|(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_or_pd(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend simd_mask operator^(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_xor_pd(a.m_data[i], b.m_data[i]);
			return result;
		}

		friend simd_mask &operator&=(simd_mask &a, const simd_mask &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_and_pd(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend simd_mask &operator|=(simd_mask &a, const simd_mask &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_or_pd(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend simd_mask &operator^=(simd_mask &a, const simd_mask &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_xor_pd(a.m_data[i], b.m_data[i]);
			return a;
		}

		[[nodiscard]] friend simd_mask operator&&(const simd_mask &a, const simd_mask &b) noexcept { return a & b; }
		[[nodiscard]] friend simd_mask operator||(const simd_mask &a, const simd_mask &b) noexcept { return a | b; }

		[[nodiscard]] friend simd_mask operator==(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			for (std::size_t i = 0; i < data_size; ++i)
			{
				const auto va = std::bit_cast<__m128i>(a.m_data[i]);
				const auto vb = std::bit_cast<__m128i>(b.m_data[i]);
				result.m_data[i] = std::bit_cast<__m128d>(_mm_cmpeq_epi32(va, vb));
			}
			return result;
		}
		[[nodiscard]] friend simd_mask operator!=(const simd_mask &a, const simd_mask &b) noexcept
		{
			return !(a == b);
		}

	private:
		alignas(alignment) data_type m_data;
	};

	template<std::size_t N, std::size_t A> requires detail::x86_overload_m128<double, N, A>
	class const_where_expression<simd_mask<double, detail::avec<N, A>>, simd_mask<double, detail::avec<N, A>>>
	{
		template<typename U, typename Abi, typename K>
		friend inline simd<U, Abi> ext::blend(const simd<U, Abi> &, const const_where_expression<K, simd<U, Abi>> &);
		template<typename U, typename Abi, typename K>
		friend inline simd_mask<U, Abi> ext::blend(const simd_mask<U, Abi> &, const const_where_expression<K, simd_mask<U, Abi>> &);

		template<typename U>
		friend inline U ext::blend(const U &, const const_where_expression<bool, U> &);

	protected:
		using mask_t = simd_mask<double, detail::avec<N, A>>;
		using value_type = bool;

	public:
		const_where_expression(const const_where_expression &) = delete;
		const_where_expression &operator=(const const_where_expression &) = delete;

		const_where_expression(mask_t mask, mask_t &data) noexcept : m_mask(mask), m_data(data) {}
		const_where_expression(mask_t mask, const mask_t &data) noexcept : m_mask(mask), m_data(const_cast<mask_t &>(data)) {}

		/** Copies selected elements to \a mem. */
		template<typename Flags>
		void copy_to(bool *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
			const auto v_mask = ext::to_native_data(m_data);
			const auto v_data = ext::to_native_data(m_data);
			for (std::size_t i = 0; i < mask_t::size(); ++i)
			{
				const auto bits = _mm_movemask_pd(_mm_and_pd(v_data[i / 2], v_mask[i / 2]));
				switch (mask_t::size() - i)
				{
					default: mem[i + 1] = bits & 0b10; [[fallthrough]];
					case 1: mem[i] = bits & 0b01;
				}
			}
		}

	protected:
		mask_t m_mask;
		mask_t &m_data;
	};
	template<std::size_t N, std::size_t A> requires detail::x86_overload_m128<double, N, A>
	class where_expression<simd_mask<double, detail::avec<N, A>>, simd_mask<double, detail::avec<N, A>>>
			: public const_where_expression<simd_mask<double, detail::avec<N, A>>, simd_mask<double, detail::avec<N, A>>>
	{
		using base_expr = const_where_expression<simd_mask<double, detail::avec<N, A>>, simd_mask<double, detail::avec<N, A>>>;
		using value_type = typename base_expr::value_type;
		using mask_t = typename base_expr::mask_t;

		using base_expr::m_mask;
		using base_expr::m_data;

	public:
		using base_expr::const_where_expression;

		template<std::convertible_to<value_type> U>
		void operator=(U &&value) && noexcept { m_data = ext::blend(m_data, mask_t{std::forward<U>(value)}, m_mask); }

		template<typename U>
		void operator&=(U &&value) && noexcept requires std::convertible_to<U, value_type>
		{
			const auto new_data = m_data & mask_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		void operator|=(U &&value) && noexcept requires std::convertible_to<U, value_type>
		{
			const auto new_data = m_data | mask_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		void operator^=(U &&value) && noexcept requires std::convertible_to<U, value_type>
		{
			const auto new_data = m_data ^ mask_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}

		/** Copies selected elements from \a mem. */
		template<typename Flags>
		void copy_from(const bool *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
			const auto v_mask = ext::to_native_data(m_data);
			const auto v_data = ext::to_native_data(m_data);
			for (std::size_t i = 0; i < mask_t::size(); ++i)
			{
				double f0 = 0.0, f1 = 0.0;
				switch (mask_t::size() - i)
				{
					default: f1 = std::bit_cast<double>(detail::extend_bool<std::int64_t>(mem[i + 1])); [[fallthrough]];
					case 1: f0 = std::bit_cast<double>(detail::extend_bool<std::int64_t>(mem[i]));
				}
				v_data[i / 2] = _mm_and_pd(_mm_set_pd(f1, f0), v_mask[i / 2]);
			}
		}
	};

	namespace detail
	{
		template<std::size_t N, std::size_t A> requires detail::x86_overload_m128<double, N, A>
		struct native_access<simd_mask<double, avec<N, A>>>
		{
			using mask_t = simd_mask<double, avec<N, A>>;

			[[nodiscard]] static std::span<__m128d, mask_t::data_size> to_native_data(mask_t &x) noexcept { return {x.m_data}; }
			[[nodiscard]] static std::span<const __m128d, mask_t::data_size> to_native_data(const mask_t &x) noexcept { return {x.m_data}; }
		};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a x. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128d, detail::align_data<double, N, 16>()> to_native_data(simd_mask<double, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<double, N, A>
		{
			return detail::native_access<simd_mask<double, detail::avec<N, A>>>::to_native_data(x);
		}
		/** Returns a constant span of the underlying SSE vectors for \a x. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128d, detail::align_data<double, N, 16>()> to_native_data(const simd_mask<double, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<double, N, A>
		{
			return detail::native_access<simd_mask<double, detail::avec<N, A>>>::to_native_data(x);
		}

#ifdef DPM_HAS_SSE4_1
		/** Replaces elements of mask \a a with elements of mask \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline simd_mask<double, detail::avec<N, A>> blend(
				const simd_mask<double, detail::avec<N, A>> &a,
				const simd_mask<double, detail::avec<N, A>> &b,
				const simd_mask<double, detail::avec<N, A>> &m)
		noexcept requires detail::x86_overload_m128<double, N, A>
		{
			constexpr auto data_size = native_data_size_v<simd_mask<double, detail::avec<N, A>>>;

			simd_mask<double, detail::avec<N, A>> result;
			auto result_data = to_native_data(result);
			const auto a_data = to_native_data(a);
			const auto b_data = to_native_data(b);
			const auto m_data = to_native_data(m);

			for (std::size_t i = 0; i < data_size; ++i) result_data[i] = _mm_blendv_pd(a_data[i], b_data[i], m_data[i]);
			return result;
		}
#endif
	}

#pragma region "simd_mask reductions"
	/** Returns `true` if all of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool all_of(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		using mask_t = simd_mask<double, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

#ifdef DPM_HAS_SSE4_1
		if constexpr (ext::native_data_size_v<mask_t> == 1)
		{
			const auto vm = detail::x86_maskone_vector_f64(mask_t::size(), mask_data[0]);
			return _mm_test_all_ones(std::bit_cast<__m128i>(vm));
		}
#endif

		auto result = _mm_set1_pd(std::bit_cast<double>(0xffff'ffff'ffff'ffff));
		for (std::size_t i = 0; i < mask_t::size(); i += 2)
		{
			const auto vm = detail::x86_maskone_vector_f64(mask_t::size() - i, mask_data[i / 2]);
			result = _mm_and_pd(result, vm);
		}
		return _mm_movemask_pd(result) == 0b11;
	}
	/** Returns `true` if at least one of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool any_of(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		using mask_t = simd_mask<double, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		auto result = _mm_setzero_pd();
		for (std::size_t i = 0; i < mask_t::size(); i += 2)
		{
			const auto vm = detail::x86_maskone_vector_f64(mask_t::size() - i, mask_data[i / 2]);
			result = _mm_or_pd(result, vm);
		}
#ifdef DPM_HAS_SSE4_1
		const auto vi = std::bit_cast<__m128i>(result);
		return !_mm_testz_si128(vi, vi);
#else
		return _mm_movemask_pd(result);
#endif
	}
	/** Returns `true` if at none of the elements of the \a mask is `true`. Otherwise returns `false`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool none_of(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		using mask_t = simd_mask<double, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		auto result = _mm_setzero_pd();
		for (std::size_t i = 0; i < mask_t::size(); i += 2)
		{
			const auto vm = x86_maskzero_f64(mask_t::size() - i, mask_data[i / 2]);
			result = _mm_or_pd(result, vm);
		}
#ifdef DPM_HAS_SSE4_1
		const auto vi = std::bit_cast<__m128i>(result);
		return _mm_testz_si128(vi, vi);
#else
		return !_mm_movemask_pd(result);
#endif
	}
	/** Returns `true` if at least one of the elements of the \a mask is `true` and at least one is `false`. Otherwise returns `false`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool some_of(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		using mask_t = simd_mask<double, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		auto any_mask = _mm_setzero_pd(), all_mask = _mm_set1_pd(std::bit_cast<double>(0xffff'ffff'ffff'ffff));
		for (std::size_t i = 0; i < mask_t::size(); i += 2)
		{
			const auto vm = mask_data[i / 2];
			const auto vmz = x86_maskzero_f64(mask_t::size() - i, vm);
			const auto vmo = x86_maskone_vector_f64(mask_t::size() - i, vm);

			all_mask = _mm_and_pd(all_mask, vmo);
			any_mask = _mm_or_pd(any_mask, vmz);
		}
#ifdef DPM_HAS_SSE4_1
		const auto any_vi = std::bit_cast<__m128i>(any_mask);
		const auto all_vi = std::bit_cast<__m128i>(all_mask);
		return !_mm_testz_si128(any_vi, any_vi) && !_mm_test_all_ones(all_vi);
#else
		return _mm_movemask_pd(any_mask) && _mm_movemask_pd(all_mask) != 0b11;
#endif
	}

	/** Returns the number of `true` elements of \a mask. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t popcount(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		using mask_t = simd_mask<double, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		std::size_t result = 0;
		for (std::size_t i = 0; i < mask_t::size(); i += 2)
		{
			const auto vm = detail::x86_maskzero_f64(mask_t::size() - i, mask_data[i / 2]);
			result += std::popcount(static_cast<std::uint32_t>(_mm_movemask_pd(vm)));
		}
		return result;
	}
	/** Returns the index of the first `true` element of \a mask. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t find_first_set(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		using mask_t = simd_mask<double, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto bits = static_cast<std::uint8_t>(_mm_movemask_pd(mask_data[i]));
			if (bits) return std::countr_zero(bits) + i * 2;
		}
		DPM_UNREACHABLE();
	}
	/** Returns the index of the last `true` element of \a mask. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t find_last_set(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		using mask_t = simd_mask<double, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		for (std::size_t i = ext::native_data_size_v<mask_t>, k; (k = i--) != 0;)
		{
			auto bits = static_cast<std::uint8_t>(_mm_movemask_pd(mask_data[i]));
			switch (mask_t::size() - i * 2)
			{
				case 1: bits <<= 1; [[fallthrough]];
				default: bits <<= 2;
			}
			if (bits) return (k * 2 - 1) - std::countl_zero(bits);
		}
		DPM_UNREACHABLE();
	}
#pragma endregion

#pragma region "simd_mask casts"
	/** Converts SIMD mask \a x to it's fixed-size equivalent for value type `T`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd_mask<double, simd_abi::fixed_size<N>> to_fixed_size(const simd_mask<double, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		return simd_mask<double, simd_abi::fixed_size<N>>{x};
	}
	/** Converts SIMD mask \a x to it's native ABI equivalent for value type `T`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline native_simd_mask<double> to_native(const simd_mask<double, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		return native_simd_mask<double>{x};
	}
	/** Converts SIMD mask \a x to it's compatible ABI equivalent for value type `T`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd_mask<double> to_compatible(const simd_mask<double, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		return simd_mask<double>{x};
	}

	/** Concatenates elements of \a values into a single SIMD mask. */
	template<detail::x86_simd_abi_m128<double>... Abis>
	[[nodiscard]] inline auto concat(const simd_mask<double, Abis> &...values) noexcept
	{
		if constexpr (sizeof...(values) == 1)
			return (values, ...);
		else
		{
			simd_mask<double, simd_abi::deduce_t<double, (simd_size_v<double, Abis> + ...)>> result;
			detail::concat_impl<0>(result, values...);
			return result;
		}
	}
	/** Concatenates elements of \a values into a single SIMD mask. */
	template<std::size_t N, std::size_t A, std::size_t M>
	[[nodiscard]] inline auto concat(const std::array<simd_mask<double, detail::avec<N, A>>, M> &values) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		if constexpr (M == 1)
			return values[0];
		else
		{
			simd_mask<double, detail::avec<N * M, A>> result;
			auto *data = reinterpret_cast<double *>(ext::to_native_data(result).data());
			for (std::size_t i = 0; i < M; ++i)
			{
				if ((i * N) % sizeof(__m128d) != 0)
					values[i].copy_to(data + i * N, element_aligned);
				else
					values[i].copy_to(data + i * N, vector_aligned);
			}
			return result;
		}
	}
#pragma endregion

	DPM_DECLARE_EXT_NAMESPACE
	{
		template<std::size_t N, std::size_t Align> requires detail::x86_overload_m128<double, N, Align>
		struct native_data_type<simd<double, detail::avec<N, Align>>> { using type = __m128d; };
		template<std::size_t N, std::size_t Align> requires detail::x86_overload_m128<double, N, Align>
		struct native_data_size<simd<double, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<double, N, 16>()> {};

		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128d, detail::align_data<double, N, 16>()> to_native_data(simd<double, detail::avec<N, A>> &) noexcept requires detail::x86_overload_m128<double, N, A>;
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128d, detail::align_data<double, N, 16>()> to_native_data(const simd<double, detail::avec<N, A>> &) noexcept requires detail::x86_overload_m128<double, N, A>;
	}

	template<std::size_t N, std::size_t Align> requires detail::x86_overload_m128<double, N, Align>
	class simd<double, detail::avec<N, Align>>
	{
		friend struct detail::native_access<simd>;

		constexpr static auto data_size = ext::native_data_size_v<simd>;
		constexpr static auto alignment = std::max(Align, alignof(__m128d));

		using data_type = __m128d[data_size];

	public:
		using value_type = double;
		using reference = value_type &;

		using abi_type = detail::avec<N, Align>;
		using mask_type = simd_mask<double, abi_type>;

		/** Returns width of the SIMD type. */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	public:
		constexpr simd() noexcept = default;
		constexpr simd(const simd &) noexcept = default;
		constexpr simd &operator=(const simd &) noexcept = default;
		constexpr simd(simd &&) noexcept = default;
		constexpr simd &operator=(simd &&) noexcept = default;

		/** Initializes the SIMD vector with a native SSE vector.
		 * @note This constructor is available for overload resolution only when the SIMD vector contains a single SSE vector. */
		constexpr simd(__m128d native) noexcept requires (data_size == 1) { m_data[0] = native; }
		/** Initializes the SIMD vector with an array of native SSE vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd) / sizeof(__m128d)`. */
		constexpr simd(const __m128d (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

		/** Initializes the underlying elements with \a value. */
		template<detail::compatible_element<value_type> U>
		simd(U &&value) noexcept
		{
			const auto vec = _mm_set1_pd(static_cast<double>(value));
			std::fill_n(m_data, data_size, vec);
		}
		/** Initializes the underlying elements with values provided by the generator \a gen. */
		template<detail::element_generator<value_type, size()> G>
		simd(G &&gen) noexcept
		{
			detail::generate_n<data_size>(m_data, [&gen]<std::size_t I>(std::integral_constant<std::size_t, I>)
			{
				double f0 = 0.0f, f1 = 0.0f;
				switch (constexpr auto value_idx = I * 2; size() - value_idx)
				{
					default: f1 = std::invoke(gen, std::integral_constant<std::size_t, value_idx + 1>());
					case 1: f0 = std::invoke(gen, std::integral_constant<std::size_t, value_idx>());
				}
				return _mm_set_pd(f1, f0);
			});
		}

		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		simd(const simd<U, detail::avec<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (constexpr auto other_alignment = alignof(decltype(other)); other_alignment >= alignment)
				copy_from(reinterpret_cast<const U *>(ext::to_native_data(other).data()), vector_aligned);
			else if constexpr (other_alignment != alignof(value_type))
				copy_from(reinterpret_cast<const U *>(ext::to_native_data(other).data()), overaligned<other_alignment>);
			else
				copy_from(reinterpret_cast<const U *>(ext::to_native_data(other).data()), element_aligned);
		}
		/** Initializes the underlying elements from \a mem. */
		template<typename U, typename Flags>
		simd(const U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying elements from \a mem. */
		template<typename U, typename Flags>
		void copy_from(const U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags>
		{
			if constexpr (detail::aligned_tag<Flags, alignof(__m128d)>)
			{
				if constexpr (std::same_as<std::remove_cvref_t<U>, double>)
				{
					for (std::size_t i = 0; i < size(); i += 2)
					{
						if (size() - i > 1)
							m_data[i / 2] = reinterpret_cast<const __m128d *>(mem)[i / 2];
						else
							operator[](i) = mem[i];
					}
					return;
				}
				else if constexpr (detail::signed_integral_of_size<std::remove_cvref_t<U>, 8>)
				{
					for (std::size_t i = 0; i < size(); i += 2)
					{
						if (size() - i > 1)
							m_data[i / 2] = detail::x86_cvt_i64_f64(reinterpret_cast<const __m128i *>(mem)[i / 2]);
						else
							operator[](i) = static_cast<double>(mem[i]);
					}
					return;
				}
				else if constexpr (detail::unsigned_integral_of_size<std::remove_cvref_t<U>, 8>)
				{
					for (std::size_t i = 0; i < size(); i += 2)
					{
						if (size() - i > 1)
							m_data[i / 2] = detail::x86_cvt_u64_f64(reinterpret_cast<const __m128i *>(mem)[i / 2]);
						else
							operator[](i) = static_cast<double>(mem[i]);
					}
					return;
				}
			}
			for (std::size_t i = 0; i < size(); ++i) operator[](i) = static_cast<double>(mem[i]);
		}
		/** Copies the underlying elements to \a mem. */
		template<typename U, typename Flags>
		void copy_to(U *mem, Flags) const noexcept requires is_simd_flag_type_v<Flags>
		{
			if constexpr (detail::aligned_tag<Flags, alignof(__m128d)>)
			{
				if constexpr (std::same_as<std::remove_cvref_t<U>, double>)
				{
					for (std::size_t i = 0; i < size(); i += 2)
					{
						if (size() - i != 1)
							reinterpret_cast<__m128d *>(mem)[i / 2] = m_data[i / 2];
						else
							mem[i] = operator[](i);
					}
					return;
				}
				else if constexpr (detail::signed_integral_of_size<std::remove_cvref_t<U>, 8>)
				{
					for (std::size_t i = 0; i < size(); i += 2)
					{
						if (size() - i != i)
							reinterpret_cast<__m128i *>(mem)[i / 2] = detail::x86_cvt_f64_i64(m_data[i / 2]);
						else
							mem[i] = static_cast<U>(operator[](i));
					}
					return;
				}
				else if constexpr (detail::unsigned_integral_of_size<std::remove_cvref_t<U>, 8>)
				{
					for (std::size_t i = 0; i < size(); i += 2)
					{
						if (size() - i != i)
							reinterpret_cast<__m128i *>(mem)[i / 2] = detail::x86_cvt_f64_u64(m_data[i / 2]);
						else
							mem[i] = static_cast<U>(operator[](i));
					}
					return;
				}
			}
			for (std::size_t i = 0; i < size(); ++i) mem[i] = static_cast<U>(operator[](i));
		}

		[[nodiscard]] reference operator[](std::size_t i) noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<double *>(m_data)[i];
		}
		[[nodiscard]] value_type operator[](std::size_t i) const noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<const double *>(m_data)[i];
		}

		simd operator++(int) noexcept
		{
			auto tmp = *this;
			operator++();
			return tmp;
		}
		simd operator--(int) noexcept
		{
			auto tmp = *this;
			operator--();
			return tmp;
		}
		simd &operator++() noexcept
		{
			const auto one = _mm_set1_pd(1.0);
			for (std::size_t i = 0; i < data_size; ++i) m_data[i] = _mm_add_pd(m_data[i], one);
			return *this;
		}
		simd &operator--() noexcept
		{
			const auto one = _mm_set1_pd(1.0);
			for (std::size_t i = 0; i < data_size; ++i) m_data[i] = _mm_sub_pd(m_data[i], one);
			return *this;
		}

		[[nodiscard]] simd operator+() const noexcept { return *this; }
		[[nodiscard]] simd operator-() const noexcept
		{
			simd result;
			const auto mask = _mm_set1_pd(-0.0);
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_xor_pd(m_data[i], mask);
			return result;
		}

		[[nodiscard]] friend simd operator+(const simd &a, const simd &b) noexcept
		{
			simd result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_add_pd(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend simd operator-(const simd &a, const simd &b) noexcept
		{
			simd result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_sub_pd(a.m_data[i], b.m_data[i]);
			return result;
		}

		friend simd &operator+=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_add_pd(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend simd &operator-=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_sub_pd(a.m_data[i], b.m_data[i]);
			return a;
		}

		[[nodiscard]] friend simd operator*(const simd &a, const simd &b) noexcept
		{
			simd result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_mul_pd(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend simd operator/(const simd &a, const simd &b) noexcept
		{
			simd result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_div_pd(a.m_data[i], b.m_data[i]);
			return result;
		}

		friend simd &operator*=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_mul_pd(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend simd &operator/=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_div_pd(a.m_data[i], b.m_data[i]);
			return a;
		}

		[[nodiscard]] friend mask_type operator==(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpeq_pd(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend mask_type operator<=(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmple_pd(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend mask_type operator>=(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpge_pd(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend mask_type operator<(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmplt_pd(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend mask_type operator>(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpgt_pd(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend mask_type operator!=(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpneq_pd(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}

	private:
		alignas(alignment) data_type m_data;
	};

	template<std::size_t N, std::size_t A> requires detail::x86_overload_m128<double, N, A>
	class const_where_expression<simd_mask<double, detail::avec<N, A>>, simd<double, detail::avec<N, A>>>
	{
		template<typename U, typename Abi, typename K>
		friend inline simd<U, Abi> ext::blend(const simd<U, Abi> &, const const_where_expression<K, simd<U, Abi>> &);
		template<typename U, typename Abi, typename K>
		friend inline simd_mask<U, Abi> ext::blend(const simd_mask<U, Abi> &, const const_where_expression<K, simd_mask<U, Abi>> &);

		template<typename U>
		friend inline U ext::blend(const U &, const const_where_expression<bool, U> &);

	protected:
		using mask_t = simd_mask<double, detail::avec<N, A>>;
		using simd_t = simd<double, detail::avec<N, A>>;
		using value_type = double;

	public:
		const_where_expression(const const_where_expression &) = delete;
		const_where_expression &operator=(const const_where_expression &) = delete;

		const_where_expression(mask_t mask, simd_t &data) noexcept : m_mask(mask), m_data(data) {}
		const_where_expression(mask_t mask, const simd_t &data) noexcept : m_mask(mask), m_data(const_cast<simd_t &>(data)) {}

		[[nodiscard]] simd_t operator-() const && noexcept { return ext::blend(m_data, -m_data, m_mask); }
		[[nodiscard]] simd_t operator+() const && noexcept { return ext::blend(m_data, +m_data, m_mask); }

		/** Copies selected elements to \a mem. */
		template<typename U, typename Flags>
		void copy_to(U *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
			const auto v_mask = ext::to_native_data(m_data);
			const auto v_data = ext::to_native_data(m_data);
			if constexpr (std::same_as<std::remove_cvref_t<U>, double>)
				for (std::size_t i = 0; i < mask_t::size(); i += 2)
				{
#ifdef DPM_HAS_AVX
					if constexpr (detail::aligned_tag<Flags, alignof(__m128d)>)
					{
						const auto mi = std::bit_cast<__m128i>(detail::x86_maskzero_f64(mask_t::size() - i, v_mask[i / 2]));
						_mm_maskstore_pd(mem, mi, v_data[i / 2]);
					}
					else
#endif
					{
						const auto mi = std::bit_cast<__m128i>(detail::x86_maskzero_f64(mask_t::size() - i, v_mask[i / 2]));
						const auto vi = std::bit_cast<__m128i>(v_data[i / 2]);
						_mm_maskmoveu_si128(vi, mi, reinterpret_cast<char *>(mem));
					}
				}
			else if constexpr (detail::signed_integral_of_size<std::remove_cvref_t<U>, 8>)
				for (std::size_t i = 0; i < mask_t::size(); i += 2)
				{
#ifdef DPM_HAS_AVX
					if constexpr (detail::aligned_tag<Flags, alignof(__m128d)>)
					{
						const auto mi = std::bit_cast<__m128i>(detail::x86_maskzero_f64(mask_t::size() - i, v_mask[i / 2]));
						_mm_maskstore_pd(mem, mi, std::bit_cast<__m128d>(detail::x86_cvt_f64_i64(v_data[i / 2])));
					}
					else
#endif
					{
						const auto mi = std::bit_cast<__m128i>(detail::x86_maskzero_f64(mask_t::size() - i, v_mask[i / 2]));
						const auto vi = detail::x86_cvt_f64_i64(v_data[i / 2]);
						_mm_maskmoveu_si128(vi, mi, reinterpret_cast<char *>(mem));
					}
				}
			else if constexpr (detail::unsigned_integral_of_size<std::remove_cvref_t<U>, 8>)
				for (std::size_t i = 0; i < mask_t::size(); i += 2)
				{
#ifdef DPM_HAS_AVX
					if constexpr (detail::aligned_tag<Flags, alignof(__m128d)>)
					{
						const auto mi = std::bit_cast<__m128i>(detail::x86_maskzero_f64(mask_t::size() - i, v_mask[i / 2]));
						_mm_maskstore_pd(mem, mi, std::bit_cast<__m128d>(detail::x86_cvt_f64_u64(v_data[i / 2])));
					}
					else
#endif
					{
						const auto mi = std::bit_cast<__m128i>(detail::x86_maskzero_f64(mask_t::size() - i, v_mask[i / 2]));
						const auto vi = detail::x86_cvt_f64_u64(v_data[i / 2]);
						_mm_maskmoveu_si128(vi, mi, reinterpret_cast<char *>(mem));
					}
				}
			else
				for (std::size_t i = 0; i < mask_t::size(); ++i) if (m_mask[i]) mem[i] = static_cast<U>(m_data[i]);
		}

	protected:
		mask_t m_mask;
		simd_t &m_data;
	};
	template<std::size_t N, std::size_t A> requires detail::x86_overload_m128<double, N, A>
	class where_expression<simd_mask<double, detail::avec<N, A>>, simd<double, detail::avec<N, A>>>
			: public const_where_expression<simd_mask<double, detail::avec<N, A>>, simd<double, detail::avec<N, A>>>
	{
		using base_expr = const_where_expression<simd_mask<double, detail::avec<N, A>>, simd<double, detail::avec<N, A>>>;
		using value_type = typename base_expr::value_type;
		using simd_t = typename base_expr::simd_t;
		using mask_t = typename base_expr::mask_t;

		using base_expr::m_mask;
		using base_expr::m_data;

	public:
		using base_expr::const_where_expression;

		template<std::convertible_to<value_type> U>
		void operator=(U &&value) && noexcept { m_data = ext::blend(m_data, simd_t{std::forward<U>(value)}, m_mask); }

		void operator++() && noexcept
		{
			const auto old_data = m_data;
			const auto new_data = ++old_data;
			m_data = ext::blend(old_data, new_data, m_mask);
		}
		void operator--() && noexcept
		{
			const auto old_data = m_data;
			const auto new_data = --old_data;
			m_data = ext::blend(old_data, new_data, m_mask);
		}
		void operator++(int) && noexcept
		{
			const auto old_data = m_data++;
			m_data = ext::blend(old_data, m_data, m_mask);
		}
		void operator--(int) && noexcept
		{
			const auto old_data = m_data--;
			m_data = ext::blend(old_data, m_data, m_mask);
		}

		template<typename U>
		void operator+=(U &&value) && noexcept requires std::convertible_to<U, value_type>
		{
			const auto new_data = m_data + simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		void operator-=(U &&value) && noexcept requires std::convertible_to<U, value_type>
		{
			const auto new_data = m_data - simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}

		template<typename U>
		void operator*=(U &&value) && noexcept requires std::convertible_to<U, value_type>
		{
			const auto new_data = m_data * simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		void operator/=(U &&value) && noexcept requires std::convertible_to<U, value_type>
		{
			const auto new_data = m_data / simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}

		/** Copies selected elements from \a mem. */
		template<typename U, typename Flags>
		void copy_from(const U *mem, Flags) && noexcept requires is_simd_flag_type_v<Flags>
		{
			const auto v_mask = ext::to_native_data(m_data);
			const auto v_data = ext::to_native_data(m_data);
			if constexpr (std::same_as<std::remove_cvref_t<U>, double>)
				for (std::size_t i = 0; i < mask_t::size(); i += 2)
				{
#ifdef DPM_HAS_AVX
					if constexpr (detail::aligned_tag<Flags, alignof(__m128d)>)
					{
						const auto mi = std::bit_cast<__m128i>(detail::x86_maskzero_f64(mask_t::size() - i, v_mask[i / 2]));
						v_data[i / 2] = _mm_maskload_pd(mem + i, mi);
					}
					else
#endif
					{
						const auto mi = std::bit_cast<__m128i>(detail::x86_maskzero_f64(mask_t::size() - i, v_mask[i / 2]));
						const auto vi = std::bit_cast<__m128i>(v_data[i / 2]);
						_mm_maskmoveu_si128(vi, mi, reinterpret_cast<char *>(mem + i));
					}
				}
			else if constexpr (detail::signed_integral_of_size<std::remove_cvref_t<U>, 8>)
				for (std::size_t i = 0; i < mask_t::size(); i += 2)
				{
#ifdef DPM_HAS_AVX
					if constexpr (detail::aligned_tag<Flags, alignof(__m128d)>)
					{
						const auto mi = std::bit_cast<__m128i>(detail::x86_maskzero_f64(mask_t::size() - i, v_mask[i / 2]));
						v_data[i / 2] = detail::x86_cvt_i64_f64(_mm_maskload_pd(mem + i, mi));
					}
					else
#endif
					{
						const auto mi = std::bit_cast<__m128i>(detail::x86_maskzero_f64(mask_t::size() - i, v_mask[i / 2]));
						const auto vi = detail::x86_cvt_i64_f64(v_data[i / 2]);
						_mm_maskmoveu_si128(vi, mi, reinterpret_cast<char *>(mem + i));
					}
				}
			else if constexpr (detail::unsigned_integral_of_size<std::remove_cvref_t<U>, 8>)
				for (std::size_t i = 0; i < mask_t::size(); i += 2)
				{
#ifdef DPM_HAS_AVX
					if constexpr (detail::aligned_tag<Flags, alignof(__m128d)>)
					{
						const auto mi = std::bit_cast<__m128i>(detail::x86_maskzero_f64(mask_t::size() - i, v_mask[i / 2]));
						v_data[i / 2] = detail::x86_cvt_u64_f64(_mm_maskload_pd(mem + i, mi));
					}
					else
#endif
					{
						const auto mi = std::bit_cast<__m128i>(detail::x86_maskzero_f64(mask_t::size() - i, v_mask[i / 2]));
						const auto vi = detail::x86_cvt_u64_f64(v_data[i / 2]);
						_mm_maskmoveu_si128(vi, mi, reinterpret_cast<char *>(mem + i));
					}
				}
			else
				for (std::size_t i = 0; i < mask_t::size(); ++i) if (m_mask[i]) m_data[i] = static_cast<double>(mem[i]);
		}
	};

	namespace detail
	{
		template<std::size_t N, std::size_t A> requires detail::x86_overload_m128<double, N, A>
		struct native_access<simd<double, avec<N, A>>>
		{
			using simd_t = simd<double, avec<N, A>>;

			[[nodiscard]] static std::span<__m128d, simd_t::data_size> to_native_data(simd_t &x) noexcept { return {x.m_data}; }
			[[nodiscard]] static std::span<const __m128d, simd_t::data_size> to_native_data(const simd_t &x) noexcept { return {x.m_data}; }
		};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a x. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128d, detail::align_data<double, N, 16>()> to_native_data(simd<double, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<double, N, A>
		{
			return detail::native_access<simd<double, detail::avec<N, A>>>::to_native_data(x);
		}
		/** Returns a constant span of the underlying SSE vectors for \a x. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128d, detail::align_data<double, N, 16>()> to_native_data(const simd<double, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<double, N, A>
		{
			return detail::native_access<simd<double, detail::avec<N, A>>>::to_native_data(x);
		}

#ifdef DPM_HAS_SSE4_1
		/** Replaces elements of vector \a a with elements of vector \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline simd<double, detail::avec<N, A>> blend(
				const simd<double, detail::avec<N, A>> &a,
				const simd<double, detail::avec<N, A>> &b,
				const simd_mask<double, detail::avec<N, A>> &m)
		noexcept requires detail::x86_overload_m128<double, N, A>
		{
			constexpr auto data_size = native_data_size_v<simd<double, detail::avec<N, A>>>;

			simd<double, detail::avec<N, A>> result;
			auto result_data = to_native_data(result);
			const auto a_data = to_native_data(a);
			const auto b_data = to_native_data(b);
			const auto m_data = to_native_data(m);

			for (std::size_t i = 0; i < data_size; ++i) result_data[i] = _mm_blendv_pd(a_data[i], b_data[i], m_data[i]);
			return result;
		}
#endif
	}

#pragma region "simd casts"
	/** Implicitly converts elements of SIMD vector \a x to the `To` type, where `To` is either `typename T::value_type` or `T` if `T` is a scalar. */
	template<typename T, std::size_t N, std::size_t A, typename To = typename detail::deduce_cast<T>::type>
	[[nodiscard]] inline auto simd_cast(const simd<double, detail::avec<N, A>> &x) noexcept
	requires (detail::valid_simd_cast<T, double, detail::avec<N, A>> &&
	          detail::x86_overload_m128<double, N, A> &&
	          detail::x86_overload_any<To, N, A>)
	{
		detail::cast_return_t<T, double, detail::avec<N, A>, simd<double, detail::avec<N, A>>::size()> result;
		x.copy_to(reinterpret_cast<To *>(ext::to_native_data(result).data()), vector_aligned);
		return result;
	}
	/** Explicitly converts elements of SIMD vector \a x to the `To` type, where `To` is either `typename T::value_type` or `T` if `T` is a scalar. */
	template<typename T, std::size_t N, std::size_t A, typename To = typename detail::deduce_cast<T>::type>
	[[nodiscard]] inline auto static_simd_cast(const simd<double, detail::avec<N, A>> &x) noexcept
	requires (detail::valid_simd_static_cast<T, double, detail::avec<N, A>> &&
	          detail::x86_overload_m128<double, N, A> &&
	          detail::x86_overload_any<To, N, A>)
	{
		typename detail::static_cast_return<T, double, detail::avec<N, A>, simd<double, detail::avec<N, A>>::size()>::type result;
		x.copy_to(reinterpret_cast<To *>(ext::to_native_data(result).data()), vector_aligned);
		return result;
	}

	/** Concatenates elements of \a values into a single SIMD vector. */
	template<detail::x86_simd_abi_m128<double>... Abis>
	[[nodiscard]] inline auto concat(const simd<double, Abis> &...values) noexcept
	{
		if constexpr (sizeof...(values) == 1)
			return (values, ...);
		else
		{
			simd<double, simd_abi::deduce_t<double, (simd_size_v<double, Abis> + ...)>> result;
			detail::concat_impl<0>(result, values...);
			return result;
		}
	}
	/** Concatenates elements of \a values into a single SIMD vector. */
	template<std::size_t N, std::size_t A, std::size_t M>
	[[nodiscard]] inline auto concat(const std::array<simd<double, detail::avec<N, A>>, M> &values) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		if constexpr (M == 1)
			return values[0];
		else
		{
			simd<double, detail::avec<N * M, A>> result;
			auto *data = reinterpret_cast<double *>(ext::to_native_data(result).data());
			for (std::size_t i = 0; i < M; ++i)
			{
				if ((i * N) % sizeof(__m128d) != 0)
					values[i].copy_to(data + i * N, element_aligned);
				else
					values[i].copy_to(data + i * N, vector_aligned);
			}
			return result;
		}
	}
#pragma endregion

#pragma region "simd algorithms"
	/** Returns an SIMD vector of minimum elements of \a a and \a b. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<double, detail::avec<N, A>> min(
			const simd<double, detail::avec<N, A>> &a,
			const simd<double, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_m128<double, N, A>
	{
		constexpr auto data_size = native_data_size_v<simd<double, detail::avec<N, A>>>;

		simd<double, detail::avec<N, A>> result;
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < data_size; ++i) result_data[i] = _mm_min_pd(a_data[i], b_data[i]);
		return result;
	}
	/** Returns an SIMD vector of maximum elements of \a a and \a b. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<double, detail::avec<N, A>> max(
			const simd<double, detail::avec<N, A>> &a,
			const simd<double, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_m128<double, N, A>
	{
		constexpr auto data_size = native_data_size_v<simd<double, detail::avec<N, A>>>;

		simd<double, detail::avec<N, A>> result;
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < data_size; ++i) result_data[i] = _mm_max_pd(a_data[i], b_data[i]);
		return result;
	}

	/** Returns a pair of SIMD vectors of minimum and maximum elements of \a a and \a b. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::pair<simd<double, detail::avec<N, A>>, simd<double, detail::avec<N, A>>> minmax(
			const simd<double, detail::avec<N, A>> &a,
			const simd<double, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_m128<double, N, A>
	{
		constexpr auto data_size = native_data_size_v<simd<double, detail::avec<N, A>>>;

		std::pair<simd<double, detail::avec<N, A>>, simd<double, detail::avec<N, A>>> result;
		auto min_data = ext::to_native_data(result.first), max_data = ext::to_native_data(result.second);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < data_size; ++i)
		{
			min_data[i] = _mm_min_pd(a_data[i], b_data[i]);
			max_data[i] = _mm_max_pd(a_data[i], b_data[i]);
		}
		return result;
	}

	/** Clamps elements of \a x between corresponding elements of \a ming and \a max. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<double, detail::avec<N, A>> clamp(
			const simd<double, detail::avec<N, A>> &x,
			const simd<double, detail::avec<N, A>> &min,
			const simd<double, detail::avec<N, A>> &max)
	noexcept requires detail::x86_overload_m128<double, N, A>
	{
		constexpr auto data_size = native_data_size_v<simd<double, detail::avec<N, A>>>;

		simd<double, detail::avec<N, A>> result;
		auto result_data = ext::to_native_data(result);
		const auto min_data = ext::to_native_data(min);
		const auto max_data = ext::to_native_data(max);
		const auto x_data = ext::to_native_data(x);

		for (std::size_t i = 0; i < data_size; ++i) result_data[i] = _mm_min_pd(_mm_max_pd(x_data[i], min_data[i]), max_data[i]);
		return result;
	}
#pragma endregion
}

#endif