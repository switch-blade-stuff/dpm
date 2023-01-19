/*
 * Created by switchblade on 2023-01-06.
 */

#pragma once

#include "../../fwd.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE)

#include "utility.hpp"

namespace dpm
{
	namespace detail
	{
		template<std::size_t I, typename OutAbi, typename XAbi, typename... Abis>
		[[nodiscard]] inline DPM_SAFE_ARRAY auto concat_impl(simd_mask<float, OutAbi> &out, const simd_mask<float, XAbi> &x, const simd_mask<float, Abis> &...rest) noexcept
		{
			auto *data = reinterpret_cast<float *>(ext::to_native_data(out).data());
			if constexpr (I % sizeof(__m128) != 0)
				x.copy_to(data + I, element_aligned);
			else
				x.copy_to(data + I, vector_aligned);

			if constexpr (sizeof...(Abis) != 0) concat_impl<I + simd_mask<float, XAbi>::size()>(out, rest...);
		}
		template<std::size_t I, typename OutAbi, typename XAbi, typename... Abis>
		[[nodiscard]] inline DPM_SAFE_ARRAY auto concat_impl(simd<float, OutAbi> &out, const simd<float, XAbi> &x, const simd<float, Abis> &...rest) noexcept
		{
			auto *data = reinterpret_cast<float *>(ext::to_native_data(out).data());
			if constexpr (I % sizeof(__m128) != 0)
				x.copy_to(data + I, element_aligned);
			else
				x.copy_to(data + I, vector_aligned);

			if constexpr (sizeof...(Abis) != 0) concat_impl<I + simd<float, XAbi>::size()>(out, rest...);
		}

		[[nodiscard]] inline __m128 x86_maskzero_vector_f32(std::size_t n, __m128 v) noexcept
		{
			switch ([[maybe_unused]] const auto mask = std::bit_cast<float>(0xffff'ffff); n)
			{
#ifdef DPM_HAS_SSE4_1
				case 3: return _mm_blend_ps(v, _mm_setzero_ps(), 0b1000);
				case 2: return _mm_blend_ps(v, _mm_setzero_ps(), 0b1100);
				case 1: return _mm_blend_ps(v, _mm_setzero_ps(), 0b1110);
#else
				case 3: return _mm_and_ps(v, _mm_set_ps(0.0f, mask, mask, mask));
				case 2: return _mm_and_ps(v, _mm_set_ps(0.0f, 0.0f, mask, mask));
				case 1: return _mm_and_ps(v, _mm_set_ps(0.0f, 0.0f, 0.0f, mask));
#endif
				default: return v;
			}
		}
		[[nodiscard]] inline __m128 x86_maskone_vector_f32(std::size_t n, __m128 v) noexcept
		{
			switch (const auto mask = std::bit_cast<float>(0xffff'ffff); n)
			{
#ifdef DPM_HAS_SSE4_1
				case 3: return _mm_blend_ps(v, _mm_set1_ps(mask), 0b1000);
				case 2: return _mm_blend_ps(v, _mm_set1_ps(mask), 0b1100);
				case 1: return _mm_blend_ps(v, _mm_set1_ps(mask), 0b1110);
#else
				case 3: return _mm_or_ps(v, _mm_set_ps(mask, 0.0f, 0.0f, 0.0f));
				case 2: return _mm_or_ps(v, _mm_set_ps(mask, mask, 0.0f, 0.0f));
				case 1: return _mm_or_ps(v, _mm_set_ps(mask, mask, mask, 0.0f));
#endif
				default: return v;
			}
		}
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		template<std::size_t N, std::size_t Align> requires detail::x86_overload_m128<float, N, Align>
		struct native_data_type<simd_mask<float, detail::avec<N, Align>>> { using type = __m128; };
		template<std::size_t N, std::size_t Align> requires detail::x86_overload_m128<float, N, Align>
		struct native_data_size<simd_mask<float, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<float, N, 16>()> {};

		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128, detail::align_data<float, N, 16>()> to_native_data(simd_mask<float, detail::avec<N, A>> &) noexcept requires detail::x86_overload_m128<float, N, A>;
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128, detail::align_data<float, N, 16>()> to_native_data(const simd_mask<float, detail::avec<N, A>> &) noexcept requires detail::x86_overload_m128<float, N, A>;
	}

	template<std::size_t N, std::size_t Align> requires detail::x86_overload_m128<float, N, Align>
	class simd_mask<float, detail::avec<N, Align>>
	{
		friend struct detail::native_access<simd_mask>;

		constexpr static auto data_size = ext::native_data_size_v<simd_mask>;
		constexpr static auto alignment = std::max(Align, alignof(__m128));

		using data_type = __m128[data_size];

	public:
		using value_type = bool;
		using reference = detail::mask_reference<std::int32_t>;

		using abi_type = detail::avec<N, Align>;
		using simd_type = simd<float, abi_type>;

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
		constexpr DPM_SAFE_ARRAY simd_mask(__m128 native) noexcept requires (data_size == 1) { m_data[0] = native; }
		/** Initializes the SIMD mask object with an array of native SSE mask vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd_mask) / sizeof(__m128)`. */
		constexpr DPM_SAFE_ARRAY simd_mask(const __m128 (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

		/** Initializes the underlying elements with \a value. */
		DPM_SAFE_ARRAY simd_mask(value_type value) noexcept
		{
			const auto v = value ? _mm_set1_ps(std::bit_cast<float>(0xffff'ffff)) : _mm_setzero_ps();
			std::fill_n(m_data, data_size, v);
		}
		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		DPM_SAFE_ARRAY simd_mask(const simd_mask<U, detail::avec<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (std::same_as<U, value_type> && (OtherAlign == 0 || OtherAlign >= alignment))
				std::copy_n(reinterpret_cast<const __m128 *>(ext::to_native_data(other).data()), data_size, m_data);
			else
				for (std::size_t i = 0; i < size(); ++i) operator[](i) = other[i];
		}
		/** Initializes the underlying elements from \a mem. */
		template<typename Flags>
		simd_mask(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying elements from \a mem. */
		template<typename Flags>
		DPM_SAFE_ARRAY void copy_from(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags>
		{
			for (std::size_t i = 0; i < size(); i += 4)
			{
				float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
				switch (size() - i)
				{
					default: f3 = std::bit_cast<float>(detail::extend_bool<std::int32_t>(mem[i + 3])); [[fallthrough]];
					case 3: f2 = std::bit_cast<float>(detail::extend_bool<std::int32_t>(mem[i + 2])); [[fallthrough]];
					case 2: f1 = std::bit_cast<float>(detail::extend_bool<std::int32_t>(mem[i + 1])); [[fallthrough]];
					case 1: f0 = std::bit_cast<float>(detail::extend_bool<std::int32_t>(mem[i]));
				}
				m_data[i / 4] = _mm_set_ps(f3, f2, f1, f0);
			}
		}
		/** Copies the underlying elements to \a mem. */
		template<typename Flags>
		DPM_SAFE_ARRAY void copy_to(value_type *mem, Flags) const noexcept requires is_simd_flag_type_v<Flags>
		{
			for (std::size_t i = 0; i < size(); i += 4)
				switch (const auto bits = _mm_movemask_ps(m_data[i / 4]); size() - i)
				{
					default: mem[i + 3] = bits & 0b1000; [[fallthrough]];
					case 3: mem[i + 2] = bits & 0b0100; [[fallthrough]];
					case 2: mem[i + 1] = bits & 0b0010; [[fallthrough]];
					case 1: mem[i] = bits & 0b0001;
				}
		}

		[[nodiscard]] reference operator[](std::size_t i) noexcept
		{
			DPM_ASSERT(i < size());
			return reference{reinterpret_cast<std::int32_t *>(m_data)[i]};
		}
		[[nodiscard]] value_type operator[](std::size_t i) const noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<const std::int32_t *>(m_data)[i];
		}

		[[nodiscard]] DPM_SAFE_ARRAY simd_mask operator!() const noexcept
		{
			simd_mask result;
			const auto mask = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_xor_ps(m_data[i], mask);
			return result;
		}

		[[nodiscard]] friend DPM_SAFE_ARRAY simd_mask operator&(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_and_ps(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY simd_mask operator|(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_or_ps(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY simd_mask operator^(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_xor_ps(a.m_data[i], b.m_data[i]);
			return result;
		}

		friend DPM_SAFE_ARRAY simd_mask &operator&=(simd_mask &a, const simd_mask &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_and_ps(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend DPM_SAFE_ARRAY simd_mask &operator|=(simd_mask &a, const simd_mask &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_or_ps(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend DPM_SAFE_ARRAY simd_mask &operator^=(simd_mask &a, const simd_mask &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_xor_ps(a.m_data[i], b.m_data[i]);
			return a;
		}

		[[nodiscard]] friend DPM_SAFE_ARRAY simd_mask operator&&(const simd_mask &a, const simd_mask &b) noexcept { return a & b; }
		[[nodiscard]] friend DPM_SAFE_ARRAY simd_mask operator||(const simd_mask &a, const simd_mask &b) noexcept { return a | b; }

		[[nodiscard]] friend DPM_SAFE_ARRAY simd_mask operator==(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
#ifdef DPM_HAS_SSE2
			for (std::size_t i = 0; i < data_size; ++i)
			{
				const auto va = std::bit_cast<__m128i>(a.m_data[i]);
				const auto vb = std::bit_cast<__m128i>(b.m_data[i]);
				result.m_data[i] = std::bit_cast<__m128>(_mm_cmpeq_epi32(va, vb));
			}
#else
			const auto nan_mask = _mm_set1_ps(std::bit_cast<float>(0x3fff'ffff));
			for (std::size_t i = 0; i < M; ++i)
			{
				const auto va = _mm_and_ps(a.m_data[i], nan_mask);
				const auto vb = _mm_and_ps(b.m_data[i], nan_mask);
				result.m_data[i] = _mm_cmpeq_ps(va, vb);
			}
#endif
			return result;
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY simd_mask operator!=(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
#ifdef DPM_HAS_SSE2
			const auto inv_mask = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
			for (std::size_t i = 0; i < data_size; ++i)
			{
				const auto va = std::bit_cast<__m128i>(a.m_data[i]);
				const auto vb = std::bit_cast<__m128i>(b.m_data[i]);
				result.m_data[i] = _mm_xor_ps(std::bit_cast<__m128>(_mm_cmpeq_epi32(va, vb)), inv_mask);
			}
#else
			const auto nan_mask = _mm_set1_ps(std::bit_cast<float>(0x3fff'ffff));
			for (std::size_t i = 0; i < data_size; ++i)
			{
				const auto va = _mm_and_ps(a.m_data[i], nan_mask);
				const auto vb = _mm_and_ps(b.m_data[i], nan_mask);
				result.m_data[i] = _mm_cmpneq_ps(va, vb);
			}
#endif
			return result;
		}

	private:
		alignas(alignment) data_type m_data;
	};

	template<std::size_t N, std::size_t A> requires detail::x86_overload_m128<float, N, A>
	class const_where_expression<simd_mask<float, detail::avec<N, A>>, simd_mask<float, detail::avec<N, A>>>
	{
		template<typename U, typename Abi, typename K>
		friend inline simd<U, Abi> ext::blend(const simd<U, Abi> &, const const_where_expression<K, simd<U, Abi>> &);
		template<typename U, typename Abi, typename K>
		friend inline simd_mask<U, Abi> ext::blend(const simd_mask<U, Abi> &, const const_where_expression<K, simd_mask<U, Abi>> &);

		template<typename U>
		friend inline U ext::blend(const U &, const const_where_expression<bool, U> &);

	protected:
		using mask_t = simd_mask<float, detail::avec<N, A>>;
		using value_type = bool;

	public:
		const_where_expression(const const_where_expression &) = delete;
		const_where_expression &operator=(const const_where_expression &) = delete;

		const_where_expression(mask_t mask, mask_t &data) noexcept : m_mask(mask), m_data(data) {}
		const_where_expression(mask_t mask, const mask_t &data) noexcept : m_mask(mask), m_data(const_cast<mask_t &>(data)) {}

		/** Copies selected elements to \a mem. */
		template<typename Flags>
		DPM_SAFE_ARRAY void copy_to(bool *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
			const auto v_mask = ext::to_native_data(m_data);
			const auto v_data = ext::to_native_data(m_data);
			for (std::size_t i = 0; i < mask_t::size(); ++i)
			{
				const auto bits = _mm_movemask_ps(_mm_and_pd(v_data[i / 4], v_mask[i / 4]));
				switch (mask_t::size() - i)
				{
					default: mem[i + 3] = bits & 0b1000; [[fallthrough]];
					case 3: mem[i + 2] = bits & 0b0100; [[fallthrough]];
					case 2: mem[i + 1] = bits & 0b0010; [[fallthrough]];
					case 1: mem[i] = bits & 0b0001;
				}
			}
		}

	protected:
		mask_t m_mask;
		mask_t &m_data;
	};
	template<std::size_t N, std::size_t A> requires detail::x86_overload_m128<float, N, A>
	class where_expression<simd_mask<float, detail::avec<N, A>>, simd_mask<float, detail::avec<N, A>>>
			: public const_where_expression<simd_mask<float, detail::avec<N, A>>, simd_mask<float, detail::avec<N, A>>>
	{
		using base_expr = const_where_expression<simd_mask<float, detail::avec<N, A>>, simd_mask<float, detail::avec<N, A>>>;
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
		DPM_SAFE_ARRAY void copy_from(const bool *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
			const auto v_mask = ext::to_native_data(m_data);
			const auto v_data = ext::to_native_data(m_data);
			for (std::size_t i = 0; i < mask_t::size(); ++i)
			{
				float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
				switch (mask_t::size() - i)
				{
					default: f3 = std::bit_cast<float>(detail::extend_bool<std::int32_t>(mem[i + 3])); [[fallthrough]];
					case 3: f2 = std::bit_cast<float>(detail::extend_bool<std::int32_t>(mem[i + 2])); [[fallthrough]];
					case 2: f1 = std::bit_cast<float>(detail::extend_bool<std::int32_t>(mem[i + 1])); [[fallthrough]];
					case 1: f0 = std::bit_cast<float>(detail::extend_bool<std::int32_t>(mem[i]));
				}
				v_data[i / 4] = _mm_and_ps(_mm_set_ps(f3, f2, f1, f0), v_mask[i / 4]);
			}
		}
	};

	namespace detail
	{
		template<std::size_t N, std::size_t A> requires detail::x86_overload_m128<float, N, A>
		struct native_access<simd_mask<float, avec<N, A>>>
		{
			using mask_t = simd_mask<float, avec<N, A>>;

			[[nodiscard]] static std::span<__m128, mask_t::data_size> to_native_data(mask_t &x) noexcept { return {x.m_data}; }
			[[nodiscard]] static std::span<const __m128, mask_t::data_size> to_native_data(const mask_t &x) noexcept { return {x.m_data}; }
		};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a x. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128, detail::align_data<float, N, 16>()> to_native_data(simd_mask<float, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<float, N, A>
		{
			return detail::native_access<simd_mask<float, detail::avec<N, A>>>::to_native_data(x);
		}
		/** Returns a constant span of the underlying SSE vectors for \a x. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128, detail::align_data<float, N, 16>()> to_native_data(const simd_mask<float, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<float, N, A>
		{
			return detail::native_access<simd_mask<float, detail::avec<N, A>>>::to_native_data(x);
		}

#ifdef DPM_HAS_SSE4_1
		/** Replaces elements of mask \a a with elements of mask \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline DPM_SAFE_ARRAY simd_mask<float, detail::avec<N, A>> blend(
				const simd_mask<float, detail::avec<N, A>> &a,
				const simd_mask<float, detail::avec<N, A>> &b,
				const simd_mask<float, detail::avec<N, A>> &m)
		noexcept requires detail::x86_overload_m128<float, N, A>
		{
			constexpr auto data_size = native_data_size_v<simd_mask<float, detail::avec<N, A>>>;

			simd_mask<float, detail::avec<N, A>> result;
			auto result_data = to_native_data(result)();
			const auto a_data = to_native_data(a)();
			const auto b_data = to_native_data(b)();
			const auto m_data = to_native_data(m)();

			for (std::size_t i = 0; i < data_size; ++i) result_data[i] = _mm_blendv_ps(a_data[i], b_data[i], m_data[i]);
			return result;
		}
#endif
	}

#pragma region "simd_mask reductions"
	/** Returns `true` if all of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY bool all_of(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		using mask_t = simd_mask<float, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

#ifdef DPM_HAS_SSE4_1
		if constexpr (ext::native_data_size_v<mask_t> == 1)
		{
			const auto vm = detail::x86_maskone_vector_f32(mask_t::size(), mask_data[0]);
			return _mm_test_all_ones(std::bit_cast<__m128i>(vm));
		}
#endif
		auto result = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
		for (std::size_t i = 0; i < mask_t::size(); i += 4)
		{
			const auto vm = detail::x86_maskone_vector_f32(mask_t::size() - i, mask_data[i / 4]);
			result = _mm_and_ps(result, vm);
		}
		return _mm_movemask_ps(result) == 0b1111;
	}
	/** Returns `true` if at least one of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY bool any_of(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		using mask_t = simd_mask<float, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		auto result = _mm_setzero_ps();
		for (std::size_t i = 0; i < mask_t::size(); i += 4)
		{
			const auto vm = detail::x86_maskone_vector_f32(mask_t::size() - i, mask_data[i / 4]);
			result = _mm_or_ps(result, vm);
		}
#ifdef DPM_HAS_SSE4_1
		const auto vi = std::bit_cast<__m128i>(result);
		return !_mm_testz_si128(vi, vi);
#else
		return _mm_movemask_ps(result);
#endif
	}
	/** Returns `true` if at none of the elements of the \a mask is `true`. Otherwise returns `false`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY bool none_of(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		using mask_t = simd_mask<float, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		auto result = _mm_setzero_ps();
		for (std::size_t i = 0; i < mask_t::size(); i += 4)
		{
			const auto vm = detail::x86_maskone_vector_f32(mask_t::size() - i, mask_data[i / 4]);
			result = _mm_or_ps(result, vm);
		}
#ifdef DPM_HAS_SSE4_1
		const auto vi = std::bit_cast<__m128i>(result);
		return _mm_testz_si128(vi, vi);
#else
		return !_mm_movemask_ps(result);
#endif
	}
	/** Returns `true` if at least one of the elements of the \a mask is `true` and at least one is `false`. Otherwise returns `false`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY bool some_of(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		using mask_t = simd_mask<float, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		auto any_mask = _mm_setzero_ps(), all_mask = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
		for (std::size_t i = 0; i < mask_t::size(); i += 4)
		{
			const auto vm = mask_data[i / 4];
			const auto vmz = detail::x86_maskzero_vector_f32(mask_t::size() - i, vm);
			const auto vmo = detail::x86_maskone_vector_f32(mask_t::size() - i, vm);

			all_mask = _mm_and_ps(all_mask, vmo);
			any_mask = _mm_or_ps(any_mask, vmz);
		}
#ifdef DPM_HAS_SSE4_1
		const auto any_vi = std::bit_cast<__m128i>(any_mask);
		const auto all_vi = std::bit_cast<__m128i>(all_mask);
		return !_mm_testz_si128(any_vi, any_vi) && !_mm_test_all_ones(all_vi);
#else
		return _mm_movemask_ps(any_mask) && _mm_movemask_ps(all_mask) != 0b1111;
#endif
	}

	/** Returns the number of `true` elements of \a mask. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY std::size_t popcount(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		using mask_t = simd_mask<float, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		std::size_t result = 0;
		for (std::size_t i = 0; i < mask_t::size(); i += 4)
		{
			const auto vm = detail::x86_maskone_vector_f32(mask_t::size() - i, mask_data[i / 4]);
			result += std::popcount(static_cast<std::uint32_t>(_mm_movemask_ps(vm)));
		}
		return result;
	}
	/** Returns the index of the first `true` element of \a mask. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY std::size_t find_first_set(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		using mask_t = simd_mask<float, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto bits = static_cast<std::uint8_t>(_mm_movemask_ps(mask_data[i]));
			if (bits) return std::countr_zero(bits) + i * 4;
		}
		DPM_UNREACHABLE();
	}
	/** Returns the index of the last `true` element of \a mask. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY std::size_t find_last_set(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		using mask_t = simd_mask<float, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		for (std::size_t i = ext::native_data_size_v<mask_t>, k; (k = i--) != 0;)
		{
			auto bits = static_cast<std::uint8_t>(_mm_movemask_ps(mask_data[i]));
			switch (mask_t::size() - i * 4)
			{
				case 1: bits <<= 1; [[fallthrough]];
				case 2: bits <<= 1; [[fallthrough]];
				case 3: bits <<= 1; [[fallthrough]];
				default: bits <<= 4;
			}
			if (bits) return (k * 4 - 1) - std::countl_zero(bits);
		}
		DPM_UNREACHABLE();
	}
#pragma endregion

#pragma region "simd casts"
	/** Converts SIMD mask \a x to it's fixed-size equivalent for value type `T`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd_mask<float, simd_abi::fixed_size<N>> to_fixed_size(const simd_mask<float, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		return simd_mask<float, simd_abi::fixed_size<N>>{x};
	}
	/** Converts SIMD mask \a x to it's native ABI equivalent for value type `T`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY native_simd_mask<float> to_native(const simd_mask<float, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		return native_simd_mask<float>{x};
	}
	/** Converts SIMD mask \a x to it's compatible ABI equivalent for value type `T`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd_mask<float> to_compatible(const simd_mask<float, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		return simd_mask<float>{x};
	}

	/** Concatenates elements of \a values into a single SIMD mask. */
	template<detail::x86_simd_abi_m128<float>... Abis>
	[[nodiscard]] inline DPM_SAFE_ARRAY auto concat(const simd_mask<float, Abis> &...values) noexcept
	{
		if constexpr (sizeof...(values) == 1)
			return (values, ...);
		else
		{
			simd_mask<float, simd_abi::deduce_t<float, (simd_size_v<float, Abis> + ...)>> result;
			detail::concat_impl<0>(result, values...);
			return result;
		}
	}
	/** Concatenates elements of \a values into a single SIMD mask. */
	template<std::size_t N, std::size_t A, std::size_t M>
	[[nodiscard]] inline DPM_SAFE_ARRAY auto concat(const std::array<simd_mask<float, detail::avec<N, A>>, M> &values) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		if constexpr (M == 1)
			return values[0];
		else
		{
			simd_mask<float, detail::avec<N * M, A>> result;
			auto *data = reinterpret_cast<float *>(ext::to_native_data(result).data());
			for (std::size_t i = 0; i < M; ++i)
			{
				if ((i * N) % sizeof(__m128) != 0)
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
		template<std::size_t N, std::size_t Align> requires detail::x86_overload_m128<float, N, Align>
		struct native_data_type<simd<float, detail::avec<N, Align>>> { using type = __m128; };
		template<std::size_t N, std::size_t Align> requires detail::x86_overload_m128<float, N, Align>
		struct native_data_size<simd<float, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<float, N, 16>()> {};

		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128, detail::align_data<float, N, 16>()> to_native_data(simd<float, detail::avec<N, A>> &) noexcept requires detail::x86_overload_m128<float, N, A>;
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128, detail::align_data<float, N, 16>()> to_native_data(const simd<float, detail::avec<N, A>> &) noexcept requires detail::x86_overload_m128<float, N, A>;
	}

	template<std::size_t N, std::size_t Align> requires detail::x86_overload_m128<float, N, Align>
	class simd<float, detail::avec<N, Align>>
	{
		friend struct detail::native_access<simd>;

		constexpr static auto data_size = ext::native_data_size_v<simd>;
		constexpr static auto alignment = std::max(Align, alignof(__m128));

		using data_type = __m128[data_size];

	public:
		using value_type = float;
		using reference = detail::simd_reference<value_type>;

		using abi_type = detail::avec<N, Align>;
		using mask_type = simd_mask<float, abi_type>;

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
		constexpr DPM_SAFE_ARRAY simd(__m128 native) noexcept requires (data_size == 1) { m_data[0] = native; }
		/** Initializes the SIMD vector with an array of native SSE vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd) / sizeof(__m128)`. */
		constexpr DPM_SAFE_ARRAY simd(const __m128 (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

		/** Initializes the underlying elements with \a value. */
		template<detail::compatible_element<value_type> U>
		DPM_SAFE_ARRAY simd(U &&value) noexcept
		{
			const auto vec = _mm_set1_ps(static_cast<float>(value));
			std::fill_n(m_data, data_size, vec);
		}
		/** Initializes the underlying elements with values provided by the generator \a gen. */
		template<detail::element_generator<value_type, size()> G>
		DPM_SAFE_ARRAY simd(G &&gen) noexcept
		{
			detail::generate_n<data_size>(m_data, [&gen]<std::size_t I>(std::integral_constant<std::size_t, I>)
			{
				float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
				switch (constexpr auto value_idx = I * 4; size() - value_idx)
				{
					default: f3 = std::invoke(gen, std::integral_constant<std::size_t, value_idx + 3>());
					case 3: f2 = std::invoke(gen, std::integral_constant<std::size_t, value_idx + 2>());
					case 2: f1 = std::invoke(gen, std::integral_constant<std::size_t, value_idx + 1>());
					case 1: f0 = std::invoke(gen, std::integral_constant<std::size_t, value_idx>());
				}
				return _mm_set_ps(f3, f2, f1, f0);
			});
		}

		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		DPM_SAFE_ARRAY simd(const simd<U, detail::avec<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (OtherAlign == 0 || OtherAlign >= alignment)
				copy_from(reinterpret_cast<const U *>(ext::to_native_data(other).data()), vector_aligned);
			else if constexpr (OtherAlign != alignof(value_type))
				copy_from(reinterpret_cast<const U *>(ext::to_native_data(other).data()), overaligned<OtherAlign>);
			else
				copy_from(reinterpret_cast<const U *>(ext::to_native_data(other).data()), element_aligned);
		}
		/** Initializes the underlying elements from \a mem. */
		template<typename U, typename Flags>
		simd(const U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying elements from \a mem. */
		template<typename U, typename Flags>
		DPM_SAFE_ARRAY void copy_from(const U *mem, Flags) noexcept requires std::convertible_to<U, float> && is_simd_flag_type_v<Flags>
		{
			if constexpr (detail::aligned_tag<Flags, alignof(__m128)>)
			{
				if constexpr (std::same_as<std::remove_cvref_t<U>, float>)
				{
					for (std::size_t i = 0; i < size(); i += 4)
						switch (size() - i)
						{
							default: m_data[i / 4] = reinterpret_cast<const __m128 *>(mem)[i / 4];
								break;
							case 3: operator[](i + 2) = mem[i + 2]; [[fallthrough]];
							case 2: operator[](i + 1) = mem[i + 1]; [[fallthrough]];
							case 1: operator[](i) = mem[i];
						}
					return;
				}
#ifdef DPM_HAS_SSE2
				if constexpr (detail::signed_integral_of_size<std::remove_cvref_t<U>, 4>)
				{
					for (std::size_t i = 0; i < size(); i += 4)
						switch (size() - i)
						{
							default: m_data[i / 4] = _mm_cvtepi32_ps(reinterpret_cast<const __m128i *>(mem)[i / 4]);
								break;
							case 3: operator[](i + 2) = static_cast<float>(mem[i + 2]); [[fallthrough]];
							case 2: operator[](i + 1) = static_cast<float>(mem[i + 1]); [[fallthrough]];
							case 1: operator[](i) = static_cast<float>(mem[i]);
						}
					return;
				}
#endif
			}
			for (std::size_t i = 0; i < size(); ++i) operator[](i) = static_cast<float>(mem[i]);
		}
		/** Copies the underlying elements to \a mem. */
		template<typename U, typename Flags>
		DPM_SAFE_ARRAY void copy_to(U *mem, Flags) const noexcept requires std::convertible_to<float, U> && is_simd_flag_type_v<Flags>
		{
			if constexpr (detail::aligned_tag<Flags, alignof(__m128)>)
			{
				if constexpr (std::same_as<std::remove_cvref_t<U>, float>)
				{
					for (std::size_t i = 0; i < size(); i += 4)
						switch (size() - i)
						{
							default: reinterpret_cast<__m128 *>(mem)[i / 4] = m_data[i / 4];
								break;
							case 3: mem[i + 2] = operator[](i + 2); [[fallthrough]];
							case 2: mem[i + 1] = operator[](i + 1); [[fallthrough]];
							case 1: mem[i] = operator[](i);
						}
					return;
				}
#ifdef DPM_HAS_SSE2
				if constexpr (detail::signed_integral_of_size<std::remove_cvref_t<U>, 4>)
				{
					for (std::size_t i = 0; i < size(); i += 4)
						switch (size() - i)
						{
							default: reinterpret_cast<__m128i *>(mem)[i / 4] = _mm_cvtps_epi32(m_data[i / 4]);
								break;
							case 3: mem[i + 2] = static_cast<std::int32_t>(operator[](i + 2)); [[fallthrough]];
							case 2: mem[i + 1] = static_cast<std::int32_t>(operator[](i + 1)); [[fallthrough]];
							case 1: mem[i] = static_cast<std::int32_t>(operator[](i));
						}
					return;
				}
#endif
			}
			for (std::size_t i = 0; i < size(); ++i) mem[i] = static_cast<U>(operator[](i));
		}

		[[nodiscard]] reference operator[](std::size_t i) noexcept
		{
			DPM_ASSERT(i < size());
			return reference{reinterpret_cast<float *>(m_data)[i]};
		}
		[[nodiscard]] value_type operator[](std::size_t i) const noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<const float *>(m_data)[i];
		}

		DPM_SAFE_ARRAY simd operator++(int) noexcept
		{
			auto tmp = *this;
			operator++();
			return tmp;
		}
		DPM_SAFE_ARRAY simd operator--(int) noexcept
		{
			auto tmp = *this;
			operator--();
			return tmp;
		}
		DPM_SAFE_ARRAY simd &operator++() noexcept
		{
			const auto one = _mm_set1_ps(1.0f);
			for (std::size_t i = 0; i < data_size; ++i) m_data[i] = _mm_add_ps(m_data[i], one);
			return *this;
		}
		DPM_SAFE_ARRAY simd &operator--() noexcept
		{
			const auto one = _mm_set1_ps(1.0f);
			for (std::size_t i = 0; i < data_size; ++i) m_data[i] = _mm_sub_ps(m_data[i], one);
			return *this;
		}

		[[nodiscard]] simd DPM_SAFE_ARRAY operator+() const noexcept { return *this; }
		[[nodiscard]] simd DPM_SAFE_ARRAY operator-() const noexcept
		{
			simd result;
			const auto mask = _mm_set1_ps(-0.0f);
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_xor_ps(m_data[i], mask);
			return result;
		}

		[[nodiscard]] friend DPM_SAFE_ARRAY simd operator+(const simd &a, const simd &b) noexcept
		{
			simd result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_add_ps(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY simd operator-(const simd &a, const simd &b) noexcept
		{
			simd result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_sub_ps(a.m_data[i], b.m_data[i]);
			return result;
		}

		friend DPM_SAFE_ARRAY simd &operator+=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_add_ps(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend DPM_SAFE_ARRAY simd &operator-=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_sub_ps(a.m_data[i], b.m_data[i]);
			return a;
		}

		[[nodiscard]] friend DPM_SAFE_ARRAY simd operator*(const simd &a, const simd &b) noexcept
		{
			simd result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_mul_ps(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY simd operator/(const simd &a, const simd &b) noexcept
		{
			simd result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_div_ps(a.m_data[i], b.m_data[i]);
			return result;
		}

		friend DPM_SAFE_ARRAY simd &operator*=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_mul_ps(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend DPM_SAFE_ARRAY simd &operator/=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_div_ps(a.m_data[i], b.m_data[i]);
			return a;
		}

		[[nodiscard]] friend DPM_SAFE_ARRAY mask_type operator==(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpeq_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY mask_type operator<=(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmple_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY mask_type operator>=(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpge_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY mask_type operator<(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmplt_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY mask_type operator>(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpgt_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY mask_type operator!=(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpneq_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}

	private:
		alignas(alignment) data_type m_data;
	};

	template<std::size_t N, std::size_t A> requires detail::x86_overload_m128<float, N, A>
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

		[[nodiscard]] simd_t operator-() const && noexcept { return ext::blend(m_data, -m_data, m_mask); }
		[[nodiscard]] simd_t operator+() const && noexcept { return ext::blend(m_data, +m_data, m_mask); }

		/** Copies selected elements to \a mem. */
		template<typename U, typename Flags>
		DPM_SAFE_ARRAY void copy_to(U *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
#ifdef DPM_HAS_AVX
			if constexpr (aligned_tag<Flags, alignof(__m128)>)
			{
				const auto v_mask = ext::to_native_data(m_data);
				const auto v_data = ext::to_native_data(m_data);
				if constexpr (std::same_as<std::remove_cvref_t<U>, float>)
				{
					for (std::size_t i = 0; i < mask_t::size(); i += 4)
					{
						const auto mi = std::bit_cast<__m128i>(detail::x86_maskzero_vector_f32(mask_t::size() - i, v_mask[i / 4]));
						_mm_maskstore_ps(mem + i, mi, v_data[i / 4]);
					}
					return;
				}
#ifdef DPM_HAS_AVX2
				else if constexpr (detail::signed_integral_of_size<std::remove_cvref_t<U>, 4>)
				{
					for (std::size_t i = 0; i < mask_t::size(); i += 4)
					{
						const auto mi = std::bit_cast<__m128i>(detail::x86_maskzero_vector_f32(mask_t::size() - i, v_mask[i / 4]));
						_mm_maskstore_epi32(mem + i, mi, _mm_cvtps_epi32(v_data[i / 4]));
					}
					return;
				}
#endif
			}
			else
#endif
			for (std::size_t i = 0; i < mask_t::size(); ++i) if (m_mask[i]) mem[i] = static_cast<U>(m_data[i]);
		}

	protected:
		mask_t m_mask;
		simd_t &m_data;
	};
	template<std::size_t N, std::size_t A> requires detail::x86_overload_m128<float, N, A>
	class where_expression<simd_mask<float, detail::avec<N, A>>, simd<float, detail::avec<N, A>>>
			: public const_where_expression<simd_mask<float, detail::avec<N, A>>, simd<float, detail::avec<N, A>>>
	{
		using base_expr = const_where_expression<simd_mask<float, detail::avec<N, A>>, simd<float, detail::avec<N, A>>>;
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
		DPM_SAFE_ARRAY void copy_from(const U *mem, Flags) && noexcept requires is_simd_flag_type_v<Flags>
		{
#ifdef DPM_HAS_AVX
			if constexpr (aligned_tag<Flags, alignof(__m128)>)
			{
				const auto v_mask = ext::to_native_data(m_data);
				const auto v_data = ext::to_native_data(m_data);
				if constexpr (std::same_as<std::remove_cvref_t<U>, float>)
				{
					for (std::size_t i = 0; i < mask_t::size(); i += 4)
					{
						const auto mi = std::bit_cast<__m128i>(detail::x86_maskzero_vector_f32(mask_t::size() - i, v_mask[i / 4]));
						v_data[i / 4] = _mm_maskload_ps(mem + i, mi);
					}
					return;
				}
#ifdef DPM_HAS_AVX2
				else if constexpr (detail::signed_integral_of_size<std::remove_cvref_t<U>, 4>)
				{
					for (std::size_t i = 0; i < mask_t::size(); i += 4)
					{
						const auto mi = std::bit_cast<__m128i>(detail::x86_maskzero_vector_f32(mask_t::size() - i, v_mask[i / 4]));
						v_data[i / 4] = _mm_cvtepi32_ps(_mm_maskload_epi32(mem + i, mi));
					}
					return;
				}
#endif
			}
			else
#endif
			for (std::size_t i = 0; i < mask_t::size(); ++i) if (m_mask[i]) m_data[i] = static_cast<float>(mem[i]);
		}
	};

	namespace detail
	{
		template<std::size_t N, std::size_t A> requires detail::x86_overload_m128<float, N, A>
		struct native_access<simd<float, avec<N, A>>>
		{
			using simd_t = simd<float, avec<N, A>>;

			[[nodiscard]] static std::span<__m128, simd_t::data_size> to_native_data(simd_t &x) noexcept { return {x.m_data}; }
			[[nodiscard]] static std::span<const __m128, simd_t::data_size> to_native_data(const simd_t &x) noexcept { return {x.m_data}; }
		};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a x. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128, detail::align_data<float, N, 16>()> to_native_data(simd<float, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<float, N, A>
		{
			return detail::native_access<simd<float, detail::avec<N, A>>>::to_native_data(x);
		}
		/** Returns a constant span of the underlying SSE vectors for \a x. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128, detail::align_data<float, N, 16>()> to_native_data(const simd<float, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<float, N, A>
		{
			return detail::native_access<simd<float, detail::avec<N, A>>>::to_native_data(x);
		}

#ifdef DPM_HAS_SSE4_1
		/** Replaces elements of vector \a a with elements of vector \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline DPM_SAFE_ARRAY simd<float, detail::avec<N, A>> blend(
				const simd<float, detail::avec<N, A>> &a,
				const simd<float, detail::avec<N, A>> &b,
				const simd_mask<float, detail::avec<N, A>> &m)
		noexcept requires detail::x86_overload_m128<float, N, A>
		{
			constexpr auto data_size = native_data_size_v<simd<float, detail::avec<N, A>>>;

			simd<float, detail::avec<N, A>> result;
			auto result_data = to_native_data(result)();
			const auto a_data = to_native_data(a)();
			const auto b_data = to_native_data(b)();
			const auto m_data = to_native_data(m)();

			for (std::size_t i = 0; i < data_size; ++i) result_data[i] = _mm_blendv_ps(a_data[i], b_data[i], m_data[i]);
			return result;
		}
#endif
	}

#pragma region "simd casts"
	/** Implicitly converts elements of SIMD vector \a x to the `To` type, where `To` is either `typename T::value_type` or `T` if `T` is a scalar. */
	template<typename T, std::size_t N, std::size_t A, typename To = typename detail::deduce_cast<T>::type>
	[[nodiscard]] inline DPM_SAFE_ARRAY auto simd_cast(const simd<float, detail::avec<N, A>> &x) noexcept
	requires (detail::valid_simd_cast<T, float, detail::avec<N, A>> &&
	          detail::x86_overload_m128<float, N, A> &&
	          detail::x86_overload_any<To, N, A>)
	{
		typename detail::cast_return<T, float, detail::avec<N, A>, simd<float, detail::avec<N, A>>::size()>::type result;
		x.copy_to(reinterpret_cast<To *>(ext::to_native_data(result).data()), vector_aligned);
		return result;
	}
	/** Explicitly converts elements of SIMD vector \a x to the `To` type, where `To` is either `typename T::value_type` or `T` if `T` is a scalar. */
	template<typename T, std::size_t N, std::size_t A, typename To = typename detail::deduce_cast<T>::type>
	[[nodiscard]] inline DPM_SAFE_ARRAY auto static_simd_cast(const simd<float, detail::avec<N, A>> &x) noexcept
	requires (detail::valid_simd_static_cast<T, float, detail::avec<N, A>> &&
	          detail::x86_overload_m128<float, N, A> &&
	          detail::x86_overload_any<To, N, A>)
	{
		typename detail::static_cast_return<T, float, detail::avec<N, A>, simd<float, detail::avec<N, A>>::size()>::type result;
		x.copy_to(reinterpret_cast<To *>(ext::to_native_data(result).data()), vector_aligned);
		return result;
	}

	/** Concatenates elements of \a values into a single SIMD vector. */
	template<detail::x86_simd_abi_m128<float>... Abis>
	[[nodiscard]] inline DPM_SAFE_ARRAY auto concat(const simd<float, Abis> &...values) noexcept
	{
		if constexpr (sizeof...(values) == 1)
			return (values, ...);
		else
		{
			simd<float, simd_abi::deduce_t<float, (simd_size_v<float, Abis> + ...)>> result;
			detail::concat_impl<0>(result, values...);
			return result;
		}
	}
	/** Concatenates elements of \a values into a single SIMD vector. */
	template<std::size_t N, std::size_t A, std::size_t M>
	[[nodiscard]] inline DPM_SAFE_ARRAY auto concat(const std::array<simd<float, detail::avec<N, A>>, M> &values) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		if constexpr (M == 1)
			return values[0];
		else
		{
			simd<float, detail::avec<N * M, A>> result;
			auto *data = reinterpret_cast<float *>(ext::to_native_data(result).data());
			for (std::size_t i = 0; i < M; ++i)
			{
				if ((i * N) % sizeof(__m128) != 0)
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
	[[nodiscard]] inline DPM_SAFE_ARRAY simd<float, detail::avec<N, A>> min(
			const simd<float, detail::avec<N, A>> &a,
			const simd<float, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_m128<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<float, detail::avec<N, A>>>;

		simd<float, detail::avec<N, A>> result;
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < data_size; ++i) result_data[i] = _mm_min_ps(a_data[i], b_data[i]);
		return result;
	}
	/** Returns an SIMD vector of maximum elements of \a a and \a b. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd<float, detail::avec<N, A>> max(
			const simd<float, detail::avec<N, A>> &a,
			const simd<float, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_m128<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<float, detail::avec<N, A>>>;

		simd<float, detail::avec<N, A>> result;
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < data_size; ++i) result_data[i] = _mm_max_ps(a_data[i], b_data[i]);
		return result;
	}

	/** Returns a pair of SIMD vectors of minimum and maximum elements of \a a and \a b. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY std::pair<simd<float, detail::avec<N, A>>, simd<float, detail::avec<N, A>>> minmax(
			const simd<float, detail::avec<N, A>> &a,
			const simd<float, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_m128<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<float, detail::avec<N, A>>>;

		std::pair<simd<float, detail::avec<N, A>>, simd<float, detail::avec<N, A>>> result;
		auto min_data = ext::to_native_data(result.first), max_data = ext::to_native_data(result.second);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < data_size; ++i)
		{
			max_data[i] = _mm_min_ps(a[i], b[i]);
			max_data[i] = _mm_max_ps(a[i], b[i]);
		}
		return result;
	}

	/** Clamps elements \f \a x between corresponding elements of \a ming and \a max. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd<float, detail::avec<N, A>> clamp(
			const simd<float, detail::avec<N, A>> &x,
			const simd<float, detail::avec<N, A>> &min,
			const simd<float, detail::avec<N, A>> &max)
	noexcept requires detail::x86_overload_m128<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<float, detail::avec<N, A>>>;

		simd<float, detail::avec<N, A>> result;
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