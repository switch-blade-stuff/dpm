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

	template<std::size_t N, std::size_t Align> requires detail::x86_overload_128<float, N, Align>
	class simd<float, detail::avec<N, Align>>
	{
		friend struct detail::native_access<simd>;

		constexpr static auto alignment = std::max<std::size_t>(Align, 16);
		constexpr static auto data_size = ext::native_data_size_v<simd>;

		using value_alias = detail::alias_t<float>;
		using storage_type = __m128[data_size];

	public:
		using value_type = float;
		using reference = value_alias &;

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
		constexpr simd(__m128 native) noexcept requires (data_size == 1) { m_data[0] = native; }
		/** Initializes the SIMD vector with an array of native SSE vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd) / sizeof(__m128)`. */
		constexpr simd(const __m128 (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

		/** Initializes the underlying elements with \a value. */
		template<detail::compatible_element<value_type> U>
		simd(U &&value) noexcept
		{
			const auto vec = _mm_set1_ps(static_cast<float>(value));
			std::fill_n(m_data, data_size, vec);
		}
		/** Initializes the underlying elements with values provided by the generator \a gen. */
		template<detail::element_generator<value_type, size()> G>
		simd(G &&gen) noexcept
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
		simd(const simd<U, detail::avec<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (constexpr auto other_alignment = alignof(decltype(other)); other_alignment >= alignment)
				copy_from(reinterpret_cast<const detail::alias_t<U> *>(ext::to_native_data(other).data()), vector_aligned);
			else if constexpr (other_alignment != alignof(value_type))
				copy_from(reinterpret_cast<const detail::alias_t<U> *>(ext::to_native_data(other).data()), overaligned<other_alignment>);
			else
				copy_from(reinterpret_cast<const detail::alias_t<U> *>(ext::to_native_data(other).data()), element_aligned);
		}
		/** Initializes the underlying elements from \a mem. */
		template<typename U, typename Flags>
		simd(const U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying elements from \a mem. */
		template<typename U, typename Flags>
		DPM_FORCEINLINE void copy_from(const U *mem, Flags) noexcept requires std::convertible_to<U, float> && is_simd_flag_type_v<Flags>
		{
			if constexpr (detail::aligned_tag<Flags, 16> && std::same_as<std::remove_volatile_t<U>, float>)
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
			}
#ifdef DPM_HAS_SSE2
			else if constexpr (detail::aligned_tag<Flags, 16> && detail::integral_of_size<std::remove_volatile_t<U>, 4>)
			{
				for (std::size_t i = 0; i < size(); i += 4)
					switch (size() - i)
					{
						default:
						{
							if constexpr (!std::is_signed_v<std::remove_volatile_t<U>>)
								m_data[i / 4] = detail::cvt_u32_f32(reinterpret_cast<const __m128i *>(mem)[i / 4]);
							else
								m_data[i / 4] = _mm_cvtepi32_ps(reinterpret_cast<const __m128i *>(mem)[i / 4]);
							break;
						}
						case 3: operator[](i + 2) = static_cast<float>(mem[i + 2]); [[fallthrough]];
						case 2: operator[](i + 1) = static_cast<float>(mem[i + 1]); [[fallthrough]];
						case 1: operator[](i) = static_cast<float>(mem[i]);
					}
			}
			else
#endif
				for (std::size_t i = 0; i < size(); ++i) operator[](i) = static_cast<float>(mem[i]);
		}
		/** Copies the underlying elements to \a mem. */
		template<typename U, typename Flags>
		DPM_FORCEINLINE void copy_to(U *mem, Flags) const noexcept requires std::convertible_to<float, U> && is_simd_flag_type_v<Flags>
		{
			if constexpr (detail::aligned_tag<Flags, 16> && std::same_as<std::remove_volatile_t<U>, float>)
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
			}
#ifdef DPM_HAS_SSE2
			else if constexpr (detail::aligned_tag<Flags, 16> && detail::integral_of_size<std::remove_volatile_t<U>, 4>)
			{
				for (std::size_t i = 0; i < size(); i += 4)
					switch (size() - i)
					{
						default:
						{
							if constexpr (!std::is_signed_v<std::remove_volatile_t<U>>)
								reinterpret_cast<__m128i *>(mem)[i / 4] = detail::cvt_f32_u32(m_data[i / 4]);
							else
								reinterpret_cast<__m128i *>(mem)[i / 4] = _mm_cvtps_epi32(m_data[i / 4]);
							break;
						}
						case 3: mem[i + 2] = static_cast<std::int32_t>(operator[](i + 2)); [[fallthrough]];
						case 2: mem[i + 1] = static_cast<std::int32_t>(operator[](i + 1)); [[fallthrough]];
						case 1: mem[i] = static_cast<std::int32_t>(operator[](i));
					}
			}
			else
#endif
				for (std::size_t i = 0; i < size(); ++i) mem[i] = static_cast<U>(operator[](i));
		}

		[[nodiscard]] DPM_FORCEINLINE reference operator[](std::size_t i) noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<value_alias *>(m_data)[i];
		}
		[[nodiscard]] DPM_FORCEINLINE value_type operator[](std::size_t i) const noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<const value_alias *>(m_data)[i];
		}

		DPM_FORCEINLINE simd operator++(int) noexcept
		{
			auto tmp = *this;
			operator++();
			return tmp;
		}
		DPM_FORCEINLINE simd operator--(int) noexcept
		{
			auto tmp = *this;
			operator--();
			return tmp;
		}
		DPM_FORCEINLINE simd &operator++() noexcept
		{
			const auto one = _mm_set1_ps(1.0f);
			for (std::size_t i = 0; i < data_size; ++i) m_data[i] = _mm_add_ps(m_data[i], one);
			return *this;
		}
		DPM_FORCEINLINE simd &operator--() noexcept
		{
			const auto one = _mm_set1_ps(1.0f);
			for (std::size_t i = 0; i < data_size; ++i) m_data[i] = _mm_sub_ps(m_data[i], one);
			return *this;
		}

		[[nodiscard]] DPM_FORCEINLINE simd operator+() const noexcept { return *this; }
		[[nodiscard]] DPM_FORCEINLINE simd operator-() const noexcept
		{
			simd result = {};
			const auto mask = _mm_set1_ps(-0.0f);
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_xor_ps(m_data[i], mask);
			return result;
		}

		[[nodiscard]] friend DPM_FORCEINLINE simd operator+(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_add_ps(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend DPM_FORCEINLINE simd operator-(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_sub_ps(a.m_data[i], b.m_data[i]);
			return result;
		}

		friend DPM_FORCEINLINE simd &operator+=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_add_ps(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend DPM_FORCEINLINE simd &operator-=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_sub_ps(a.m_data[i], b.m_data[i]);
			return a;
		}

		[[nodiscard]] friend DPM_FORCEINLINE simd operator*(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_mul_ps(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend DPM_FORCEINLINE simd operator/(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_div_ps(a.m_data[i], b.m_data[i]);
			return result;
		}

		friend DPM_FORCEINLINE simd &operator*=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_mul_ps(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend DPM_FORCEINLINE simd &operator/=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_div_ps(a.m_data[i], b.m_data[i]);
			return a;
		}

		[[nodiscard]] friend DPM_FORCEINLINE mask_type operator==(const simd &a, const simd &b) noexcept
		{
			storage_type mask_data = {};
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpeq_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend DPM_FORCEINLINE mask_type operator<=(const simd &a, const simd &b) noexcept
		{
			storage_type mask_data = {};
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmple_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend DPM_FORCEINLINE mask_type operator>=(const simd &a, const simd &b) noexcept
		{
			storage_type mask_data = {};
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpge_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend DPM_FORCEINLINE mask_type operator<(const simd &a, const simd &b) noexcept
		{
			storage_type mask_data = {};
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmplt_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend DPM_FORCEINLINE mask_type operator>(const simd &a, const simd &b) noexcept
		{
			storage_type mask_data = {};
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpgt_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend DPM_FORCEINLINE mask_type operator!=(const simd &a, const simd &b) noexcept
		{
			storage_type mask_data = {};
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpneq_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}

	private:
		alignas(alignment) storage_type m_data;
	};

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

		DPM_FORCEINLINE void operator++() && noexcept
		{
			const auto old_data = m_data;
			const auto new_data = ++old_data;
			m_data = ext::blend(old_data, new_data, m_mask);
		}
		DPM_FORCEINLINE void operator--() && noexcept
		{
			const auto old_data = m_data;
			const auto new_data = --old_data;
			m_data = ext::blend(old_data, new_data, m_mask);
		}
		DPM_FORCEINLINE void operator++(int) && noexcept
		{
			const auto old_data = m_data++;
			m_data = ext::blend(old_data, m_data, m_mask);
		}
		DPM_FORCEINLINE void operator--(int) && noexcept
		{
			const auto old_data = m_data--;
			m_data = ext::blend(old_data, m_data, m_mask);
		}

		template<typename U>
		DPM_FORCEINLINE void operator+=(U &&value) && noexcept requires std::is_convertible_v<U, value_type>
		{
			const auto new_data = m_data + simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		DPM_FORCEINLINE void operator-=(U &&value) && noexcept requires std::is_convertible_v<U, value_type>
		{
			const auto new_data = m_data - simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}

		template<typename U>
		DPM_FORCEINLINE void operator*=(U &&value) && noexcept requires std::is_convertible_v<U, value_type>
		{
			const auto new_data = m_data * simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		DPM_FORCEINLINE void operator/=(U &&value) && noexcept requires std::is_convertible_v<U, value_type>
		{
			const auto new_data = m_data / simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}

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