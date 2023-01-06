/*
 * Created by switchblade on 2023-01-06.
 */

#pragma once

#include "../generic/type.hpp"
#include "../utility.hpp"

#include "abi.hpp"

#if defined(SVM_ARCH_X86) && (defined(SVM_HAS_SSE) || defined(SVM_DYNAMIC_DISPATCH))

#ifndef SVM_USE_IMPORT

#include <bit>

#endif

namespace svm
{
	namespace detail
	{
		using simd_abi::detail::x86_sse_overload;
		using simd_abi::detail::x86_avx_overload;
		using simd_abi::detail::x86_avx512_overload;

		template<std::size_t N, std::size_t A>
		using avec = simd_abi::ext::aligned_vector<N, A>;
	}

	/* Where expressions need to be specialized for SIMD operations. Underlying implementation is handled via
	 * detail::simd_access, as such where expressions are specialized for all 3 SIMD levels using the same type. */
	template<typename T, std::size_t N, std::size_t A> requires (detail::x86_sse_overload<T, N, A> || detail::x86_avx_overload<T, N, A> || detail::x86_avx512_overload<T, N, A>)
	class const_where_expression<simd_mask<T, detail::avec<N, A>>, simd<T, detail::avec<N, A>>>
	{
		template<typename U, typename Abi, typename K>
		friend inline simd<U, Abi> ext::blend(const simd<U, Abi> &, const const_where_expression<K, simd<U, Abi>> &);
		template<typename U, typename Abi, typename K>
		friend inline simd_mask<U, Abi> ext::blend(const simd_mask<U, Abi> &, const const_where_expression<K, simd_mask<U, Abi>> &);

		template<typename U>
		friend inline U ext::blend(const U &, const const_where_expression<bool, U> &);

	protected:
		using mask_t = simd_mask<T, detail::avec<N, A>>;
		using simd_t = simd<T, detail::avec<N, A>>;
		using value_type = T;

	public:
		const_where_expression(const const_where_expression &) = delete;
		const_where_expression &operator=(const const_where_expression &) = delete;

		const_where_expression(mask_t mask, simd_t &data) noexcept : m_mask(mask), m_data(data) {}
		const_where_expression(mask_t mask, const simd_t &data) noexcept : m_mask(mask), m_data(const_cast<simd_t &>(data)) {}

		[[nodiscard]] simd_t operator-() const && noexcept { return ext::blend(m_data, -m_data, m_mask); }
		[[nodiscard]] simd_t operator+() const && noexcept { return ext::blend(m_data, +m_data, m_mask); }

		/** Copies selected elements to \a mem. */
		template<typename U, typename Flags>
		inline void copy_to(U *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
			if constexpr (std::same_as<U, value_type>)
				detail::simd_access<simd_t>::copy_to_masked(m_data, m_mask, mem, Flags{});
			else
				for (std::size_t i = 0; i < simd_t::size(); ++i) if (m_mask[i]) mem[i] = static_cast<U>(m_data[i]);
		}

	protected:
		mask_t m_mask;
		simd_t &m_data;
	};
	template<typename T, std::size_t N, std::size_t A> requires (detail::x86_sse_overload<T, N, A> || detail::x86_avx_overload<T, N, A> || detail::x86_avx512_overload<T, N, A>)
	class where_expression<simd_mask<T, detail::avec<N, A>>, simd<T, detail::avec<N, A>>> : public const_where_expression<simd_mask<T, detail::avec<N, A>>, simd<T, detail::avec<N, A>>>
	{
		using base_expr = const_where_expression<simd_mask<T, detail::avec<N, A>>, simd<T, detail::avec<N, A>>>;
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
		void copy_from(U *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
			if constexpr (std::same_as<U, value_type>)
				detail::simd_access<simd_t>::copy_from_masked(m_data, m_mask, mem, Flags{});
			else
				for (std::size_t i = 0; i < simd_t::size(); ++i) if (m_mask[i]) m_data[i] = static_cast<value_type>(mem[i]);
		}
	};

	template<typename T, std::size_t N, std::size_t A> requires (detail::x86_sse_overload<T, N, A> || detail::x86_avx_overload<T, N, A> || detail::x86_avx512_overload<T, N, A>)
	class const_where_expression<simd_mask<T, detail::avec<N, A>>, simd_mask<T, detail::avec<N, A>>>
	{
		template<typename U, typename Abi, typename K>
		friend inline simd<U, Abi> ext::blend(const simd<U, Abi> &, const const_where_expression<K, simd<U, Abi>> &);
		template<typename U, typename Abi, typename K>
		friend inline simd_mask<U, Abi> ext::blend(const simd_mask<U, Abi> &, const const_where_expression<K, simd_mask<U, Abi>> &);

		template<typename U>
		friend inline U ext::blend(const U &, const const_where_expression<bool, U> &);

	protected:
		using mask_t = simd_mask<T, detail::avec<N, A>>;
		using value_type = bool;

	public:
		const_where_expression(const const_where_expression &) = delete;
		const_where_expression &operator=(const const_where_expression &) = delete;

		const_where_expression(mask_t mask, mask_t &data) noexcept : m_mask(mask), m_data(data) {}
		const_where_expression(mask_t mask, const mask_t &data) noexcept : m_mask(mask), m_data(const_cast<mask_t &>(data)) {}

		/** Copies selected elements to \a mem. */
		template<typename U, typename Flags>
		inline void copy_to(U *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
			if constexpr (std::same_as<U, value_type>)
				detail::simd_access<mask_t>::copy_to_masked(m_data, m_mask, mem, Flags{});
			else
				for (std::size_t i = 0; i < mask_t::size(); ++i) if (m_mask[i]) mem[i] = static_cast<U>(m_data[i]);
		}

	protected:
		mask_t m_mask;
		mask_t &m_data;
	};
	template<typename T, std::size_t N, std::size_t A> requires (detail::x86_sse_overload<T, N, A> || detail::x86_avx_overload<T, N, A> || detail::x86_avx512_overload<T, N, A>)
	class where_expression<simd_mask<T, detail::avec<N, A>>, simd_mask<T, detail::avec<N, A>>> : public const_where_expression<simd_mask<T, detail::avec<N, A>>, simd_mask<T, detail::avec<N, A>>>
	{
		using base_expr = const_where_expression<simd_mask<T, detail::avec<N, A>>, simd_mask<T, detail::avec<N, A>>>;
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
		template<typename U, typename Flags>
		void copy_from(U *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
			if constexpr (std::same_as<U, value_type>)
				detail::simd_access<mask_t>::copy_from_masked(m_data, m_mask, mem, Flags{});
			else
				for (std::size_t i = 0; i < mask_t::size(); ++i) if (m_mask[i]) m_data[i] = static_cast<value_type>(mem[i]);
		}
	};
}

#endif