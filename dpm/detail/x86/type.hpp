/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include "type_fwd.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE)

#include "transform.hpp"
#include "bitwise.hpp"
#include "addsub.hpp"
#include "muldiv.hpp"
#include "minmax.hpp"
#include "cmp.hpp"
#include "cvt.hpp"

namespace dpm
{
	DPM_DECLARE_EXT_NAMESPACE
	{
		template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_128<T, N, Align>
		struct native_data_type<detail::x86_mask<T, N, Align>> { using type = typename detail::select_vector<T, 16>::type; };
		template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_128<T, N, Align>
		struct native_data_size<detail::x86_mask<T, N, Align>> : std::integral_constant<std::size_t, detail::align_data<T, N, 16>()> {};

		template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_128<T, N, Align>
		struct native_data_type<detail::x86_simd<T, N, Align>> { using type = typename detail::select_vector<T, 16>::type; };
		template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_128<T, N, Align>
		struct native_data_size<detail::x86_simd<T, N, Align>> : std::integral_constant<std::size_t, detail::align_data<T, N, 16>()> {};

#ifdef DPM_HAS_AVX
		template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_256<T, N, Align>
		struct native_data_type<detail::x86_mask<T, N, Align>> { using type = typename detail::select_vector<T, 32>::type; };
		template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_256<T, N, Align>
		struct native_data_size<detail::x86_mask<T, N, Align>> : std::integral_constant<std::size_t, detail::align_data<T, N, 32>()> {};

		template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_256<T, N, Align>
		struct native_data_type<detail::x86_simd<T, N, Align>> { using type = typename detail::select_vector<T, 32>::type; };
		template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_256<T, N, Align>
		struct native_data_size<detail::x86_simd<T, N, Align>> : std::integral_constant<std::size_t, detail::align_data<T, N, 32>()> {};
#endif
	}

	namespace detail
	{
		template<typename F, typename V, typename... Vs>
		DPM_FORCEINLINE void vectorize(F f, V &&v, Vs &&...vs) noexcept
		{
			for (std::size_t i = 0; i < ext::native_data_size_v<std::remove_cvref_t<V>>; ++i)
				f(ext::to_native_data(v)[i], ext::to_native_data(vs)[i]...);
		}
	}

	template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_any<T, N, Align>
	class simd_mask<T, detail::avec<N, Align>>
	{
		friend struct detail::native_access<simd_mask>;

		using native_type = ext::native_data_type_t<simd_mask>;

		constexpr static auto alignment = std::max<std::size_t>(Align, sizeof(native_type));
		constexpr static auto native_extent = sizeof(native_type) / sizeof(T);
		constexpr static auto data_size = ext::native_data_size_v<simd_mask>;

		using value_alias = typename detail::sized_mask<sizeof(T)>::type;
		using storage_type = std::array<native_type, data_size>;

	public:
		using value_type = bool;
		using reference = value_alias &;

		using abi_type = detail::avec<N, Align>;
		using simd_type = simd<T, abi_type>;

		/** Returns width of the SIMD mask. */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	public:
		constexpr simd_mask() noexcept = default;
		constexpr simd_mask(const simd_mask &) noexcept = default;
		constexpr simd_mask &operator=(const simd_mask &) noexcept = default;
		constexpr simd_mask(simd_mask &&) noexcept = default;
		constexpr simd_mask &operator=(simd_mask &&) noexcept = default;

		/** Initializes the SIMD mask object with a native mask vector. */
		constexpr simd_mask(native_type native) noexcept requires (data_size == 1) { m_data[0] = native; }
		/** Initializes the SIMD mask object with an array of native mask vectors. */
		constexpr simd_mask(const native_type (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

		/** Initializes the underlying elements with \a value. */
		DPM_FORCEINLINE simd_mask(value_type value) noexcept { m_data.fill(value ? detail::setones<native_type>() : detail::setzero<native_type>()); }
		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		DPM_FORCEINLINE simd_mask(const simd_mask<U, detail::avec<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (std::same_as<U, value_type> && alignof(decltype(other)) >= alignment)
				std::copy_n(reinterpret_cast<const native_type *>(ext::to_native_data(other).data()), data_size, m_data);
			else
				for (std::size_t i = 0; i < size(); ++i) operator[](i) = other[i];
		}
		/** Initializes the underlying elements from \a mem. */
		template<typename Flags>
		DPM_FORCEINLINE simd_mask(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying elements from \a mem. */
		template<typename Flags>
		DPM_FORCEINLINE void copy_from(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags>
		{
			for (std::size_t i = 0; i < size(); i += native_extent)
			{
				alignas(native_type) T values[native_extent] = {};
				for (std::size_t j = 0; i + j < size() && j < native_extent; ++j)
				{
					const auto extended = detail::extend_bool<detail::int_of_size_t<sizeof(T)>>(mem[i + j]);
					values[j] = std::bit_cast<T>(extended);
				}
				m_data[i / native_extent] = detail::set<native_type>(values);
			}
		}
		/** Copies the underlying elements to \a mem. */
		template<typename Flags>
		DPM_FORCEINLINE void copy_to(value_type *mem, Flags) const noexcept requires is_simd_flag_type_v<Flags>
		{
			for (std::size_t i = 0; i < size();)
			{
				const auto bits = detail::movemask<T>(m_data[i / native_extent]);
				for (std::size_t j = 0; i < size() && j < native_extent; ++j, ++i)
					mem[i] = bits & (1 << (j * detail::movemask_bits_v<T>));
			}
		}

		[[nodiscard]] DPM_FORCEINLINE reference operator[](std::size_t i) noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<value_alias *>(m_data.data())[i];
		}
		[[nodiscard]] DPM_FORCEINLINE value_type operator[](std::size_t i) const noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<const value_alias *>(m_data.data())[i];
		}

	private:
		alignas(alignment) storage_type m_data;
	};

	namespace detail
	{
		template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
		struct native_access<x86_mask<T, N, A>>
		{
			using mask_t = x86_mask<T, N, A>;
			static std::span<ext::native_data_type_t<mask_t>, mask_t::data_size> to_native_data(mask_t &x) noexcept { return {x.m_data}; }
			static std::span<const ext::native_data_type_t<mask_t>, mask_t::data_size> to_native_data(const mask_t &x) noexcept { return {x.m_data}; }
		};

		template<std::size_t I, typename T, typename OutAbi, typename XAbi, typename... Abis>
		DPM_FORCEINLINE void concat_impl(simd_mask<T, OutAbi> &out, const simd_mask<T, XAbi> &x, const simd_mask<T, Abis> &...rest) noexcept
		{
			auto *data = reinterpret_cast<T *>(ext::to_native_data(out).data());
			if constexpr (I % sizeof(ext::native_data_type_t<simd_mask<T, OutAbi>>) != 0)
				x.copy_to(data + I, element_aligned);
			else
				x.copy_to(data + I, vector_aligned);

			if constexpr (sizeof...(Abis) != 0) concat_impl<I + simd_mask<T, XAbi>::size()>(out, rest...);
		}
	}

	template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_any<T, N, Align>
	class const_where_expression<detail::x86_mask<T, N, Align>, detail::x86_mask<T, N, Align>>
	{
		template<typename U, typename Abi, typename K>
		friend inline simd<U, Abi> ext::blend(const simd<U, Abi> &, const const_where_expression<K, simd<U, Abi>> &);
		template<typename U, typename Abi, typename K>
		friend inline simd_mask<U, Abi> ext::blend(const simd_mask<U, Abi> &, const const_where_expression<K, simd_mask<U, Abi>> &);

	protected:
		using mask_t = detail::x86_mask<T, N, Align>;
		using native_type = ext::native_data_type_t<mask_t>;
		using value_type = bool;

	public:
		const_where_expression(const const_where_expression &) = delete;
		const_where_expression &operator=(const const_where_expression &) = delete;

		const_where_expression(mask_t mask, mask_t &data) noexcept : m_mask(mask), m_data(data) {}
		const_where_expression(mask_t mask, const mask_t &data) noexcept : m_mask(mask), m_data(const_cast<mask_t &>(data)) {}

		/** Copies selected elements to \a mem. */
		template<typename Flags>
		DPM_FORCEINLINE void copy_to(value_type *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
			constexpr auto native_extent = sizeof(native_type) / sizeof(T);
			const auto v_mask = ext::to_native_data(m_mask);
			const auto v_data = ext::to_native_data(m_data);
			for (std::size_t i = 0; i < mask_t::size(); i += native_extent)
			{
				const auto mask_bits = detail::movemask<T>(v_mask[i / native_extent]);
				if (!mask_bits) [[unlikely]] continue;

				const auto data_bits = detail::movemask<T>(v_data[i / native_extent]);
				for (std::size_t j = 0; i + j < mask_t::size() && j < native_extent; ++j)
				{
					const auto j_bit = 1 << (j * detail::movemask_bits_v<T>);
					if (mask_bits & j_bit) mem[i + j] = data_bits & j_bit;
				}
			}
		}

	protected:
		mask_t m_mask;
		mask_t &m_data;
	};
	template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_any<T, N, Align>
	class where_expression<detail::x86_mask<T, N, Align>, detail::x86_mask<T, N, Align>> : public const_where_expression<detail::x86_mask<T, N, Align>, detail::x86_mask<T, N, Align>>
	{
		using base_expr = const_where_expression<detail::x86_mask<T, N, Align>, detail::x86_mask<T, N, Align>>;
		using native_type = typename base_expr::native_type;
		using value_type = typename base_expr::value_type;
		using mask_t = typename base_expr::mask_t;

		using base_expr::m_mask;
		using base_expr::m_data;

	public:
		using base_expr::const_where_expression;

		template<typename U>
		DPM_FORCEINLINE void operator=(U &&value) && noexcept requires std::is_convertible_v<U, value_type>
		{
			m_data = ext::blend(m_data, mask_t{std::forward<U>(value)}, m_mask);
		}

		template<typename U>
		DPM_FORCEINLINE void operator&=(U &&value) && noexcept requires std::is_convertible_v<U, value_type>
		{
			const auto new_data = m_data & mask_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		DPM_FORCEINLINE void operator|=(U &&value) && noexcept requires std::is_convertible_v<U, value_type>
		{
			const auto new_data = m_data | mask_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		DPM_FORCEINLINE void operator^=(U &&value) && noexcept requires std::is_convertible_v<U, value_type>
		{
			const auto new_data = m_data ^ mask_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}

		/** Copies selected elements from \a mem. */
		template<typename Flags>
		DPM_FORCEINLINE void copy_from(const bool *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
			constexpr auto native_extent = sizeof(native_type) / sizeof(T);
			const auto v_mask = ext::to_native_data(m_mask);
			const auto v_data = ext::to_native_data(m_data);
			for (std::size_t i = 0; i < mask_t::size(); i += native_extent)
			{
				const auto &mask = v_mask[i / native_extent];
				const auto mask_bits = detail::movemask<T>(mask);
				if (!mask_bits) [[unlikely]] continue;

				alignas(native_type) T values[native_extent] = {};
				for (std::size_t j = 0; j < native_extent && i + j < mask_t::size(); ++j)
					if (const auto j_bit = 1ull << (j * detail::movemask_bits_v<T>); mask_bits & j_bit)
					{
						const auto extended = detail::extend_bool<detail::int_of_size_t<sizeof(T)>>(mem[i + j]);
						values[j] = std::bit_cast<T>(extended);
					}
				auto &data = v_data[i / native_extent];
				data = detail::blendv<T>(data, detail::set<native_type>(values), mask);
			}
		}
	};

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a x. */
		template<typename T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE auto to_native_data(detail::x86_mask<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
		{
			return detail::native_access<detail::x86_mask<T, N, A>>::to_native_data(x);
		}
		/** Returns a constant span of the underlying SSE vectors for \a x. */
		template<typename T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE auto to_native_data(const detail::x86_mask<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
		{
			return detail::native_access<detail::x86_mask<T, N, A>>::to_native_data(x);
		}

		/** Replaces elements of mask \a a with elements of mask \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<typename T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> blend(
				const detail::x86_mask<T, N, A> &a,
				const detail::x86_mask<T, N, A> &b,
				const detail::x86_mask<T, N, A> &m)
		noexcept requires detail::x86_overload_any<T, N, A>
		{
			detail::x86_mask<T, N, A> result = {};
			auto result_data = to_native_data(result);
			const auto a_data = to_native_data(a);
			const auto b_data = to_native_data(b);
			const auto m_data = to_native_data(m);

			for (std::size_t i = 0; i < native_data_size_v<detail::x86_mask<T, N, A>>; ++i)
				result_data[i] = detail::blendv<T>(a_data[i], b_data[i], m_data[i]);
			return result;
		}

		/** Shuffles elements of mask \a x into a new mask according to the specified indices. */
		template<std::size_t... Is, typename T, std::size_t N, std::size_t A, std::size_t M = sizeof...(Is)>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, M, A> shuffle(const detail::x86_mask<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A> && detail::x86_overload_any<T, M, A>
		{
			detail::x86_mask<T, M, A> result = {};
			auto result_data = to_native_data(result).data();
			const auto x_data = to_native_data(x).data();

			detail::shuffle<T, 0>(std::index_sequence<Is...>{}, result_data, x_data);
			return result;
		}
	}

#pragma region "simd_mask operators"
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator&(const detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_and(a, b); }, result, a, b);
		return result;
	}
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator^(const detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_xor(a, b); }, result, a, b);
		return result;
	}
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator|(const detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_or(a, b); }, result, a, b);
		return result;
	}

	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> &operator&=(detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::bit_and(a, b); }, a, b);
		return a;
	}
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> &operator^=(detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::bit_xor(a, b); }, a, b);
		return a;
	}
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> &operator|=(detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::bit_or(a, b); }, a, b);
		return a;
	}

	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator!(const detail::x86_mask<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::bit_not(x); }, result, x);
		return result;
	}

	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator&&(const detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return a & b;
	}
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator||(const detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return a | b;
	}

	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator==(const detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::mask_eq<T>(a, b); }, result, a, b);
		return result;
	}
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator!=(const detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return !(a == b);
	}
#pragma endregion

#pragma region "simd_mask reductions"
	/** Returns `true` if all of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool all_of(const detail::x86_mask<T, N, A> &mask) noexcept requires detail::x86_overload_any<T, N, A>
	{
		using mask_t = detail::x86_mask<T, N, A>;
		const auto mask_data = ext::to_native_data(mask);

		auto result = detail::setones<ext::native_data_type_t<mask_t>>();
		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto vm = detail::maskone<T>(mask_data[i], mask_t::size() - i * sizeof(result) / sizeof(T));
			result = detail::bit_and(result, vm);
		}
		return detail::movemask<T>(result) == detail::fill_bits<(sizeof(result) / sizeof(T)) * detail::movemask_bits_v<T>>();
	}
	/** Returns `true` if at least one of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool any_of(const detail::x86_mask<T, N, A> &mask) noexcept requires detail::x86_overload_any<T, N, A>
	{
		using mask_t = detail::x86_mask<T, N, A>;
		const auto mask_data = ext::to_native_data(mask);

		auto result = detail::setzero<ext::native_data_type_t<mask_t>>();
		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto vm = detail::maskzero<T>(mask_data[i], mask_t::size() - i * sizeof(result) / sizeof(T));
			result = detail::bit_or(result, vm);
		}
		return detail::movemask<T>(result);
	}
	/** Returns `true` if at none of the elements of the \a mask is `true`. Otherwise returns `false`. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool none_of(const detail::x86_mask<T, N, A> &mask) noexcept requires detail::x86_overload_any<T, N, A>
	{
		using mask_t = detail::x86_mask<T, N, A>;
		const auto mask_data = ext::to_native_data(mask);

		auto result = detail::setzero<ext::native_data_type_t<mask_t>>();
		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto vm = detail::maskzero<T>(mask_data[i], mask_t::size() - i * sizeof(result) / sizeof(T));
			result = detail::bit_or(result, vm);
		}
		return !detail::movemask<T>(result);
	}
	/** Returns `true` if at least one of the elements of the \a mask is `true` and at least one is `false`. Otherwise returns `false`. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool some_of(const detail::x86_mask<T, N, A> &mask) noexcept requires detail::x86_overload_any<T, N, A>
	{
		using mask_t = detail::x86_mask<T, N, A>;
		const auto mask_data = ext::to_native_data(mask);

		auto any_mask = detail::setzero<ext::native_data_type_t<mask_t>>(), all_mask = detail::setones<ext::native_data_type_t<mask_t>>();
		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto vm = mask_data[i];
			const auto n = mask_t::size() - i * sizeof(vm) / sizeof(T);
			const auto vmz = detail::maskzero<T>(vm, n);
			const auto vmo = detail::maskone<T>(vm, n);

			all_mask = detail::bit_and(all_mask, vmo);
			any_mask = detail::bit_or(any_mask, vmz);
		}
		return detail::movemask<T>(any_mask) && detail::movemask<T>(all_mask) != detail::fill_bits<(sizeof(all_mask) / sizeof(T)) * detail::movemask_bits_v<T>>();
	}

	/** Returns the number of `true` elements of \a mask. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::size_t popcount(const detail::x86_mask<T, N, A> &mask) noexcept requires detail::x86_overload_any<T, N, A>
	{
		using mask_t = detail::x86_mask<T, N, A>;
		const auto mask_data = ext::to_native_data(mask);

		std::size_t result = 0;
		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto vm = detail::maskzero<T>(mask_data[i], mask_t::size() - i * sizeof(mask_data[i]) / sizeof(T));
			result += std::popcount(detail::movemask<T>(vm));
		}
		return result / detail::movemask_bits_v<T>;
	}
	/** Returns the index of the first `true` element of \a mask. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::size_t find_first_set(const detail::x86_mask<T, N, A> &mask) noexcept requires detail::x86_overload_any<T, N, A>
	{
		using mask_t = detail::x86_mask<T, N, A>;
		const auto mask_data = ext::to_native_data(mask);
		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto bits = detail::movemask<T>(mask_data[i]);
			if (bits) return std::countr_zero(bits) / detail::movemask_bits_v<T> + i * sizeof(mask_data[i]) / sizeof(T);
		}
		DPM_UNREACHABLE();
	}
	/** Returns the index of the last `true` element of \a mask. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::size_t find_last_set(const detail::x86_mask<T, N, A> &mask) noexcept requires detail::x86_overload_any<T, N, A>
	{
		using mask_t = detail::x86_mask<T, N, A>;
		const auto mask_data = ext::to_native_data(mask);
		for (std::size_t i = ext::native_data_size_v<mask_t>, j; (j = i--) != 0;)
		{
			constexpr auto native_extent = sizeof(mask_data[i]) / sizeof(T);
			const auto bits = detail::movemask_l<T>(mask_data[i], mask_t::size() - i * native_extent);
			if (bits) return (j * native_extent - 1) - std::countl_zero(bits) / detail::movemask_bits_v<T>;
		}
		DPM_UNREACHABLE();
	}
#pragma endregion

#pragma region "simd_mask casts"
	/** Concatenates elements of \a values into a single SIMD mask. */
	template<typename T, typename... Abis>
	[[nodiscard]] DPM_FORCEINLINE auto concat(const simd_mask<T, Abis> &...values) noexcept requires ((detail::x86_simd_abi_any<Abis, T> && ...))
	{
		if constexpr (sizeof...(values) == 1)
			return (values, ...);
		else
		{
			simd_mask<T, simd_abi::deduce_t<T, (simd_size_v<T, Abis> + ...)>> result = {};
			detail::concat_impl<0>(result, values...);
			return result;
		}
	}
	/** Concatenates elements of \a values into a single SIMD mask. */
	template<typename T, std::size_t N, std::size_t A, std::size_t M>
	[[nodiscard]] DPM_FORCEINLINE auto concat(const std::array<detail::x86_mask<T, N, A>, M> &values) noexcept requires detail::x86_overload_any<T, N, A> && detail::x86_overload_any<T, M, A>
	{
		using result_t = detail::x86_mask<T, N * M, A>;
		if constexpr (M == 1)
			return values[0];
		else
		{
			result_t result = {};
			for (std::size_t i = 0; i < M; ++i)
			{
				auto *data = reinterpret_cast<T *>(ext::to_native_data(result).data());
				if ((i * N) % sizeof(ext::native_data_type_t<result_t>) != 0)
					values[i].copy_to(data + i * N, element_aligned);
				else
					values[i].copy_to(data + i * N, vector_aligned);
			}
			return result;
		}
	}

	/** Returns an array of SIMD masks where every `i`th element of the `j`th mask a copy of the `i + j * V::size()`th element from \a x.
	 * @note Size of \a x must be a multiple of `V::size()`. */
	template<typename V, std::size_t N, std::size_t A, typename U = typename V::simd_type::value_type>
	[[nodiscard]] DPM_FORCEINLINE auto split(const simd_mask<U, detail::avec<N, A>> &x) noexcept requires detail::can_split_mask<V, detail::avec<N, A>> && detail::x86_overload_any<U, N, A>
	{
		std::array<V, simd_size_v<U, detail::avec<N, A>> / V::size()> result = {};
		for (std::size_t j = 0; j < result.size(); ++j)
		{
			const auto *data = reinterpret_cast<const U *>(ext::to_native_data(x).data());
			result[j].copy_from(data + j * V::size(), element_aligned);
		}
		return result;
	}
	/** Returns an array of SIMD masks where every `i`th element of the `j`th mask is a copy of the `i + j * (simd_size_v<T, Abi> / N)`th element from \a x. */
	template<std::size_t N, typename T, std::size_t M, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE auto split_by(const detail::x86_mask<T, M, A> &x) noexcept requires (M % N == 0 && detail::x86_overload_any<T, M, A>)
	{
		return split<resize_simd_t<M / N, detail::x86_mask<T, M, A>>>(x);
	}
#pragma endregion

	template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_any<T, N, Align>
	class simd<T, detail::avec<N, Align>>
	{
		friend struct detail::native_access<simd>;

		using native_type = ext::native_data_type_t<simd>;

		constexpr static auto alignment = std::max<std::size_t>(Align, sizeof(native_type));
		constexpr static auto native_extent = sizeof(native_type) / sizeof(T);
		constexpr static auto data_size = ext::native_data_size_v<simd>;

		using storage_type = std::array<native_type, data_size>;
		using value_alias = detail::alias_t<T>;

	public:
		using value_type = T;
		using reference = value_alias &;

		using abi_type = detail::avec<N, Align>;
		using mask_type = simd_mask<T, abi_type>;

		/** Returns width of the SIMD type. */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	public:
		constexpr simd() noexcept = default;
		constexpr simd(const simd &) noexcept = default;
		constexpr simd &operator=(const simd &) noexcept = default;
		constexpr simd(simd &&) noexcept = default;
		constexpr simd &operator=(simd &&) noexcept = default;

		/** Initializes the SIMD vector with a native vector. */
		constexpr simd(native_type native) noexcept requires (data_size == 1) { m_data[0] = native; }
		/** Initializes the SIMD vector with an array of native vectors. */
		constexpr simd(const native_type (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

		/** Initializes the underlying elements with \a value. */
		template<detail::compatible_element<value_type> U>
		DPM_FORCEINLINE simd(U &&value) noexcept { m_data.fill(detail::fill<native_type>(static_cast<T>(value))); }
		/** Initializes the underlying elements with values provided by the generator \a gen. */
		template<detail::element_generator<value_type, size()> G>
		DPM_FORCEINLINE simd(G &&gen) noexcept
		{
			const auto invoke_gen = [&]<std::size_t J>(std::integral_constant<std::size_t, J> i) -> T
			{
				if constexpr (J < size())
					return std::invoke(gen, i);
				else
					return T{};
			};
			const auto gen_proxy = [&]<std::size_t J, std::size_t... Is>(std::integral_constant<std::size_t, J>, std::index_sequence<Is...>)
			{
				return detail::set<native_type>(invoke_gen(std::integral_constant<std::size_t, J + Is>())...);
			};
			detail::generate_n<data_size>(m_data, [&]<std::size_t I>(std::integral_constant<std::size_t, I>)
			{
				constexpr auto base_pos = I * native_extent;
				return gen_proxy(std::integral_constant<std::size_t, base_pos>{}, std::make_index_sequence<native_extent>{});
			});
		}

		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		DPM_FORCEINLINE simd(const simd<U, detail::avec<size(), OtherAlign>> &other) noexcept
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
		DPM_FORCEINLINE simd(const U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying elements from \a mem. */
		template<typename U, typename Flags>
		DPM_FORCEINLINE void copy_from(const U *mem, Flags) noexcept requires std::convertible_to<U, T> && is_simd_flag_type_v<Flags>
		{
			std::size_t i = 0;
			if constexpr (detail::aligned_tag<Flags, alignof(native_type)> && sizeof(U) == sizeof(value_type))
				for (; i + native_extent < size(); i += native_extent)
				{
					auto &dst = m_data[i / native_extent];
					detail::cast_copy<T, U>(dst, mem + i);
				}
			for (; i < size(); ++i) operator[](i) = static_cast<T>(mem[i]);
		}
		/** Copies the underlying elements to \a mem. */
		template<typename U, typename Flags>
		DPM_FORCEINLINE void copy_to(U *mem, Flags) const noexcept requires std::convertible_to<T, U> && is_simd_flag_type_v<Flags>
		{
			std::size_t i = 0;
			if constexpr (detail::aligned_tag<Flags, alignof(native_type)> && sizeof(U) == sizeof(value_type))
				for (; i + native_extent < size(); i += native_extent)
				{
					const auto &src = m_data[i / native_extent];
					detail::cast_copy<U, T>(mem + i, src);
				}
			for (; i < size(); ++i) mem[i] = static_cast<U>(operator[](i));
		}

		[[nodiscard]] DPM_FORCEINLINE reference operator[](std::size_t i) noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<value_alias *>(m_data.data())[i];
		}
		[[nodiscard]] DPM_FORCEINLINE value_type operator[](std::size_t i) const noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<const value_alias *>(m_data.data())[i];
		}

	private:
		alignas(alignment) storage_type m_data;
	};

	template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_any<T, N, Align>
	class const_where_expression<detail::x86_mask<T, N, Align>, detail::x86_simd<T, N, Align>>
	{
		template<typename U, typename Abi, typename K>
		friend inline simd<U, Abi> ext::blend(const simd<U, Abi> &, const const_where_expression<K, simd<U, Abi>> &);
		template<typename U, typename Abi, typename K>
		friend inline simd_mask<U, Abi> ext::blend(const simd_mask<U, Abi> &, const const_where_expression<K, simd_mask<U, Abi>> &);

	protected:
		using simd_t = detail::x86_simd<T, N, Align>;
		using mask_t = detail::x86_mask<T, N, Align>;
		using native_type = ext::native_data_type_t<simd_t>;
		using alias_type = detail::alias_t<T>;

	public:
		const_where_expression(const const_where_expression &) = delete;
		const_where_expression &operator=(const const_where_expression &) = delete;

		const_where_expression(mask_t mask, simd_t &data) noexcept : m_mask(mask), m_data(data) {}
		const_where_expression(mask_t mask, const simd_t &data) noexcept : m_mask(mask), m_data(const_cast<simd_t &>(data)) {}

		[[nodiscard]] DPM_FORCEINLINE T operator+() const && noexcept { return ext::blend(m_data, +m_data, m_mask); }
		[[nodiscard]] DPM_FORCEINLINE T operator-() const && noexcept { return ext::blend(m_data, -m_data, m_mask); }
		[[nodiscard]] DPM_FORCEINLINE T operator~() const && noexcept requires std::integral<T> { return ext::blend(m_data, ~m_data, m_mask); }

		/** Copies selected elements to \a mem. */
		template<typename U, typename Flags>
		DPM_FORCEINLINE void copy_to(U *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
			constexpr auto native_extent = sizeof(native_type) / sizeof(T);
			for (std::size_t i = 0; i < mask_t::size(); i += native_extent)
			{
				const auto &mask = ext::to_native_data(m_mask)[i / native_extent];
#ifdef DPM_HAS_SSE2
				/* If we have masked move intrinsics, try to convert the vector & use masked move. */
				if constexpr (sizeof(U) == sizeof(T))
				{
					typename detail::select_vector<U, sizeof(native_type)>::type tmp;
					detail::cast_copy<U, T>(tmp, ext::to_native_data(m_data)[i / native_extent]);

					if constexpr (!detail::aligned_tag<Flags, alignof(native_type)>)
						detail::maskstoreu<native_type>(mem + i, tmp, mask);
					else
						detail::maskstore<native_type>(mem + i, tmp, mask);
					continue;
				}
#endif
				const auto mask_bits = detail::movemask<T>(mask);
				if (!mask_bits) [[unlikely]] continue;

				for (std::size_t j = 0; i + j < mask_t::size() && j < native_extent; ++j)
				{
					const auto j_bit = 1 << (j * detail::movemask_bits_v<T>);
					if (mask_bits & j_bit) mem[i + j] = static_cast<U>(m_data[i + j]);
				}
			}
		}

	protected:
		mask_t m_mask;
		simd_t &m_data;
	};
	template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_any<T, N, Align>
	class where_expression<detail::x86_mask<T, N, Align>, detail::x86_simd<T, N, Align>> : public const_where_expression<detail::x86_mask<T, N, Align>, detail::x86_simd<T, N, Align>>
	{
		using base_expr = const_where_expression<detail::x86_mask<T, N, Align>, detail::x86_simd<T, N, Align>>;
		using native_type = typename base_expr::native_type;
		using alias_type = typename base_expr::alias_type;
		using mask_t = typename base_expr::mask_t;
		using simd_t = typename base_expr::simd_t;

		using base_expr::m_mask;
		using base_expr::m_data;

	public:
		using base_expr::const_where_expression;

		template<typename U>
		DPM_FORCEINLINE void operator=(U &&value) && noexcept requires std::is_convertible_v<U, T>
		{
			m_data = ext::blend(m_data, mask_t{std::forward<U>(value)}, m_mask);
		}

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
		DPM_FORCEINLINE void operator+=(U &&value) && noexcept requires requires(T &a, U b) {{ a += b } -> std::convertible_to<T>; }
		{
			const auto new_data = m_data + simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		DPM_FORCEINLINE void operator-=(U &&value) && noexcept requires requires(T &a, U b) {{ a -= b } -> std::convertible_to<T>; }
		{
			const auto new_data = m_data - simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}

		template<typename U>
		DPM_FORCEINLINE void operator*=(U &&value) && noexcept requires requires(T &a, U b) {{ a *= b } -> std::convertible_to<T>; }
		{
			const auto new_data = m_data * simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		DPM_FORCEINLINE void operator/=(U &&value) && noexcept requires requires(T &a, U b) {{ a /= b } -> std::convertible_to<T>; }
		{
			const auto new_data = m_data / simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		DPM_FORCEINLINE void operator%=(U &&value) && noexcept requires requires(T &a, U b) {{ a %= b } -> std::convertible_to<T>; }
		{
			const auto new_data = m_data % simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}

		template<typename U>
		DPM_FORCEINLINE void operator&=(U &&value) && noexcept requires requires(T &a, U b) {{ a &= b } -> std::convertible_to<T>; }
		{
			const auto new_data = m_data & simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		DPM_FORCEINLINE void operator|=(U &&value) && noexcept requires requires(T &a, U b) {{ a |= b } -> std::convertible_to<T>; }
		{
			const auto new_data = m_data | simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		DPM_FORCEINLINE void operator^=(U &&value) && noexcept requires requires(T &a, U b) {{ a ^= b } -> std::convertible_to<T>; }
		{
			const auto new_data = m_data ^ simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		DPM_FORCEINLINE void operator<<=(U &&value) && noexcept requires requires(T &a, U b) {{ a >>= b } -> std::convertible_to<T>; }
		{
			const auto new_data = m_data << mask_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		DPM_FORCEINLINE void operator>>=(U &&value) && noexcept requires requires(T &a, U b) {{ a <<= b } -> std::convertible_to<T>; }
		{
			const auto new_data = m_data >> mask_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}

		/** Copies selected elements from \a mem. */
		template<typename U, typename Flags>
		DPM_FORCEINLINE void copy_from(const U *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
			constexpr auto native_extent = sizeof(native_type) / sizeof(T);
			for (std::size_t i = 0; i < mask_t::size(); i += native_extent)
			{
				const auto &mask = ext::to_native_data(m_mask)[i / native_extent];
				native_type new_data;
#ifdef DPM_HAS_AVX2
				/* If we have masked load intrinsics, try to use masked load & convert. */
				if constexpr (detail::aligned_tag<Flags, alignof(native_type)> && sizeof(U) == sizeof(T) && sizeof(T) >= 4)
				{
					using src_vector = typename detail::select_vector<U, sizeof(native_type)>::type;
					detail::cast_copy<T, U>(new_data, detail::maskload<src_vector>(mem + i, mask));
				}
				else
#endif
				{
					const auto mask_bits = detail::movemask<T>(mask);
					if (!mask_bits) [[unlikely]] continue;

					alignas(native_type) T values[native_extent] = {};
					for (std::size_t j = 0; j < native_extent && i + j < mask_t::size(); ++j)
					{
						const auto j_bit = 1ull << (j * detail::movemask_bits_v<T>);
						if (mask_bits & j_bit) values[j] = static_cast<T>(mem[i + j]);
					}
					new_data = detail::set<native_type>(values);
				}
				auto &data = ext::to_native_data(m_data)[i / native_extent];
				data = detail::blendv<T>(data, new_data, mask);
			}
		}
	};

	namespace detail
	{
		template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
		struct native_access<x86_simd<T, N, A>>
		{
			using simd_t = x86_simd<T, N, A>;
			static std::span<ext::native_data_type_t<simd_t>, simd_t::data_size> to_native_data(simd_t &x) noexcept { return {x.m_data}; }
			static std::span<const ext::native_data_type_t<simd_t>, simd_t::data_size> to_native_data(const simd_t &x) noexcept { return {x.m_data}; }
		};

		template<typename To, typename ToAbi, typename From, typename FromAbi>
		DPM_FORCEINLINE void cast_impl(simd<To, ToAbi> &to, const simd<From, FromAbi> &from) noexcept
		{
			const auto from_data = reinterpret_cast<const From *>(ext::to_native_data(from).data());
			constexpr auto from_align = alignof(decltype(from));
			constexpr auto to_align = alignof(decltype(to));

			if constexpr (to_align > from_align)
				to.copy_from(from_data, overaligned<to_align>);
			else if constexpr (to_align < from_align)
				to.copy_from(from_data, element_aligned);
			else
				to.copy_from(from_data, vector_aligned);
		}
		template<std::size_t I, typename T, typename OutAbi, typename XAbi, typename... Abis>
		DPM_FORCEINLINE void concat_impl(simd<T, OutAbi> &out, const simd<T, XAbi> &x, const simd<T, Abis> &...rest) noexcept
		{
			auto *data = reinterpret_cast<T *>(ext::to_native_data(out).data());
			if constexpr (I % sizeof(ext::native_data_type_t<simd<T, OutAbi>>) != 0)
				x.copy_to(data + I, element_aligned);
			else
				x.copy_to(data + I, vector_aligned);

			if constexpr (sizeof...(Abis) != 0) concat_impl<I + simd<T, XAbi>::size()>(out, rest...);
		}
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a x. */
		template<typename T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE auto to_native_data(detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
		{
			return detail::native_access<detail::x86_simd<T, N, A>>::to_native_data(x);
		}
		/** Returns a constant span of the underlying SSE vectors for \a x. */
		template<typename T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE auto to_native_data(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
		{
			return detail::native_access<detail::x86_simd<T, N, A>>::to_native_data(x);
		}

		/** Replaces elements of vector \a a with elements of vector \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<typename T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> blend(
				const detail::x86_simd<T, N, A> &a,
				const detail::x86_simd<T, N, A> &b,
				const detail::x86_mask<T, N, A> &m)
		noexcept requires detail::x86_overload_any<T, N, A>
		{
			detail::x86_simd<T, N, A> result = {};
			auto result_data = to_native_data(result);
			const auto a_data = to_native_data(a);
			const auto b_data = to_native_data(b);
			const auto m_data = to_native_data(m);

			for (std::size_t i = 0; i < native_data_size_v<detail::x86_simd<T, N, A>>; ++i)
				result_data[i] = detail::blendv<T>(a_data[i], b_data[i], m_data[i]);
			return result;
		}

		/** Shuffles elements of vector \a x into a new vector according to the specified indices. */
		template<std::size_t... Is, typename T, std::size_t N, std::size_t A, std::size_t M = sizeof...(Is)>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, M, A> shuffle(const detail::x86_simd<T, N, A> &x) noexcept
		requires (DPM_NOT_SSSE3(sizeof(T) >= 4 &&)
				detail::x86_overload_any<T, N, A> &&
				detail::x86_overload_any<T, M, A>)
		{
			detail::x86_simd<T, M, A> result = {};
			auto result_data = to_native_data(result).data();
			const auto x_data = to_native_data(x).data();

			detail::shuffle<T, 0>(std::index_sequence<Is...>{}, result_data, x_data);
			return result;
		}
	}

#pragma region "simd operators"
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator+(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return x;
	}
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator-(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::negate<T>(x); }, result, x);
		return result;
	}

	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator++(detail::x86_simd<T, N, A> &x, int) noexcept requires detail::x86_overload_any<T, N, A>
	{
		auto tmp = x;
		operator++(x);
		return tmp;
	}
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator--(detail::x86_simd<T, N, A> &x, int) noexcept requires detail::x86_overload_any<T, N, A>
	{
		auto tmp = x;
		operator--(x);
		return tmp;
	}
	template<typename T, std::size_t N, std::size_t A>
	DPM_FORCEINLINE detail::x86_simd<T, N, A> &operator++(detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::inc<T>(b); }, x, x);
		return x;
	}
	template<typename T, std::size_t N, std::size_t A>
	DPM_FORCEINLINE detail::x86_simd<T, N, A> &operator--(detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::dec<T>(b); }, x, x);
		return x;
	}

	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator+(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::add<T>(a, b); }, result, a, b);
		return result;
	}
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator-(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::sub<T>(a, b); }, result, a, b);
		return result;
	}
	template<typename T, std::size_t N, std::size_t A>
	DPM_FORCEINLINE detail::x86_simd<T, N, A> &operator+=(detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::add<T>(a, b); }, a, b);
		return a;
	}
	template<typename T, std::size_t N, std::size_t A>
	DPM_FORCEINLINE detail::x86_simd<T, N, A> &operator-=(detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::sub<T>(a, b); }, a, b);
		return a;
	}

	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator*(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::mul<T>(a, b); }, result, a, b);
		return result;
	}
	template<std::floating_point  T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator/(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::div<T>(a, b); }, result, a, b);
		return result;
	}
	template<std::floating_point T, std::size_t N, std::size_t A>
	DPM_FORCEINLINE detail::x86_simd<T, N, A> &operator*=(detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::mul<T>(a, b); }, a, b);
		return a;
	}
	template<std::floating_point T, std::size_t N, std::size_t A>
	DPM_FORCEINLINE detail::x86_simd<T, N, A> &operator/=(detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::div<T>(a, b); }, a, b);
		return a;
	}

	template<detail::integral_of_size<2> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator*(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::mul<T>(a, b); }, result, a, b);
		return result;
	}
	template<detail::integral_of_size<2> T, std::size_t N, std::size_t A>
	DPM_FORCEINLINE detail::x86_simd<T, N, A> &operator*=(detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::mul<T>(a, b); }, a, b);
		return a;
	}

#ifdef DPM_HAS_SSE4_1
	template<detail::integral_of_size<4> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator*(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::mul<T>(a, b); }, result, a, b);
		return result;
	}
	template<detail::integral_of_size<4> T, std::size_t N, std::size_t A>
	DPM_FORCEINLINE detail::x86_simd<T, N, A> &operator*=(detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::mul<T>(a, b); }, a, b);
		return a;
	}
#endif

	template<std::integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator&(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_and(a, b); }, result, a, b);
		return result;
	}
	template<std::integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator^(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_xor(a, b); }, result, a, b);
		return result;
	}
	template<std::integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator|(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_or(a, b); }, result, a, b);
		return result;
	}
	template<std::integral T, std::size_t N, std::size_t A>
	DPM_FORCEINLINE detail::x86_simd<T, N, A> &operator&=(detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::bit_and(a, b); }, a, b);
		return a;
	}
	template<std::integral T, std::size_t N, std::size_t A>
	DPM_FORCEINLINE detail::x86_simd<T, N, A> &operator^=(detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::bit_xor(a, b); }, a, b);
		return a;
	}
	template<std::integral T, std::size_t N, std::size_t A>
	DPM_FORCEINLINE detail::x86_simd<T, N, A> &operator|=(detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::bit_or(a, b); }, a, b);
		return a;
	}

	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator<<(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_shiftl<T>(a, b); }, result, a, b);
		return result;
	}
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator>>(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_shiftr<T>(a, b); }, result, a, b);
		return result;
	}
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A>
	DPM_FORCEINLINE detail::x86_simd<T, N, A> &operator<<=(detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::bit_shiftl<T>(a, b); }, a, b);
		return a;
	}
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A>
	DPM_FORCEINLINE detail::x86_simd<T, N, A> &operator>>=(detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::bit_shiftr<T>(a, b); }, a, b);
		return a;
	}

#ifdef DPM_HAS_AVX2
	template<detail::integral_of_size<4> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator<<(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_shiftl<T>(a, b); }, result, a, b);
		return result;
	}
	template<detail::integral_of_size<4> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator>>(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_shiftr<T>(a, b); }, result, a, b);
		return result;
	}
	template<detail::integral_of_size<4> T, std::size_t N, std::size_t A>
	DPM_FORCEINLINE detail::x86_simd<T, N, A> &operator<<=(detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::bit_shiftl<T>(a, b); }, a, b);
		return a;
	}
	template<detail::integral_of_size<4> T, std::size_t N, std::size_t A>
	DPM_FORCEINLINE detail::x86_simd<T, N, A> &operator>>=(detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::bit_shiftr<T>(a, b); }, a, b);
		return a;
	}
#endif

#if defined(DPM_HAS_AVX512DQ) && defined(DPM_HAS_AVX512LV)
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator*(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::mul<T>(a, b); }, result, a, b);
		return result;
	}
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A>
	DPM_FORCEINLINE detail::x86_simd<T, N, A> &operator*=(detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::mul<T>(a, b); }, a, b);
		return a;
	}
#endif

#if defined(DPM_HAS_AVX512BW) && defined(DPM_HAS_AVX512LV)
	template<detail::integral_of_size<2> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator<<(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_shiftl<T>(a, b); }, result, a, b);
		return result;
	}
	template<detail::integral_of_size<2> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> operator>>(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_shiftr<T>(a, b); }, result, a, b);
		return result;
	}
	template<detail::integral_of_size<2> T, std::size_t N, std::size_t A>
	DPM_FORCEINLINE detail::x86_simd<T, N, A> &operator<<=(detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::bit_shiftl<T>(a, b); }, a, b);
		return a;
	}
	template<detail::integral_of_size<2> T, std::size_t N, std::size_t A>
	DPM_FORCEINLINE detail::x86_simd<T, N, A> &operator>>=(detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::vectorize([](auto &a, auto b) { a = detail::bit_shiftr<T>(a, b); }, a, b);
		return a;
	}
#endif

	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator==(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_eq<T>(a, b); }, result, a, b);
		return result;
	}
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator!=(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_ne<T>(a, b); }, result, a, b);
		return result;
	}

	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator>(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_gt<T>(a, b); }, result, a, b);
		return result;
	}
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator<(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_lt<T>(a, b); }, result, a, b);
		return result;
	}
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator>=(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_ge<T>(a, b); }, result, a, b);
		return result;
	}
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator<=(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_le<T>(a, b); }, result, a, b);
		return result;
	}

#ifdef DPM_HAS_SSE4_1
	template<std::signed_integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator>(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_gt<T>(a, b); }, result, a, b);
		return result;
	}
	template<std::signed_integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator<(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return b > a;
	}
#endif

	template<std::signed_integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator>=(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return (a > b) | (a == b);
	}
	template<std::signed_integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator<=(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return (a < b) | (a == b);
	}
#pragma endregion

#pragma region "simd reductions"
	namespace detail
	{
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_add(const V *data) noexcept { return reduce<T, N>(data, setzero<V>(), [](auto a, auto b) { return add<T>(a, b); }); }
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_mul(const V *data) noexcept { return reduce<T, N>(data, fill<V>(T{1}), [](auto a, auto b) { return mul<T>(a, b); }); }
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_min(const V *data) noexcept { return reduce<T, N>(data, fill<V>(std::numeric_limits<T>::max()), [](auto a, auto b) { return min<T>(a, b); }); }
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_max(const V *data) noexcept { return reduce<T, N>(data, fill<V>(std::numeric_limits<T>::min()), [](auto a, auto b) { return max<T>(a, b); }); }

#ifdef DPM_HAS_SSE2
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_and(const V *data) noexcept { return reduce<T, N>(data, setones<V>(), [](auto a, auto b) { return bit_and(a, b); }); }
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_xor(const V *data) noexcept { return reduce<T, N>(data, setzero<V>(), [](auto a, auto b) { return bit_xor(a, b); }); }
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_or(const V *data) noexcept { return reduce<T, N>(data, setzero<V>(), [](auto a, auto b) { return bit_or(a, b); }); }
#endif

		/* On 256-bit integral reductions without AVX2, reduce via 128-bit instead. */
#ifndef DPM_HAS_AVX2
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_add(const V *data) noexcept requires (sizeof(V) == 32) { return reduce_add<T, N>(reinterpret_cast<const select_vector_t<T, 16> *>(data)); }
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_mul(const V *data) noexcept requires (sizeof(V) == 32) { return reduce_mul<T, N>(reinterpret_cast<const select_vector_t<T, 16> *>(data)); }
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_min(const V *data) noexcept requires (sizeof(V) == 32) { return reduce_min<T, N>(reinterpret_cast<const select_vector_t<T, 16> *>(data)); }
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_max(const V *data) noexcept requires (sizeof(V) == 32) { return reduce_max<T, N>(reinterpret_cast<const select_vector_t<T, 16> *>(data)); }
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_and(const V *data) noexcept requires (sizeof(V) == 32) { return reduce_and<T, N>(reinterpret_cast<const select_vector_t<T, 16> *>(data)); }
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_xor(const V *data) noexcept requires (sizeof(V) == 32) { return reduce_xor<T, N>(reinterpret_cast<const select_vector_t<T, 16> *>(data)); }
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_or(const V *data) noexcept requires (sizeof(V) == 32) { return reduce_or<T, N>(reinterpret_cast<const select_vector_t<T, 16> *>(data)); }
#endif
	}

	/** Horizontally reduced elements of \a x using operation `Op`. */
	template<typename T, std::size_t N, std::size_t A, typename Op = std::plus<>>
	[[nodiscard]] DPM_FORCEINLINE T reduce(const detail::x86_simd<T, N, A> &x, Op op = {}) noexcept requires (detail::x86_overload_any<T, N, A> && sizeof(T) >= 4)
	{
		constexpr auto size = detail::x86_simd<T, N, A>::size();
		const auto x_data = ext::to_native_data(x).data();

		if constexpr (detail::template_instance<Op, std::plus>)
			return detail::reduce_add<T, size>(x_data);
		if constexpr (detail::template_instance<Op, std::multiplies>)
		{
			if constexpr (std::floating_point<T>)
				return detail::reduce_mul<T, size>(x_data);
#ifdef DPM_HAS_SSE4_1
			if (detail::integral_of_size<T, 4>)
				return detail::reduce_mul<T, size>(x_data);
#endif
#if defined(DPM_HAS_AVX512DQ) && defined(DPM_HAS_AVX512LV)
			if (detail::integral_of_size<T, 8>)
				return detail::reduce_mul<T, size>(x_data);
#endif
		}
		if constexpr (std::integral<T>)
		{
			if constexpr (detail::template_instance<Op, std::bit_and>)
				return detail::reduce_and<T, size>(x_data);
			if constexpr (detail::template_instance<Op, std::bit_xor>)
				return detail::reduce_xor<T, size>(x_data);
			if constexpr (detail::template_instance<Op, std::bit_or>)
				return detail::reduce_or<T, size>(x_data);
		}
		return detail::reduce_impl<size>(x, op);
	}
#ifdef DPM_HAS_SSSE3
	/** @copydoc reduce */
	template<std::integral T, std::size_t N, std::size_t A, typename Op = std::plus<>>
	[[nodiscard]] DPM_FORCEINLINE T reduce(const detail::x86_simd<T, N, A> &x, Op op = {}) noexcept requires (detail::x86_overload_any<T, N, A> && sizeof(T) <= 2)
	{
		constexpr auto size = detail::x86_simd<T, N, A>::size();
		const auto x_data = ext::to_native_data(x).data();

		if constexpr (detail::template_instance<Op, std::plus>)
			return detail::reduce_add<T, size>(x_data);
		else if constexpr (detail::template_instance<Op, std::multiplies> && sizeof(T) == 2)
			return detail::reduce_mul<T, size>(x_data);
		else if constexpr (detail::template_instance<Op, std::bit_and>)
			return detail::reduce_and<T, size>(x_data);
		else if constexpr (detail::template_instance<Op, std::bit_xor>)
			return detail::reduce_xor<T, size>(x_data);
		else if constexpr (detail::template_instance<Op, std::bit_or>)
			return detail::reduce_or<T, size>(x_data);
		else
			return detail::reduce_impl<size>(x, op);
	}
#endif

	/** Calculates horizontal minimum of elements of \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE T hmin(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return detail::reduce_min<T, N>(ext::to_native_data(x).data());
	}
	/** Calculates horizontal maximum of elements of \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE T hmax(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return detail::reduce_max<T, N>(ext::to_native_data(x).data());
	}
#if defined(DPM_HAS_SSE4_1)
	/** @copydoc hmin */
	template<std::integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE T hmin(const detail::x86_simd<T, N, A> &x) noexcept requires (detail::x86_overload_any<T, N, A> && sizeof(T) < 8)
	{
		return detail::reduce_min<T, N>(ext::to_native_data(x).data());
	}
	/** @copydoc hmax */
	template<std::integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE T hmax(const detail::x86_simd<T, N, A> &x) noexcept requires (detail::x86_overload_any<T, N, A> && sizeof(T) < 8)
	{
		return detail::reduce_max<T, N>(ext::to_native_data(x).data());
	}
#elif defined(DPM_HAS_SSSE3)
	/** @copydoc hmin */
	template<detail::unsigned_integral_of_size<1> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE T hmin(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return detail::reduce_min<T, N>(ext::to_native_data(x).data());
	}
	/** @copydoc hmin */
	template<detail::unsigned_integral_of_size<1> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE T hmax(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return detail::reduce_max<T, N>(ext::to_native_data(x).data());
	}
	/** @copydoc hmin */
	template<detail::signed_integral_of_size<2> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE T hmin(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return detail::reduce_min<T, N>(ext::to_native_data(x).data());
	}
	/** @copydoc hmax */
	template<detail::signed_integral_of_size<2> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE T hmax(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return detail::reduce_max<T, N>(ext::to_native_data(x).data());
	}
#endif
#if defined(DPM_HAS_AVX512F) && defined(DPM_HAS_AVX512LV)
	/** @copydoc hmin */
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE T hmin(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return detail::reduce_min<T, N>(ext::to_native_data(x).data());
	}
	/** @copydoc hmax */
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE T hmax(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return detail::reduce_max<T, N>(ext::to_native_data(x).data());
	}
#endif
#pragma endregion

#pragma region "simd algorithms"
	/** Returns an SIMD vector of minimum elements of \a a and \a b. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> min(
			const detail::x86_simd<T, N, A> &a,
			const detail::x86_simd<T, N, A> &b)
	noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::min<T>(a, b); }, result, a, b);
		return result;
	}
	/** Returns an SIMD vector of maximum elements of \a a and \a b. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> max(
			const detail::x86_simd<T, N, A> &a,
			const detail::x86_simd<T, N, A> &b)
	noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::max<T>(a, b); }, result, a, b);
		return result;
	}

	/** Returns a pair of SIMD vectors of minimum and maximum elements of \a a and \a b. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::pair<detail::x86_simd<T, N, A>, detail::x86_simd<T, N, A>> minmax(
			const detail::x86_simd<T, N, A> &a,
			const detail::x86_simd<T, N, A> &b)
	noexcept requires detail::x86_overload_any<T, N, A>
	{
		return {min(a, b), max(a, b)};
	}
	/** Clamps elements \f \a x between corresponding elements of \a ming and \a max. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> clamp(
			const detail::x86_simd<T, N, A> &x,
			const detail::x86_simd<T, N, A> &min,
			const detail::x86_simd<T, N, A> &max)
	noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x, auto min, auto max) { res = detail::min<T>(detail::max<T>(x, min), max); }, result, x, min, max);
		return result;
	}

#if defined(DPM_HAS_SSE4_1)
	/** @copydoc min */
	template<std::integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> min(
			const detail::x86_simd<T, N, A> &a,
			const detail::x86_simd<T, N, A> &b)
	noexcept requires (detail::x86_overload_any<T, N, A> && sizeof(T) < 8)
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::min<T>(a, b); }, result, a, b);
		return result;
	}
	/** @copydoc max */
	template<std::integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> max(
			const detail::x86_simd<T, N, A> &a,
			const detail::x86_simd<T, N, A> &b)
	noexcept requires (detail::x86_overload_any<T, N, A> && sizeof(T) < 8)
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::min<T>(a, b); }, result, a, b);
		return result;
	}
	/** @copydoc minmax */
	template<std::integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::pair<detail::x86_simd<T, N, A>, detail::x86_simd<T, N, A>> minmax(
			const detail::x86_simd<T, N, A> &a,
			const detail::x86_simd<T, N, A> &b)
	noexcept requires (detail::x86_overload_any<T, N, A> && sizeof(T) < 8)
	{
		return {min(a, b), max(a, b)};
	}
	/** @copydoc clamp */
	template<std::integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> clamp(
			const detail::x86_simd<T, N, A> &x,
			const detail::x86_simd<T, N, A> &min,
			const detail::x86_simd<T, N, A> &max)
	noexcept requires (detail::x86_overload_any<T, N, A> && sizeof(T) < 8)
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x, auto min, auto max) { res = detail::min<T>(detail::max<T>(x, min), max); }, result, x, min, max);
		return result;
	}
#elif defined(DPM_HAS_SSSE3)
	/** @copydoc min */
	template<std::integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> min(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b)
	noexcept requires (detail::x86_overload_any<T, N, A> && (detail::unsigned_integral_of_size<T, 1> || detail::signed_integral_of_size<T, 2>))
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::min<T>(a, b); }, result, a, b);
		return result;
	}
	/** @copydoc max */
	template<std::integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> max(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b)
	noexcept requires (detail::x86_overload_any<T, N, A> && (detail::unsigned_integral_of_size<T, 1> || detail::signed_integral_of_size<T, 2>))
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::max<T>(a, b); }, result, a, b);
		return result;
	}
	/** @copydoc minmax */
	template<std::integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::pair<detail::x86_simd<T, N, A>, detail::x86_simd<T, N, A>> minmax(
			const detail::x86_simd<T, N, A> &a,
			const detail::x86_simd<T, N, A> &b)
	noexcept requires (detail::x86_overload_any<T, N, A> && (detail::unsigned_integral_of_size<T, 1> || detail::signed_integral_of_size<T, 2>))
	{
		return {min(a, b), max(a, b)};
	}
	/** @copydoc clamp */
	template<std::integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> clamp(
			const detail::x86_simd<T, N, A> &x,
			const detail::x86_simd<T, N, A> &min,
			const detail::x86_simd<T, N, A> &max)
	noexcept requires (detail::x86_overload_any<T, N, A> && (detail::unsigned_integral_of_size<T, 1> || detail::signed_integral_of_size<T, 2>))
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x, auto min, auto max) { res = detail::min<T>(detail::max<T>(x, min), max); }, result, x, min, max);
		return result;
	}
#endif
#if defined(DPM_HAS_AVX512F) && defined(DPM_HAS_AVX512LV)
	/** @copydoc min */
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> min(
			const detail::x86_simd<T, N, A> &a,
			const detail::x86_simd<T, N, A> &b)
	noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::min<T>(a, b); }, result, a, b);
		return result;
	}
	/** @copydoc max */
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> max(
			const detail::x86_simd<T, N, A> &a,
			const detail::x86_simd<T, N, A> &b)
	noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::max<T>(a, b); }, result, a, b);
		return result;
	}
	/** @copydoc minmax */
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::pair<detail::x86_simd<T, N, A>, detail::x86_simd<T, N, A>> minmax(
			const detail::x86_simd<T, N, A> &a,
			const detail::x86_simd<T, N, A> &b)
	noexcept requires detail::x86_overload_any<T, N, A>
	{
		return {min(a, b), max(a, b)};
	}
	/** @copydoc clamp */
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> clamp(
			const detail::x86_simd<T, N, A> &x,
			const detail::x86_simd<T, N, A> &min,
			const detail::x86_simd<T, N, A> &max)
	noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x, auto min, auto max) { res = detail::min<T>(detail::max<T>(x, min), max); }, result, x, min, max);
		return result;
	}
#endif
#pragma endregion

#pragma region "simd casts"
	/** Implicitly converts elements of SIMD vector \a x to the `To` type, where `To` is either `typename T::value_type` or `T` if `T` is a scalar. */
	template<typename T, typename U, std::size_t N, std::size_t A, typename To = typename detail::deduce_cast<T>::type>
	[[nodiscard]] DPM_FORCEINLINE auto simd_cast(const simd<U, detail::avec<N, A>> &x) noexcept
	requires (detail::valid_simd_cast<T, U, detail::avec<N, A>> &&
	          detail::x86_overload_any<To, N, A> &&
	          detail::x86_overload_any<U, N, A>)
	{
		detail::cast_return_t<T, U, detail::avec<N, A>, simd<U, detail::avec<N, A>>::size()> result = {};
		detail::cast_impl(result, x);
		return result;
	}
	/** Explicitly converts elements of SIMD vector \a x to the `To` type, where `To` is either `typename T::value_type` or `T` if `T` is a scalar. */
	template<typename T, typename U, std::size_t N, std::size_t A, typename To = typename detail::deduce_cast<T>::type>
	[[nodiscard]] DPM_FORCEINLINE auto static_simd_cast(const simd<U, detail::avec<N, A>> &x) noexcept
	requires (detail::valid_simd_cast<T, U, detail::avec<N, A>> &&
	          detail::x86_overload_any<To, N, A> &&
	          detail::x86_overload_any<U, N, A>)
	{
		detail::static_cast_return_t<T, U, detail::avec<N, A>, simd<U, detail::avec<N, A>>::size()> result = {};
		detail::cast_impl(result, x);
		return result;
	}

	/** Concatenates elements of \a values into a single SIMD vector. */
	template<typename T, typename... Abis>
	[[nodiscard]] DPM_FORCEINLINE auto concat(const simd<T, Abis> &...values) noexcept requires ((detail::x86_simd_abi_any<Abis, T> && ...))
	{
		if constexpr (sizeof...(values) == 1)
			return (values, ...);
		else
		{
			simd<T, simd_abi::deduce_t<T, (simd_size_v<T, Abis> + ...)>> result = {};
			detail::concat_impl<0>(result, values...);
			return result;
		}
	}
	/** Concatenates elements of \a values into a single SIMD mask. */
	template<typename T, std::size_t N, std::size_t A, std::size_t M>
	[[nodiscard]] DPM_FORCEINLINE auto concat(const std::array<detail::x86_simd<T, N, A>, M> &values) noexcept requires detail::x86_overload_any<T, N, A> && detail::x86_overload_any<T, M, A>
	{
		using result_t = detail::x86_simd<T, N * M, A>;
		if constexpr (M == 1)
			return values[0];
		else
		{
			result_t result = {};
			for (std::size_t i = 0; i < M; ++i)
			{
				auto *data = reinterpret_cast<T *>(ext::to_native_data(result).data());
				if ((i * N) % sizeof(ext::native_data_type_t<result_t>) != 0)
					values[i].copy_to(data + i * N, element_aligned);
				else
					values[i].copy_to(data + i * N, vector_aligned);
			}
			return result;
		}
	}

	/** Returns an array of SIMD vectors where every `i`th element of the `j`th vector is a copy of the `i + j * V::size()`th element from \a x.
	 * @note Size of \a x must be a multiple of `V::size()`. */
	template<typename V, std::size_t N, std::size_t A, typename U = typename V::simd_type::value_type>
	[[nodiscard]] DPM_FORCEINLINE auto split(const simd<U, detail::avec<N, A>> &x) noexcept requires detail::can_split_simd<V, detail::avec<N, A>> && detail::x86_overload_any<U, N, A>
	{
		std::array<V, simd_size_v<U, detail::avec<N, A>> / V::size()> result = {};
		for (std::size_t j = 0; j < result.size(); ++j)
		{
			const auto *data = reinterpret_cast<const U *>(ext::to_native_data(x).data());
			result[j].copy_from(data + j * V::size(), element_aligned);
		}
		return result;
	}
	/** Returns an array of SIMD vectors where every `i`th element of the `j`th vector is a copy of the `i + j * (simd_size_v<T, Abi> / N)`th element from \a x. */
	template<std::size_t N, typename T, std::size_t M, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE auto split_by(const detail::x86_simd<T, M, A> &x) noexcept requires (M % N == 0 && detail::x86_overload_any<T, M, A>)
	{
		return split<resize_simd_t<M / N, detail::x86_simd<T, M, A>>>(x);
	}
#pragma endregion
}

#endif

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif