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
		constexpr DPM_FORCEINLINE void vectorize(F f, V &&v, Vs &&...vs) noexcept
		{
			for (std::size_t i = 0; i < ext::native_data_size_v<std::remove_cvref_t<V>>; ++i)
				f(ext::to_native_data(v)[i], ext::to_native_data(vs)[i]...);
		}
	}

	template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_any<T, N, Align>
	class simd_mask<T, detail::avec<N, Align>>
	{
		friend detail::native_access<simd_mask>;
		template<typename, typename>
		friend class simd_mask;

		using native_type = ext::native_data_type_t<simd_mask>;

		constexpr static auto alignment = std::max<std::size_t>(Align, sizeof(native_type));
		constexpr static auto native_extent = sizeof(native_type) / sizeof(T);
		constexpr static auto data_size = ext::native_data_size_v<simd_mask>;

		using value_alias = typename detail::sized_mask<sizeof(T)>::type;
		using storage_type = std::array<native_type, data_size>;

	public:
		using value_type = bool;
		using abi_type = detail::avec<N, Align>;

		using reference = value_alias &;
		using simd_type = simd<T, abi_type>;
		using mask_type = simd_mask<T, abi_type>;

		/** Returns width of the SIMD mask. */
		static constexpr std::size_t size() noexcept { return abi_type::size(); }

	public:
		constexpr simd_mask() noexcept = default;

		/** Initializes the underlying elements with \a value. */
		constexpr simd_mask(value_type value) noexcept
		{
			std::array<value_alias, native_extent> buff = {};
			std::ranges::fill(buff, static_cast<value_alias>(value));
			std::ranges::fill(_data, detail::native_bit_cast<native_type>(buff));
		}

		/** Initializes the SIMD mask with a native mask vector. */
		constexpr simd_mask(native_type native) noexcept { std::ranges::fill(_data, native); }
		/** Initializes the SIMD mask with an array of native mask vectors. */
		constexpr simd_mask(const native_type (&native)[data_size]) noexcept { std::copy_n(native, data_size, _data); }
		/** Initializes the SIMD mask with a span of native mask vectors. */
		constexpr simd_mask(std::span<const native_type, data_size> native) noexcept { std::copy_n(native.begin(), data_size, _data); }

		/** Initializes the underlying elements from \a mem. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr simd_mask(I mem, Flags) noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<std::iter_value_t<I>, value_type> { copy_from(mem, Flags{}); }
		/** Initializes the underlying elements from \a mem if the corresponding element of mask \a m evaluates to `true`. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr simd_mask(I mem, const mask_type &m, Flags) noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<std::iter_value_t<I>, value_type> { copy_from(mem, m, Flags{}); }

		/** Initializes the underlying elements with values provided by the generator \a gen. */
		template<detail::element_generator<value_type, size()> G>
		constexpr simd_mask(G &&gen) noexcept
		{
			for (std::size_t i = 0, j = 0; i < size(); i += native_extent, ++j)
			{
				std::array<value_alias, native_extent> buff = {};
				for (std::size_t k = 0; k < buff.size() && i + k < size(); ++k)
					buff[k] = detail::extend_bool<value_alias>(std::invoke(gen, i + k));
				_data[j] = detail::native_bit_cast<native_type>(buff);
			}
		}

		template<typename U, typename OtherAbi> requires(simd_size_v<T, OtherAbi> == size())
		constexpr operator simd_mask<U, OtherAbi>() const noexcept
		{
			constexpr auto align = std::max(alignof(simd_mask), alignof(simd_mask<U, OtherAbi>));
			alignas(align) std::array<value_type, size()> buff = {};
			copy_to(buff.data(), overaligned<align>);
			return {buff.data(), overaligned<align>};
		}

		/** Copies the underlying elements from \a mem. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr void copy_from(I mem, Flags) noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<std::iter_value_t<I>, value_type>
		{
			for (std::size_t i = 0; i < size(); ++i)
				_data[i] = static_cast<value_type>(mem[i]);
		}
		/** Copies the underlying elements to \a mem. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr void copy_to(I mem, Flags) const noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<value_type, std::iter_value_t<I>>
		{
			for (std::size_t i = 0; i < size(); ++i)
				mem[i] = static_cast<std::iter_value_t<I>>(_data[i]);
		}

		/** Copies the underlying elements from \a mem if the corresponding element of mask \a m evaluates to `true`. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr void copy_from(I mem, const mask_type &m, Flags) noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<std::iter_value_t<I>, value_type>
		{
			for (std::size_t i = 0; i < size(); ++i)
				if (m[i]) _data[i] = static_cast<value_type>(mem[i]);
		}
		/** Copies the underlying elements to \a mem if the corresponding element of mask \a m evaluates to `true`. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr void copy_to(I mem, const mask_type &m, Flags) const noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<value_type, std::iter_value_t<I>>
		{
			for (std::size_t i = 0; i < size(); ++i)
				if (m[i]) mem[i] = static_cast<std::iter_value_t<I>>(_data[i]);
		}

		[[nodiscard]] constexpr reference operator[](std::size_t i) & noexcept
		{
			return reinterpret_cast<value_alias *>(_data.data())[i];
		}
		[[nodiscard]] constexpr value_type operator[](std::size_t i) const & noexcept
		{
			if (std::is_constant_evaluated())
				return detail::native_bit_cast<std::array<value_alias, native_extent>>(_data[i / native_extent])[i];
			else
				return reinterpret_cast<const value_alias *>(_data.data())[i];
		}

	private:
		alignas(alignment) storage_type _data;
	};

	namespace detail
	{
		template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
		struct native_access<x86_mask<T, N, A>>
		{
			using mask_t = x86_mask<T, N, A>;
			static constexpr std::span<ext::native_data_type_t<mask_t>, mask_t::data_size> to_native_data(mask_t &x) noexcept { return {x._data}; }
			static constexpr std::span<const ext::native_data_type_t<mask_t>, mask_t::data_size> to_native_data(const mask_t &x) noexcept { return {x._data}; }
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

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a x. */
		template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
		[[nodiscard]] constexpr auto to_native_data(detail::x86_mask<T, N, A> &x) noexcept { return detail::native_access<detail::x86_mask<T, N, A>>::to_native_data(x); }
		/** Returns a constant span of the underlying SSE vectors for \a x. */
		template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
		[[nodiscard]] constexpr auto to_native_data(const detail::x86_mask<T, N, A> &x) noexcept { return detail::native_access<detail::x86_mask<T, N, A>>::to_native_data(x); }

		/** Shuffles elements of mask \a x into a new mask according to the specified indices. */
		template<std::size_t... Is, typename T, std::size_t N, std::size_t A, std::size_t M = sizeof...(Is)>
		[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, M, A> shuffle(const detail::x86_mask<T, N, A> &x) noexcept requires(DPM_NOT_SSSE3(sizeof(T) >= 4 &&) detail::x86_overload_any<T, N, A> && detail::x86_overload_any<T, M, A>)
		{
			if (std::is_constant_evaluated())
			{
				const auto packed = static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(x);
				return static_cast<detail::x86_mask<T, N, A>>(shuffle<Is...>(packed));
			}
			else
			{
				detail::x86_mask<T, M, A> result = {};
				auto result_data = to_native_data(result).data();
				const auto x_data = to_native_data(x).data();

				detail::shuffle<T, 0>(std::index_sequence<Is...>{}, result_data, x_data);
				return result;
			}
		}
	}

	/** Replaces elements of mask \a a with elements of mask \a b using mask \a m. Elements of \a a are selected if the corresponding element of \a m evaluates to `true`. */
	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> blend(const detail::x86_mask<T, N, A> &m, const detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_m = static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(m);
			const auto packed_a = static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(blend(packed_m, packed_a, packed_b));
		}
		else
		{
			detail::x86_mask<T, N, A> result = {};
			auto result_data = to_native_data(result);
			const auto m_data = to_native_data(m);
			const auto a_data = to_native_data(a);
			const auto b_data = to_native_data(b);

			const auto zero = detail::setzero<native_data_type_t<detail::x86_mask<T, N, A>>>();
			for (std::size_t i = 0; i < native_data_size_v<detail::x86_mask<T, N, A>>; ++i)
			{
				/* Since mask can be modified via to_native_data, need to make sure the entire mask is filled. */
				const auto inv_mask = detail::mask_eq<T>(m_data[i], zero);
				result_data[i] = detail::blendv<T>(a_data[i], b_data[i], inv_mask);
			}
			return result;
		}
	}

#pragma region "simd_mask operators"
	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator&(const detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(packed_a & packed_b);
		}
		else
		{
			detail::x86_mask<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_and(a, b); }, result, a, b);
			return result;
		}
	}
	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator^(const detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(packed_a ^ packed_b);
		}
		else
		{
			detail::x86_mask<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_xor(a, b); }, result, a, b);
			return result;
		}
	}
	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator|(const detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(packed_a | packed_b);
		}
		else
		{
			detail::x86_mask<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_or(a, b); }, result, a, b);
			return result;
		}
	}

	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator~(const detail::x86_mask<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_mask<T, N, A>>(~packed);
		}
		else
		{
			detail::x86_mask<T, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::bit_not(x); }, result, x);
			return result;
		}
	}
	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator!(const detail::x86_mask<T, N, A> &x) noexcept { return ~x; }

	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator&&(const detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(packed_a && packed_b);
		}
		else
		{
			detail::x86_mask<T, N, A> result = {};
			const auto zero = detail::setzero<ext::native_data_type_t<detail::x86_mask<T, N, A>>>();
			detail::vectorize([z = zero](auto &res, auto a, auto b)
			{
				res = detail::bit_andnot(detail::cmp_eq<T>(a, z), res);
				res = detail::bit_andnot(detail::cmp_eq<T>(b, z), res);
			}, result, a, b);
			return result;
		}
	}
	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator||(const detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept { return (a | b) != detail::x86_mask<T, N, A>{}; }

	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator==(const detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(packed_a == packed_b);
		}
		else
		{
			detail::x86_mask<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::mask_eq<T>(a, b); }, result, a, b);
			return result;
		}
	}
	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator!=(const detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept { return !(a == b); }
#pragma endregion

#pragma region "simd_mask reductions"
	/** Returns `true` if all of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE bool all_of(const detail::x86_mask<T, N, A> &mask) noexcept
	{
		if (std::is_constant_evaluated())
		{
			bool result = true;
			for (std::size_t i = 0; i < N; ++i)
				result &= mask[i];
			return result;
		}
		else
		{
			using mask_t = detail::x86_mask<T, N, A>;
			const auto mask_data = ext::to_native_data(mask);

			auto result = detail::setones<ext::native_data_type_t<mask_t>>();
			const auto zero = detail::setzero<ext::native_data_type_t<mask_t>>();
			for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
			{
				const auto vm = detail::maskone<T>(mask_data[i], mask_t::size() - i * sizeof(result) / sizeof(T));
				result = detail::bit_andnot(detail::cmp_eq<T>(vm, zero), result);
			}
			return detail::movemask<T>(result) == detail::fill_bits<(sizeof(result) / sizeof(T)) * detail::movemask_bits_v<T>>();
		}
	}
	/** Returns `true` if at least one of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE bool any_of(const detail::x86_mask<T, N, A> &mask) noexcept
	{
		if (std::is_constant_evaluated())
		{
			bool result = false;
			for (std::size_t i = 0; i < N; ++i)
				result |= mask[i];
			return result;
		}
		else
		{
			using mask_t = detail::x86_mask<T, N, A>;
			const auto mask_data = ext::to_native_data(mask);

			auto result = detail::setzero<ext::native_data_type_t<mask_t>>();
			for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
			{
				const auto vm = detail::maskzero<T>(mask_data[i], mask_t::size() - i * sizeof(result) / sizeof(T));
				result = detail::bit_or(result, vm);
			}
#ifndef DPM_HAS_SSE4_1
			result = detail::cmp_ne<T>(result, detail::setzero<ext::native_data_type_t<mask_t>>());
#endif
			return detail::test_mask(result);
		}
	}
	/** Returns `true` if at none of the elements of the \a mask is `true`. Otherwise returns `false`. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] constexpr DPM_FORCEINLINE bool none_of(const detail::x86_mask<T, N, A> &mask) noexcept
	{
		if (std::is_constant_evaluated())
			return !any_of(mask);
		else
		{
			using mask_t = detail::x86_mask<T, N, A>;
			const auto mask_data = ext::to_native_data(mask);

			auto result = detail::setzero<ext::native_data_type_t<mask_t>>();
			for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
			{
				const auto vm = detail::maskzero<T>(mask_data[i], mask_t::size() - i * sizeof(result) / sizeof(T));
				result = detail::bit_or(result, vm);
			}
#ifndef DPM_HAS_SSE4_1
			result = detail::cmp_ne<T>(result, detail::setzero<ext::native_data_type_t<mask_t>>());
#endif
			return !detail::test_mask(result);
		}
	}
	/** Returns `true` if at least one of the elements of the \a mask is `true` and at least one is `false`. Otherwise returns `false`. */
	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE bool some_of(const detail::x86_mask<T, N, A> &mask) noexcept
	{
		if (std::is_constant_evaluated())
			return any_of(mask) && !all_of(mask);
		else
		{
			using mask_t = detail::x86_mask<T, N, A>;
			const auto mask_data = ext::to_native_data(mask);

			const auto zero = detail::setzero<ext::native_data_type_t<mask_t>>();
			auto any_mask = detail::setzero<ext::native_data_type_t<mask_t>>(), all_mask = detail::setones<ext::native_data_type_t<mask_t>>();
			for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
			{
				const auto vm = mask_data[i];
				const auto n = mask_t::size() - i * sizeof(vm) / sizeof(T);
				const auto vmz = detail::maskzero<T>(vm, n);
				const auto vmo = detail::maskone<T>(vm, n);

				all_mask = detail::bit_andnot(detail::cmp_eq<T>(vmo, zero), all_mask);
				any_mask = detail::bit_or(any_mask, vmz);
			}
#ifndef DPM_HAS_SSE4_1
			any_mask = detail::cmp_ne<T>(any_mask, detail::setzero<ext::native_data_type_t<mask_t>>());
#endif
			return detail::test_mask(any_mask) && detail::movemask<T>(all_mask) != detail::fill_bits<(sizeof(all_mask) / sizeof(T)) * detail::movemask_bits_v<T>>();
		}
	}

	/** Returns the number of `true` elements of \a mask. */
	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE std::size_t reduce_count(const detail::x86_mask<T, N, A> &mask) noexcept
	{
		if (std::is_constant_evaluated())
		{
			std::size_t result = 0;
			for (std::size_t i = 0; i < N; ++i)
				result += mask[i];
			return result;
		}
		else
		{
			using mask_t = detail::x86_mask<T, N, A>;
			const auto mask_data = ext::to_native_data(mask);

			std::size_t result = 0;
			const auto zero = detail::setzero<ext::native_data_type_t<mask_t>>();
			for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
			{
				const auto vm = detail::maskzero<T>(mask_data[i], mask_t::size() - i * sizeof(mask_data[i]) / sizeof(T));
				result += std::popcount(detail::movemask<T>(detail::cmp_ne<T>(vm, zero)));
			}
			return result / detail::movemask_bits_v<T>;
		}
	}
	/** Returns the index of the first `true` element of \a mask. */
	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE std::size_t reduce_min_index(const detail::x86_mask<T, N, A> &mask) noexcept
	{
		if (std::is_constant_evaluated())
		{
			for (std::size_t i = 0; i != N; ++i)
				if (mask[i]) return i;
			return N;
		}
		else
		{
			using mask_t = detail::x86_mask<T, N, A>;
			const auto mask_data = ext::to_native_data(mask);
			const auto zero = detail::setzero<ext::native_data_type_t<mask_t>>();
			for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
			{
				const auto bits = detail::movemask<T>(detail::cmp_ne<T>(mask_data[i], zero));
				if (bits) return std::countr_zero(bits) / detail::movemask_bits_v<T> + i * sizeof(mask_data[i]) / sizeof(T);
			}
			DPM_UNREACHABLE();
		}
	}
	/** Returns the index of the last `true` element of \a mask. */
	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE std::size_t reduce_max_index(const detail::x86_mask<T, N, A> &mask) noexcept
	{
		if (std::is_constant_evaluated())
		{
			for (std::size_t i = N; --i != 0;)
				if (mask[i]) return i;
			return N;
		}
		else
		{
			using mask_t = detail::x86_mask<T, N, A>;
			const auto mask_data = ext::to_native_data(mask);
			const auto zero = detail::setzero<ext::native_data_type_t<mask_t>>();
			for (std::size_t i = ext::native_data_size_v<mask_t>, j; (j = i--) != 0;)
			{
				constexpr auto native_extent = sizeof(mask_data[i]) / sizeof(T);
				const auto bits = detail::movemask_l<T>(detail::cmp_ne<T>(mask_data[i], zero), mask_t::size() - i * native_extent);
				if (bits) return (j * native_extent - 1) - std::countl_zero(bits) / detail::movemask_bits_v<T>;
			}
			DPM_UNREACHABLE();
		}
	}
#pragma endregion

	template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_any<T, N, Align>
	class simd<T, detail::avec<N, Align>>
	{
		friend detail::native_access<simd>;
		template<typename, typename>
		friend class simd;

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
		static constexpr std::size_t size() noexcept { return abi_type::size(); }

	public:
		constexpr simd() noexcept = default;

		/** Initializes the underlying elements with \a value. */
		template<detail::compatible_element<value_type> U>
		constexpr simd(U value) noexcept
		{
			std::array<value_type, native_extent> buff = {};
			std::ranges::fill(buff, static_cast<value_type>(value));
			std::ranges::fill(_data, detail::native_bit_cast<native_type>(buff));
		}

		/** Initializes the SIMD vector with a native vector. */
		constexpr simd(native_type native) noexcept { std::ranges::fill(_data, native); }
		/** Initializes the SIMD vector with an array of native vectors. */
		constexpr simd(const native_type (&native)[data_size]) noexcept { std::copy_n(native, data_size, _data); }
		/** Initializes the SIMD vector with a span of native mask vectors. */
		constexpr simd(std::span<const native_type, data_size> native) noexcept { std::copy_n(native.begin(), data_size, _data); }

		/** Initializes the underlying elements from \a mem. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr simd(I mem, Flags) noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<std::iter_value_t<I>, value_type> { copy_from(mem, Flags{}); }
		/** Initializes the underlying elements from \a mem if the corresponding element of mask \a m evaluates to `true`. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr simd(I mem, const mask_type &m, Flags) noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<std::iter_value_t<I>, value_type> { copy_from(mem, m, Flags{}); }

		/** Initializes the underlying elements with values provided by the generator \a gen. */
		template<detail::element_generator<value_type, size()> G>
		constexpr simd(G &&gen) noexcept
		{
			for (std::size_t i = 0, j = 0; i < size(); i += native_extent, ++j)
			{
				std::array<value_type, native_extent> buff = {};
				for (std::size_t k = 0; k < buff.size() && i + k < size(); ++k)
					buff[k] = static_cast<T>(std::invoke(gen, i + k));
				_data[j] = detail::native_bit_cast<native_type>(buff);
			}
		}

		template<typename U, typename OtherAbi> requires(simd_size_v<T, OtherAbi> == size() && std::is_convertible_v<T, U>)
		constexpr explicit(!std::convertible_to<T, U>) operator simd<U, OtherAbi>() const noexcept
		{
			constexpr auto align = std::max(alignof(simd), alignof(simd<U, OtherAbi>));
			if constexpr (std::same_as<U, T>)
			{
				alignas(align) std::array<T, size()> buff = {};
				copy_to(buff.data(), vector_aligned);
				return {buff.data(), vector_aligned};
			}
			else
			{
				alignas(align) std::array<T, size()> buff_a = {};
				alignas(align) std::array<U, size()> buff_b = {};
				copy_to(buff_a.data(), vector_aligned);
				for (std::size_t i = 0; i < size(); ++i)
					buff_b[i] = static_cast<U>(buff_a[i]);
				return {buff_b.data(), vector_aligned};
			}
		}

		/** Copies the underlying elements from \a mem. */
		template<std::contiguous_iterator I, typename Flags, typename U = std::iter_value_t<I>>
		constexpr void copy_from(I mem, Flags) noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<U, T>
		{
			std::size_t i = 0;
			if constexpr (detail::aligned_tag<Flags, alignof(native_type)> && sizeof(U) == sizeof(value_type))
				if (!std::is_constant_evaluated())
				{
					const auto src = std::to_address(mem);
					for (std::size_t j = 0; j < data_size; i += native_extent, ++j)
						detail::native_vec_cast<T, U>(_data[j], src + i);
				}
			for (; i < size(); ++i) operator[](i) = static_cast<T>(mem[i]);
		}
		/** Copies the underlying elements to \a mem. */
		template<std::contiguous_iterator I, typename Flags, typename U = std::iter_value_t<I>>
		constexpr void copy_to(I mem, Flags) const noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<T, U>
		{
			std::size_t i = 0;
			if constexpr (detail::aligned_tag<Flags, alignof(native_type)> && sizeof(U) == sizeof(value_type))
				if (!std::is_constant_evaluated())
				{
					const auto dst = std::to_address(mem);
					for (std::size_t j = 0; j < data_size; i += native_extent, ++j)
						detail::native_vec_cast<U, T>(dst + i, _data[j]);
				}
			for (; i < size(); ++i) mem[i] = static_cast<U>(operator[](i));
		}

		/** Copies the underlying elements from \a mem if the corresponding element of mask \a m evaluates to `true`. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr void copy_from(I mem, const mask_type &m, Flags) noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<std::iter_value_t<I>, value_type>
		{
			for (std::size_t i = 0; i < size(); ++i)
				if (m[i]) _data[i] = static_cast<value_type>(mem[i]);
		}
		/** Copies the underlying elements to \a mem if the corresponding element of mask \a m evaluates to `true`. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr void copy_to(I mem, const mask_type &m, Flags) const noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<value_type, std::iter_value_t<I>>
		{
			for (std::size_t i = 0; i < size(); ++i)
				if (m[i]) mem[i] = static_cast<std::iter_value_t<I>>(_data[i]);
		}

		[[nodiscard]] constexpr reference operator[](std::size_t i) & noexcept
		{
			return reinterpret_cast<value_alias *>(_data.data())[i];
		}
		[[nodiscard]] constexpr value_type operator[](std::size_t i) const & noexcept
		{
			if (std::is_constant_evaluated())
				return detail::native_bit_cast<std::array<value_type, native_extent>>(_data[i / native_extent])[i];
			else
				return reinterpret_cast<const value_alias *>(_data.data())[i];
		}

	private:
		alignas(alignment) storage_type _data;
	};

	namespace detail
	{
		template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
		struct native_access<x86_simd<T, N, A>>
		{
			using simd_t = x86_simd<T, N, A>;
			static constexpr std::span<ext::native_data_type_t<simd_t>, simd_t::data_size> to_native_data(simd_t &x) noexcept { return {x._data}; }
			static constexpr std::span<const ext::native_data_type_t<simd_t>, simd_t::data_size> to_native_data(const simd_t &x) noexcept { return {x._data}; }
		};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a x. */
		template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
		[[nodiscard]] constexpr auto to_native_data(detail::x86_simd<T, N, A> &x) noexcept { return detail::native_access<detail::x86_simd<T, N, A>>::to_native_data(x); }
		/** Returns a constant span of the underlying SSE vectors for \a x. */
		template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
		[[nodiscard]] constexpr auto to_native_data(const detail::x86_simd<T, N, A> &x) noexcept { return detail::native_access<detail::x86_simd<T, N, A>>::to_native_data(x); }

		/** Shuffles elements of vector \a x into a new vector according to the specified indices. */
		template<std::size_t... Is, typename T, std::size_t N, std::size_t A, std::size_t M = sizeof...(Is)>
		[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, M, A> shuffle(const detail::x86_simd<T, N, A> &x) noexcept requires(DPM_NOT_SSSE3(sizeof(T) >= 4 &&) detail::x86_overload_any<T, N, A> && detail::x86_overload_any<T, M, A>)
		{
			if (std::is_constant_evaluated())
			{
				const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
				return static_cast<detail::x86_simd<T, N, A>>(shuffle<Is...>(packed));
			}
			else
			{
				detail::x86_simd<T, M, A> result = {};
				auto result_data = to_native_data(result).data();
				const auto x_data = to_native_data(x).data();

				detail::shuffle<T, 0>(std::index_sequence<Is...>{}, result_data, x_data);
				return result;
			}
		}
	}

	/** Replaces elements of vector \a a with elements of vector \a b using mask \a m. Elements of \a a are selected if the corresponding element of \a m evaluates to `true`. */
	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> blend(const detail::x86_mask<T, N, A> &m, const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_m = static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(m);
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(blend(packed_m, packed_a, packed_b));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			auto result_data = to_native_data(result);
			const auto a_data = to_native_data(a);
			const auto b_data = to_native_data(b);
			const auto m_data = to_native_data(m);

			const auto zero = detail::setzero<native_data_type_t<detail::x86_mask<T, N, A>>>();
			for (std::size_t i = 0; i < native_data_size_v<detail::x86_simd<T, N, A>>; ++i)
			{
				/* Since mask can be modified via to_native_data, need to make sure the entire mask is filled. */
				const auto inv_mask = detail::mask_eq<T>(m_data[i], zero);
				result_data[i] = detail::blendv<T>(a_data[i], b_data[i], inv_mask);
			}
			return result;
		}
	}

#pragma region "simd operators"
	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator-(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(-packed);
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::negate<T>(x); }, result, x);
			return result;
		}
	}

	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator+(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(packed_a + packed_b);
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::add<T>(a, b); }, result, a, b);
			return result;
		}
	}
	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator-(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(packed_a - packed_b);
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::sub<T>(a, b); }, result, a, b);
			return result;
		}
	}

	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator+(const detail::x86_simd<T, N, A> &a, T b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			return static_cast<detail::x86_simd<T, N, A>>(packed + b);
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
			detail::vectorize([b = b_vec](auto &res, auto a) { res = detail::add<T>(a, b); }, result, a);
			return result;
		}
	}
	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator-(const detail::x86_simd<T, N, A> &a, T b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			return static_cast<detail::x86_simd<T, N, A>>(packed - b);
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
			detail::vectorize([b = b_vec](auto &res, auto a) { res = detail::sub<T>(a, b); }, result, a);
			return result;
		}
	}

	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator*(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(packed_a * packed_b);
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::mul<T>(a, b); }, result, a, b);
			return result;
		}
	}
	template<std::floating_point  T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator/(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(packed_a / packed_b);
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::div<T>(a, b); }, result, a, b);
			return result;
		}
	}

	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator*(const detail::x86_simd<T, N, A> &a, T b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			return static_cast<detail::x86_simd<T, N, A>>(packed * b);
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
			detail::vectorize([b = b_vec](auto &res, auto a) { res = detail::mul<T>(a, b); }, result, a);
			return result;
		}
	}
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator/(const detail::x86_simd<T, N, A> &a, T b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			return static_cast<detail::x86_simd<T, N, A>>(packed / b);
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
			detail::vectorize([b = b_vec](auto &res, auto a) { res = detail::div<T>(a, b); }, result, a);
			return result;
		}
	}

	template<std::integral T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator*(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
#if defined(DPM_HAS_AVX512DQ) && defined(DPM_HAS_AVX512LV)
		if constexpr(!std::is_constant_evaluated() && sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8)
#elif defined(DPM_HAS_SSE4_1)
		if constexpr(!std::is_constant_evaluated() && sizeof(T) == 2 || sizeof(T) == 4)
#else
		if constexpr(!std::is_constant_evaluated() && sizeof(T) == 2)
#endif
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::mul<T>(a, b); }, result, a, b);
			return result;
		}
		else
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(packed_a * packed_b);
		}
	}
	template<std::integral T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator*(const detail::x86_simd<T, N, A> &a, T b) noexcept
	{
#if defined(DPM_HAS_AVX512DQ) && defined(DPM_HAS_AVX512LV)
		if constexpr(!std::is_constant_evaluated() && sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8)
#elif defined(DPM_HAS_SSE4_1)
		if constexpr(!std::is_constant_evaluated() && sizeof(T) == 2 || sizeof(T) == 4)
#else
		if constexpr(!std::is_constant_evaluated() && sizeof(T) == 2)
#endif
		{
			detail::x86_simd<T, N, A> result = {};
			const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
			detail::vectorize([b = b_vec](auto &res, auto a) { res = detail::mul<T>(a, b); }, result, a);
			return result;
		}
		else
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			return static_cast<detail::x86_simd<T, N, A>>(packed * b);
		}
	}

	template<std::integral T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator!(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_mask<T, N, A>>(!packed);
		}
		else
		{
			detail::x86_mask<T, N, A> result = {};
			const auto zero = detail::setzero<ext::native_data_type_t<detail::x86_simd<T, N, A>>>();
			detail::vectorize([z = zero](auto &result, auto x) { result = detail::cmp_eq<T>(x, z); }, result, x);
			return result;
		}
	}

	template<std::integral T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator&(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(packed_a & packed_b);
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_and(a, b); }, result, a, b);
			return result;
		}
	}
	template<std::integral T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator^(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(packed_a ^ packed_b);
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_xor(a, b); }, result, a, b);
			return result;
		}
	}
	template<std::integral T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator|(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(packed_a | packed_b);
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_or(a, b); }, result, a, b);
			return result;
		}
	}

	template<std::integral T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator&(const detail::x86_simd<T, N, A> &a, T b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			return static_cast<detail::x86_simd<T, N, A>>(packed & b);
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
			detail::vectorize([b = b_vec](auto &res, auto a) { res = detail::bit_and(a, b); }, result, a);
			return result;
		}
	}
	template<std::integral T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator^(const detail::x86_simd<T, N, A> &a, T b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			return static_cast<detail::x86_simd<T, N, A>>(packed ^ b);
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
			detail::vectorize([b = b_vec](auto &res, auto a) { res = detail::bit_xor(a, b); }, result, a);
			return result;
		}
	}
	template<std::integral T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator|(const detail::x86_simd<T, N, A> &a, T b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			return static_cast<detail::x86_simd<T, N, A>>(packed | b);
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
			detail::vectorize([b = b_vec](auto &res, auto a) { res = detail::bit_or(a, b); }, result, a);
			return result;
		}
	}

	template<std::integral T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator<<(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
#if defined(DPM_HAS_AVX512BW) && defined(DPM_HAS_AVX512LV)
		if constexpr(!std::is_constant_evaluated() && sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8)
#elif defined(DPM_HAS_SSE4_1)
		if constexpr(!std::is_constant_evaluated() && sizeof(T) == 4 || sizeof(T) == 8)
#else
		if constexpr(!std::is_constant_evaluated() && sizeof(T) == 8)
#endif
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_shiftl<T>(a, b); }, result, a, b);
			return result;
		}
		else
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(packed_a << packed_b);
		}
	}
	template<std::integral T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator>>(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
#if defined(DPM_HAS_AVX512BW) && defined(DPM_HAS_AVX512LV)
		if constexpr(!std::is_constant_evaluated() && sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8)
#elif defined(DPM_HAS_SSE4_1)
		if constexpr(!std::is_constant_evaluated() && sizeof(T) == 4 || sizeof(T) == 8)
#else
		if constexpr(!std::is_constant_evaluated() && sizeof(T) == 8)
#endif
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_shiftr<T>(a, b); }, result, a, b);
			return result;
		}
		else
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(packed_a << packed_b);
		}
	}

	template<std::integral T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator<<(const detail::x86_simd<T, N, A> &a, T n) noexcept
	{
#if defined(DPM_HAS_AVX512BW) && defined(DPM_HAS_AVX512LV)
		if constexpr(!std::is_constant_evaluated() && sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8)
#elif defined(DPM_HAS_SSE4_1)
		if constexpr(!std::is_constant_evaluated() && sizeof(T) == 4 || sizeof(T) == 8)
#else
		if constexpr(!std::is_constant_evaluated() && sizeof(T) == 8)
#endif
		{
			detail::x86_simd<T, N, A> result = {};
			const auto n_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(n);
			detail::vectorize([n = n_vec](auto &res, auto a) { res = detail::bit_shiftl<T>(a, n); }, result, a);
			return result;
		}
		else
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			return static_cast<detail::x86_simd<T, N, A>>(packed << n);
		}
	}
	template<std::integral T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> operator>>(const detail::x86_simd<T, N, A> &a, T n) noexcept
	{
#if defined(DPM_HAS_AVX512BW) && defined(DPM_HAS_AVX512LV)
		if constexpr(!std::is_constant_evaluated() && sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8)
#elif defined(DPM_HAS_SSE4_1)
		if constexpr(!std::is_constant_evaluated() && sizeof(T) == 4 || sizeof(T) == 8)
#else
		if constexpr(!std::is_constant_evaluated() && sizeof(T) == 8)
#endif
		{
			detail::x86_simd<T, N, A> result = {};
			const auto n_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(n);
			detail::vectorize([n = n_vec](auto &res, auto a) { res = detail::bit_shiftr<T>(a, n); }, result, a);
			return result;
		}
		else
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			return static_cast<detail::x86_simd<T, N, A>>(packed >> n);
		}
	}

	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator==(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(packed_a == packed_b);
		}
		else
		{
			detail::x86_mask<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_eq<T>(a, b); }, result, a, b);
			return result;
		}
	}
	template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator!=(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(packed_a != packed_b);
		}
		else
		{
			detail::x86_mask<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_ne<T>(a, b); }, result, a, b);
			return result;
		}
	}

	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator>(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(packed_a > packed_b);
		}
		else
		{
			detail::x86_mask<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_gt<T>(a, b); }, result, a, b);
			return result;
		}
	}
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator<(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(packed_a < packed_b);
		}
		else
		{
			detail::x86_mask<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_lt<T>(a, b); }, result, a, b);
			return result;
		}
	}
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator>=(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(packed_a >= packed_b);
		}
		else
		{
			detail::x86_mask<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_ge<T>(a, b); }, result, a, b);
			return result;
		}
	}
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator<=(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(packed_a <= packed_b);
		}
		else
		{
			detail::x86_mask<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_le<T>(a, b); }, result, a, b);
			return result;
		}
	}

#ifdef DPM_HAS_SSE4_1
	template<std::signed_integral T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator>(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(packed_a > packed_b);
		}
		else
		{
			detail::x86_mask<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_gt<T>(a, b); }, result, a, b);
			return result;
		}
	}
	template<std::integral T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator<(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept { return b > a; }
	template<std::integral T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator>=(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept { return (a > b) | (a == b); }
	template<std::integral T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> operator<=(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept { return (a < b) | (a == b); }
#endif

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Logically shifts elements of vector \a x left by a constant number of bits \a N. */
		template<std::size_t N, std::integral T, std::size_t M, std::size_t A> requires(detail::x86_overload_any<T, M, A> && sizeof(T) > 1 && N < std::numeric_limits<T>::digits)
		[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, M, A> lsl(const detail::x86_simd<T, M, A> &x) noexcept
		{
			if (std::is_constant_evaluated())
			{
				const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
				return static_cast<detail::x86_mask<T, N, A>>(lsl<N>(packed));
			}
			else
			{
				detail::x86_simd<T, M, A> result = {};
				detail::vectorize([](auto &res, auto x) { res = detail::bit_shiftl<T, N>(x); }, result, x);
				return result;
			}
		}
		/** Logically shifts elements of vector \a x right by a constant number of bits \a N. */
		template<std::size_t N, std::integral T, std::size_t M, std::size_t A> requires(detail::x86_overload_any<T, M, A> && sizeof(T) > 1 && N < std::numeric_limits<T>::digits)
		[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, M, A> lsr(const detail::x86_simd<T, M, A> &x) noexcept
		{
			if (std::is_constant_evaluated())
			{
				const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
				return static_cast<detail::x86_mask<T, N, A>>(lsr<N>(packed));
			}
			else
			{
				detail::x86_simd<T, M, A> result = {};
				detail::vectorize([](auto &res, auto x) { res = detail::bit_shiftr<T, N>(x); }, result, x);
				return result;
			}
		}

		/** Arithmetically shifts elements of vector \a x left by a constant number of bits \a N. */
		template<std::size_t N, std::signed_integral T, std::size_t M, std::size_t A> requires(detail::x86_overload_any<T, M, A> && sizeof(T) > 1 && N < std::numeric_limits<T>::digits)
		[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, M, A> asl(const detail::x86_simd<T, M, A> &x) noexcept
		{
			if (std::is_constant_evaluated())
			{
				const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
				return static_cast<detail::x86_mask<T, N, A>>(asl<N>(packed));
			}
			else
			{
				detail::x86_simd<T, M, A> result = {};
				detail::vectorize([](auto &res, auto x) { res = detail::bit_shiftl<T, N>(x); }, result, x);
				return result;
			}
		}
		/** Arithmetically shifts elements of vector \a x right by a constant number of bits \a N. */
		template<std::size_t N, std::signed_integral T, std::size_t M, std::size_t A> requires(detail::x86_overload_any<T, M, A> && sizeof(T) > 1 && N < std::numeric_limits<T>::digits)
		[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, M, A> asr(const detail::x86_simd<T, M, A> &x) noexcept
		{
			if (std::is_constant_evaluated())
			{
				const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
				return static_cast<detail::x86_mask<T, N, A>>(asr<N>(packed));
			}
			else
			{
				detail::x86_simd<T, M, A> result = {};
				detail::vectorize([](auto &res, auto x) { res = detail::bit_ashiftr<T, N>(x); }, result, x);
				return result;
			}
		}
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
		[[nodiscard]] DPM_FORCEINLINE T reduce_max(const V *data) noexcept { return reduce<T, N>(data, fill<V>(std::numeric_limits<T>::lowest()), [](auto a, auto b) { return max<T>(a, b); }); }

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
		[[nodiscard]] DPM_FORCEINLINE T reduce_add(const V *data) noexcept requires(sizeof(V) == 32) { return reduce_add<T, N>(reinterpret_cast<const select_vector_t<T, 16> *>(data)); }
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_mul(const V *data) noexcept requires(sizeof(V) == 32) { return reduce_mul<T, N>(reinterpret_cast<const select_vector_t<T, 16> *>(data)); }
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_min(const V *data) noexcept requires(sizeof(V) == 32) { return reduce_min<T, N>(reinterpret_cast<const select_vector_t<T, 16> *>(data)); }
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_max(const V *data) noexcept requires(sizeof(V) == 32) { return reduce_max<T, N>(reinterpret_cast<const select_vector_t<T, 16> *>(data)); }
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_and(const V *data) noexcept requires(sizeof(V) == 32) { return reduce_and<T, N>(reinterpret_cast<const select_vector_t<T, 16> *>(data)); }
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_xor(const V *data) noexcept requires(sizeof(V) == 32) { return reduce_xor<T, N>(reinterpret_cast<const select_vector_t<T, 16> *>(data)); }
		template<typename T, std::size_t N, typename V>
		[[nodiscard]] DPM_FORCEINLINE T reduce_or(const V *data) noexcept requires(sizeof(V) == 32) { return reduce_or<T, N>(reinterpret_cast<const select_vector_t<T, 16> *>(data)); }
#endif
	}

	/** Horizontally reduced elements of \a x using operation \a Op. */
	template<typename T, std::size_t N, std::size_t A, typename Op = std::plus<>> requires(detail::x86_overload_any<T, N, A> && sizeof(T) >= 4)
	[[nodiscard]] constexpr DPM_FORCEINLINE T reduce(const detail::x86_simd<T, N, A> &x, Op op = {}) noexcept(std::is_nothrow_invocable_v<Op, T, T>)
	{
		constexpr auto size = detail::x86_simd<T, N, A>::size();
		const auto x_data = ext::to_native_data(x).data();

		if (std::is_constant_evaluated())
			return detail::reduce_impl<size>(x, op);
		else if constexpr (detail::template_instance<Op, std::plus>)
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
		else
			return detail::reduce_impl<size>(x, op);
	}
#ifdef DPM_HAS_SSSE3
	/** @copydoc reduce */
	template<std::integral T, std::size_t N, std::size_t A, typename Op = std::plus<>> requires(detail::x86_overload_any<T, N, A> && sizeof(T) <= 2)
	[[nodiscard]] constexpr DPM_FORCEINLINE T reduce(const detail::x86_simd<T, N, A> &x, Op op = {}) noexcept
	{
		constexpr auto size = detail::x86_simd<T, N, A>::size();
		const auto x_data = ext::to_native_data(x).data();

		if (std::is_constant_evaluated())
			return detail::reduce_impl<size>(x, op);
		else if constexpr (detail::template_instance<Op, std::plus>)
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
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE T reduce_min(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
			return reduce_min(static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(x));
		else
			return detail::reduce_min<T, N>(ext::to_native_data(x).data());
	}
	/** Calculates horizontal maximum of elements of \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE T reduce_max(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
			return reduce_max(static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(x));
		else
			return detail::reduce_max<T, N>(ext::to_native_data(x).data());
	}
#if defined(DPM_HAS_SSE4_1)
	/** @copydoc reduce_min */
	template<std::integral T, std::size_t N, std::size_t A> requires(detail::x86_overload_any<T, N, A> && sizeof(T) < 8)
	[[nodiscard]] constexpr DPM_FORCEINLINE T reduce_min(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
			return reduce_min(static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(x));
		else
			return detail::reduce_min<T, N>(ext::to_native_data(x).data());
	}
	/** @copydoc reduce_max */
	template<std::integral T, std::size_t N, std::size_t A> requires(detail::x86_overload_any<T, N, A> && sizeof(T) < 8)
	[[nodiscard]] constexpr DPM_FORCEINLINE T reduce_max(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
			return reduce_max(static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(x));
		else
			return detail::reduce_max<T, N>(ext::to_native_data(x).data());
	}
#elif defined(DPM_HAS_SSSE3)
	/** @copydoc reduce_min */
	template<detail::unsigned_integral_of_size<1> T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE T reduce_min(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
			return reduce_min(static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(x));
		else
			return detail::reduce_min<T, N>(ext::to_native_data(x).data());
	}
	/** @copydoc reduce_min */
	template<detail::unsigned_integral_of_size<1> T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE T reduce_max(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
			return reduce_max(static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(x));
		else
			return detail::reduce_max<T, N>(ext::to_native_data(x).data());
	}
	/** @copydoc reduce_min */
	template<detail::signed_integral_of_size<2> T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE T reduce_min(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
			return reduce_min(static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(x));
		else
			return detail::reduce_min<T, N>(ext::to_native_data(x).data());
	}
	/** @copydoc reduce_max */
	template<detail::signed_integral_of_size<2> T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE T reduce_max(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
			return reduce_max(static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(x));
		else
			return detail::reduce_max<T, N>(ext::to_native_data(x).data());
	}
#endif
#if defined(DPM_HAS_AVX512F) && defined(DPM_HAS_AVX512LV)
	/** @copydoc reduce_min */
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE T reduce_min(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
			return reduce_min(static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(x));
		else
			return detail::reduce_min<T, N>(ext::to_native_data(x).data());
	}
	/** @copydoc reduce_max */
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE T reduce_max(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
			return reduce_max(static_cast<simd_mask<T, simd_abi::ext::packed_buffer<N>>>(x));
		else
			return detail::reduce_max<T, N>(ext::to_native_data(x).data());
	}
#endif
#pragma endregion

#pragma region "simd algorithms"
	/** Returns an SIMD vector of minimum elements of \a a and \a b. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> min(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(min(packed_a, packed_b));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::min<T>(a, b); }, result, a, b);
			return result;
		}
	}
	/** Returns an SIMD vector of maximum elements of \a a and \a b. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> max(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(max(packed_a, packed_b));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::max<T>(a, b); }, result, a, b);
			return result;
		}
	}

	/** Returns a pair of SIMD vectors of minimum and maximum elements of \a a and \a b. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE std::pair<detail::x86_simd<T, N, A>, detail::x86_simd<T, N, A>> minmax(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept { return {min(a, b), max(a, b)}; }
	/** Clamps elements \f \a x between corresponding elements of \a min and \a max. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> clamp(const detail::x86_simd<T, N, A> &x, const detail::x86_simd<T, N, A> &min, const detail::x86_simd<T, N, A> &max) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_x = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			const auto packed_min = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(min);
			const auto packed_max = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(max);
			return static_cast<detail::x86_simd<T, N, A>>(clamp(packed_x, packed_min, packed_max));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto x, auto min, auto max) { res = detail::min<T>(detail::max<T>(x, min), max); }, result, x, min, max);
			return result;
		}
	}

#if defined(DPM_HAS_SSE4_1)
	/** @copydoc min */
	template<std::integral T, std::size_t N, std::size_t A> requires(detail::x86_overload_any<T, N, A> && sizeof(T) < 8)
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> min(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(min(packed_a, packed_b));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::min<T>(a, b); }, result, a, b);
			return result;
		}
	}
	/** @copydoc max */
	template<std::integral T, std::size_t N, std::size_t A> requires(detail::x86_overload_any<T, N, A> && sizeof(T) < 8)
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> max(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(max(packed_a, packed_b));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::max<T>(a, b); }, result, a, b);
			return result;	
		}
	}
	/** @copydoc minmax */
	template<std::integral T, std::size_t N, std::size_t A> requires(detail::x86_overload_any<T, N, A> && sizeof(T) < 8)
	[[nodiscard]] constexpr DPM_FORCEINLINE std::pair<detail::x86_simd<T, N, A>, detail::x86_simd<T, N, A>> minmax(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept { return {min(a, b), max(a, b)}; }
	/** @copydoc clamp */
	template<std::integral T, std::size_t N, std::size_t A> requires(detail::x86_overload_any<T, N, A> && sizeof(T) < 8)
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> clamp(const detail::x86_simd<T, N, A> &x, const detail::x86_simd<T, N, A> &min, const detail::x86_simd<T, N, A> &max) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_x = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			const auto packed_min = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(min);
			const auto packed_max = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(max);
			return static_cast<detail::x86_simd<T, N, A>>(clamp(packed_x, packed_min, packed_max));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto x, auto min, auto max) { res = detail::min<T>(detail::max<T>(x, min), max); }, result, x, min, max);
			return result;
		}
	}
#elif defined(DPM_HAS_SSSE3)
	/** @copydoc min */
	template<std::integral T, std::size_t N, std::size_t A> requires(detail::x86_overload_any<T, N, A> && (detail::unsigned_integral_of_size<T, 1> || detail::signed_integral_of_size<T, 2>))
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> min(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(min(packed_a, packed_b));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::min<T>(a, b); }, result, a, b);
			return result;
		}
	}
	/** @copydoc max */
	template<std::integral T, std::size_t N, std::size_t A> requires(detail::x86_overload_any<T, N, A> && (detail::unsigned_integral_of_size<T, 1> || detail::signed_integral_of_size<T, 2>))
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> max(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(max(packed_a, packed_b));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::max<T>(a, b); }, result, a, b);
			return result;	
		}
	}
	/** @copydoc minmax */
	template<std::integral T, std::size_t N, std::size_t A> requires(detail::x86_overload_any<T, N, A> && (detail::unsigned_integral_of_size<T, 1> || detail::signed_integral_of_size<T, 2>))
	[[nodiscard]] constexpr DPM_FORCEINLINE std::pair<detail::x86_simd<T, N, A>, detail::x86_simd<T, N, A>> minmax(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept { return {min(a, b), max(a, b)}; }
	/** @copydoc clamp */
	template<std::integral T, std::size_t N, std::size_t A> requires(detail::x86_overload_any<T, N, A> && (detail::unsigned_integral_of_size<T, 1> || detail::signed_integral_of_size<T, 2>))
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> clamp(const detail::x86_simd<T, N, A> &x, const detail::x86_simd<T, N, A> &min, const detail::x86_simd<T, N, A> &max) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_x = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			const auto packed_min = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(min);
			const auto packed_max = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(max);
			return static_cast<detail::x86_simd<T, N, A>>(clamp(packed_x, packed_min, packed_max));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto x, auto min, auto max) { res = detail::min<T>(detail::max<T>(x, min), max); }, result, x, min, max);
			return result;
		}
	}
#endif
#if defined(DPM_HAS_AVX512F) && defined(DPM_HAS_AVX512LV)
	/** @copydoc min */
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> min(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(min(packed_a, packed_b));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::min<T>(a, b); }, result, a, b);
			return result;
		}
	}
	/** @copydoc max */
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> max(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(max(packed_a, packed_b));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::max<T>(a, b); }, result, a, b);
			return result;	
		}
	}
	/** @copydoc minmax */
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE std::pair<detail::x86_simd<T, N, A>, detail::x86_simd<T, N, A>> minmax(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept { return {min(a, b), max(a, b)}; }
	/** @copydoc clamp */
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> clamp(const detail::x86_simd<T, N, A> &x, const detail::x86_simd<T, N, A> &min, const detail::x86_simd<T, N, A> &max) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_x = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			const auto packed_min = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(min);
			const auto packed_max = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(max);
			return static_cast<detail::x86_simd<T, N, A>>(clamp(packed_x, packed_min, packed_max));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto x, auto min, auto max) { res = detail::min<T>(detail::max<T>(x, min), max); }, result, x, min, max);
			return result;
		}
	}
#endif

	/** Returns an SIMD vector of minimum elements of \a a and scalar \a b. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> min(const detail::x86_simd<T, N, A> &a, T b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			return static_cast<detail::x86_simd<T, N, A>>(min(packed, b));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
			detail::vectorize([b = b_vec](auto &res, auto a) { res = detail::min<T>(a, b); }, result, a);
			return result;
		}
	}
	/** Returns an SIMD vector of maximum elements of \a a and scalar \a b. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> max(const detail::x86_simd<T, N, A> &a, T b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			return static_cast<detail::x86_simd<T, N, A>>(max(packed, b));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
			detail::vectorize([b = b_vec](auto &res, auto a) { res = detail::max<T>(a, b); }, result, a);
			return result;
		}
	}
	/** Returns a pair of SIMD vectors of minimum and maximum elements of \a a and scalar \a b. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE std::pair<detail::x86_simd<T, N, A>, detail::x86_simd<T, N, A>> minmax(const detail::x86_simd<T, N, A> &a, T b) noexcept { return {min(a, b), max(a, b)}; }
	/** Clamps elements \f \a x between \a min and \a max. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> clamp(const detail::x86_simd<T, N, A> &x, T min, T max) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(clamp(packed, min, max));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto min_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(min);
			const auto max_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(max);
			detail::vectorize([min = min_vec, max = max_vec](auto &res, auto x) { res = detail::min<T>(detail::max<T>(x, min), max); }, result, x);
			return result;
		}
	}

#if defined(DPM_HAS_SSE4_1)
	/** @copydoc min */
	template<std::integral T, std::size_t N, std::size_t A> requires(detail::x86_overload_any<T, N, A> && sizeof(T) < 8)
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> min(const detail::x86_simd<T, N, A> &a, T b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			return static_cast<detail::x86_simd<T, N, A>>(min(packed, b));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
			detail::vectorize([b = b_vec](auto &res, auto a) { res = detail::min<T>(a, b); }, result, a);
			return result;
		}
	}
	/** @copydoc max */
	template<std::integral T, std::size_t N, std::size_t A> requires(detail::x86_overload_any<T, N, A> && sizeof(T) < 8)
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> max(const detail::x86_simd<T, N, A> &a, T b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			return static_cast<detail::x86_simd<T, N, A>>(max(packed, b));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
			detail::vectorize([b = b_vec](auto &res, auto a) { res = detail::max<T>(a, b); }, result, a);
			return result;
		}
	}
	/** @copydoc minmax */
	template<std::integral T, std::size_t N, std::size_t A> requires(detail::x86_overload_any<T, N, A> && sizeof(T) < 8)
	[[nodiscard]] constexpr DPM_FORCEINLINE std::pair<detail::x86_simd<T, N, A>, detail::x86_simd<T, N, A>> minmax(const detail::x86_simd<T, N, A> &a, T b) noexcept { return {min(a, b), max(a, b)}; }
	/** @copydoc clamp */
	template<std::integral T, std::size_t N, std::size_t A> requires(detail::x86_overload_any<T, N, A> && sizeof(T) < 8)
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> clamp(const detail::x86_simd<T, N, A> &x, T min, T max) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(clamp(packed, min, max));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto min_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(min);
			const auto max_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(max);
			detail::vectorize([min = min_vec, max = max_vec](auto &res, auto x) { res = detail::min<T>(detail::max<T>(x, min), max); }, result, x);
			return result;
		}
	}
#elif defined(DPM_HAS_SSSE3)
	/** @copydoc min */
	template<std::integral T, std::size_t N, std::size_t A> requires(detail::x86_overload_any<T, N, A> && (detail::unsigned_integral_of_size<T, 1> || detail::signed_integral_of_size<T, 2>))
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> min(const detail::x86_simd<T, N, A> &a, T b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			return static_cast<detail::x86_simd<T, N, A>>(min(packed, b));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
			detail::vectorize([b = b_vec](auto &res, auto a) { res = detail::min<T>(a, b); }, result, a);
			return result;
		}
	}
	/** @copydoc max */
	template<std::integral T, std::size_t N, std::size_t A> requires(detail::x86_overload_any<T, N, A> && (detail::unsigned_integral_of_size<T, 1> || detail::signed_integral_of_size<T, 2>))
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> max(const detail::x86_simd<T, N, A> &a, T b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			return static_cast<detail::x86_simd<T, N, A>>(max(packed, b));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
			detail::vectorize([b = b_vec](auto &res, auto a) { res = detail::max<T>(a, b); }, result, a);
			return result;
		}
	}
	/** @copydoc minmax */
	template<std::integral T, std::size_t N, std::size_t A> requires(detail::x86_overload_any<T, N, A> && (detail::unsigned_integral_of_size<T, 1> || detail::signed_integral_of_size<T, 2>))
	[[nodiscard]] constexpr DPM_FORCEINLINE std::pair<detail::x86_simd<T, N, A>, detail::x86_simd<T, N, A>> minmax(const detail::x86_simd<T, N, A> &a, T b) noexcept { return {min(a, b), max(a, b)}; }
	/** @copydoc clamp */
	template<std::integral T, std::size_t N, std::size_t A> requires(detail::x86_overload_any<T, N, A> && (detail::unsigned_integral_of_size<T, 1> || detail::signed_integral_of_size<T, 2>))
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> clamp(const detail::x86_simd<T, N, A> &x, T min, T max) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(clamp(packed, min, max));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto min_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(min);
			const auto max_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(max);
			detail::vectorize([min = min_vec, max = max_vec](auto &res, auto x) { res = detail::min<T>(detail::max<T>(x, min), max); }, result, x);
			return result;
		}
	}
#endif
#if defined(DPM_HAS_AVX512F) && defined(DPM_HAS_AVX512LV)
	/** @copydoc min */
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> min(const detail::x86_simd<T, N, A> &a, T b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			return static_cast<detail::x86_simd<T, N, A>>(min(packed, b));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
			detail::vectorize([b = b_vec](auto &res, auto a) { res = detail::min<T>(a, b); }, result, a);
			return result;
		}
	}
	/** @copydoc max */
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> max(const detail::x86_simd<T, N, A> &a, T b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			return static_cast<detail::x86_simd<T, N, A>>(max(packed, b));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
			detail::vectorize([b = b_vec](auto &res, auto a) { res = detail::max<T>(a, b); }, result, a);
			return result;
		}
	}
	/** @copydoc minmax */
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE std::pair<detail::x86_simd<T, N, A>, detail::x86_simd<T, N, A>> minmax(const detail::x86_simd<T, N, A> &a, T b) noexcept { return {min(a, b), max(a, b)}; }
	/** @copydoc clamp */
	template<detail::integral_of_size<8> T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> clamp(const detail::x86_simd<T, N, A> &x, T min, T max) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(clamp(packed, min, max));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			const auto min_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(min);
			const auto max_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(max);
			detail::vectorize([min = min_vec, max = max_vec](auto &res, auto x) { res = detail::min<T>(detail::max<T>(x, min), max); }, result, x);
			return result;
		}
	}
#endif
#pragma endregion
}

#endif

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif