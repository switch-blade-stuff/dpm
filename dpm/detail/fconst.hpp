/*
 * Created by switchblade on 2023-01-10.
 */

#pragma once

#include "../utility.hpp"

#include <numbers>
#include <limits>
#include <span>
#include <bit>

namespace dpm::detail
{
	template<std::floating_point T>
	static constexpr T pio32 = std::numbers::pi_v<T> / T{32.0};
	template<std::floating_point T>
	static constexpr T pio4 = std::numbers::pi_v<T> / T{4.0};
	template<std::floating_point T>
	static constexpr T pio2 = std::numbers::pi_v<T> / T{2.0};
	template<std::floating_point T>
	static constexpr T tan3pio8 = T{2.41421356237309504880};
	template<std::floating_point T>
	static constexpr T ln2 = T{0x1.62e42fefa39efp-1};

	template<std::floating_point T>
	static constexpr T exp_multm = std::same_as<T, float> ? T{0x1p-25} : T{0x1p-54};
	template<std::floating_point T>
	static constexpr T exp_mult = std::same_as<T, float> ? T{0x1p25} : T{0x1p54};

	/* In some cases x86 intrinsics generate extraneous casts if literals are used with intrinsics. As such, define commonly used values here. */
	template<std::floating_point T>
	static constexpr T five_eights = T{0.625};
	template<std::floating_point T>
	static constexpr T sign_bit = T{-0.0};
	template<std::floating_point T>
	static constexpr T half = T{0.5};
	template<std::floating_point T>
	static constexpr T p66 = T{0.66};
	template<std::floating_point T>
	static constexpr T one = T{1.0};
	template<std::floating_point T>
	static constexpr T two = T{2.0};

	template<typename>
	struct range_vals;
	template<typename T> requires std::same_as<double, T>
	struct range_vals<T>
	{
		static constexpr T huge = 1.0e+300;
		static constexpr T tiny = 1.0e-300;
	};
	template<typename T> requires std::same_as<float, T>
	struct range_vals<T>
	{
		static constexpr T huge = 1.0e+30f;
		static constexpr T tiny = 1.0e-30f;
	};

	template<std::floating_point T>
	static constexpr T huge = range_vals<T>::huge;
	template<std::floating_point T>
	static constexpr T tiny = range_vals<T>::tiny;

	template<typename T>
	static constexpr T exp_middle = sizeof(T) == sizeof(double) ? 0x3fe : 0x7e;
	template<typename T>
	static constexpr T exp_mask = sizeof(T) == sizeof(double) ? 0x7ff : 0xff;
	template<typename T>
	static constexpr T exp_off = sizeof(T) == sizeof(double) ? 1023 : 127;
	template<typename T>
	static constexpr T mant_bits = sizeof(T) == sizeof(double) ? 52 : 23;
	template<typename T>
	static constexpr T exp_bits = sizeof(T) == sizeof(double) ? 11 : 8;
	template<typename T>
	static constexpr T max_scalbn = 50000;

	template<std::floating_point T>
	static constexpr T dp_sincos[] = {T{-7.85398125648498535156e-1}, T{-3.77489470793079817668e-8}, T{-2.69515142907905952645e-15}};

	template<std::floating_point T>
	static constexpr T sincof[] = {
			T{1.58962301576546568060e-10}, T{-2.50507477628578072866e-8},
			T{2.75573136213857245213e-6}, T{-1.98412698295895385996e-4},
			T{8.33333333332211858878e-3}, T{-1.66666666666666307295e-1}
	};
	template<std::floating_point T>
	static constexpr T coscof[] = {
			T{-1.13585365213876817300e-11}, T{2.08757008419747316778e-9},
			T{-2.75573141792967388112e-7}, T{2.48015872888517045348e-5},
			T{-1.38888888888730564116e-3}, T{4.16666666666665929218e-2}
	};

	template<std::floating_point T>
	static constexpr T tancof[] = {T{1.0}, T{3.3333333332780246e-1}, T{1.3333333883976731e-1}, T{5.3966541627173827e-2}, T{2.2079624737833755e-2}};

	template<std::floating_point T>
	static constexpr T asin_r[] = {T{2.967721961301243206100e-3}, T{-5.634242780008963776856e-1}, T{6.968710824104713396794e0}, T{-2.556901049652824852289e1}, T{2.853665548261061424989e1}};
	template<std::floating_point T>
	static constexpr T asin_s[] = {T{1.000000000000000000000e0}, T{-2.194779531642920639778e1}, T{1.470656354026814941758e2}, T{-3.838770957603691357202e2}, T{3.424398657913078477438e2}};
	template<std::floating_point T>
	static constexpr T asin_p[] = {T{4.253011369004428248960e-3}, T{-6.019598008014123785661e-1}, T{5.444622390564711410273e0}, T{-1.626247967210700244449e1}, T{1.956261983317594739197e1}, T{-8.198089802484824371615e0}};
	template<std::floating_point T>
	static constexpr T asin_q[] = {T{1.0}, T{-1.474091372988853791896e1}, T{7.049610280856842141659e1}, T{-1.471791292232726029859e2}, T{1.395105614657485689735e2}, T{-4.918853881490881290097e1}};
	template<std::floating_point T>
	static constexpr T atan_q[] = {T{1.0}, T{2.485846490142306297962e1}, T{1.650270098316988542046e2}, T{4.328810604912902668951e2}, T{4.853903996359136964868e2}, T{1.945506571482613964425e2}};
	template<std::floating_point T>
	static constexpr T atan_p[] = {T{-8.750608600031904122785e-1}, T{-1.615753718733365076637e1}, T{-7.500855792314704667340e1}, T{-1.228866684490136173410e2}, T{-6.485021904942025371773e1}};
	template<std::floating_point T>
	static constexpr T asin_pmin = std::same_as<T, float> ? T{1.0e-4} : T{1.0e-8};
	template<std::floating_point T>
	static constexpr T asin_off = T{6.123233995736765886130e-17};

	static constexpr double logcoff_f64[] = {-0x1.ffffffffffff7p-2, 0x1.55555555170d4p-2, -0x1.0000000399c27p-2, 0x1.999b2e90e94cap-3, -0x1.554e550bd501ep-3};
	static constexpr float logcoff_f32[] = {-0x1.00ea348b88334p-2f, 0x1.5575b0be00b6ap-2f, -0x1.ffffef20a4123p-2f};

	static constexpr std::size_t logtab_bits_f32 = 4;
	static constexpr std::size_t logtab_size_f32 = 1 << (logtab_bits_f32 + 1);
	extern const DPM_API_HIDDEN float logtab_f32[logtab_size_f32];

	static constexpr std::size_t logtab_bits_f64 = 7;
	static constexpr std::size_t logtab_size_f64 = 1 << (logtab_bits_f64 + 1);
	extern const DPM_API_HIDDEN double logtab_f64[logtab_size_f64];

	template<typename>
	struct logtab;
	template<>
	struct logtab<float> { static constexpr auto value = std::span{logtab_f32}; };
	template<>
	struct logtab<double> { static constexpr auto value = std::span{logtab_f64}; };
	template<typename T>
	static constexpr auto logtab_v = logtab<T>::value;

	static constexpr double log2coff_f64[] = {-0x1.71547652b8339p-1, 0x1.ec709dc3a04bep-2, -0x1.7154764702ffbp-2, 0x1.2776c50034c48p-2, -0x1.ec7b328ea92bcp-3, 0x1.a6225e117f92ep-3};
	static constexpr float log2coff_f32[] = {-0x1.712b6f70a7e4dp-2f, 0x1.ecabf496832ep-2f, -0x1.715479ffae3dep-1f, 0x1.715475f35c8b8p0f};

	static constexpr std::size_t log2tab_bits_f32 = 4;
	static constexpr std::size_t log2tab_size_f32 = 1 << (log2tab_bits_f32 + 1);
	extern const DPM_API_HIDDEN float log2tab_f32[log2tab_size_f32];

	static constexpr std::size_t log2tab_bits_f64 = 6;
	static constexpr std::size_t log2tab_size_f64 = 1 << (log2tab_bits_f64 + 1);
	extern const DPM_API_HIDDEN double log2tab_f64[log2tab_size_f64];

	template<typename>
	struct log2tab;
	template<>
	struct log2tab<float> { static constexpr auto value = std::span{log2tab_f32}; };
	template<>
	struct log2tab<double> { static constexpr auto value = std::span{log2tab_f64}; };
	template<typename T>
	static constexpr auto log2tab_v = log2tab<T>::value;

	template<std::floating_point T>
	static constexpr T log10_2l = std::same_as<T, float> ? T{7.9034151668e-7} : T{3.69423907715893078616e-13};
	template<std::floating_point T>
	static constexpr T log10_2h = std::same_as<T, float> ? T{3.0102920532e-1} : T{3.01029995663611771306e-1};
	template<std::floating_point T>
	static constexpr T ivln10 = T{4.34294481903251816668e-1};
	template<std::floating_point T>
	static constexpr T invln2l = T{0x1.705fc2eefa200p-33};
	template<std::floating_point T>
	static constexpr T invln2h = T{0x1.7154765200000p+0};
}