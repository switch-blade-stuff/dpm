/*
 * Created by switchblade on 2023-01-10.
 */

#pragma once

#include "define.hpp"

#ifndef DPM_USE_IMPORT

#include <numbers>
#include <limits>

#endif

namespace dpm::detail
{
	template<typename T>
	constexpr static int mant_bits = std::same_as<T, double> ? 52 : 23;
	template<typename T>
	constexpr static int exp_bits = std::same_as<T, double> ? 11 : 8;

	/* In some cases x86 intrinsics generate extraneous casts if literals are used with intrinsics. As such, define commonly used values here. */
	template<std::floating_point T>
	static constexpr T five_eights = 0.625;
	template<std::floating_point T>
	static constexpr T sign_bit = -0.0;
	template<std::floating_point T>
	static constexpr T half = 0.5;
	template<std::floating_point T>
	static constexpr T one = 1.0;
	template<std::floating_point T>
	static constexpr T two = 2.0;

	template<std::floating_point T>
	static constexpr T dp_sincos[] = {T{-7.85398125648498535156e-1}, T{-3.77489470793079817668e-8}, T{-2.69515142907905952645e-15}};
	template<std::floating_point T>
	static constexpr T pio32 = std::numbers::pi_v<T> / T{32.0};
	template<std::floating_point T>
	static constexpr T pio4 = std::numbers::pi_v<T> / T{4.0};

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
	static constexpr T asin_p[] = {
			T{4.253011369004428248960e-3}, T{-6.019598008014123785661e-1},
			T{5.444622390564711410273e0}, T{-1.626247967210700244449e1},
			T{1.956261983317594739197e1}, T{-8.198089802484824371615e0}
	};
	template<std::floating_point T>
	static constexpr T asin_q[] = {
			T{1.000000000000000000000e0}, T{-1.474091372988853791896e1},
			T{7.049610280856842141659e1}, T{-1.471791292232726029859e2},
			T{1.395105614657485689735e2}, T{-4.918853881490881290097e1}
	};
	template<std::floating_point T>
	static constexpr T asin_pmin = std::same_as<T, float> ? T{1.0e-4} : T{1.0e-8};
	template<std::floating_point T>
	static constexpr T asin_off = T{6.123233995736765886130e-17};
}