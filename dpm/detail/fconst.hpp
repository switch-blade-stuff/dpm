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
	template<std::floating_point T>
	static constexpr T dp_sincos[] = {T{-7.85398125648498535156e-1}, T{-3.77489470793079817668e-8}, T{-2.69515142907905952645e-15}};
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
	static constexpr T dp_tancot[] = {T{-7.853981554508209228515625e-1L}, T{-7.853981554508209228515625e-9L}, T{-7.853981554508209228515625e-17L}};
	template<std::floating_point T>
	static constexpr T tancot_q[] = {T{1.00000000000000000000e0L}, T{1.3681296347069295467845e4L}, T{-1.3208923444021096744731e6L}, T{2.5008380182335791583922e7L}, T{-5.3869575592945462988123e7L}};
	template<std::floating_point T>
	static constexpr T tancot_p[] = {T{-1.3093693918138377764608e4L}, T{1.1535166483858741613983e6L}, T{-1.7956525197648487798769e7L}};
	template<std::floating_point T>
	static constexpr T tancot_pmin = std::same_as<T, float> ? T{1.0e-4} : std::same_as<T, double> ? T{1.0e-14} : T{1.0e-20L};

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