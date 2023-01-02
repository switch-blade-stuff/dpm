/*
 * Created by switchblade on 2022-12-31.
 */

#pragma once

#include "define.hpp"

#ifdef SVM_DYNAMIC_DISPATCH
#ifndef SVM_USE_IMPORT

#include <atomic>

#endif

namespace svm::detail
{
	template<typename DispFunc> requires std::is_function_v<std::remove_pointer_t<std::invoke_result_t<DispFunc>>>
	class dispatcher : DispFunc
	{
		using func_t = std::remove_pointer_t<std::invoke_result_t<DispFunc>>;

	public:
		constexpr dispatcher() noexcept = default;
		constexpr dispatcher(const DispFunc &disp) : DispFunc(disp) {}
		constexpr dispatcher(DispFunc &&disp) : DispFunc(std::forward<DispFunc>(disp)) {}

		template<typename... Args>
		constexpr std::invoke_result_t<func_t, Args &&...> operator()(Args &&...args) requires std::is_invocable_v<func_t, Args &&...>
		{
			/* NOTE: It is fine to avoid thread synchronization here (or use thread locals).
			 * Synchronization will introduce unwanted overhead (bad for efficiency-oriented SIMD functions),
			 * and at worst we will see multiple threads dispatching the same function several times. */
			auto func = m_func.load();
			if (func == nullptr) [[unlikely]]
			{
				func = DispFunc{}();
				m_func.store(func);
			}
			return func(std::forward<Args>(args)...);
		}

	private:
		std::atomic<func_t *> m_func = nullptr;
	};

	template<typename DispFunc>
	dispatcher(DispFunc &&) -> dispatcher<DispFunc>;
}

#endif