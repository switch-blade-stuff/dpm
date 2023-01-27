# Data-Parralel Math

This library implements a feature-complete version of the C++ data-parallel math library TS with optional extensions.

## SIMD intrinsics support

DPM currently supports the following architectures for SIMD operations:

- x86
    - SSE
    - SSE2
    - SSE3
    - SSSE3
    - SSE4.1
    - SSE4.2
    - AVX
    - AVX2
    - FMA
    - AVX512 (see notes)
- ARM Neon

On architectures without SIMD intrinsic support, vectorization is emulated via scalar operations.

## Library options

<table>
  <tr><th>#define macro</th><th>CMake option</th><th>Default value</th><th>Description</th></tr>
  <tr>
    <td>DPM_USE_MODULES</td>
    <td>-DDPM_USE_MODULES</td>
    <td>OFF</td>
    <td>Toggles support for C++20 modules</td>
  </tr>
  <tr>
    <td>DPM_INLINE_EXTENSIONS</td>
    <td>-DDPM_INLINE_EXTENSIONS</td>
    <td>ON</td>
    <td>Toggles inlining of the library extension namespace (see notes)</td>
  </tr>
  <tr>
    <td>DPM_DYNAMIC_DISPATCH</td>
    <td>-DDPM_DYNAMIC_DISPATCH</td>
    <td>ON</td>
    <td>Toggles runtime detection & dispatch of SIMD operations</td>
  </tr>
  <tr>
    <td>DPM_HANDLE_ERRORS</td>
    <td>-DDPM_HANDLE_ERRORS</td>
    <td>ON</td>
    <td>Toggles detection & reporting of math errors via <a hred="https://en.cppreference.com/w/cpp/numeric/math/math_errhandling">math_errhandling</a> (see notes)</td>
  </tr>
  <tr>
    <td>DPM_PROPAGATE_NAN</td>
    <td>-DDPM_PROPAGATE_NAN</td>
    <td>ON</td>
    <td>Toggles guaranteed propagation of NaN (see notes)</td>
  </tr>
  <tr>
    <td>N/A</td>
    <td>-DDPM_USE_IPO</td>
    <td>ON</td>
    <td>Toggles support for inter-procedural optimization</td>
  </tr>
  <tr>
    <td>N/A</td>
    <td>-DDPM_TESTS</td>
    <td>OFF</td>
    <td>Enables unit test target</td>
  </tr>
</table>

Note that the default value applies only to CMake options.

## Build & usage instructions

In order to build the library using CMake, run the following commands:

```shell
mkdir build
cd build
cmake ..
cmake --build .
```

Build artifacts will be found in `build/bin` and `build/lib`. Minimum required CMake version is 3.20

In order to use the library as a CMake link dependency, you must link to one of the following targets:

* `dpm-interface` - interface headers of the library.
* `dpm` - static or shared library target, depending on the value of `BUILD_SHARED_LIBS`.

## Notes

### Extensions

DPM provides the following extensions to the standard API:

* ABI tags
    * `aligned_vector`
    * `packed_buffer`
    * `common`
    * x86
        * `sse`
        * `avx`
    * ARM
        * `neon`
* Storage traits & accessors
    * `native_data_type`
    * `native_data_size`
    * `std::span to_native_data(simd &)`
    * `std::span to_native_data(const simd &)`
    * `std::span to_native_data(simd_mask &)`
    * `std::span to_native_data(const simd_mask &)`
* Blend functions
    * `simd blend(const simd &, const simd &, const simd_mask &)`
    * `simd blend(const simd &, const const_where_expression &)`
    * `simd_mask blend(const simd_mask &, const simd_mask &, const simd_mask &)`
    * `simd_mask blend(const simd_mask &, const const_where_expression &)`
    * `V blend(const V &, const V &, /* bool-wrapper */)`
    * `V blend(const V &, const const_where_expression &)`
* Shuffle functions
    * `simd shuffle<Is...>(const simd &)`
    * `simd_mask shuffle<Is...>(const simd_mask &)`
    * `simd_mask shuffle<Is...>(const V &)`
* Other utilities
    * `cpuid`

All extensions are available from the `dpm::ext` and `dpm::simd_abi::ext` namespaces. If `DPM_INLINE_EXTENSIONS` option
is enabled, the `ext` namespaces are declared as inline.

### Dynamic dispatch

By default, DPM will attempt to automatically detect SIMD support and dispatch vectorized functions appropriately on
supported CPUs. This enables the library to support multiple tiers of hardware with the same binary. This, however, may
increase binary size and add additional overhead to every dispatched operation. In order to disable dynamic dispatch,
use the `DPM_DYNAMIC_DISPATCH` option.

Note that inline functions such as operators, reductions, min/max algorithms, etc. are never dispatched. Additionally,
dispatched functions require a minimum SIMD level to be used for ABI compatibility. For example, functions operating
on AVX vectors require the consumer to have AVX enabled at compile-time, regardless of runtime availability.

**Do not manually specify architecture options for DPM target if `DPM_DYNAMIC_DISPATCH` is enabled. Doing so may mess up
code generation and lead to crashes.** This happens because most compilers give preference to the highest architecture
level specified. As such, for example, if AVX is enabled from command-line arguments, all functions will be allowed to
use AVX instructions, even if those functions are meant to be SSE2-only, causing unintended behavior in situations
like `vaddpd` instruction being emitted for SSE2 functions when `addpd` was intended instead.
See [this godbolt example](https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGISVykrgAyeAyYAHI%2BAEaYxBKBAA6oCoRODB7evv5JKWkCoeFRLLHxXIF2mA7pQgRMxASZPn4Btpj2jgK19QSFkTFxCbZ1DU3ZrQojvWH9JYPlAJS2qF7EyOwcJhoAguYAzGHI3lgA1CZ7bngsLGEExGEAdAjn2Fu7OwD6HyxcZgAc6BOXyYBDueGiXgImC%2BUBGwEwBAg5jMCiUZmRC0xJyY6HQEC%2BP3%2BgKYpCB31%2BAJO0QWbxMAHYrDsTsyTgB6ABUJwAkoJXJhAWFJsQvNUBCc8ApsbjEoD2ay3iyTsQEasGEDrh8cegPjKICSqTS9ozdnSACJvAkUwHA0H3CFQmEQOEIpFmMxMABuqgxWK1%2BPJROxpMtgeptIZCpZHO5vIYWAFDCFIs6aolJw9WplJzlkeZyoIqvVLE1uJ1eP1YaN4dNHCWtE4AFZeH4OFpSKhOG5rNYTgoVmtMGczHseKQCJpa0sANYgABsZgee0kfz%2BezMGg0dL%2BAS4AE5Z/pOJJeCwJBvSC22x2OLwFCANGOJ0s4LAYIgUKgWIk6HFyJQ0F%2BP7xEchjAFwDbnlgHp4OsABqeCYAA7gA8okjCcKONC0FCxB3hA0QTqQ0RhPUACeGG8MRzDEKRyHRNoVTjtwvAAWwgjIQwtDka2vBYCwoHiDxpD4Mq1Qepgd5CZgqhVJCGyjrc7SEbQ4LEGRHhYIRYKnsxSxUAYwAKPBSGoehzEyIIIhiOwUgWfIShqIRuhmPooEoN2lj6OCd6QEsqCJCmkkALSTOg5ymqYljWOuJxBchZixfxXqJaimBmLe7SMekLhxmMfiBCEMzFKUejJKkKZ5aVeQpn0xXzG0HQ1FMlUVJlooMN0DS1QMZTDD0LV9V1RU9RISx9qs6yjYeHBNhehHXicqh/LOQWzpIJwgUYJzgQ8Gi7ScEC4IQJBDiOCy8ExWiYqQM57HSDyzg25SznSu67mY4Eri59YcMepCnhBc1Cdet73o%2BPHPm%2BEBIAB370GQFAQLDQEgGBbrCZg0FwQhKFoS2mF0DheEEUJVFkRRREkTRdEMQ4FOsYwBAcVxhF8QJGxtiJWXiZJbbSbJUIU4pP1tip0RqTRGkcxd9w6TwekGUZOOmfjvD8JZojiLZ6v2So6hCbogQGEY7lRZ5Ys%2BRAfkBekwWheFkVWJYMVxQlQVJaoKVohljXOBArgDYVRQjbk5XpANZX5Aw3VzL1lTtZ1jSeM0ejxymicxyVFTNcn2TZz0mfzGN/aTVwdaNs282cIty2retwDIMg20Lglh34EQxCnWXF1PksCCYDigxW9Nf0Aw%2Bl68CDthg5dk43SAd0PU9XAvW9H0Nl9017JXwOcD3EPlxw6VA1ee/g1dSzibh2WSEAA)
for details.

### Error handling & NaN

The standard specifies that floating-point math functions such as `sin`, `cos`, etc. must report math errors via the
mechanism specified in [math_errhandling](https://en.cppreference.com/w/cpp/numeric/math/math_errhandling).
When `DPM_HANDLE_ERRORS` option is enabled, DPM catches and reports errors as specified by the standard, however this
does reduce efficiency of math functions due to the additional safety checks. If performance is preferred over accuracy,
disable `DPM_HANDLE_ERRORS`.

Additionally, if `DPM_PROPAGATE_NAN` option is enabled, the library guarantees that math functions will propagate any
NaN inputs, which may be used as a form of error handling. If `DPM_PROPAGATE_NAN` is disabled, invoking math functions
with `NaN` as input will result in undefined behavior unless otherwise specified.

### AVX512

While DPM does utilize AVX512 instructions for 128- and 256-bit operations, there is no support for 512-wide vector data
types. The main reasons being the increased complexity of implementation due to both the fracturing of AVX512 standard,
and complexity of most 512-bit wide instructions (ex. there is no 512 blend, and it must be emulated via more complex
fused operations); as well as relative inefficiency of 512-bit wide registers (for general-purpose use cases) on certain
CPUs. See the following articles for details:

- [https://lemire.me/blog/2018/09/07/avx-512-when-and-how-to-use-these-new-instructions/](https://lemire.me/blog/2018/09/07/avx-512-when-and-how-to-use-these-new-instructions/)
- [https://news.ycombinator.com/item?id=21031905](https://news.ycombinator.com/item?id=21031905)
- [https://www.phoronix.com/news/Linus-Torvalds-On-AVX-512](https://www.phoronix.com/news/Linus-Torvalds-On-AVX-512)