#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(feature = "simd", feature(avx512_target_feature))]
#![cfg_attr(
    all(feature = "simd", target_arch = "x86_64"),
    feature(stdarch_x86_avx512)
)]

pub mod arithmetic;
pub mod comparisons;
pub mod filter;
pub mod min_max;

pub mod arity;
