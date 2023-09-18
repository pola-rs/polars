use std::convert::TryInto;

use super::{set, Simd8, Simd8Lanes, Simd8PartialEq, Simd8PartialOrd};
use crate::types::{days_ms, f16, i256, months_days_ns};

simd8_native_all!(u8);
simd8_native_all!(u16);
simd8_native_all!(u32);
simd8_native_all!(u64);
simd8_native_all!(i8);
simd8_native_all!(i16);
simd8_native_all!(i32);
simd8_native_all!(i128);
simd8_native_all!(i256);
simd8_native_all!(i64);
simd8_native!(f16);
simd8_native_partial_eq!(f16);
simd8_native_all!(f32);
simd8_native_all!(f64);
simd8_native!(days_ms);
simd8_native_partial_eq!(days_ms);
simd8_native!(months_days_ns);
simd8_native_partial_eq!(months_days_ns);
