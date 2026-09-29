use crate::float16::pf16;

/// (De)serialization for floats that round-trips non-finite values (NaN, inf, -inf) through
/// human-readable formats like JSON.
///
/// serde_json (and other human-readable formats) writes any non-finite f32/f64 as `null`, since
/// JSON itself has no literal for them, and then fails to read that `null` back as a float. To
/// avoid that, non-finite values are written as their `Display` string (e.g. "NaN", "inf",
/// "-inf") for human-readable formats only; binary formats are untouched, as they already encode
/// the value's bits directly and round-trip non-finite values without help.
#[cfg(feature = "serde")]
pub mod serde_nonfinite {
    macro_rules! impl_nonfinite_float_serde {
        ($mod_name:ident, $ty:ty, $serialize_fn:ident, $deserialize_fn:ident) => {
            pub mod $mod_name {
                use std::fmt;

                use serde::de::{self, Visitor};
                use serde::{Deserializer, Serializer};

                pub fn serialize<S>(v: &$ty, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: Serializer,
                {
                    if serializer.is_human_readable() && !v.is_finite() {
                        serializer.serialize_str(&v.to_string())
                    } else {
                        serializer.$serialize_fn(*v)
                    }
                }

                struct FloatVisitor;

                impl<'de> Visitor<'de> for FloatVisitor {
                    type Value = $ty;

                    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                        f.write_str(
                            "a float, or a string encoding one (e.g. \"NaN\", \"inf\", \"-inf\")",
                        )
                    }

                    fn visit_f32<E: de::Error>(self, v: f32) -> Result<Self::Value, E> {
                        Ok(v as $ty)
                    }

                    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
                        Ok(v as $ty)
                    }

                    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
                        Ok(v as $ty)
                    }

                    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                        Ok(v as $ty)
                    }

                    fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                        v.parse::<$ty>()
                            .map_err(|_| de::Error::invalid_value(de::Unexpected::Str(v), &self))
                    }
                }

                pub fn deserialize<'de, D>(deserializer: D) -> Result<$ty, D::Error>
                where
                    D: Deserializer<'de>,
                {
                    if deserializer.is_human_readable() {
                        deserializer.deserialize_any(FloatVisitor)
                    } else {
                        deserializer.$deserialize_fn(FloatVisitor)
                    }
                }
            }
        };
    }

    impl_nonfinite_float_serde!(serde_f32, f32, serialize_f32, deserialize_f32);
    impl_nonfinite_float_serde!(serde_f64, f64, serialize_f64, deserialize_f64);
}

/// # Safety
/// Unsafe code downstream relies on the correct is_float call.
pub unsafe trait IsFloat: private::Sealed + Sized {
    #[inline]
    fn is_float() -> bool {
        false
    }

    #[inline]
    fn is_f16() -> bool {
        false
    }

    #[inline]
    fn is_f32() -> bool {
        false
    }

    #[inline]
    fn is_f64() -> bool {
        false
    }

    fn nan_value() -> Self {
        unimplemented!()
    }

    fn pos_inf_value() -> Self {
        unimplemented!()
    }

    fn neg_inf_value() -> Self {
        unimplemented!()
    }

    #[allow(clippy::wrong_self_convention)]
    #[inline]
    fn is_nan(&self) -> bool
    where
        Self: Sized,
    {
        false
    }
    #[allow(clippy::wrong_self_convention)]
    #[inline]
    fn is_finite(&self) -> bool
    where
        Self: Sized,
    {
        true
    }
}

unsafe impl IsFloat for i8 {}
unsafe impl IsFloat for i16 {}
unsafe impl IsFloat for i32 {}
unsafe impl IsFloat for i64 {}
unsafe impl IsFloat for i128 {}
unsafe impl IsFloat for u8 {}
unsafe impl IsFloat for u16 {}
unsafe impl IsFloat for u32 {}
unsafe impl IsFloat for u64 {}
unsafe impl IsFloat for u128 {}
unsafe impl IsFloat for usize {}
unsafe impl IsFloat for &str {}
unsafe impl IsFloat for &[u8] {}
unsafe impl IsFloat for bool {}
unsafe impl<T: IsFloat> IsFloat for Option<T> {}

mod private {
    use super::*;

    pub trait Sealed {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for i128 {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for u128 {}
    impl Sealed for usize {}
    impl Sealed for pf16 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for &str {}
    impl Sealed for &[u8] {}
    impl Sealed for bool {}
    impl<T: Sealed> Sealed for Option<T> {}
}

macro_rules! impl_is_float {
    ($tp:ty, $is_f16:literal, $is_f32:literal, $is_f64:literal) => {
        unsafe impl IsFloat for $tp {
            #[inline]
            fn is_float() -> bool {
                true
            }

            #[inline]
            fn is_f16() -> bool {
                $is_f16
            }

            #[inline]
            fn is_f32() -> bool {
                $is_f32
            }

            #[inline]
            fn is_f64() -> bool {
                $is_f64
            }

            #[inline]
            fn nan_value() -> Self {
                Self::NAN
            }

            #[inline]
            fn pos_inf_value() -> Self {
                Self::INFINITY
            }

            #[inline]
            fn neg_inf_value() -> Self {
                Self::NEG_INFINITY
            }

            #[inline]
            fn is_nan(&self) -> bool {
                <$tp>::is_nan(*self)
            }

            #[inline]
            fn is_finite(&self) -> bool {
                <$tp>::is_finite(*self)
            }
        }
    };
}

impl_is_float!(pf16, true, false, false);
impl_is_float!(f32, false, true, false);
impl_is_float!(f64, false, false, true);
