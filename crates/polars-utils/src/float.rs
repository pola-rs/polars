use std::cmp::Ordering;

use num_traits::ToPrimitive;

use crate::ord::compare_fn_nan_max;

/// # Safety
/// unsafe code downstream relies on the correct is_float call
pub unsafe trait IsFloat: private::Sealed {
    fn is_float() -> bool {
        false
    }

    fn is_f32() -> bool {
        false
    }

    fn is_f64() -> bool {
        false
    }

    #[allow(clippy::wrong_self_convention)]
    fn is_nan(&self) -> bool
    where
        Self: Sized,
    {
        false
    }
    #[allow(clippy::wrong_self_convention)]
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
unsafe impl IsFloat for &str {}
unsafe impl IsFloat for &[u8] {}
unsafe impl IsFloat for bool {}
unsafe impl<T: IsFloat> IsFloat for Option<T> {}

mod private {
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
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for &str {}
    impl Sealed for &[u8] {}
    impl Sealed for bool {}
    impl<T: Sealed> Sealed for Option<T> {}
}

macro_rules! impl_is_float {
    ($tp:ty, $is_f32:literal, $is_f64:literal) => {
        unsafe impl IsFloat for $tp {
            #[inline]
            fn is_float() -> bool {
                true
            }

            fn is_f32() -> bool {
                $is_f32
            }

            fn is_f64() -> bool {
                $is_f64
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

impl_is_float!(f32, true, false);
impl_is_float!(f64, false, true);

/// A utility type that make floats Ord by
/// nan == nan == true
/// nan > float::max == true
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct OrdFloat<T>(T);

impl<T: IsFloat + PartialEq + PartialOrd> PartialOrd for OrdFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: IsFloat + PartialEq + PartialOrd> Ord for OrdFloat<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_fn_nan_max(&self.0, &other.0)
    }
}

impl<T: IsFloat + PartialEq> PartialEq for OrdFloat<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self.0.is_nan(), other.0.is_nan()) {
            (true, true) => true,
            _ => self.0 == other.0,
        }
    }
}

impl<T: PartialEq + IsFloat> Eq for OrdFloat<T> {}

impl<T: ToPrimitive> ToPrimitive for OrdFloat<T> {
    fn to_isize(&self) -> Option<isize> {
        self.0.to_isize()
    }

    fn to_i8(&self) -> Option<i8> {
        self.0.to_i8()
    }

    fn to_i16(&self) -> Option<i16> {
        self.0.to_i16()
    }

    fn to_i32(&self) -> Option<i32> {
        self.0.to_i32()
    }

    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    fn to_usize(&self) -> Option<usize> {
        self.0.to_usize()
    }

    fn to_u8(&self) -> Option<u8> {
        self.0.to_u8()
    }

    fn to_u16(&self) -> Option<u16> {
        self.0.to_u16()
    }

    fn to_u32(&self) -> Option<u32> {
        self.0.to_u32()
    }

    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }

    fn to_f32(&self) -> Option<f32> {
        self.0.to_f32()
    }

    fn to_f64(&self) -> Option<f64> {
        self.0.to_f64()
    }
}

pub fn f32_to_ordablef32(vals: &mut [f32]) -> &mut [OrdFloat<f32>] {
    unsafe { std::mem::transmute(vals) }
}

pub fn f64_to_ordablef64(vals: &mut [f64]) -> &mut [OrdFloat<f64>] {
    unsafe { std::mem::transmute(vals) }
}
