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
