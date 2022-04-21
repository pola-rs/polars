pub trait IsFloat {
    fn is_float() -> bool {
        false
    }

    #[allow(clippy::wrong_self_convention)]
    fn is_nan(self) -> bool
    where
        Self: Sized,
    {
        false
    }
}

impl IsFloat for i8 {}
impl IsFloat for i16 {}
impl IsFloat for i32 {}
impl IsFloat for i64 {}
impl IsFloat for u8 {}
impl IsFloat for u16 {}
impl IsFloat for u32 {}
impl IsFloat for u64 {}

macro_rules! impl_is_float {
    ($tp:ty) => {
        impl IsFloat for $tp {
            fn is_float() -> bool {
                true
            }

            fn is_nan(self) -> bool {
                <$tp>::is_nan(self)
            }
        }
    };
}

impl_is_float!(f32);
impl_is_float!(f64);
