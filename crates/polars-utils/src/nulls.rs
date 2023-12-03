pub trait IsNull {
    fn is_null(&self) -> bool;
}

impl<T> IsNull for Option<T> {
    #[inline(always)]
    fn is_null(&self) -> bool {
        self.is_none()
    }
}

macro_rules! impl_is_null (
    ($($ty:tt)*) => {
        impl IsNull for $($ty)* {
            #[inline(always)]
            fn is_null(&self) -> bool {
                false
            }
        }
    };
);

impl_is_null!(bool);
impl_is_null!(f32);
impl_is_null!(f64);
impl_is_null!(i8);
impl_is_null!(i16);
impl_is_null!(i32);
impl_is_null!(i64);
impl_is_null!(i128);
impl_is_null!(u8);
impl_is_null!(u16);
impl_is_null!(u32);
impl_is_null!(u64);
impl_is_null!(&[u8]);

impl<T: IsNull + ?Sized> IsNull for &T {
    fn is_null(&self) -> bool {
        (*self).is_null()
    }
}
