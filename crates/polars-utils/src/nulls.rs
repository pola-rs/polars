pub trait IsNull {
    const HAS_NULLS: bool;
    type Inner;

    fn is_null(&self) -> bool;

    fn unwrap_inner(self) -> Self::Inner;
}

impl<T> IsNull for Option<T> {
    const HAS_NULLS: bool = true;
    type Inner = T;

    #[inline(always)]
    fn is_null(&self) -> bool {
        self.is_none()
    }

    #[inline(always)]
    fn unwrap_inner(self) -> Self::Inner {
        Option::unwrap(self)
    }
}

macro_rules! impl_is_null (
    ($($ty:tt)*) => {
        impl IsNull for $($ty)* {
            const HAS_NULLS: bool = false;
            type Inner = $($ty)*;


            #[inline(always)]
            fn is_null(&self) -> bool {
                false
            }

            #[inline(always)]
            fn unwrap_inner(self) -> $($ty)* {
                self
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
impl_is_null!(u128);

impl<'a> IsNull for &'a [u8] {
    const HAS_NULLS: bool = false;
    type Inner = &'a [u8];

    #[inline(always)]
    fn is_null(&self) -> bool {
        false
    }

    #[inline(always)]
    fn unwrap_inner(self) -> Self::Inner {
        self
    }
}

impl<'a, T: IsNull + ?Sized> IsNull for &'a T {
    const HAS_NULLS: bool = false;
    type Inner = &'a T;

    fn is_null(&self) -> bool {
        (*self).is_null()
    }

    fn unwrap_inner(self) -> Self::Inner {
        self
    }
}
