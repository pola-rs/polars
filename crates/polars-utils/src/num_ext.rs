use num_traits::Num;

pub trait NumExt {
    type Abs: Num + PartialOrd + Copy + std::fmt::Debug + Send + Sync;

    fn floor_div(self, other: Self) -> Self;

    fn max_abs_diff() -> Self::Abs;
    fn abs_diff(self, other: Self) -> Self::Abs;
}

macro_rules! impl_float_num_ext {
    ($T: ty) => {
        impl NumExt for $T {
            type Abs = $T;

            fn floor_div(self, other: Self) -> Self {
                (self / other).floor()
            }

            fn max_abs_diff() -> Self::Abs {
                <$T>::INFINITY
            }

            fn abs_diff(self, other: Self) -> Self::Abs {
                if self > other {
                    self - other
                } else {
                    other - self
                }
            }
        }
    };
}

macro_rules! impl_unsigned_num_ext {
    ($T: ty) => {
        impl NumExt for $T {
            type Abs = $T;

            fn floor_div(self, other: Self) -> Self {
                self / other
            }

            fn max_abs_diff() -> Self::Abs {
                <$T>::MAX
            }

            fn abs_diff(self, other: Self) -> Self::Abs {
                if self > other {
                    self - other
                } else {
                    other - self
                }
            }
        }
    };
}

macro_rules! impl_signed_num_ext {
    ($T: ty, $U: ty) => {
        impl NumExt for $T {
            type Abs = $U;

            fn floor_div(self, other: Self) -> Self {
                let d = self / other;
                let r = self % other;
                if (r > 0 && other < 0) || (r < 0 && other > 0) {
                    d - 1
                } else {
                    d
                }
            }

            fn max_abs_diff() -> Self::Abs {
                <$U>::MAX
            }

            fn abs_diff(self, other: Self) -> Self::Abs {
                self.abs_diff(other)
            }
        }
    };
}

impl_unsigned_num_ext!(u8);
impl_unsigned_num_ext!(u16);
impl_unsigned_num_ext!(u32);
impl_unsigned_num_ext!(u64);
impl_unsigned_num_ext!(u128);
impl_unsigned_num_ext!(usize);
impl_float_num_ext!(f32);
impl_float_num_ext!(f64);
impl_signed_num_ext!(i8, u8);
impl_signed_num_ext!(i16, u16);
impl_signed_num_ext!(i32, u32);
impl_signed_num_ext!(i64, u64);
impl_signed_num_ext!(i128, u128);
impl_signed_num_ext!(isize, usize);
