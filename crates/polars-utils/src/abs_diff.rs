use num_traits::Num;

pub trait AbsDiff {
    type Abs: Num + PartialOrd + Copy + std::fmt::Debug + Send + Sync;

    fn max_abs_diff() -> Self::Abs;
    fn abs_diff(self, other: Self) -> Self::Abs;
}

macro_rules! impl_trivial_abs_diff {
    ($T: ty, $max: expr) => {
        impl AbsDiff for $T {
            type Abs = $T;

            fn max_abs_diff() -> Self::Abs {
                $max
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

macro_rules! impl_signed_abs_diff {
    ($T: ty, $U: ty) => {
        impl AbsDiff for $T {
            type Abs = $U;

            fn max_abs_diff() -> Self::Abs {
                <$U>::MAX
            }

            fn abs_diff(self, other: Self) -> Self::Abs {
                self.abs_diff(other)
            }
        }
    };
}

impl_trivial_abs_diff!(u8, u8::MAX);
impl_trivial_abs_diff!(u16, u16::MAX);
impl_trivial_abs_diff!(u32, u32::MAX);
impl_trivial_abs_diff!(u64, u64::MAX);
impl_trivial_abs_diff!(u128, u128::MAX);
impl_trivial_abs_diff!(usize, usize::MAX);
impl_trivial_abs_diff!(f32, f32::INFINITY);
impl_trivial_abs_diff!(f64, f64::INFINITY);
impl_signed_abs_diff!(i8, u8);
impl_signed_abs_diff!(i16, u16);
impl_signed_abs_diff!(i32, u32);
impl_signed_abs_diff!(i64, u64);
impl_signed_abs_diff!(i128, u128);
impl_signed_abs_diff!(isize, usize);
