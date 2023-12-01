// These min/max operators don't follow our total order strictly. Instead
// if exactly one of the two arguments is NaN the skip_nan varieties returns
// the non-nan argument, whereas the propagate_nan varieties give the nan
// argument. If both/neither argument is NaN these extrema follow the normal
// total order.
//
// They also violate the regular total order for Option<T>: on top of the
// above rules None's are always ignored, so only if both arguments are
// None is the output None.
pub trait MinMax: Sized {
    fn min_ignore_nan(self, other: Self) -> Self;
    fn max_ignore_nan(self, other: Self) -> Self;
    fn min_propagate_nan(self, other: Self) -> Self;
    fn max_propagate_nan(self, other: Self) -> Self;
}

macro_rules! impl_trivial_min_max {
    ($T: ty) => {
        impl MinMax for $T {
            #[inline(always)]
            fn min_ignore_nan(self, other: Self) -> Self {
                self.min(other)
            }

            #[inline(always)]
            fn max_ignore_nan(self, other: Self) -> Self {
                self.max(other)
            }

            #[inline(always)]
            fn min_propagate_nan(self, other: Self) -> Self {
                self.min(other)
            }

            #[inline(always)]
            fn max_propagate_nan(self, other: Self) -> Self {
                self.max(other)
            }
        }
    };
}

// We can't do a blanket impl because Rust complains f32 might implement
// Ord someday.
impl_trivial_min_max!(bool);
impl_trivial_min_max!(u8);
impl_trivial_min_max!(u16);
impl_trivial_min_max!(u32);
impl_trivial_min_max!(u64);
impl_trivial_min_max!(u128);
impl_trivial_min_max!(usize);
impl_trivial_min_max!(i8);
impl_trivial_min_max!(i16);
impl_trivial_min_max!(i32);
impl_trivial_min_max!(i64);
impl_trivial_min_max!(i128);
impl_trivial_min_max!(isize);
impl_trivial_min_max!(char);
impl_trivial_min_max!(&str);
impl_trivial_min_max!(&[u8]);
impl_trivial_min_max!(String);

macro_rules! impl_float_min_max {
    ($T: ty) => {
        impl MinMax for $T {
            #[inline(always)]
            fn min_ignore_nan(self, other: Self) -> Self {
                <$T>::min(self, other)
            }

            #[inline(always)]
            fn max_ignore_nan(self, other: Self) -> Self {
                <$T>::max(self, other)
            }

            #[inline(always)]
            fn min_propagate_nan(self, other: Self) -> Self {
                if (self < other) | self.is_nan() {
                    self
                } else {
                    other
                }
            }

            #[inline(always)]
            fn max_propagate_nan(self, other: Self) -> Self {
                if (self > other) | self.is_nan() {
                    self
                } else {
                    other
                }
            }
        }
    };
}

impl_float_min_max!(f32);
impl_float_min_max!(f64);

#[inline(always)]
pub fn reduce_option<T, F: Fn(T, T) -> T>(a: Option<T>, b: Option<T>, f: F) -> Option<T> {
    match (a, b) {
        (Some(l), Some(r)) => Some(f(l, r)),
        (Some(l), None) => Some(l),
        (None, Some(r)) => Some(r),
        (None, None) => None,
    }
}

impl<T: MinMax> MinMax for Option<T> {
    #[inline(always)]
    fn min_ignore_nan(self, other: Self) -> Self {
        reduce_option(self, other, MinMax::min_ignore_nan)
    }

    #[inline(always)]
    fn max_ignore_nan(self, other: Self) -> Self {
        reduce_option(self, other, MinMax::max_ignore_nan)
    }

    #[inline(always)]
    fn min_propagate_nan(self, other: Self) -> Self {
        reduce_option(self, other, MinMax::min_propagate_nan)
    }

    #[inline(always)]
    fn max_propagate_nan(self, other: Self) -> Self {
        reduce_option(self, other, MinMax::max_propagate_nan)
    }
}
