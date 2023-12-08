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
    // Comparison operators that either consider nan to be the smallest, or the
    // largest possible value. Use tot_eq for equality. Prefer directly using
    // min/max, they're slightly faster.
    fn nan_min_lt(&self, other: &Self) -> bool;
    fn nan_max_lt(&self, other: &Self) -> bool;

    // Binary operators that return either the minimum or maximum.
    #[inline(always)]
    fn min_propagate_nan(self, other: Self) -> Self {
        if self.nan_min_lt(&other) {
            self
        } else {
            other
        }
    }

    #[inline(always)]
    fn max_propagate_nan(self, other: Self) -> Self {
        if self.nan_max_lt(&other) {
            other
        } else {
            self
        }
    }

    #[inline(always)]
    fn min_ignore_nan(self, other: Self) -> Self {
        if self.nan_max_lt(&other) {
            self
        } else {
            other
        }
    }

    #[inline(always)]
    fn max_ignore_nan(self, other: Self) -> Self {
        if self.nan_min_lt(&other) {
            other
        } else {
            self
        }
    }
}

macro_rules! impl_trivial_min_max {
    ($T: ty) => {
        impl MinMax for $T {
            #[inline(always)]
            fn nan_min_lt(&self, other: &Self) -> bool {
                self < other
            }

            #[inline(always)]
            fn nan_max_lt(&self, other: &Self) -> bool {
                self < other
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
            fn nan_min_lt(&self, other: &Self) -> bool {
                !(other.is_nan() | (self >= other))
            }

            #[inline(always)]
            fn nan_max_lt(&self, other: &Self) -> bool {
                !(self.is_nan() | (self >= other))
            }

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
