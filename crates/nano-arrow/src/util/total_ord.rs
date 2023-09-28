use std::cmp::Ordering;

use bytemuck::TransparentWrapper;

use crate::array::Array;

/// Alternative trait for Ord. By consistently using this we can still be
/// generic w.r.t Ord while getting a total ordering for floats.
pub trait TotalEq {
    fn tot_eq(&self, other: &Self) -> bool;

    #[inline(always)]
    fn tot_ne(&self, other: &Self) -> bool {
        !(self.tot_eq(other))
    }
}

/// Alternative traits for Eq. By consistently using this we can still be
/// generic w.r.t Eq while getting a total ordering for floats.
pub trait TotalOrd: TotalEq {
    fn tot_cmp(&self, other: &Self) -> Ordering;

    #[inline(always)]
    fn tot_lt(&self, other: &Self) -> bool {
        self.tot_cmp(other) == Ordering::Less
    }

    #[inline(always)]
    fn tot_gt(&self, other: &Self) -> bool {
        self.tot_cmp(other) == Ordering::Greater
    }

    #[inline(always)]
    fn tot_le(&self, other: &Self) -> bool {
        self.tot_cmp(other) != Ordering::Greater
    }

    #[inline(always)]
    fn tot_ge(&self, other: &Self) -> bool {
        self.tot_cmp(other) != Ordering::Less
    }
}

#[repr(transparent)]
pub struct TotalOrdWrap<T>(T);
unsafe impl<T> TransparentWrapper<T> for TotalOrdWrap<T> {}

impl<T: TotalEq + TotalOrd> PartialOrd for TotalOrdWrap<T> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }

    #[inline(always)]
    fn lt(&self, other: &Self) -> bool {
        self.0.tot_lt(&other.0)
    }

    #[inline(always)]
    fn le(&self, other: &Self) -> bool {
        self.0.tot_le(&other.0)
    }

    #[inline(always)]
    fn gt(&self, other: &Self) -> bool {
        self.0.tot_gt(&other.0)
    }

    #[inline(always)]
    fn ge(&self, other: &Self) -> bool {
        self.0.tot_ge(&other.0)
    }
}

impl<T: TotalEq + TotalOrd> Ord for TotalOrdWrap<T> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.tot_cmp(&other.0)
    }
}

impl<T: TotalEq + TotalOrd> PartialEq for TotalOrdWrap<T> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0.tot_eq(&other.0)
    }

    #[inline(always)]
    #[allow(clippy::partialeq_ne_impl)]
    fn ne(&self, other: &Self) -> bool {
        self.0.tot_ne(&other.0)
    }
}

impl<T: TotalEq + TotalOrd> Eq for TotalOrdWrap<T> {}

macro_rules! impl_trivial_eq {
    ($T: ty) => {
        impl TotalEq for $T {
            #[inline(always)]
            fn tot_eq(&self, other: &Self) -> bool {
                self == other
            }

            #[inline(always)]
            fn tot_ne(&self, other: &Self) -> bool {
                self != other
            }
        }
    };
}

macro_rules! impl_trivial_eq_ord {
    ($T: ty) => {
        impl_trivial_eq!($T);

        impl TotalOrd for $T {
            #[inline(always)]
            fn tot_cmp(&self, other: &Self) -> Ordering {
                self.cmp(other)
            }

            #[inline(always)]
            fn tot_lt(&self, other: &Self) -> bool {
                self < other
            }

            #[inline(always)]
            fn tot_gt(&self, other: &Self) -> bool {
                self > other
            }

            #[inline(always)]
            fn tot_le(&self, other: &Self) -> bool {
                self <= other
            }

            #[inline(always)]
            fn tot_ge(&self, other: &Self) -> bool {
                self >= other
            }
        }
    };
}

// We can't do a blanket impl because Rust complains f32 might implement
// Ord / Eq someday.
impl_trivial_eq_ord!(bool);
impl_trivial_eq_ord!(u8);
impl_trivial_eq_ord!(u16);
impl_trivial_eq_ord!(u32);
impl_trivial_eq_ord!(u64);
impl_trivial_eq_ord!(u128);
impl_trivial_eq_ord!(usize);
impl_trivial_eq_ord!(i8);
impl_trivial_eq_ord!(i16);
impl_trivial_eq_ord!(i32);
impl_trivial_eq_ord!(i64);
impl_trivial_eq_ord!(i128);
impl_trivial_eq_ord!(isize);
impl_trivial_eq_ord!(char);
impl_trivial_eq_ord!(&str);
impl_trivial_eq_ord!(&[u8]);
impl_trivial_eq_ord!(String);
impl_trivial_eq!(&dyn Array);
impl_trivial_eq!(Box<dyn Array>);

macro_rules! impl_polars_eq_ord_float {
    ($f:ty) => {
        impl TotalEq for $f {
            #[inline(always)]
            fn tot_eq(&self, other: &Self) -> bool {
                self.to_bits() == other.to_bits()
            }

            #[inline(always)]
            fn tot_ne(&self, other: &Self) -> bool {
                self.to_bits() != other.to_bits()
            }
        }

        impl TotalOrd for $f {
            #[inline(always)]
            fn tot_cmp(&self, other: &Self) -> Ordering {
                self.total_cmp(other)
            }

            #[inline(always)]
            fn tot_lt(&self, other: &Self) -> bool {
                self.total_cmp(other) == Ordering::Less
            }

            #[inline(always)]
            fn tot_gt(&self, other: &Self) -> bool {
                self.total_cmp(other) == Ordering::Greater
            }

            #[inline(always)]
            fn tot_le(&self, other: &Self) -> bool {
                self.total_cmp(other) != Ordering::Greater
            }

            #[inline(always)]
            fn tot_ge(&self, other: &Self) -> bool {
                self.total_cmp(other) != Ordering::Less
            }
        }
    };
}

impl_polars_eq_ord_float!(f32);
impl_polars_eq_ord_float!(f64);

// Blanket implementations.
impl<T: TotalEq> TotalEq for Option<T> {
    #[inline(always)]
    fn tot_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (None, None) => true,
            (Some(a), Some(b)) => a.tot_eq(b),
            _ => false,
        }
    }

    #[inline(always)]
    fn tot_ne(&self, other: &Self) -> bool {
        match (self, other) {
            (None, None) => false,
            (Some(a), Some(b)) => a.tot_ne(b),
            _ => true,
        }
    }
}

impl<T: TotalOrd> TotalOrd for Option<T> {
    #[inline(always)]
    fn tot_cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            (Some(a), Some(b)) => a.tot_cmp(b),
        }
    }

    #[inline(always)]
    fn tot_lt(&self, other: &Self) -> bool {
        match (self, other) {
            (None, Some(_)) => true,
            (Some(a), Some(b)) => a.tot_lt(b),
            _ => false,
        }
    }

    #[inline(always)]
    fn tot_gt(&self, other: &Self) -> bool {
        other.tot_lt(self)
    }

    #[inline(always)]
    fn tot_le(&self, other: &Self) -> bool {
        match (self, other) {
            (Some(_), None) => false,
            (Some(a), Some(b)) => a.tot_lt(b),
            _ => true,
        }
    }

    #[inline(always)]
    fn tot_ge(&self, other: &Self) -> bool {
        other.tot_le(self)
    }
}

impl<T: TotalEq + ?Sized> TotalEq for &T {
    #[inline(always)]
    fn tot_eq(&self, other: &Self) -> bool {
        (*self).tot_eq(*other)
    }

    #[inline(always)]
    fn tot_ne(&self, other: &Self) -> bool {
        (*self).tot_ne(*other)
    }
}

impl<T: TotalOrd + ?Sized> TotalOrd for &T {
    #[inline(always)]
    fn tot_cmp(&self, other: &Self) -> Ordering {
        (*self).tot_cmp(*other)
    }

    #[inline(always)]
    fn tot_lt(&self, other: &Self) -> bool {
        (*self).tot_lt(*other)
    }

    #[inline(always)]
    fn tot_gt(&self, other: &Self) -> bool {
        (*self).tot_gt(*other)
    }

    #[inline(always)]
    fn tot_le(&self, other: &Self) -> bool {
        (*self).tot_le(*other)
    }

    #[inline(always)]
    fn tot_ge(&self, other: &Self) -> bool {
        (*self).tot_ge(*other)
    }
}
