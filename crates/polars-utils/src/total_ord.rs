use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use bytemuck::TransparentWrapper;

/// Converts an f32 into a canonical form, where -0 == 0 and all NaNs map to
/// the same value.
pub fn canonical_f32(x: f32) -> f32 {
    // -0.0 + 0.0 becomes 0.0.
    let convert_zero = x + 0.0;
    if convert_zero.is_nan() {
        f32::from_bits(0x7fc00000) // Canonical quiet NaN.
    } else {
        convert_zero
    }
}

/// Converts an f64 into a canonical form, where -0 == 0 and all NaNs map to
/// the same value.
pub fn canonical_f64(x: f64) -> f64 {
    // -0.0 + 0.0 becomes 0.0.
    let convert_zero = x + 0.0;
    if convert_zero.is_nan() {
        f64::from_bits(0x7ff8000000000000) // Canonical quiet NaN.
    } else {
        convert_zero
    }
}

/// Alternative trait for Eq. By consistently using this we can still be
/// generic w.r.t Eq while getting a total ordering for floats.
pub trait TotalEq {
    fn tot_eq(&self, other: &Self) -> bool;

    #[inline(always)]
    fn tot_ne(&self, other: &Self) -> bool {
        !(self.tot_eq(other))
    }
}

/// Alternative trait for Ord. By consistently using this we can still be
/// generic w.r.t Ord while getting a total ordering for floats.
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

/// Alternative trait for Hash. By consistently using this we can still be
/// generic w.r.t Hash while being able to hash floats.
pub trait TotalHash {
    fn tot_hash<H>(&self, state: &mut H)
    where
        H: Hasher;

    fn tot_hash_slice<H>(data: &[Self], state: &mut H)
    where
        H: Hasher,
        Self: Sized,
    {
        for piece in data {
            piece.tot_hash(state)
        }
    }
}

#[repr(transparent)]
pub struct TotalOrdWrap<T>(pub T);
unsafe impl<T> TransparentWrapper<T> for TotalOrdWrap<T> {}

impl<T: TotalOrd> PartialOrd for TotalOrdWrap<T> {
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

impl<T: TotalOrd> Ord for TotalOrdWrap<T> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.tot_cmp(&other.0)
    }
}

impl<T: TotalEq> PartialEq for TotalOrdWrap<T> {
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

impl<T: TotalEq> Eq for TotalOrdWrap<T> {}

impl<T: TotalHash> Hash for TotalOrdWrap<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.tot_hash(state);
    }
}

macro_rules! impl_trivial_total {
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

        impl TotalHash for $T {
            fn tot_hash<H>(&self, state: &mut H)
            where
                H: Hasher,
            {
                self.hash(state);
            }
        }
    };
}

// We can't do a blanket impl because Rust complains f32 might implement
// Ord / Eq someday.
impl_trivial_total!(bool);
impl_trivial_total!(u8);
impl_trivial_total!(u16);
impl_trivial_total!(u32);
impl_trivial_total!(u64);
impl_trivial_total!(u128);
impl_trivial_total!(usize);
impl_trivial_total!(i8);
impl_trivial_total!(i16);
impl_trivial_total!(i32);
impl_trivial_total!(i64);
impl_trivial_total!(i128);
impl_trivial_total!(isize);
impl_trivial_total!(char);
impl_trivial_total!(&str);
impl_trivial_total!(&[u8]);
impl_trivial_total!(String);

macro_rules! impl_float_eq_ord {
    ($T:ty) => {
        impl TotalEq for $T {
            #[inline(always)]
            fn tot_eq(&self, other: &Self) -> bool {
                if self.is_nan() {
                    other.is_nan()
                } else {
                    self == other
                }
            }
        }

        impl TotalOrd for $T {
            #[inline(always)]
            fn tot_cmp(&self, other: &Self) -> Ordering {
                if self.tot_lt(other) {
                    Ordering::Less
                } else if self.tot_gt(other) {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            }

            #[inline(always)]
            fn tot_lt(&self, other: &Self) -> bool {
                !self.tot_ge(other)
            }

            #[inline(always)]
            fn tot_gt(&self, other: &Self) -> bool {
                other.tot_lt(self)
            }

            #[inline(always)]
            fn tot_le(&self, other: &Self) -> bool {
                other.tot_ge(self)
            }

            #[inline(always)]
            fn tot_ge(&self, other: &Self) -> bool {
                // We consider all NaNs equal, and NaN is the largest possible
                // value. Thus if self is NaN we always return true. Otherwise
                // self >= other is correct. If other is not NaN it is trivially
                // correct, and if it is we note that nothing can be greater or
                // equal to NaN except NaN itself, which we already handled earlier.
                self.is_nan() | (self >= other)
            }
        }
    };
}

impl_float_eq_ord!(f32);
impl_float_eq_ord!(f64);

impl TotalHash for f32 {
    fn tot_hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        canonical_f32(*self).to_bits().hash(state)
    }
}

impl TotalHash for f64 {
    fn tot_hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        canonical_f64(*self).to_bits().hash(state)
    }
}

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

impl<T: TotalHash> TotalHash for Option<T> {
    fn tot_hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.is_some().tot_hash(state);
        if let Some(slf) = self {
            slf.tot_hash(state)
        }
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

impl<T: TotalHash + ?Sized> TotalHash for &T {
    fn tot_hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        (*self).tot_hash(state)
    }
}
