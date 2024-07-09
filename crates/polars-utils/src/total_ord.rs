use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use bytemuck::TransparentWrapper;

use crate::hashing::{BytesHash, DirtyHash};
use crate::nulls::IsNull;

/// Converts an f32 into a canonical form, where -0 == 0 and all NaNs map to
/// the same value.
#[inline]
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
#[inline]
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

    #[inline]
    fn tot_ne(&self, other: &Self) -> bool {
        !(self.tot_eq(other))
    }
}

/// Alternative trait for Ord. By consistently using this we can still be
/// generic w.r.t Ord while getting a total ordering for floats.
pub trait TotalOrd: TotalEq {
    fn tot_cmp(&self, other: &Self) -> Ordering;

    #[inline]
    fn tot_lt(&self, other: &Self) -> bool {
        self.tot_cmp(other) == Ordering::Less
    }

    #[inline]
    fn tot_gt(&self, other: &Self) -> bool {
        self.tot_cmp(other) == Ordering::Greater
    }

    #[inline]
    fn tot_le(&self, other: &Self) -> bool {
        self.tot_cmp(other) != Ordering::Greater
    }

    #[inline]
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
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.tot_hash(state);
    }
}

impl<T: Clone> Clone for TotalOrdWrap<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Copy> Copy for TotalOrdWrap<T> {}

impl<T: IsNull> IsNull for TotalOrdWrap<T> {
    const HAS_NULLS: bool = T::HAS_NULLS;
    type Inner = T::Inner;

    #[inline(always)]
    fn is_null(&self) -> bool {
        self.0.is_null()
    }

    #[inline(always)]
    fn unwrap_inner(self) -> Self::Inner {
        self.0.unwrap_inner()
    }
}

impl DirtyHash for f32 {
    #[inline(always)]
    fn dirty_hash(&self) -> u64 {
        canonical_f32(*self).to_bits().dirty_hash()
    }
}

impl DirtyHash for f64 {
    #[inline(always)]
    fn dirty_hash(&self) -> u64 {
        canonical_f64(*self).to_bits().dirty_hash()
    }
}

impl<T: DirtyHash> DirtyHash for TotalOrdWrap<T> {
    #[inline(always)]
    fn dirty_hash(&self) -> u64 {
        self.0.dirty_hash()
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
            #[inline(always)]
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
            #[inline]
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
    #[inline(always)]
    fn tot_hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        canonical_f32(*self).to_bits().hash(state)
    }
}

impl TotalHash for f64 {
    #[inline(always)]
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
    #[inline]
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
    #[inline(always)]
    fn tot_hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        (*self).tot_hash(state)
    }
}

impl<T: TotalEq, U: TotalEq> TotalEq for (T, U) {
    #[inline]
    fn tot_eq(&self, other: &Self) -> bool {
        self.0.tot_eq(&other.0) && self.1.tot_eq(&other.1)
    }
}

impl<T: TotalOrd, U: TotalOrd> TotalOrd for (T, U) {
    #[inline]
    fn tot_cmp(&self, other: &Self) -> Ordering {
        self.0
            .tot_cmp(&other.0)
            .then_with(|| self.1.tot_cmp(&other.1))
    }
}

impl<'a> TotalHash for BytesHash<'a> {
    #[inline(always)]
    fn tot_hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.hash(state)
    }
}

impl<'a> TotalEq for BytesHash<'a> {
    #[inline(always)]
    fn tot_eq(&self, other: &Self) -> bool {
        self == other
    }
}

/// This elides creating a [`TotalOrdWrap`] for types that don't need it.
pub trait ToTotalOrd {
    type TotalOrdItem;
    type SourceItem;

    fn to_total_ord(&self) -> Self::TotalOrdItem;

    fn peel_total_ord(ord_item: Self::TotalOrdItem) -> Self::SourceItem;
}

macro_rules! impl_to_total_ord_identity {
    ($T: ty) => {
        impl ToTotalOrd for $T {
            type TotalOrdItem = $T;
            type SourceItem = $T;

            #[inline]
            fn to_total_ord(&self) -> Self::TotalOrdItem {
                self.clone()
            }

            #[inline]
            fn peel_total_ord(ord_item: Self::TotalOrdItem) -> Self::SourceItem {
                ord_item
            }
        }
    };
}

impl_to_total_ord_identity!(bool);
impl_to_total_ord_identity!(u8);
impl_to_total_ord_identity!(u16);
impl_to_total_ord_identity!(u32);
impl_to_total_ord_identity!(u64);
impl_to_total_ord_identity!(u128);
impl_to_total_ord_identity!(usize);
impl_to_total_ord_identity!(i8);
impl_to_total_ord_identity!(i16);
impl_to_total_ord_identity!(i32);
impl_to_total_ord_identity!(i64);
impl_to_total_ord_identity!(i128);
impl_to_total_ord_identity!(isize);
impl_to_total_ord_identity!(char);
impl_to_total_ord_identity!(String);

macro_rules! impl_to_total_ord_lifetimed_ref_identity {
    ($T: ty) => {
        impl<'a> ToTotalOrd for &'a $T {
            type TotalOrdItem = &'a $T;
            type SourceItem = &'a $T;

            #[inline]
            fn to_total_ord(&self) -> Self::TotalOrdItem {
                *self
            }

            #[inline]
            fn peel_total_ord(ord_item: Self::TotalOrdItem) -> Self::SourceItem {
                ord_item
            }
        }
    };
}

impl_to_total_ord_lifetimed_ref_identity!(str);
impl_to_total_ord_lifetimed_ref_identity!([u8]);

macro_rules! impl_to_total_ord_wrapped {
    ($T: ty) => {
        impl ToTotalOrd for $T {
            type TotalOrdItem = TotalOrdWrap<$T>;
            type SourceItem = $T;

            #[inline]
            fn to_total_ord(&self) -> Self::TotalOrdItem {
                TotalOrdWrap(self.clone())
            }

            #[inline]
            fn peel_total_ord(ord_item: Self::TotalOrdItem) -> Self::SourceItem {
                ord_item.0
            }
        }
    };
}

impl_to_total_ord_wrapped!(f32);
impl_to_total_ord_wrapped!(f64);

/// This is safe without needing to map the option value to TotalOrdWrap, since
/// for example:
/// `TotalOrdWrap<Option<T>>` implements `Eq + Hash`, iff:
/// `Option<T>` implements `TotalEq + TotalHash`, iff:
/// `T` implements `TotalEq + TotalHash`
impl<T: Copy> ToTotalOrd for Option<T> {
    type TotalOrdItem = TotalOrdWrap<Option<T>>;
    type SourceItem = Option<T>;

    #[inline]
    fn to_total_ord(&self) -> Self::TotalOrdItem {
        TotalOrdWrap(*self)
    }

    #[inline]
    fn peel_total_ord(ord_item: Self::TotalOrdItem) -> Self::SourceItem {
        ord_item.0
    }
}

impl<T: ToTotalOrd> ToTotalOrd for &T {
    type TotalOrdItem = T::TotalOrdItem;
    type SourceItem = T::SourceItem;

    #[inline]
    fn to_total_ord(&self) -> Self::TotalOrdItem {
        (*self).to_total_ord()
    }

    #[inline]
    fn peel_total_ord(ord_item: Self::TotalOrdItem) -> Self::SourceItem {
        T::peel_total_ord(ord_item)
    }
}

impl<'a> ToTotalOrd for BytesHash<'a> {
    type TotalOrdItem = BytesHash<'a>;
    type SourceItem = BytesHash<'a>;

    #[inline]
    fn to_total_ord(&self) -> Self::TotalOrdItem {
        *self
    }

    #[inline]
    fn peel_total_ord(ord_item: Self::TotalOrdItem) -> Self::SourceItem {
        ord_item
    }
}
