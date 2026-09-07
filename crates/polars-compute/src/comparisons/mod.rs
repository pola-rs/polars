use arrow::bitmap::{self, Bitmap};
use polars_array::{PlBitmap, PlBitmapRef};

pub trait TotalEqKernel: Sized {
    type Scalar: ?Sized;

    // The validity mask, with one bit per element. This is what `Array::validity` hands out for
    // an Arrow array; the arrays of `polars-array` implement these kernels in their flat
    // representation, whose mask is flat in turn. An array whose mask may repeat a single bit
    // implements `PlTotalEqKernel` instead.
    fn validity_mask(&self) -> Option<&Bitmap>;

    // These kernels ignore validity entirely (results for nulls are unspecified
    // but initialized).
    fn tot_eq_kernel(&self, other: &Self) -> Bitmap;
    fn tot_ne_kernel(&self, other: &Self) -> Bitmap;
    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;

    // These kernels treat null as any other value equal to itself but unequal
    // to anything else.
    fn tot_eq_missing_kernel(&self, other: &Self) -> Bitmap {
        let q = self.tot_eq_kernel(other);
        match (self.validity_mask(), other.validity_mask()) {
            (None, None) => q,
            (None, Some(r)) => &q & r,
            (Some(l), None) => &q & l,
            (Some(l), Some(r)) => bitmap::ternary(&q, l, r, |q, l, r| (q & l & r) | !(l | r)),
        }
    }

    fn tot_ne_missing_kernel(&self, other: &Self) -> Bitmap {
        let q = self.tot_ne_kernel(other);
        match (self.validity_mask(), other.validity_mask()) {
            (None, None) => q,
            (None, Some(r)) => &q | &!r,
            (Some(l), None) => &q | &!l,
            (Some(l), Some(r)) => bitmap::ternary(&q, l, r, |q, l, r| (q & l & r) | (l ^ r)),
        }
    }
    fn tot_eq_missing_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        let q = self.tot_eq_kernel_broadcast(other);
        if let Some(valid) = self.validity_mask() {
            bitmap::binary(&q, valid, |q, v| q & v)
        } else {
            q
        }
    }

    fn tot_ne_missing_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        let q = self.tot_ne_kernel_broadcast(other);
        if let Some(valid) = self.validity_mask() {
            bitmap::binary(&q, valid, |q, v| q | !v)
        } else {
            q
        }
    }
}

// Low-level comparison kernel.
pub trait TotalOrdKernel: Sized {
    type Scalar: ?Sized;

    // These kernels ignore validity entirely (results for nulls are unspecified
    // but initialized).
    fn tot_lt_kernel(&self, other: &Self) -> Bitmap;
    fn tot_le_kernel(&self, other: &Self) -> Bitmap;
    fn tot_gt_kernel(&self, other: &Self) -> Bitmap {
        other.tot_lt_kernel(self)
    }
    fn tot_ge_kernel(&self, other: &Self) -> Bitmap {
        other.tot_le_kernel(self)
    }

    // These kernels ignore validity entirely (results for nulls are unspecified
    // but initialized).
    fn tot_lt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
    fn tot_le_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
    fn tot_gt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
    fn tot_ge_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
}

/// What a validity mask leaves for the missing-aware kernels of [`PlTotalEqKernel`] to combine.
enum Validity<'a> {
    /// Every element is there: no mask at all, or one repeating a set bit.
    AllValid,
    /// Every element is null, which a mask repeating an unset bit says in a single bit.
    AllNull,
    /// One bit per element.
    Flat(&'a Bitmap),
}

fn validity_of(mask: Option<PlBitmapRef<'_>>) -> Validity<'_> {
    match mask {
        None => Validity::AllValid,
        Some(mask) => match mask.scalar_value() {
            Some(true) => Validity::AllValid,
            Some(false) => Validity::AllNull,
            None => Validity::Flat(mask.flat_bitmap().unwrap()),
        },
    }
}

/// `q & mask`, where `mask` holds one bit per element.
fn and_mask(q: PlBitmap, mask: &Bitmap) -> PlBitmap {
    match q.scalar_value() {
        Some(false) => q,
        Some(true) => PlBitmap::from_bitmap(mask.clone()),
        None => PlBitmap::from_bitmap(bitmap::binary(q.flat_bitmap().unwrap(), mask, |q, m| q & m)),
    }
}

/// `q | !mask`, where `mask` holds one bit per element.
fn or_not_mask(q: PlBitmap, mask: &Bitmap) -> PlBitmap {
    match q.scalar_value() {
        Some(true) => q,
        Some(false) => PlBitmap::from_bitmap(!mask),
        None => PlBitmap::from_bitmap(bitmap::binary(q.flat_bitmap().unwrap(), mask, |q, m| {
            q | !m
        })),
    }
}

/// The equality kernels over an array whose buffers may repeat a single slot.
pub trait PlTotalEqKernel: Sized {
    type Scalar: ?Sized;

    /// The validity mask, in whichever representation it is in.
    fn validity_mask(&self) -> Option<PlBitmapRef<'_>>;

    // These kernels ignore validity entirely (results for nulls are unspecified
    // but initialized).
    fn tot_eq_kernel(&self, other: &Self) -> PlBitmap;
    fn tot_ne_kernel(&self, other: &Self) -> PlBitmap;
    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap;
    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap;

    // These kernels treat null as any other value equal to itself but unequal
    // to anything else.
    fn tot_eq_missing_kernel(&self, other: &Self) -> PlBitmap {
        use Validity::*;

        let q = self.tot_eq_kernel(other);
        let length = q.len();

        match (
            validity_of(self.validity_mask()),
            validity_of(other.validity_mask()),
        ) {
            (AllValid, AllValid) => q,
            // A null is equal to a null and to nothing else, so a side that is null throughout
            // answers for every element at once, with no value read on either side.
            (AllNull, AllNull) => PlBitmap::new_scalar(true, length),
            (AllValid, AllNull) | (AllNull, AllValid) => PlBitmap::new_scalar(false, length),
            // One side is null throughout, so the answer is where the other side is null too.
            (AllNull, Flat(r)) => PlBitmap::from_bitmap(!r),
            (Flat(l), AllNull) => PlBitmap::from_bitmap(!l),
            (AllValid, Flat(r)) => and_mask(q, r),
            (Flat(l), AllValid) => and_mask(q, l),
            (Flat(l), Flat(r)) => {
                PlBitmap::from_bitmap(bitmap::ternary(&q.into_bitmap(), l, r, |q, l, r| {
                    (q & l & r) | !(l | r)
                }))
            },
        }
    }

    fn tot_ne_missing_kernel(&self, other: &Self) -> PlBitmap {
        use Validity::*;

        let q = self.tot_ne_kernel(other);
        let length = q.len();

        // The complement of `tot_eq_missing_kernel`, arm for arm.
        match (
            validity_of(self.validity_mask()),
            validity_of(other.validity_mask()),
        ) {
            (AllValid, AllValid) => q,
            (AllNull, AllNull) => PlBitmap::new_scalar(false, length),
            (AllValid, AllNull) | (AllNull, AllValid) => PlBitmap::new_scalar(true, length),
            (AllNull, Flat(r)) => PlBitmap::from_bitmap(r.clone()),
            (Flat(l), AllNull) => PlBitmap::from_bitmap(l.clone()),
            (AllValid, Flat(r)) => or_not_mask(q, r),
            (Flat(l), AllValid) => or_not_mask(q, l),
            (Flat(l), Flat(r)) => {
                PlBitmap::from_bitmap(bitmap::ternary(&q.into_bitmap(), l, r, |q, l, r| {
                    (q & l & r) | (l ^ r)
                }))
            },
        }
    }

    fn tot_eq_missing_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        let q = self.tot_eq_kernel_broadcast(other);
        let length = q.len();

        match validity_of(self.validity_mask()) {
            Validity::AllValid => q,
            // The scalar is a value and every element is null, so none of them is equal to it.
            Validity::AllNull => PlBitmap::new_scalar(false, length),
            Validity::Flat(valid) => and_mask(q, valid),
        }
    }

    fn tot_ne_missing_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        let q = self.tot_ne_kernel_broadcast(other);
        let length = q.len();

        match validity_of(self.validity_mask()) {
            Validity::AllValid => q,
            Validity::AllNull => PlBitmap::new_scalar(true, length),
            Validity::Flat(valid) => or_not_mask(q, valid),
        }
    }
}

/// The ordering kernels over an array whose buffers may repeat a single slot.
pub trait PlTotalOrdKernel: Sized {
    type Scalar: ?Sized;

    // These kernels ignore validity entirely (results for nulls are unspecified
    // but initialized).
    fn tot_lt_kernel(&self, other: &Self) -> PlBitmap;
    fn tot_le_kernel(&self, other: &Self) -> PlBitmap;
    fn tot_gt_kernel(&self, other: &Self) -> PlBitmap {
        other.tot_lt_kernel(self)
    }
    fn tot_ge_kernel(&self, other: &Self) -> PlBitmap {
        other.tot_le_kernel(self)
    }

    fn tot_lt_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap;
    fn tot_le_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap;
    fn tot_gt_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap;
    fn tot_ge_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap;
}

mod binary;
mod boolean;
mod dictionary;
mod dyn_array;
mod list;
mod null;
mod pl_array;
mod pl_primitive;
mod scalar;
mod struct_;
mod utf8;
mod view;

#[cfg(feature = "simd")]
mod _simd_dtypes {
    use arrow::types::{days_ms, i256, months_days_ns};

    use crate::NotSimdPrimitive;

    impl NotSimdPrimitive for i256 {}
    impl NotSimdPrimitive for days_ms {}
    impl NotSimdPrimitive for months_days_ns {}
}

#[cfg(feature = "simd")]
mod simd;

#[cfg(feature = "dtype-array")]
mod array;
