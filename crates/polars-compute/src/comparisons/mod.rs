use arrow::bitmap::{self, Bitmap};

pub trait TotalEqKernel: Sized {
    type Scalar: ?Sized;

    // The validity mask, with one bit per element. This is what `Array::validity` hands out for
    // an Arrow array; the arrays of `polars-array` implement these kernels in their flat
    // representation, whose mask is flat in turn.
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

mod binary;
mod boolean;
mod dictionary;
mod dyn_array;
mod list;
mod null;
mod pl_array;
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
