use arrow::array::Array;
use arrow::bitmap::{self, Bitmap};

// Low-level comparison kernel.
pub trait TotalOrdKernel: Sized + Array {
    type Scalar: ?Sized;

    // These kernels ignore validity entirely (results for nulls are unspecified
    // but initialized).
    fn tot_eq_kernel(&self, other: &Self) -> Bitmap;
    fn tot_ne_kernel(&self, other: &Self) -> Bitmap;
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
    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
    fn tot_lt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
    fn tot_le_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
    fn tot_gt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
    fn tot_ge_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;

    // These kernels treat null as any other value equal to itself but unequal
    // to anything else.
    fn tot_eq_missing_kernel(&self, other: &Self) -> Bitmap {
        let q = self.tot_eq_kernel(other);
        let combined = match (self.validity(), other.validity()) {
            (None, None) => q,
            (None, Some(r)) => &q & r,
            (Some(l), None) => &q & l,
            (Some(l), Some(r)) => bitmap::ternary(&q, l, r, |q, l, r| (q & l & r) | !(l | r)),
        };
        combined
    }

    fn tot_ne_missing_kernel(&self, other: &Self) -> Bitmap {
        let q = self.tot_ne_kernel(other);
        let combined = match (self.validity(), other.validity()) {
            (None, None) => q,
            (None, Some(r)) => &q | &!r,
            (Some(l), None) => &q | &!l,
            (Some(l), Some(r)) => bitmap::ternary(&q, l, r, |q, l, r| (q & l & r) | (l ^ r)),
        };
        combined
    }

    // These kernels treat null as any other value equal to itself but unequal
    // to anything else. other is assumed to be non-null.
    fn tot_eq_missing_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        let q = self.tot_eq_kernel_broadcast(other);
        if let Some(valid) = self.validity() {
            bitmap::binary(&q, valid, |q, v| q & v)
        } else {
            q
        }
    }

    fn tot_ne_missing_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        let q = self.tot_ne_kernel_broadcast(other);
        if let Some(valid) = self.validity() {
            bitmap::binary(&q, valid, |q, v| q | !v)
        } else {
            q
        }
    }
}

// Trait to enable the scalar blanket implementation.
trait NotSimdPrimitive {}

#[cfg(not(feature = "simd"))]
impl<T> NotSimdPrimitive for T {}

#[cfg(feature = "simd")]
impl NotSimdPrimitive for u128 {}
#[cfg(feature = "simd")]
impl NotSimdPrimitive for i128 {}

mod scalar;

#[cfg(feature = "simd")]
mod simd;

#[cfg(feature = "dtype-array")]
mod array;
