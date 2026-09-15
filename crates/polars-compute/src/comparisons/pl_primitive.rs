//! The comparison kernels over a [`PlPrimitiveArray`] of unknown representation.

use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use polars_array::{Flat, PlBitmap, PlBitmapRef, PlPrimitiveArray};
use polars_buffer::Buffer;
use polars_utils::total_ord::{TotalEq, TotalOrd};

use super::{
    IN_PLACE_COMPARISON_LIMIT, PlTotalEqKernel, PlTotalOrdKernel, TotalEqKernel, TotalOrdKernel,
};

/// The values of a chunk as a flat array of their own, which is what the [`Flat`] kernels take.
#[inline]
fn flat_values<T: NativeType>(values: &Buffer<T>) -> Flat<PlPrimitiveArray<T>> {
    // An array built from a values buffer alone has one slot per element and no mask, so it is
    // already flat and `to_flat` hands the borrow straight back.
    PlPrimitiveArray::from_values(values.clone())
        .to_flat()
        .into_owned()
}

/// A mask of `length` copies of `value`, held in the single bit that says it.
#[inline]
fn repeated(value: bool, length: usize) -> PlBitmap {
    PlBitmap::new_scalar(value, length)
}

/// A mask of one bit per element, which is what a flat operand leaves.
#[inline]
fn written_out(bits: Bitmap) -> PlBitmap {
    PlBitmap::from_bitmap(bits)
}

/// Dispatches a binary kernel on the values representation of both operands.
macro_rules! binary_kernel {
    ($self:expr, $other:expr, $scalar:expr, $flat:path, $flat_lhs:path, $flat_rhs:path $(,)?) => {{
        let (lhs, rhs) = ($self, $other);
        assert!(lhs.len() == rhs.len());

        match (
            lhs.scalar_value_ignore_validity(),
            rhs.scalar_value_ignore_validity(),
        ) {
            // Every element of both sides holds the one value its own side repeats, so the one
            // comparison of those two values is the answer for all of them.
            (Some(l), Some(r)) => repeated($scalar(&l, &r), lhs.len()),
            (Some(l), None) => written_out($flat_rhs(&flat_values(rhs.flat_values().unwrap()), &l)),
            (None, Some(r)) => written_out($flat_lhs(&flat_values(lhs.flat_values().unwrap()), &r)),
            (None, None) => written_out($flat(
                &flat_values(lhs.flat_values().unwrap()),
                &flat_values(rhs.flat_values().unwrap()),
            )),
        }
    }};
}

/// Dispatches a broadcast kernel on the values representation of its one operand.
macro_rules! broadcast_kernel {
    ($self:expr, $other:expr, $scalar:expr, $flat:path $(,)?) => {{
        let (lhs, rhs) = ($self, $other);

        match lhs.scalar_value_ignore_validity() {
            Some(l) => repeated($scalar(&l, rhs), lhs.len()),
            None => written_out($flat(&flat_values(lhs.flat_values().unwrap()), rhs)),
        }
    }};
}

/// Defines the kernels that take two operands, each dispatching on both representations.
macro_rules! binary_kernels {
    ($($name:ident: $scalar:path, $flat:path, $flat_lhs:path, $flat_rhs:path;)*) => {
        $(
            fn $name(&self, other: &Self) -> PlBitmap {
                binary_kernel!(self, other, $scalar, $flat, $flat_lhs, $flat_rhs)
            }
        )*
    };
}

/// Defines the kernels that take one operand and a scalar, dispatching on its representation.
macro_rules! broadcast_kernels {
    ($($name:ident: $scalar:path, $flat:path;)*) => {
        $(
            fn $name(&self, other: &T) -> PlBitmap {
                broadcast_kernel!(self, other, $scalar, $flat)
            }
        )*
    };
}

impl<T> PlTotalEqKernel for PlPrimitiveArray<T>
where
    T: NativeType + TotalEq,
    Flat<PlPrimitiveArray<T>>: TotalEqKernel<Scalar = T>,
{
    type Scalar = T;

    fn validity_mask(&self) -> Option<PlBitmapRef<'_>> {
        // Whatever representation the mask is in: the missing-aware kernels resolve it themselves.
        self.validity()
    }

    // Equality is symmetric, so which side repeats its value makes no difference.
    binary_kernels! {
        tot_eq_kernel: TotalEq::tot_eq, TotalEqKernel::tot_eq_kernel,
            TotalEqKernel::tot_eq_kernel_broadcast, TotalEqKernel::tot_eq_kernel_broadcast;
        tot_ne_kernel: TotalEq::tot_ne, TotalEqKernel::tot_ne_kernel,
            TotalEqKernel::tot_ne_kernel_broadcast, TotalEqKernel::tot_ne_kernel_broadcast;
    }

    broadcast_kernels! {
        tot_eq_kernel_broadcast: TotalEq::tot_eq, TotalEqKernel::tot_eq_kernel_broadcast;
        tot_ne_kernel_broadcast: TotalEq::tot_ne, TotalEqKernel::tot_ne_kernel_broadcast;
    }

    /// The values are compared where they lie: see [`PlTotalEqKernel::tot_eq_missing_all`].
    fn tot_eq_missing_all(&self, other: &Self) -> bool {
        assert_eq!(self.len(), other.len());

        // Past this many values the written-out comparison earns its allocation back, and it is
        // the vectorised kernel that reads them.
        if self.len() > IN_PLACE_COMPARISON_LIMIT {
            return self.tot_eq_missing_kernel(other).unset_bits() == 0;
        }

        self.iter()
            .zip(other.iter())
            .all(|(lhs, rhs)| lhs.tot_eq(&rhs))
    }
}

impl<T> PlTotalOrdKernel for PlPrimitiveArray<T>
where
    T: NativeType + TotalOrd,
    Flat<PlPrimitiveArray<T>>: TotalOrdKernel<Scalar = T>,
{
    type Scalar = T;

    // A repeated left operand turns the comparison around: `l < r[i]` is `r[i] > l`.
    binary_kernels! {
        tot_lt_kernel: TotalOrd::tot_lt, TotalOrdKernel::tot_lt_kernel,
            TotalOrdKernel::tot_lt_kernel_broadcast, TotalOrdKernel::tot_gt_kernel_broadcast;
        tot_le_kernel: TotalOrd::tot_le, TotalOrdKernel::tot_le_kernel,
            TotalOrdKernel::tot_le_kernel_broadcast, TotalOrdKernel::tot_ge_kernel_broadcast;
    }

    broadcast_kernels! {
        tot_lt_kernel_broadcast: TotalOrd::tot_lt, TotalOrdKernel::tot_lt_kernel_broadcast;
        tot_le_kernel_broadcast: TotalOrd::tot_le, TotalOrdKernel::tot_le_kernel_broadcast;
        tot_gt_kernel_broadcast: TotalOrd::tot_gt, TotalOrdKernel::tot_gt_kernel_broadcast;
        tot_ge_kernel_broadcast: TotalOrd::tot_ge, TotalOrdKernel::tot_ge_kernel_broadcast;
    }
}
