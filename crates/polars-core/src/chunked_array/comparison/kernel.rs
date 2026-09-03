//! The comparison kernels of `polars-compute`, over the arrays of `polars-array`: these mirror
//! [`TotalEqKernel`] and [`TotalOrdKernel`] once [`ToArrow`] has handed the buffers over.

use arrow::bitmap::Bitmap;
use polars_array::Flat;
use polars_array::arrow::bridge::ToArrow;
use polars_compute::comparisons::{TotalEqKernel, TotalOrdKernel};

/// The counterpart of [`TotalEqKernel`] for the arrays of `polars-array`.
pub trait PlTotalEqKernel: Sized {
    /// The value a broadcasting kernel compares every element against.
    type Scalar: ?Sized;

    /// Elementwise equality, ignoring validity: the result at a null is unspecified.
    fn tot_eq_kernel(&self, other: &Self) -> Bitmap;
    /// Elementwise inequality, ignoring validity: the result at a null is unspecified.
    fn tot_ne_kernel(&self, other: &Self) -> Bitmap;
    /// Elementwise equality against one value, ignoring validity.
    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
    /// Elementwise inequality against one value, ignoring validity.
    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;

    /// Elementwise equality, with a null equal to itself and to nothing else.
    fn tot_eq_missing_kernel(&self, other: &Self) -> Bitmap;
    /// Elementwise inequality, with a null equal to itself and to nothing else.
    fn tot_ne_missing_kernel(&self, other: &Self) -> Bitmap;
    /// Elementwise equality against one value, with a null equal to nothing.
    fn tot_eq_missing_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
    /// Elementwise inequality against one value, with a null unequal to everything.
    fn tot_ne_missing_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
}

/// The counterpart of [`TotalOrdKernel`] for the arrays of `polars-array`.
pub trait PlTotalOrdKernel: Sized {
    /// The value a broadcasting kernel compares every element against.
    type Scalar: ?Sized;

    fn tot_lt_kernel(&self, other: &Self) -> Bitmap;
    fn tot_le_kernel(&self, other: &Self) -> Bitmap;
    fn tot_gt_kernel(&self, other: &Self) -> Bitmap;
    fn tot_ge_kernel(&self, other: &Self) -> Bitmap;

    fn tot_lt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
    fn tot_le_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
    fn tot_gt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
    fn tot_ge_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
}

macro_rules! binary_kernel {
    ($($name:ident),* $(,)?) => {
        $(
            #[inline]
            fn $name(&self, other: &Self) -> Bitmap {
                A::to_arrow(self).$name(&A::to_arrow(other))
            }
        )*
    };
}

macro_rules! broadcast_kernel {
    ($($name:ident),* $(,)?) => {
        $(
            #[inline]
            fn $name(&self, other: &Self::Scalar) -> Bitmap {
                A::to_arrow(self).$name(other)
            }
        )*
    };
}

impl<A> PlTotalEqKernel for Flat<A>
where
    A: ToArrow,
    A::Arrow: TotalEqKernel,
{
    type Scalar = <A::Arrow as TotalEqKernel>::Scalar;

    binary_kernel!(
        tot_eq_kernel,
        tot_ne_kernel,
        tot_eq_missing_kernel,
        tot_ne_missing_kernel,
    );
    broadcast_kernel!(
        tot_eq_kernel_broadcast,
        tot_ne_kernel_broadcast,
        tot_eq_missing_kernel_broadcast,
        tot_ne_missing_kernel_broadcast,
    );
}

impl<A> PlTotalOrdKernel for Flat<A>
where
    A: ToArrow,
    A::Arrow: TotalOrdKernel,
{
    type Scalar = <A::Arrow as TotalOrdKernel>::Scalar;

    binary_kernel!(tot_lt_kernel, tot_le_kernel, tot_gt_kernel, tot_ge_kernel);
    broadcast_kernel!(
        tot_lt_kernel_broadcast,
        tot_le_kernel_broadcast,
        tot_gt_kernel_broadcast,
        tot_ge_kernel_broadcast,
    );
}
