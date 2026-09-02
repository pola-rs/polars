use arrow::bitmap::Bitmap;
use arrow::legacy::kernels::set::set_at_nulls;
use num_traits::Float;
use polars_compute::nan::{is_nan, is_not_nan};
use polars_utils::float16::pf16;
use polars_utils::total_ord::{canonical_f16, canonical_f32, canonical_f64};

use crate::chunked_array::arrow_bridge::chunk_to_arrow;
use crate::prelude::arity::{unary_elementwise_values, unary_kernel_flat};
use crate::prelude::*;

impl<T> ChunkedArray<T>
where
    T: PolarsFloatType,
    T::Native: Float,
{
    pub fn is_nan(&self) -> BooleanChunked {
        // TODO(polars-array-scalar): whether the single value of a scalar chunk is NaN answers it
        // for every element, which this writes out to ask element by element.
        unary_kernel_flat(self, |arr| {
            let out = is_nan(arr.values()).unwrap_or_else(|| Bitmap::new_zeroed(arr.len()));
            PlBooleanArray::new(out, arr.len(), arr.validity().cloned())
        })
    }
    pub fn is_not_nan(&self) -> BooleanChunked {
        unary_kernel_flat(self, |arr| {
            let out =
                is_not_nan(arr.values()).unwrap_or_else(|| Bitmap::new_with_value(true, arr.len()));
            PlBooleanArray::new(out, arr.len(), arr.validity().cloned())
        })
    }
    pub fn is_finite(&self) -> BooleanChunked {
        unary_elementwise_values(self, |x| x.is_finite())
    }
    pub fn is_infinite(&self) -> BooleanChunked {
        unary_elementwise_values(self, |x| x.is_infinite())
    }

    #[must_use]
    /// Convert missing values to `NaN` values.
    pub fn none_to_nan(&self) -> Self {
        // The kernel is the Arrow one, so each chunk crosses over — see `arrow_bridge`.
        let chunks = self
            .downcast_iter()
            .map(|arr| ToArrow::from_arrow(&set_at_nulls(&chunk_to_arrow(arr), T::Native::nan())));
        ChunkedArray::from_chunk_iter(self.name().clone(), chunks)
    }
}

pub trait Canonical {
    fn canonical(self) -> Self;
}

impl Canonical for pf16 {
    #[inline]
    fn canonical(self) -> Self {
        canonical_f16(self)
    }
}

impl Canonical for f32 {
    #[inline]
    fn canonical(self) -> Self {
        canonical_f32(self)
    }
}

impl Canonical for f64 {
    #[inline]
    fn canonical(self) -> Self {
        canonical_f64(self)
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsFloatType,
    T::Native: Float + Canonical,
{
    pub fn to_canonical(&self) -> Self {
        unary_elementwise_values(self, |v| v.canonical())
    }
}
