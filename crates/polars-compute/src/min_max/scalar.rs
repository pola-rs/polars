use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, PrimitiveArray, Utf8Array, Utf8ViewArray,
};
use arrow::types::{NativeType, Offset};
use polars_array::arrow::bridge::ToArrow;
use polars_array::{PlBooleanArray, PlPrimitiveArray};
use polars_utils::min_max::MinMax;

use super::MinMaxKernel;
use super::pl_array::{
    fold_flat, fold_flat_min_max, min_max_ignore_nan, min_max_propagate_nan, reduce_flat,
};

/// The primitive types no vectorized kernel reduces — see `simd` for the ones that do. A chunk
/// that repeats one value is read as that value, in `O(1)`; anything else is folded one element at
/// a time.
impl<T: NativeType + MinMax + super::NotSimdPrimitive> MinMaxKernel for PlPrimitiveArray<T> {
    type Scalar<'a> = T;

    fn min_ignore_nan_kernel(&self) -> Option<T> {
        reduce_flat(
            self,
            |value| value,
            |values, validity| fold_flat(values, validity, MinMax::min_ignore_nan),
        )
    }

    fn max_ignore_nan_kernel(&self) -> Option<T> {
        reduce_flat(
            self,
            |value| value,
            |values, validity| fold_flat(values, validity, MinMax::max_ignore_nan),
        )
    }

    fn min_max_ignore_nan_kernel(&self) -> Option<(T, T)> {
        reduce_flat(
            self,
            |value| (value, value),
            |values, validity| fold_flat_min_max(values, validity, min_max_ignore_nan),
        )
    }

    fn min_propagate_nan_kernel(&self) -> Option<T> {
        reduce_flat(
            self,
            |value| value,
            |values, validity| fold_flat(values, validity, MinMax::min_propagate_nan),
        )
    }

    fn max_propagate_nan_kernel(&self) -> Option<T> {
        reduce_flat(
            self,
            |value| value,
            |values, validity| fold_flat(values, validity, MinMax::max_propagate_nan),
        )
    }

    fn min_max_propagate_nan_kernel(&self) -> Option<(T, T)> {
        reduce_flat(
            self,
            |value| (value, value),
            |values, validity| fold_flat_min_max(values, validity, min_max_propagate_nan),
        )
    }
}

/// An Arrow chunk holds one slot per element throughout, which is one of the two layouts the
/// kernel above reads: its buffers are handed over as they are, and it is that kernel that
/// reduces them.
impl<T: NativeType + MinMax + super::NotSimdPrimitive> MinMaxKernel for PrimitiveArray<T> {
    type Scalar<'a> = T;

    fn min_ignore_nan_kernel(&self) -> Option<T> {
        PlPrimitiveArray::from_arrow(self).min_ignore_nan_kernel()
    }

    fn max_ignore_nan_kernel(&self) -> Option<T> {
        PlPrimitiveArray::from_arrow(self).max_ignore_nan_kernel()
    }

    fn min_max_ignore_nan_kernel(&self) -> Option<(T, T)> {
        PlPrimitiveArray::from_arrow(self).min_max_ignore_nan_kernel()
    }

    fn min_propagate_nan_kernel(&self) -> Option<T> {
        PlPrimitiveArray::from_arrow(self).min_propagate_nan_kernel()
    }

    fn max_propagate_nan_kernel(&self) -> Option<T> {
        PlPrimitiveArray::from_arrow(self).max_propagate_nan_kernel()
    }

    fn min_max_propagate_nan_kernel(&self) -> Option<(T, T)> {
        PlPrimitiveArray::from_arrow(self).min_max_propagate_nan_kernel()
    }
}

impl<T: NativeType + MinMax + super::NotSimdPrimitive> MinMaxKernel for [T] {
    type Scalar<'a> = T;

    fn min_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        self.iter().copied().reduce(MinMax::min_ignore_nan)
    }

    fn max_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        self.iter().copied().reduce(MinMax::max_ignore_nan)
    }

    fn min_max_ignore_nan_kernel(&self) -> Option<(Self::Scalar<'_>, Self::Scalar<'_>)> {
        self.iter()
            .copied()
            .map(|v| (v, v))
            .reduce(min_max_ignore_nan)
    }

    fn min_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        self.iter().copied().reduce(MinMax::min_propagate_nan)
    }

    fn max_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        self.iter().copied().reduce(MinMax::max_propagate_nan)
    }

    fn min_max_propagate_nan_kernel(&self) -> Option<(Self::Scalar<'_>, Self::Scalar<'_>)> {
        self.iter()
            .copied()
            .map(|v| (v, v))
            .reduce(min_max_propagate_nan)
    }
}

impl MinMaxKernel for BooleanArray {
    type Scalar<'a> = bool;

    fn min_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        crate::boolean::all(&PlBooleanArray::from_arrow(self))
    }

    fn max_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        crate::boolean::any(&PlBooleanArray::from_arrow(self))
    }

    #[inline(always)]
    fn min_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        self.min_ignore_nan_kernel()
    }

    #[inline(always)]
    fn max_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        self.max_ignore_nan_kernel()
    }
}

impl MinMaxKernel for BinaryViewArray {
    type Scalar<'a> = &'a [u8];

    fn min_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        if self.null_count() == 0 {
            self.values_iter().reduce(MinMax::min_ignore_nan)
        } else {
            self.non_null_values_iter().reduce(MinMax::min_ignore_nan)
        }
    }

    fn max_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        if self.null_count() == 0 {
            self.values_iter().reduce(MinMax::max_ignore_nan)
        } else {
            self.non_null_values_iter().reduce(MinMax::max_ignore_nan)
        }
    }

    #[inline(always)]
    fn min_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        self.min_ignore_nan_kernel()
    }

    #[inline(always)]
    fn max_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        self.max_ignore_nan_kernel()
    }
}

impl MinMaxKernel for Utf8ViewArray {
    type Scalar<'a> = &'a str;

    #[inline(always)]
    fn min_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        self.to_binview().min_ignore_nan_kernel().map(|s| unsafe {
            // SAFETY: the lifetime is the same, and it is valid UTF-8.
            #[allow(clippy::transmute_bytes_to_str)]
            std::mem::transmute::<&[u8], &str>(s)
        })
    }

    #[inline(always)]
    fn max_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        self.to_binview().max_ignore_nan_kernel().map(|s| unsafe {
            // SAFETY: the lifetime is the same, and it is valid UTF-8.
            #[allow(clippy::transmute_bytes_to_str)]
            std::mem::transmute::<&[u8], &str>(s)
        })
    }

    #[inline(always)]
    fn min_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        self.min_ignore_nan_kernel()
    }

    #[inline(always)]
    fn max_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        self.max_ignore_nan_kernel()
    }
}

impl<O: Offset> MinMaxKernel for BinaryArray<O> {
    type Scalar<'a> = &'a [u8];

    fn min_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        if self.null_count() == 0 {
            self.values_iter().reduce(MinMax::min_ignore_nan)
        } else {
            self.non_null_values_iter().reduce(MinMax::min_ignore_nan)
        }
    }

    fn max_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        if self.null_count() == 0 {
            self.values_iter().reduce(MinMax::max_ignore_nan)
        } else {
            self.non_null_values_iter().reduce(MinMax::max_ignore_nan)
        }
    }

    #[inline(always)]
    fn min_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        self.min_ignore_nan_kernel()
    }

    #[inline(always)]
    fn max_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        self.max_ignore_nan_kernel()
    }
}

impl<O: Offset> MinMaxKernel for Utf8Array<O> {
    type Scalar<'a> = &'a str;

    #[inline(always)]
    fn min_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        self.to_binary().min_ignore_nan_kernel().map(|s| unsafe {
            // SAFETY: the lifetime is the same, and it is valid UTF-8.
            #[allow(clippy::transmute_bytes_to_str)]
            std::mem::transmute::<&[u8], &str>(s)
        })
    }

    #[inline(always)]
    fn max_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        self.to_binary().max_ignore_nan_kernel().map(|s| unsafe {
            // SAFETY: the lifetime is the same, and it is valid UTF-8.
            #[allow(clippy::transmute_bytes_to_str)]
            std::mem::transmute::<&[u8], &str>(s)
        })
    }

    #[inline(always)]
    fn min_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        self.min_ignore_nan_kernel()
    }

    #[inline(always)]
    fn max_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        self.max_ignore_nan_kernel()
    }
}
