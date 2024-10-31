use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, PrimitiveArray, Utf8Array, Utf8ViewArray,
};
use arrow::types::{NativeType, Offset};
use polars_utils::min_max::MinMax;

use super::MinMaxKernel;

fn min_max_ignore_nan<T: NativeType>((cur_min, cur_max): (T, T), (min, max): (T, T)) -> (T, T) {
    (
        MinMax::min_ignore_nan(cur_min, min),
        MinMax::max_ignore_nan(cur_max, max),
    )
}

fn min_max_propagate_nan<T: NativeType>((cur_min, cur_max): (T, T), (min, max): (T, T)) -> (T, T) {
    (
        MinMax::min_propagate_nan(cur_min, min),
        MinMax::max_propagate_nan(cur_max, max),
    )
}

fn reduce_vals<T, F>(v: &PrimitiveArray<T>, f: F) -> Option<T>
where
    T: NativeType,
    F: Fn(T, T) -> T,
{
    if v.null_count() == 0 {
        v.values_iter().copied().reduce(f)
    } else {
        v.non_null_values_iter().reduce(f)
    }
}

fn reduce_tuple_vals<T, F>(v: &PrimitiveArray<T>, f: F) -> Option<(T, T)>
where
    T: NativeType,
    F: Fn((T, T), (T, T)) -> (T, T),
{
    if v.null_count() == 0 {
        v.values_iter().copied().map(|v| (v, v)).reduce(f)
    } else {
        v.non_null_values_iter().map(|v| (v, v)).reduce(f)
    }
}

impl<T: NativeType + MinMax + super::NotSimdPrimitive> MinMaxKernel for PrimitiveArray<T> {
    type Scalar<'a> = T;

    fn min_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        reduce_vals(self, MinMax::min_ignore_nan)
    }

    fn max_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        reduce_vals(self, MinMax::max_ignore_nan)
    }

    fn min_max_ignore_nan_kernel(&self) -> Option<(Self::Scalar<'_>, Self::Scalar<'_>)> {
        reduce_tuple_vals(self, min_max_ignore_nan)
    }

    fn min_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        reduce_vals(self, MinMax::min_propagate_nan)
    }

    fn max_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        reduce_vals(self, MinMax::max_propagate_nan)
    }

    fn min_max_propagate_nan_kernel(&self) -> Option<(Self::Scalar<'_>, Self::Scalar<'_>)> {
        reduce_tuple_vals(self, min_max_propagate_nan)
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
        if self.len() - self.null_count() == 0 {
            return None;
        }

        let unset_bits = self.values().unset_bits();
        Some(unset_bits == 0)
    }

    fn max_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        if self.len() - self.null_count() == 0 {
            return None;
        }

        let set_bits = self.values().set_bits();
        Some(set_bits > 0)
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
