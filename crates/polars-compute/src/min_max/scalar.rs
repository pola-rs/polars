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

        if let Some(validity) = self.validity()
            && validity.unset_bits() > 0
        {
            // min is true only if every valid position has value=true.
            Some(self.values().num_intersections_with(validity) == validity.set_bits())
        } else {
            Some(self.values().unset_bits() == 0)
        }
    }

    fn max_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
        if self.len() - self.null_count() == 0 {
            return None;
        }

        if let Some(validity) = self.validity()
            && validity.unset_bits() > 0
        {
            // max is true if any valid position has value=true.
            Some(self.values().intersects_with(validity))
        } else {
            Some(self.values().set_bits() > 0)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boolean_min_max_no_nulls() {
        let all_true = BooleanArray::from_slice(&[true, true]);
        assert_eq!(all_true.min_ignore_nan_kernel(), Some(true));
        assert_eq!(all_true.max_ignore_nan_kernel(), Some(true));

        let all_false = BooleanArray::from_slice(&[false, false]);
        assert_eq!(all_false.min_ignore_nan_kernel(), Some(false));
        assert_eq!(all_false.max_ignore_nan_kernel(), Some(false));

        let mixed = BooleanArray::from_slice(&[true, false]);
        assert_eq!(mixed.min_ignore_nan_kernel(), Some(false));
        assert_eq!(mixed.max_ignore_nan_kernel(), Some(true));
    }

    #[test]
    fn test_boolean_min_max_all_null() {
        let all_null = BooleanArray::from(&[None, None, None]);
        assert_eq!(all_null.min_ignore_nan_kernel(), None);
        assert_eq!(all_null.max_ignore_nan_kernel(), None);
    }

    #[test]
    fn test_boolean_min_max_empty() {
        let empty = BooleanArray::from(&[] as &[Option<bool>]);
        assert_eq!(empty.min_ignore_nan_kernel(), None);
        assert_eq!(empty.max_ignore_nan_kernel(), None);
    }

    #[test]
    fn test_boolean_min_with_nulls() {
        let arr = BooleanArray::from(&[Some(true), Some(true), None]);
        assert_eq!(arr.min_ignore_nan_kernel(), Some(true));

        let arr = BooleanArray::from(&[Some(true), None]);
        assert_eq!(arr.min_ignore_nan_kernel(), Some(true));

        let arr = BooleanArray::from(&[Some(true), Some(false), None]);
        assert_eq!(arr.min_ignore_nan_kernel(), Some(false));

        let arr = BooleanArray::from(&[Some(false), Some(false), None]);
        assert_eq!(arr.min_ignore_nan_kernel(), Some(false));
    }

    #[test]
    fn test_boolean_max_with_nulls() {
        let arr = BooleanArray::from(&[Some(false), Some(false), None]);
        assert_eq!(arr.max_ignore_nan_kernel(), Some(false));

        let arr = BooleanArray::from(&[Some(false), None]);
        assert_eq!(arr.max_ignore_nan_kernel(), Some(false));

        let arr = BooleanArray::from(&[Some(true), Some(false), None]);
        assert_eq!(arr.max_ignore_nan_kernel(), Some(true));

        let arr = BooleanArray::from(&[Some(true), Some(true), None]);
        assert_eq!(arr.max_ignore_nan_kernel(), Some(true));
    }
}
