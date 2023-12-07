use arrow::array::{Array, PrimitiveArray};
use arrow::types::NativeType;
use polars_utils::min_max::MinMax;

use super::MinMaxKernel;

fn reduce_vals<T, F>(v: &PrimitiveArray<T>, f: F) -> Option<T>
where  
    T: NativeType,
    F: Fn(T, T) -> T
{
    if v.null_count() == 0 {
        v.values().iter().copied().reduce(f)
    } else {
        v.non_null_values_iter().copied().reduce(f)
    }
}

impl<T: NativeType + MinMax + super::NotSimdPrimitive> MinMaxKernel for PrimitiveArray<T> {
    type Scalar = T;

    fn min_ignore_nan(&self) -> Option<Self::Scalar> {
        reduce_vals(self, MinMax::min_ignore_nan)
    }

    fn max_ignore_nan(&self) -> Option<Self::Scalar> {
        reduce_vals(self, MinMax::max_ignore_nan)
    }

    fn min_propagate_nan(&self) -> Option<Self::Scalar> {
        reduce_vals(self, MinMax::min_propagate_nan)
    }

    fn max_propagate_nan(&self) -> Option<Self::Scalar> {
        reduce_vals(self, MinMax::max_propagate_nan)
    }
}
