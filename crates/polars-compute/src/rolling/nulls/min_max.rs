use polars_utils::min_max::{MaxPropagateNan, MinPropagateNan};

use super::super::min_max::MinMaxWindow;
use super::van_herk::rolling_minmax_van_herk_nulls;

pub type MinWindow<'a, T> = MinMaxWindow<'a, T, MinPropagateNan>;
pub type MaxWindow<'a, T> = MinMaxWindow<'a, T, MaxPropagateNan>;

use super::*;

macro_rules! rolling_minmax_nulls_func {
    ($rolling_m:ident, $policy:ident, $is_min:literal) => {
        pub fn $rolling_m<T>(
            arr: &PrimitiveArray<T>,
            window_size: usize,
            min_periods: usize,
            center: bool,
            weights: Option<&[f64]>,
            _params: Option<RollingFnParams>,
        ) -> ArrayRef
        where
            T: NativeType + PartialOrd + IsFloat + Bounded,
        {
            if weights.is_some() {
                panic!("weights not yet supported on array with null values")
            }
            rolling_minmax_van_herk_nulls::<$is_min, T, $policy>(
                arr,
                window_size,
                min_periods,
                center,
            )
        }
    };
}

rolling_minmax_nulls_func!(rolling_min, MinPropagateNan, true);
rolling_minmax_nulls_func!(rolling_max, MaxPropagateNan, false);
