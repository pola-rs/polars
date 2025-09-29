use core::num;

use super::super::rank::*;
use super::*;

pub fn rolling_rank<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    _params: Option<RollingFnParams>,
) -> ArrayRef
where
    T: NativeType + num_traits::Num,
{
    assert!(weights.is_none(), "weights are not supported for rank");

    // let offset_fn = match center {
    //     true => det_offsets_center,
    //     false => det_offsets,
    // };
    // if let Some(_) = weights {
    //     todo!("weighted rolling rank not implemented")
    // } else {
    //     rolling_apply_agg_window::<RankWindow<_>, _, _>(
    //         values,
    //         window_size,
    //         min_periods,
    //         offset_fn,
    //         None,
    //     )
    // }
    todo!("[amber] implement rolling rank with nulls support")
}
