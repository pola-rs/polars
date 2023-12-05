// use std::iter::zip;

// use polars_core::chunked_array::builder::ListPrimitiveChunkedBuilder;
// use polars_core::prelude::{
//     polars_bail, polars_ensure, ChunkedArray, PolarsIterator, PolarsResult, *,
// };
use polars_core::prelude::{polars_bail, polars_ensure, PolarsResult};
use polars_core::series::Series;

pub(super) fn temporal_series_to_i64_scalar(s: &Series) -> Option<i64> {
    s.to_physical_repr().get(0).unwrap().extract::<i64>()
}
pub(super) fn ensure_range_bounds_contain_exactly_one_value(
    start: &Series,
    end: &Series,
) -> PolarsResult<()> {
    polars_ensure!(
        start.len() == 1,
        ComputeError: "`start` must contain exactly one value, got {} values", start.len()
    );
    polars_ensure!(
        end.len() == 1,
        ComputeError: "`end` must contain exactly one value, got {} values", end.len()
    );
    Ok(())
}

pub(super) fn broadcast_scalar_inputs(
    start: Series,
    end: Series,
) -> PolarsResult<(Series, Series)> {
    match (start.len(), end.len()) {
        (len1, len2) if len1 == len2 => Ok((start, end)),
        (1, len2) => {
            let start_matched = start.new_from_index(0, len2);
            Ok((start_matched, end))
        },
        (len1, 1) => {
            let end_matched = end.new_from_index(0, len1);
            Ok((start, end_matched))
        },
        (len1, len2) => {
            polars_bail!(
                ComputeError:
                "lengths of `start` ({}) and `end` ({}) do not match",
                len1, len2
            )
        },
    }
}

// pub(super) fn broadcast_scalar_inputs_iter<T>(
//     start: &ChunkedArray<T>,
//     end: &ChunkedArray<T>,
//     builder: ListPrimitiveChunkedBuilder<T>,
//     range_impl: dyn Fn(T::Native, T::Native) -> PolarsResult<TimeChunked>,
// ) -> PolarsResult<Series>
// where
//     T: PolarsNumericType,
// {
//     match (start.len(), end.len()) {
//         (len_start, len_end) if len_start == len_end => {
//             for (start, end) in zip(start, end) {
//                 match (start, end) {
//                     (Some(start), Some(end)) => {
//                         let rng = range_impl(start, end)?;
//                         builder.append_slice(rng.cont_slice().unwrap())
//                     },
//                     _ => builder.append_null(),
//                 }
//             }
//         },
//         (1, len_end) => {
//             let start_scalar = unsafe { start.get_unchecked(0) };
//             match start_scalar {
//                 Some(start) => {
//                     let range_impl = |end| range_impl(start, end);
//                     for end_scalar in end {
//                         match end_scalar {
//                             Some(end) => {
//                                 let rng = range_impl(end)?;
//                                 builder.append_slice(rng.cont_slice().unwrap())
//                             },
//                             None => builder.append_null(),
//                         }
//                     }
//                 },
//                 None => {
//                     for _ in 0..len_end {
//                         builder.append_null()
//                     }
//                 },
//             }
//         },
//         (len_start, 1) => {
//             let end_scalar = unsafe { end.get_unchecked(0) };
//             match end_scalar {
//                 Some(end) => {
//                     let range_impl = |start| range_impl(start, end);
//                     for start_scalar in start {
//                         match start_scalar {
//                             Some(start) => {
//                                 let rng = range_impl(start)?;
//                                 builder.append_slice(rng.cont_slice().unwrap())
//                             },
//                             None => builder.append_null(),
//                         }
//                     }
//                 },
//                 None => {
//                     for _ in 0..len_start {
//                         builder.append_null()
//                     }
//                 },
//             }
//         },
//         (len_start, len_end) => {
//             polars_bail!(
//                 ComputeError:
//                 "lengths of `start` ({}) and `end` ({}) do not match",
//                 len_start, len_end
//             )
//         },
//     };
//     let out = builder.finish().into_series();
//     Ok(out)
// }
