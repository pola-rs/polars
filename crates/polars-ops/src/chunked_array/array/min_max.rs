use polars_array::ArrayRepr;
use polars_compute::min_max::MinMaxKernel;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

/// Reduces every run of `width` consecutive elements of `values` to one element, with `slice_agg`
/// where none of them is null and `arr_agg` otherwise.
///
/// A values buffer that repeats a single value is reduced once, in `O(1)`: every row holds the
/// same `width` copies of it, and the answer repeats a single value in turn. That reads a row of
/// one value in place of a row of `width` of them, which the aggregations here — a minimum and a
/// maximum — are free to do, since repeating a value leaves neither of them anywhere else. An
/// aggregation that counts its elements, a sum among them, would need the length instead.
///
/// Otherwise the rows are sliced off one at a time, which is `O(1)` per row and keeps whatever
/// representation the row is in.
fn array_agg<T, S, F1, F2>(
    values: &PlPrimitiveArray<T>,
    width: usize,
    slice_agg: F1,
    arr_agg: F2,
) -> PlPrimitiveArray<S>
where
    T: NumericNative,
    S: NumericNative,
    F1: Fn(&[T]) -> Option<S>,
    F2: Fn(&PlPrimitiveArray<T>) -> Option<S>,
{
    // Without a null anywhere the rows are read straight out of the values buffer, in whichever
    // representation it is in.
    if !values.has_nulls() {
        return match values.values_repr() {
            // Every row is the same `width` copies of the one value, and so reduces to it — as
            // does the answer, which repeats a single value in turn.
            ArrayRepr::Scalar(value) => {
                let reduced =
                    slice_agg(&[value]).expect("a row of one value reduces to that value");
                PlPrimitiveArray::new_scalar(reduced, values.len() / width)
            },
            // The rows are runs of the values buffer, which the kernel that reads a slice reduces
            // without a validity mask to consult.
            ArrayRepr::Flat(flat) => flat
                .as_slice()
                .chunks_exact(width)
                .map(|sl| slice_agg(sl).unwrap())
                .collect_arr(),
        };
    }

    (0..values.len())
        .step_by(width)
        .map(|start| arr_agg(&values.clone().sliced(start, width)))
        .collect_arr()
}

pub(super) enum AggType {
    Min,
    Max,
}

fn agg_min<T>(values: &PlPrimitiveArray<T>, width: usize) -> PlPrimitiveArray<T>
where
    T: NumericNative,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    array_agg(
        values,
        width,
        MinMaxKernel::min_ignore_nan_kernel,
        MinMaxKernel::min_ignore_nan_kernel,
    )
}

fn agg_max<T>(values: &PlPrimitiveArray<T>, width: usize) -> PlPrimitiveArray<T>
where
    T: NumericNative,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    array_agg(
        values,
        width,
        MinMaxKernel::max_ignore_nan_kernel,
        MinMaxKernel::max_ignore_nan_kernel,
    )
}

pub(super) fn array_dispatch(
    name: PlSmallStr,
    values: &Series,
    width: usize,
    agg_type: AggType,
) -> Series {
    with_match_physical_numeric_polars_type!(values.dtype(), |$T| {
        let ca: &ChunkedArray<$T> = values.as_ref().as_ref().as_ref();
        let chunks = ca.downcast_iter().map(|arr| {
            match agg_type {
                AggType::Min => agg_min(arr, width),
                AggType::Max => agg_max(arr, width),
            }
        });

        ChunkedArray::<$T>::from_chunk_iter(name, chunks).into_series()
    })
}

#[cfg(test)]
mod tests {
    use polars_core::prelude::*;

    use super::{AggType, array_dispatch};

    /// The chunk `array_dispatch` reduces `values` to, row by row, over runs of `width`.
    fn reduced(values: Int32Chunked, width: usize, agg_type: AggType) -> PlPrimitiveArray<i32> {
        let out = array_dispatch(PlSmallStr::EMPTY, &values.into_series(), width, agg_type);
        let ca = out.i32().expect("min and max keep the type they reduce");
        let [chunk] = ca.downcast_iter().collect::<Vec<_>>()[..] else {
            panic!("one chunk in, one chunk out");
        };
        chunk.clone()
    }

    /// Every row of a values buffer that repeats a single value is the same `width` copies of it,
    /// and so reduces to it: the answer repeats a single value in turn, and neither buffer is
    /// ever written out one slot per element.
    #[test]
    fn a_repeated_value_reduces_to_itself() {
        let values =
            Int32Chunked::with_chunk(PlSmallStr::EMPTY, PlPrimitiveArray::new_scalar(7i32, 6));

        for agg_type in [AggType::Min, AggType::Max] {
            let out = reduced(values.clone(), 3, agg_type);
            assert!(out.is_scalar(), "{out:?} was written out one slot per row");
            assert_eq!(out, PlPrimitiveArray::new_scalar(7i32, 2));
        }
    }

    /// A repeated value under a mask laid out one bit per element is reduced row by row, each row
    /// sliced off in `O(1)` — a row with a value left in it reduces to that value, and one with
    /// none left reduces to a null.
    #[test]
    fn a_repeated_value_under_a_mask_is_reduced_row_by_row() {
        let values = Int32Chunked::with_chunk(
            PlSmallStr::EMPTY,
            PlPrimitiveArray::new_scalar(7i32, 6).with_validity(Some(
                [true, false, true, false, false, false]
                    .into_iter()
                    .collect(),
            )),
        );

        for agg_type in [AggType::Min, AggType::Max] {
            assert_eq!(
                reduced(values.clone(), 3, agg_type),
                PlPrimitiveArray::from_iter([Some(7i32), None]),
            );
        }
    }

    /// Rows laid out one slot per element are reduced as they always have been, whether or not a
    /// null is in the way.
    #[test]
    fn every_row_is_reduced() {
        let flat = Int32Chunked::from_slice(PlSmallStr::EMPTY, &[3, -1, 9, 4, 8, 2]);
        assert_eq!(
            reduced(flat.clone(), 3, AggType::Min),
            PlPrimitiveArray::from_vec(vec![-1i32, 2]),
        );
        assert_eq!(
            reduced(flat, 3, AggType::Max),
            PlPrimitiveArray::from_vec(vec![9i32, 8]),
        );

        let with_nulls = Int32Chunked::from_slice_options(
            PlSmallStr::EMPTY,
            &[Some(3), None, Some(9), None, None, None],
        );
        assert_eq!(
            reduced(with_nulls.clone(), 3, AggType::Min),
            PlPrimitiveArray::from_iter([Some(3i32), None]),
        );
        assert_eq!(
            reduced(with_nulls, 3, AggType::Max),
            PlPrimitiveArray::from_iter([Some(9i32), None]),
        );
    }
}
