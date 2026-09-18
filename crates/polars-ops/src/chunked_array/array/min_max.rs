use polars_array::bitmap::combine_validities_and;
use polars_arrow::bitmap::bitmask::BitMask;
use polars_compute::min_max::MinMaxKernel;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::vec::PushUnchecked;

/// Reduces every element of `arr` — whose values `values` holds — to one element.
fn array_agg<T, S, F1, F2>(
    arr: &PlFixedSizeListArray,
    values: &PlPrimitiveArray<T>,
    slice_agg: F1,
    arr_agg: F2,
) -> PlPrimitiveArray<S>
where
    T: NumericNative,
    S: NumericNative,
    F1: Fn(&[T]) -> Option<S>,
    F2: Fn(&PlPrimitiveArray<T>) -> Option<S>,
{
    let (width, length) = (arr.width(), arr.len());

    if width == 0 {
        return PlPrimitiveArray::new_full_null(length);
    }

    if arr.values_are_scalar() {
        return match arr_agg(values) {
            Some(value) => PlPrimitiveArray::new_scalar(value, length),
            None => PlPrimitiveArray::new_full_null(length),
        };
    }

    if !values.has_nulls() {
        return match values.scalar_value_ignore_validity() {
            Some(value) => {
                let reduced =
                    slice_agg(&[value]).expect("a row of one value reduces to that value");
                PlPrimitiveArray::new_scalar(reduced, length)
            },
            None => values
                .flat_values()
                .unwrap()
                .as_slice()
                .chunks_exact(width)
                .map(|sl| slice_agg(sl).unwrap())
                .collect_arr_trusted(),
        };
    }

    let validity = values
        .validity()
        .expect("a null value is one the mask marks as not being there");
    let Some(validity) = validity.flat_bitmap() else {
        return PlPrimitiveArray::new_full_null(length);
    };

    if let Some(value) = values.scalar_value_ignore_validity() {
        let reduced = slice_agg(&[value]).expect("a row of one value reduces to that value");
        return (0..length)
            .map(|row| {
                let nulls = validity.null_count_range(row * width, width);
                (nulls < width).then_some(reduced)
            })
            .collect_arr_trusted();
    }

    let mask = BitMask::from_bitmap(validity);
    let mut row = Vec::with_capacity(width);
    values
        .flat_values()
        .unwrap()
        .as_slice()
        .chunks_exact(width)
        .enumerate()
        .map(|(index, row_values)| {
            let start = index * width;
            let Some(mut bits) = (width <= 32).then(|| {
                let word = mask.get_u32(start);
                if width == 32 {
                    word
                } else {
                    word & ((1u32 << width) - 1)
                }
            }) else {
                return match validity.null_count_range(start, width) {
                    0 => slice_agg(row_values),
                    nulls if nulls == width => None,
                    _ => {
                        row.clear();
                        row.extend(
                            (0..width)
                                .filter(|offset| validity.get_bit(start + offset))
                                .map(|offset| row_values[offset]),
                        );
                        slice_agg(&row)
                    },
                };
            };

            match bits.count_ones() as usize {
                0 => None,
                present if present == width => slice_agg(row_values),
                _ => {
                    row.clear();
                    while bits != 0 {
                        let offset = bits.trailing_zeros() as usize;
                        // SAFETY: the word covers `width` bits of the mask, one per value of the
                        // row, and `row` was built with room for that many.
                        unsafe {
                            row.push_unchecked(*row_values.get_unchecked(offset));
                        }
                        bits &= bits - 1;
                    }
                    slice_agg(&row)
                },
            }
        })
        .collect_arr_trusted()
}

pub(super) enum AggType {
    Min,
    Max,
}

fn agg_min<T>(arr: &PlFixedSizeListArray, values: &PlPrimitiveArray<T>) -> PlPrimitiveArray<T>
where
    T: NumericNative,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    array_agg(
        arr,
        values,
        MinMaxKernel::min_ignore_nan_kernel,
        MinMaxKernel::min_ignore_nan_kernel,
    )
}

fn agg_max<T>(arr: &PlFixedSizeListArray, values: &PlPrimitiveArray<T>) -> PlPrimitiveArray<T>
where
    T: NumericNative,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    array_agg(
        arr,
        values,
        MinMaxKernel::max_ignore_nan_kernel,
        MinMaxKernel::max_ignore_nan_kernel,
    )
}

/// Reduces every element of `ca` to one element.
pub(super) fn array_dispatch(ca: &ArrayChunked, agg_type: AggType) -> Series {
    with_match_physical_numeric_polars_type!(ca.inner_dtype(), |$T| {
        type N = <$T as PolarsNumericType>::Native;

        let chunks = ca.downcast_iter().map(|arr| {
            let values = arr
                .values()
                .as_any()
                .downcast_ref::<PlPrimitiveArray<N>>()
                .unwrap();

            let out = match agg_type {
                AggType::Min => agg_min(arr, values),
                AggType::Max => agg_max(arr, values),
            };

            let validity = combine_validities_and(out.validity(), arr.validity());
            out.with_validity(validity)
        });

        ChunkedArray::<$T>::from_chunk_iter(ca.name().clone(), chunks).into_series()
    })
}
