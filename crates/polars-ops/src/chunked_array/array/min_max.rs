use arrow::bitmap::bitmask::BitMask;
use polars_array::bitmap::combine_validities_and;
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

    // A row of no values holds nothing to reduce, so every element reduces to nothing.
    if width == 0 {
        return PlPrimitiveArray::new_full_null(length);
    }

    // The values hold the one list every element reads, so they all reduce to the same element:
    // that list is reduced once and the answer repeats it, rather than the list being read —
    // and reduced — once per element.
    if arr.values_are_scalar() {
        return match arr_agg(values) {
            Some(value) => PlPrimitiveArray::new_scalar(value, length),
            // The one list every element reads reduces to nothing, and so does every element.
            None => PlPrimitiveArray::new_full_null(length),
        };
    }

    // Without a null anywhere the rows are read straight out of the values buffer, in whichever
    // representation it is in.
    if !values.has_nulls() {
        return match values.scalar_value_ignore_validity() {
            // Every row is the same `width` copies of the one value, and so reduces to it — as
            // does the answer, which repeats a single value in turn.
            Some(value) => {
                let reduced =
                    slice_agg(&[value]).expect("a row of one value reduces to that value");
                PlPrimitiveArray::new_scalar(reduced, length)
            },
            // The rows are runs of the values buffer, which the kernel that reads a slice reduces
            // without a validity mask to consult.
            None => values
                .flat_values()
                .unwrap()
                .as_slice()
                .chunks_exact(width)
                .map(|sl| slice_agg(sl).unwrap())
                .collect_arr_trusted(),
        };
    }

    // A value under the rows is null, so the mask has to be consulted. It — and the values — are
    // read once here, leaving each row a slice of a buffer and a run of bits to reduce, rather
    // than an array to build, walk and drop per row.
    let validity = values
        .validity()
        .expect("a null value is one the mask marks as not being there");
    let Some(validity) = validity.flat_bitmap() else {
        // One bit stands for every value, and it says none of them is there, so no row holds a
        // value to reduce and every element reduces to nothing.
        return PlPrimitiveArray::new_full_null(length);
    };

    // Every row is `width` copies of the one value the buffer holds, so it reduces to that value
    // wherever the mask leaves it an element at all, and to nothing where it leaves none.
    if let Some(value) = values.scalar_value_ignore_validity() {
        let reduced = slice_agg(&[value]).expect("a row of one value reduces to that value");
        return (0..length)
            .map(|row| {
                let nulls = validity.null_count_range(row * width, width);
                (nulls < width).then_some(reduced)
            })
            .collect_arr_trusted();
    }

    // One row per element and one slot per value: a row is the run of the values buffer it
    // already is, and the values of it that are there are read out into `row` to be reduced as a
    // slice of their own — which is what keeps the kernel off a mask it would walk a bit at a time.
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
            // A row up to 32 wide is one load of the mask, which says how many of its values are
            // there and which ones in the same word; a wider one is counted a word at a time and
            // read a bit at a time.
            let Some(mut bits) = (width <= 32).then(|| {
                // The load reads as many bits as the mask has left, of which this row's are the
                // first `width`.
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
                // Nothing under this row is there at all, so it reduces to nothing.
                0 => None,
                // Nothing under it is null, so it reduces as the slice it is.
                present if present == width => slice_agg(row_values),
                // Some values are there and some are not: the ones that are are the bits the
                // word sets, and they are taken aside into a row of their own.
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
            // The values a chunk holds, in the representation it holds them in: reading them off
            // the chunk rather than off a flattened copy of it is what leaves a chunk that
            // repeats a single list holding that list once.
            let values = arr
                .values()
                .as_any()
                .downcast_ref::<PlPrimitiveArray<N>>()
                .unwrap();

            let out = match agg_type {
                AggType::Min => agg_min(arr, values),
                AggType::Max => agg_max(arr, values),
            };

            // An element that is null holds no values to reduce, whatever the values under it
            // read as, so it reduces to nothing in turn.
            let validity = combine_validities_and(out.validity(), arr.validity());
            out.with_validity(validity)
        });

        ChunkedArray::<$T>::from_chunk_iter(ca.name().clone(), chunks).into_series()
    })
}
