use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::types::NativeType;
use num_traits::Zero;
use polars_array::PlPrimitiveArray;
use polars_array::bitmap::PlBitmap;
use polars_compute::arithmetic::pl_num::PlNumArithmetic;
use polars_compute::sum::WrappingAdd;
use polars_core::prelude::*;

type ArrayDotKernel = fn(&ArrayChunked, &ArrayChunked, usize) -> PolarsResult<Series>;

pub fn is_supported_array_dot_dtype(dtype: &DataType) -> bool {
    array_dot_kernel(dtype).is_some()
}

#[inline]
fn multiply_then_add<T>(acc: T::Sum, lhs: T, rhs: T) -> T::Sum
where
    T: PlNumArithmetic + SumCast,
    T::Sum: WrappingAdd,
{
    let product = PlNumArithmetic::wrapping_mul(lhs, rhs).into();
    acc.wrapping_add(&product)
}

struct DotRowReducer<'a, T> {
    lhs_slice: &'a [T],
    rhs_slice: &'a [T],
    lhs_inner_validity: Option<&'a Bitmap>,
    rhs_inner_validity: Option<&'a Bitmap>,
    width: usize,
}

impl<T> DotRowReducer<'_, T>
where
    T: NativeType + PlNumArithmetic + SumCast,
    T::Sum: WrappingAdd,
{
    /// # Safety
    ///
    /// Selected row indices must be valid for their corresponding outer
    /// arrays. Child values and optional validity must cover each complete
    /// fixed-width layout; `outer_len * width` must fit in `usize`.
    #[inline(always)]
    unsafe fn dot_row(&self, lhs_idx: usize, rhs_idx: usize) -> T::Sum {
        let lhs_offset = lhs_idx * self.width;
        let rhs_offset = rhs_idx * self.width;
        let (lhs_row, rhs_row) = unsafe {
            // SAFETY: function contract guarantees both ranges are in bounds
            // and calculations above did not overflow. Width zero yields 0..0.
            (
                self.lhs_slice
                    .get_unchecked(lhs_offset..lhs_offset + self.width),
                self.rhs_slice
                    .get_unchecked(rhs_offset..rhs_offset + self.width),
            )
        };

        if self.lhs_inner_validity.is_none() && self.rhs_inner_validity.is_none() {
            lhs_row
                .iter()
                .zip(rhs_row)
                .fold(T::Sum::zero(), |acc, (&lhs, &rhs)| {
                    multiply_then_add(acc, lhs, rhs)
                })
        } else {
            let mut value = T::Sum::zero();
            for (inner_idx, (&lhs, &rhs)) in lhs_row.iter().zip(rhs_row).enumerate() {
                let lhs_valid = self.lhs_inner_validity.is_none_or(|validity| {
                    // SAFETY: inner_idx < width and lhs_offset + width <= validity.len().
                    unsafe { validity.get_bit_unchecked(lhs_offset + inner_idx) }
                });
                let rhs_valid = self.rhs_inner_validity.is_none_or(|validity| {
                    // SAFETY: inner_idx < width and rhs_offset + width <= validity.len().
                    unsafe { validity.get_bit_unchecked(rhs_offset + inner_idx) }
                });
                if lhs_valid && rhs_valid {
                    value = multiply_then_add(value, lhs, rhs);
                }
            }
            value
        }
    }
}

// Keep one outer-loop call; inline `dot_row` to avoid one call per output row.
#[inline(never)]
fn dot_outer_all_valid<T>(
    row_reducer: &DotRowReducer<T>,
    lhs_broadcast: bool,
    rhs_broadcast: bool,
    output_len: usize,
) -> Vec<T::Sum>
where
    T: NativeType + PlNumArithmetic + SumCast,
    T::Sum: WrappingAdd,
{
    let mut output = Vec::with_capacity(output_len);

    for output_idx in 0..output_len {
        let lhs_idx = if lhs_broadcast { 0 } else { output_idx };
        let rhs_idx = if rhs_broadcast { 0 } else { output_idx };
        // SAFETY: broadcast uses row 0; otherwise validated equal lengths make
        // `output_idx` valid for both operands.
        output.push(unsafe { row_reducer.dot_row(lhs_idx, rhs_idx) });
    }

    output
}

fn dot_primitive<T>(
    lhs: &ArrayChunked,
    rhs: &ArrayChunked,
    output_len: usize,
) -> PolarsResult<Series>
where
    T: NativeType + PlNumArithmetic + SumCast,
    T::Sum: WrappingAdd,
    // Every `SumCast` impl sums into a type that stands for itself; saying so lets the chunk of
    // sums be read as a `ChunkedArray` of that type.
    <T::Sum as NumericNative>::PolarsType: PolarsNumericType<Native = T::Sum>,
{
    let lhs = lhs.rechunk();
    let rhs = rhs.rechunk();
    // TODO(polars-array-scalar): both sides are read as slices, so a scalar chunk is written out
    // here rather than the single row it stands for being multiplied out once.
    let lhs_array = lhs.downcast_get(0).unwrap().to_flat();
    let rhs_array = rhs.downcast_get(0).unwrap().to_flat();
    let lhs_values = lhs_array
        .values()
        .as_any()
        .downcast_ref::<PlPrimitiveArray<T>>()
        .unwrap()
        .to_flat();
    let rhs_values = rhs_array
        .values()
        .as_any()
        .downcast_ref::<PlPrimitiveArray<T>>()
        .unwrap()
        .to_flat();

    let lhs_slice = lhs_values.as_slice();
    let rhs_slice = rhs_values.as_slice();
    let lhs_inner_validity = lhs_values.validity();
    let rhs_inner_validity = rhs_values.validity();
    let width = lhs.width();
    debug_assert!(
        lhs.len()
            .checked_mul(width)
            .is_some_and(|len| lhs_slice.len() >= len)
    );
    debug_assert!(
        rhs.len()
            .checked_mul(width)
            .is_some_and(|len| rhs_slice.len() >= len)
    );
    debug_assert!(lhs_inner_validity.is_none_or(|validity| validity.len() >= lhs_slice.len()));
    debug_assert!(rhs_inner_validity.is_none_or(|validity| validity.len() >= rhs_slice.len()));
    let row_reducer = DotRowReducer {
        lhs_slice,
        rhs_slice,
        lhs_inner_validity,
        rhs_inner_validity,
        width,
    };
    let lhs_broadcast = lhs.len() == 1 && output_len != 1;
    let rhs_broadcast = rhs.len() == 1 && output_len != 1;

    // An absent outer bitmap guarantees valid output rows without scanning.
    // Child validity only filters coordinate pairs inside `DotRowReducer`.
    if lhs_array.validity().is_none() && rhs_array.validity().is_none() {
        let output = dot_outer_all_valid(&row_reducer, lhs_broadcast, rhs_broadcast, output_len);
        let output = PlPrimitiveArray::from_vec(output);
        // The sum of a `T` is a `T::Sum`, and that is the type of the chunk just built.
        return Ok(
            ChunkedArray::<<T::Sum as NumericNative>::PolarsType>::with_chunk(
                lhs.name().clone(),
                output,
            )
            .into_series(),
        );
    }

    let mut output = Vec::with_capacity(output_len);
    let mut output_validity = BitmapBuilder::with_capacity(output_len);

    for output_idx in 0..output_len {
        let lhs_idx = if lhs_broadcast { 0 } else { output_idx };
        let rhs_idx = if rhs_broadcast { 0 } else { output_idx };
        // SAFETY: broadcast uses row 0; otherwise validated equal lengths make
        // `output_idx` valid for both operands.
        let outer_valid = unsafe {
            !lhs_array.is_null_unchecked(lhs_idx) && !rhs_array.is_null_unchecked(rhs_idx)
        };
        output_validity.push(outer_valid);

        if !outer_valid {
            output.push(T::Sum::zero());
            continue;
        }

        // SAFETY: broadcast uses row 0; otherwise validated equal lengths make
        // `output_idx` valid for both operands.
        output.push(unsafe { row_reducer.dot_row(lhs_idx, rhs_idx) });
    }

    let output = PlPrimitiveArray::from_vec(output).with_validity(
        output_validity
            .into_opt_validity()
            .map(PlBitmap::from_bitmap),
    );
    // The sum of a `T` is a `T::Sum`, and that is the type of the chunk just built.
    Ok(
        ChunkedArray::<<T::Sum as NumericNative>::PolarsType>::with_chunk(
            lhs.name().clone(),
            output,
        )
        .into_series(),
    )
}

fn array_dot_kernel(dtype: &DataType) -> Option<ArrayDotKernel> {
    let kernel = match dtype {
        DataType::Int8 => dot_primitive::<i8>,
        DataType::Int16 => dot_primitive::<i16>,
        DataType::Int32 => dot_primitive::<i32>,
        DataType::Int64 => dot_primitive::<i64>,
        #[cfg(feature = "dtype-i128")]
        DataType::Int128 => dot_primitive::<i128>,
        DataType::UInt8 => dot_primitive::<u8>,
        DataType::UInt16 => dot_primitive::<u16>,
        DataType::UInt32 => dot_primitive::<u32>,
        DataType::UInt64 => dot_primitive::<u64>,
        #[cfg(feature = "dtype-u128")]
        DataType::UInt128 => dot_primitive::<u128>,
        DataType::Float32 => dot_primitive::<f32>,
        DataType::Float64 => dot_primitive::<f64>,
        _ => return None,
    };
    Some(kernel)
}

pub(super) fn array_dot(lhs: &ArrayChunked, rhs: &ArrayChunked) -> PolarsResult<Series> {
    let (lhs_inner, lhs_width) = match lhs.dtype() {
        DataType::Array(inner, width) => (inner.as_ref(), *width),
        _ => unreachable!(),
    };
    let (rhs_inner, rhs_width) = match rhs.dtype() {
        DataType::Array(inner, width) => (inner.as_ref(), *width),
        _ => unreachable!(),
    };

    assert_eq!(
        lhs_width, rhs_width,
        "arr.dot requires equal array widths, got {lhs_width} and {rhs_width}"
    );
    assert_eq!(
        lhs_inner, rhs_inner,
        "arr.dot requires matching inner dtypes, got {lhs_inner} and {rhs_inner}"
    );
    let Some(kernel) = array_dot_kernel(lhs_inner) else {
        polars_bail!(
            InvalidOperation:
            "arr.dot does not support inner dtype {lhs_inner}"
        )
    };

    let output_len = match (lhs.len(), rhs.len()) {
        (lhs_len, rhs_len) if lhs_len == rhs_len => lhs_len,
        (1, rhs_len) => rhs_len,
        (lhs_len, 1) => lhs_len,
        (lhs_len, rhs_len) => polars_bail!(
            ShapeMismatch:
            "arr.dot requires equal row counts or one-row broadcasting, got {lhs_len} and {rhs_len}"
        ),
    };

    if output_len == 0 {
        return Ok(Series::new_empty(
            lhs.name().clone(),
            &sum_output_dtype(lhs_inner),
        ));
    }

    kernel(lhs, rhs, output_len)
}
