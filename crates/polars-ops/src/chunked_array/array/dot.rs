use std::iter::Sum;
use std::ops::{AddAssign, Mul};

use arrow::array::{Array, PrimitiveArray};
use arrow::bitmap::BitmapBuilder;
use arrow::types::NativeType;
use polars_core::prelude::*;

fn dot_primitive<T>(
    lhs: &ArrayChunked,
    rhs: &ArrayChunked,
    output_len: usize,
) -> PolarsResult<Series>
where
    T: NativeType + Default + AddAssign + Mul<Output = T> + Sum,
{
    let lhs = lhs.rechunk();
    let rhs = rhs.rechunk();
    let lhs_array = lhs.downcast_get(0).unwrap();
    let rhs_array = rhs.downcast_get(0).unwrap();
    let lhs_values = lhs_array
        .values()
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .unwrap();
    let rhs_values = rhs_array
        .values()
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .unwrap();

    let lhs_slice = lhs_values.values().as_slice();
    let rhs_slice = rhs_values.values().as_slice();
    let lhs_inner_validity = lhs_values.validity();
    let rhs_inner_validity = rhs_values.validity();
    let width = lhs.width();
    let lhs_broadcast = lhs.len() == 1 && output_len != 1;
    let rhs_broadcast = rhs.len() == 1 && output_len != 1;

    let mut output = Vec::with_capacity(output_len);
    let mut output_validity = BitmapBuilder::with_capacity(output_len);

    for output_idx in 0..output_len {
        let lhs_idx = if lhs_broadcast { 0 } else { output_idx };
        let rhs_idx = if rhs_broadcast { 0 } else { output_idx };
        let outer_valid = lhs_array.is_valid(lhs_idx) && rhs_array.is_valid(rhs_idx);
        output_validity.push(outer_valid);

        if !outer_valid {
            output.push(T::default());
            continue;
        }

        let lhs_offset = lhs_idx * width;
        let rhs_offset = rhs_idx * width;
        let lhs_row = &lhs_slice[lhs_offset..lhs_offset + width];
        let rhs_row = &rhs_slice[rhs_offset..rhs_offset + width];

        let value = if lhs_inner_validity.is_none() && rhs_inner_validity.is_none() {
            lhs_row
                .iter()
                .zip(rhs_row)
                .map(|(&lhs, &rhs)| lhs * rhs)
                .sum()
        } else {
            let mut value = T::default();
            for inner_idx in 0..width {
                let lhs_value_idx = lhs_offset + inner_idx;
                let rhs_value_idx = rhs_offset + inner_idx;
                let lhs_valid =
                    lhs_inner_validity.is_none_or(|validity| validity.get_bit(lhs_value_idx));
                let rhs_valid =
                    rhs_inner_validity.is_none_or(|validity| validity.get_bit(rhs_value_idx));
                if lhs_valid && rhs_valid {
                    value += lhs_row[inner_idx] * rhs_row[inner_idx];
                }
            }
            value
        };
        output.push(value);
    }

    let output =
        PrimitiveArray::from_data_default(output.into(), output_validity.into_opt_validity());
    Series::try_from((lhs.name().clone(), vec![Box::new(output) as ArrayRef]))
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

    polars_ensure!(
        lhs_width == rhs_width,
        ShapeMismatch:
        "arr.dot requires equal array widths, got {lhs_width} and {rhs_width}"
    );
    polars_ensure!(
        lhs_inner == rhs_inner,
        SchemaMismatch:
        "arr.dot requires matching inner dtypes, got {lhs_inner} and {rhs_inner}"
    );
    polars_ensure!(
        matches!(lhs_inner, DataType::Float32 | DataType::Float64),
        InvalidOperation:
        "arr.dot supports Float32 and Float64 arrays, got array[{lhs_inner}, {lhs_width}]"
    );

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
        return Ok(Series::new_empty(lhs.name().clone(), lhs_inner));
    }

    match lhs_inner {
        DataType::Float32 => dot_primitive::<f32>(lhs, rhs, output_len),
        DataType::Float64 => dot_primitive::<f64>(lhs, rhs, output_len),
        _ => unreachable!(),
    }
}
