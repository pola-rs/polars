//! Casting the Arrow arrays, over the arrays of `polars-array` that hold their values.

use arrow::array::*;
use arrow::datatypes::{ArrowDataType, TimeUnit};
use arrow::offset::Offset;
use polars_array::PlArray;
use polars_array::arrow::{export, import};
use polars_dtype::DataType;
use polars_error::{PolarsResult, polars_bail};

use super::CastOptionsImpl;

/// Casts an Arrow array to `to_type`, over the arrays of `polars-array` that hold its values.
///
/// # Panics
/// A dictionary, a map or a union *nested inside* `array` panics in the crossing over.
pub fn cast_arrow(
    array: &dyn Array,
    to_type: &ArrowDataType,
    options: CastOptionsImpl,
) -> PolarsResult<Box<dyn PlArray>> {
    let from_type = array.dtype();

    // A dictionary is the one Arrow array whose elements no array of `polars-array` holds, which
    // is what the crossing over would panic on — see `polars_compute::cast::cast_to_dictionary`.
    if matches!(from_type, ArrowDataType::Dictionary(..)) {
        return unsupported(from_type, to_type);
    }

    cast_crossed_over(&*import::from_arrow(array), from_type, to_type, options)
}

/// Casts an array that crossed over holding the values of `from_type` to `to_type`.
fn cast_crossed_over(
    array: &dyn PlArray,
    from_type: &ArrowDataType,
    to_type: &ArrowDataType,
    options: CastOptionsImpl,
) -> PolarsResult<Box<dyn PlArray>> {
    use ArrowDataType as A;

    // The values of both sides are laid out the same way and read the same way, which leaves the
    // array itself as the answer — including for a nested type, whose children are not walked.
    if from_type == to_type {
        return Ok(array.to_boxed());
    }

    let recurse = |values: &dyn PlArray, from: &ArrowDataType, to: &ArrowDataType| {
        cast_crossed_over(values, from, to, options)
    };

    match (from_type, to_type) {
        (
            A::List(from_field) | A::LargeList(from_field),
            A::List(to_field) | A::LargeList(to_field),
        ) => {
            let out = super::cast_list(super::downcast(array), |values| {
                recurse(values, from_field.dtype(), to_field.dtype())
            })?;
            Ok(Box::new(out))
        },
        #[cfg(feature = "dtype-array")]
        (A::FixedSizeList(from_field, from_width), A::FixedSizeList(to_field, to_width)) => {
            polars_error::polars_ensure!(
                from_width == to_width,
                InvalidOperation: "cannot cast Array to a different width"
            );
            let out = super::cast_fixed_size_list(super::downcast(array), |values| {
                recurse(values, from_field.dtype(), to_field.dtype())
            })?;
            Ok(Box::new(out))
        },
        #[cfg(feature = "dtype-array")]
        (A::List(from_field) | A::LargeList(from_field), A::FixedSizeList(to_field, width)) => {
            let out = super::list_to_fixed_size_list(super::downcast(array), *width, |values| {
                recurse(values, from_field.dtype(), to_field.dtype())
            })?;
            Ok(Box::new(out))
        },
        #[cfg(feature = "dtype-array")]
        (A::FixedSizeList(from_field, _), A::List(to_field) | A::LargeList(to_field)) => {
            let out = super::fixed_size_list_to_list(super::downcast(array), |values| {
                recurse(values, from_field.dtype(), to_field.dtype())
            })?;
            Ok(Box::new(out))
        },
        #[cfg(feature = "dtype-struct")]
        (A::Struct(from_fields), A::Struct(to_fields)) => {
            polars_error::polars_ensure!(
                from_fields.len() == to_fields.len(),
                InvalidOperation: "Cannot cast struct with different number of fields."
            );
            let out = super::cast_struct(super::downcast(array), |i, field| {
                recurse(field, from_fields[i].dtype(), to_fields[i].dtype())
            })?;
            Ok(Box::new(out))
        },

        // The bytes of an element are held one per value, which is what makes the two readable as
        // one another.
        (A::List(field) | A::LargeList(field), A::BinaryView) if field.dtype() == &A::UInt8 => Ok(
            Box::new(super::list_uint8_to_binview(super::downcast(array))?),
        ),
        (A::BinaryView, A::List(field) | A::LargeList(field)) if field.dtype() == &A::UInt8 => {
            let bytes = super::view_to_binary(super::downcast(array));
            Ok(Box::new(super::binary_to_list(&bytes)))
        },

        // A fixed size binary array crossed over as the one array no Polars type names, and the
        // bytes it holds are read as a binary's.
        (A::FixedSizeBinary(_), _) => {
            let view = super::fixed_size_binary_to_binview(super::downcast(array));
            let Some(to) = datatype_of(to_type) else {
                return unsupported(from_type, to_type);
            };
            super::cast(&view, &DataType::Binary, &to, options)
        },

        _ => {
            let (Some(from), Some(to)) = (datatype_of(from_type), datatype_of(to_type)) else {
                return unsupported(from_type, to_type);
            };

            // A nested type names the values of its children rather than any of its own, so a
            // pair of them is answered by the arms above.
            if from.is_nested() || to.is_nested() {
                return unsupported(from_type, to_type);
            }

            super::cast(array, &from, &to, options)
        },
    }
}

/// The Polars type an Arrow `dtype` crosses over as, or `None` for the ones no Polars type names.
fn datatype_of(dtype: &ArrowDataType) -> Option<DataType> {
    use ArrowDataType as A;

    Some(match dtype {
        // These crossed over as the bytes they hold, with their offsets widened.
        A::Utf8 | A::LargeUtf8 | A::Binary | A::LargeBinary => DataType::BinaryOffset,
        // A time of Polars counts nanoseconds, so a count of anything else is the count it is.
        A::Time32(_) => DataType::Int32,
        A::Time64(unit) if !matches!(unit, TimeUnit::Nanosecond) => DataType::Int64,
        // A time zone names no value of a timestamp, so which one it is stays the caller's to read
        // off the type it asked for — and rejecting one it does not know is not a cast's to do.
        A::Timestamp(unit, _) => DataType::Datetime(unit.into(), None),

        // A decimal of another width crossed over as the integer holding it, an interval as its
        // own element type, and a fixed size binary as the bytes it holds, never as a target.
        A::Decimal32(..) | A::Decimal64(..) | A::Decimal256(..) => return None,
        A::Interval(_) | A::FixedSizeBinary(_) => return None,
        // A dictionary is read as the type of its values, which is no answer for a cast *to* one:
        // packing the values is `cast_to_dictionary`'s to do, and nothing here holds a dictionary.
        A::Dictionary(..) => return None,

        dtype => DataType::from_arrow_dtype(dtype),
    })
}

fn unsupported<T>(from_type: &ArrowDataType, to_type: &ArrowDataType) -> PolarsResult<T> {
    polars_bail!(InvalidOperation: "casting from {from_type:?} to {to_type:?} not supported")
}

/// Writes the bytes every element's view reads out end to end, tagged as the UTF-8 they are.
pub fn utf8view_to_arrow_large_utf8(array: &Utf8ViewArray) -> Utf8Array<i64> {
    let bytes = super::view_to_binary(&import::utf8_view_from_arrow(array).into_binview());
    let (values, offsets, _, validity) = bytes.into_inner();
    let offsets = export::offsets_to_arrow(offsets);

    // SAFETY: the bytes came out of a UTF-8 view array, whose elements are valid UTF-8.
    unsafe { Utf8Array::new_unchecked(ArrowDataType::LargeUtf8, offsets, values, validity) }
}

/// Writes the bytes every element's view reads out end to end.
pub fn binview_to_arrow_large_binary(array: &BinaryViewArray) -> BinaryArray<i64> {
    let bytes = super::view_to_binary(&import::binary_view_from_arrow(array));
    export::binary_to_arrow_large_binary(&bytes)
}

/// Reads every element as the `row_width` bytes it holds, erroring if one holds another count.
pub fn binview_to_arrow_fixed_size_binary(
    array: &BinaryViewArray,
    row_width: usize,
) -> PolarsResult<FixedSizeBinaryArray> {
    let bytes = super::binview_to_fixed_binary(&import::binary_view_from_arrow(array), row_width)?;
    Ok(export::fixed_size_binary_to_arrow_fixed_size_binary(&bytes))
}

/// Widens the offsets of a list array, which reads its values as they are.
pub fn list_to_arrow_large_list(array: &ListArray<i32>, to_dtype: ArrowDataType) -> ListArray<i64> {
    ListArray::<i64>::new(
        to_dtype,
        array.offsets().into(),
        array.values().clone(),
        array.validity().cloned(),
    )
}

/// Reads the bytes of a UTF-8 array as the bytes they are.
pub fn utf8_to_binary<O: Offset>(from: &Utf8Array<O>, to_dtype: ArrowDataType) -> BinaryArray<O> {
    // SAFETY: erasure of an invariant is always safe
    BinaryArray::<O>::new(
        to_dtype,
        from.offsets().clone(),
        from.values().clone(),
        from.validity().cloned(),
    )
}

/// Reads a time of nanoseconds as the microsecond it falls in.
pub fn time64ns_to_time64us(from: &PrimitiveArray<i64>) -> PrimitiveArray<i64> {
    let out = super::rescale_time(
        &import::primitive_from_arrow(from),
        1_000_000_000,
        1_000_000,
    );
    export::primitive_to_arrow_primitive(&out).to(ArrowDataType::Time64(TimeUnit::Microsecond))
}
