//! Casting the view-backed byte arrays of `polars-array`, which every string is held by.

use arrow::types::NativeType;
use num_traits::FromBytes;
use polars_array::{
    PlArray, PlBinaryArray, PlBinaryViewArray, PlBitmap, PlFixedSizeBinaryArray,
    PlFixedSizeBinaryArrayBuilder, PlFixedSizeListArray, PlPrimitiveArray, StaticArrayBuilder,
};
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};

use super::binary::Parse;
use super::{CastOptionsImpl, MaskBuilder, and_validity, map_bytes_fallible};

/// Reads the text of every element as the number it stands for, leaving a null where it stands for
/// none.
pub fn binview_to_parsed<T: NativeType + Parse>(
    from: &PlBinaryViewArray,
    options: CastOptionsImpl,
) -> PlPrimitiveArray<T> {
    if options.partial {
        unimplemented!()
    }

    map_bytes(from, T::parse)
}

/// Reads the text of every element as the decimal it stands for.
#[cfg(feature = "dtype-decimal")]
pub fn binview_to_decimal(
    from: &PlBinaryViewArray,
    precision: usize,
    scale: usize,
) -> PlPrimitiveArray<i128> {
    map_bytes(from, |value| {
        crate::decimal::str_to_dec128(value, precision, scale, false)
    })
}

/// Reads the bytes of every element as the number they are the memory of.
pub fn binview_to_primitive<T>(
    from: &PlBinaryViewArray,
    is_little_endian: bool,
) -> PlPrimitiveArray<T>
where
    T: FromBytes + NativeType,
    for<'a> &'a <T as FromBytes>::Bytes: TryFrom<&'a [u8]>,
{
    map_bytes(from, |value| {
        let bytes = value.try_into().ok()?;
        Some(if is_little_endian {
            <T as FromBytes>::from_le_bytes(bytes)
        } else {
            <T as FromBytes>::from_be_bytes(bytes)
        })
    })
}

/// Applies `op` to the bytes of every element, leaving a null wherever it answers `None`.
fn map_bytes<O, F>(from: &PlBinaryViewArray, op: F) -> PlPrimitiveArray<O>
where
    O: NativeType,
    F: Fn(&[u8]) -> Option<O>,
{
    map_bytes_fallible(
        from.len(),
        from.scalar_value_ignore_validity(),
        from.broadcast_values_iter(from.len()),
        from.validity(),
        op,
    )
}

/// Writes the bytes every element's view reads out end to end, which is what an offset-backed
/// binary array holds.
pub fn view_to_binary(from: &PlBinaryViewArray) -> PlBinaryArray {
    // The one value every element of a scalar chunk reads is written once, and the offsets repeat
    // the range it lies in.
    if let Some(value) = from.scalar_value_ignore_validity() {
        return PlBinaryArray::new_scalar(value, from.len())
            .with_validity(from.validity().map(PlBitmap::from));
    }

    PlBinaryArray::from_values_iter(from.values_iter())
        .with_validity(from.validity().map(PlBitmap::from))
}

/// Reads every element as the `row_width` bytes it holds, erroring if one holds another count.
pub fn binview_to_fixed_binary(
    from: &PlBinaryViewArray,
    row_width: usize,
) -> PolarsResult<PlFixedSizeBinaryArray> {
    polars_ensure!(
        row_width != 0,
        ComputeError:
        "not implemented: FixedSizeBinary with row size of 0"
    );

    let mut out = PlFixedSizeBinaryArrayBuilder::with_capacity(row_width, from.len());
    let mut length_mismatch_idx = usize::MAX;

    for (i, bytes) in from.iter().enumerate() {
        if let Some(bytes) = bytes
            && bytes.len() == row_width
        {
            out.push_value(bytes);
        } else {
            length_mismatch_idx = usize::min(
                if bytes.is_some() { i } else { usize::MAX },
                length_mismatch_idx,
            );

            out.push_null()
        }
    }

    let out = out.freeze();

    if length_mismatch_idx != usize::MAX {
        let length = from.value(length_mismatch_idx).len();

        polars_bail!(
            ComputeError:
            "could not cast BinaryView to FixedSizeBinary({row_width}): \
            bytes at index {length_mismatch_idx} had mismatching length {length}."
        )
    }

    Ok(out)
}

/// Reads the bytes of every element as the `array_width` numbers they are the memory of, leaving a
/// null where an element holds another count of them.
pub fn binview_to_fixed_size_list<T, const IS_LITTLE_ENDIAN: bool>(
    from: &PlBinaryViewArray,
    array_width: usize,
) -> PolarsResult<PlFixedSizeListArray>
where
    T: FromBytes + NativeType,
    for<'a> &'a <T as FromBytes>::Bytes: TryFrom<&'a [u8]>,
{
    let element_size = size_of::<T>();
    // The maximum number of primitives in the result:
    let primitive_length = from.len().checked_mul(array_width).ok_or_else(|| {
        polars_err!(
            InvalidOperation:
            "array chunk length * number of items ({} * {}) is too large",
            from.len(),
            array_width
        )
    })?;
    // The size of each array, in bytes:
    let row_size_bytes = element_size.checked_mul(array_width).ok_or_else(|| {
        polars_err!(
            InvalidOperation:
            "array size in bytes ({} * {}) is too large",
            element_size,
            array_width
        )
    })?;

    let mut out: Vec<T> = vec![T::zeroed(); primitive_length];
    let mut fits = MaskBuilder::with_capacity(from.len());

    for (index, value) in from.broadcast_values_iter(from.len()).enumerate() {
        if value.len() != row_size_bytes {
            fits.push(false);
            continue;
        }
        fits.push(true);

        let out = &mut out[index * array_width..(index + 1) * array_width];
        if cfg!(target_endian = "little") && IS_LITTLE_ENDIAN {
            // Fast path: the memory of the numbers is the memory the element holds.
            let out = bytemuck::cast_slice_mut::<T, u8>(out);
            out.copy_from_slice(value);
            continue;
        }

        for (out, bytes) in out.iter_mut().zip(value.chunks_exact(element_size)) {
            // SAFETY: the chunks are `element_size` bytes wide, which is the width of `T`.
            let bytes = unsafe { bytes.try_into().unwrap_unchecked() };
            *out = if IS_LITTLE_ENDIAN {
                <T as FromBytes>::from_le_bytes(bytes)
            } else {
                <T as FromBytes>::from_be_bytes(bytes)
            };
        }
    }

    let validity = match fits.finish() {
        None => from.validity().map(PlBitmap::from),
        Some(fits) => Some(and_validity(from.validity(), fits)),
    };

    PlFixedSizeListArray::try_new(
        Box::new(PlPrimitiveArray::from_vec(out)) as Box<dyn PlArray>,
        array_width,
        from.len(),
        validity,
    )
}
