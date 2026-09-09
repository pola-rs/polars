//! Casting the nested arrays of `polars-array`, which hold the values of their children.

use arrow::array::BINVIEW_MAX_ROW_BYTE_LEN;
use polars_array::{
    PlArray, PlBinaryArray, PlBinaryViewArray, PlBitmap, PlFixedSizeListArray, PlListArray,
    PlPrimitiveArray,
};
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_ensure};
use polars_utils::IdxSize;

use super::{MaskBuilder, downcast};

/// Casts the values of a list array, which leaves every element holding as many as it did.
pub fn cast_list(
    from: &PlListArray,
    cast_values: impl FnOnce(&dyn PlArray) -> PolarsResult<Box<dyn PlArray>>,
) -> PolarsResult<PlListArray> {
    let values = cast_values(from.values())?;
    Ok(list_with_values(from, values))
}

/// Casts the values of a fixed size list array, which leaves every element as wide as it was.
#[cfg(feature = "dtype-array")]
pub fn cast_fixed_size_list(
    from: &PlFixedSizeListArray,
    cast_values: impl FnOnce(&dyn PlArray) -> PolarsResult<Box<dyn PlArray>>,
) -> PolarsResult<PlFixedSizeListArray> {
    let values = cast_values(from.values())?;
    Ok(fixed_size_list_with_values(from, values))
}

/// Casts every field of a struct array, which holds as many elements as each of them does.
#[cfg(feature = "dtype-struct")]
pub fn cast_struct(
    from: &polars_array::PlStructArray,
    cast_field: impl Fn(usize, &dyn PlArray) -> PolarsResult<Box<dyn PlArray>>,
) -> PolarsResult<polars_array::PlStructArray> {
    let fields = from
        .fields()
        .iter()
        .enumerate()
        .map(|(i, field)| cast_field(i, &**field))
        .collect::<PolarsResult<_>>()?;

    Ok(polars_array::PlStructArray::new(
        fields,
        from.len(),
        from.validity().map(PlBitmap::from),
    ))
}

/// Reads every element of a fixed size list array as a list of its own.
#[cfg(feature = "dtype-array")]
pub fn fixed_size_list_to_list(
    from: &PlFixedSizeListArray,
    cast_values: impl FnOnce(&dyn PlArray) -> PolarsResult<Box<dyn PlArray>>,
) -> PolarsResult<PlListArray> {
    let values = cast_values(from.values())?;
    let width = from.width() as u64;
    let validity = from.validity().map(PlBitmap::from);

    // The one list every element of a scalar chunk reads lies in one range, which the offsets of
    // the lists repeat rather than write out.
    if from.scalar_value_ignore_validity().is_some() {
        // SAFETY: the values are the one list every element reads, cast one for one, so the two
        // offsets are the range it covers — which is scalar for however many elements read it.
        return Ok(unsafe {
            PlListArray::new_broadcast_unchecked(
                values,
                Buffer::from(vec![0, width]),
                from.len(),
                validity,
            )
        });
    }

    let offsets = (0..=from.len() as u64)
        .map(|element| element * width)
        .collect::<Vec<_>>();

    // SAFETY: the values hold the width of every element laid end to end, cast one for one, and
    // the offsets count up by that width: one per element plus the end of the last, ascending,
    // ending exactly where the values do. Checking that back is a pass over them for nothing.
    Ok(unsafe { PlListArray::new_unchecked(values, Buffer::from(offsets), from.len(), validity) })
}

/// Reads every element of a list array as a list of `width` values, erroring on another count.
pub fn list_to_fixed_size_list(
    from: &PlListArray,
    width: usize,
    cast_values: impl FnOnce(&dyn PlArray) -> PolarsResult<Box<dyn PlArray>>,
) -> PolarsResult<PlFixedSizeListArray> {
    let validity = from.validity().map(PlBitmap::from);

    // The one list every element of a scalar chunk reads is as wide as every element then is.
    if let Some(range) = from.scalar_offsets() {
        polars_ensure!(
            range.len() == width,
            ComputeError: "not all elements have the specified width {width}"
        );
        let values = cast_values(&*from.values().sliced(range.start, range.len()))?;
        return Ok(PlFixedSizeListArray::new_broadcast(
            values,
            width,
            from.len(),
            validity,
        ));
    }

    let offsets = from.flat_offsets().unwrap();

    // Without a null element the values already lie `width` to an element, so the cast reads the
    // range they lie in as they are.
    if from.null_count() == 0 {
        let start_offset = offsets[0] as usize;
        let mut is_valid = true;
        for (i, offset) in offsets.iter().enumerate() {
            is_valid &= *offset as usize == start_offset + i * width;
        }
        polars_ensure!(is_valid, ComputeError: "not all elements have the specified width {width}");

        let length = *offsets.last().unwrap() as usize - start_offset;
        let values = cast_values(&*from.values().sliced(start_offset, length))?;
        return Ok(PlFixedSizeListArray::new(
            values,
            width,
            from.len(),
            validity,
        ));
    }

    // A null element holds no values of its own, so lining every element up `width` values to one
    // means picking out the values of the ones that do and a null for the rest.
    let mut expected_offset = offsets[0] + width as u64;
    for i in 1..=from.len() {
        let current_offset = offsets[i];
        if from.validity().is_some_and(|validity| !validity.get(i - 1)) {
            expected_offset = current_offset + width as u64;
        } else {
            polars_ensure!(
                current_offset == expected_offset,
                ComputeError: "not all elements have the specified width {width}"
            );
            expected_offset += width as u64;
        }
    }

    let mut indices = Vec::with_capacity(from.len() * width);
    let mut picked = MaskBuilder::with_capacity(from.len() * width);
    for i in 0..from.len() {
        if from.validity().is_some_and(|validity| !validity.get(i)) {
            indices.resize(indices.len() + width, 0);
            for _ in 0..width {
                picked.push(false);
            }
            continue;
        }

        let start = offsets[i];
        for j in 0..width as u64 {
            indices.push((start + j) as IdxSize);
            picked.push(true);
        }
    }

    let indices = PlPrimitiveArray::from_vec(indices);
    let indices = match picked.finish() {
        None => indices,
        Some(picked) => indices.with_validity(Some(PlBitmap::from_bitmap(picked))),
    };

    // SAFETY: every index was read off the offsets of an element, which hold ranges within the
    // values; the ones that were not are null.
    let values = unsafe { crate::gather::take_unchecked(from.values(), &indices) };
    let values = cast_values(&*values)?;

    PlFixedSizeListArray::try_new(values, width, from.len(), validity)
        .map_err(|_| polars_error::polars_err!(ComputeError: "not all elements have the specified width {width}"))
}

/// Reads the bytes every element holds one per value as the bytes of one element.
pub fn list_uint8_to_binview(from: &PlListArray) -> PolarsResult<PlBinaryViewArray> {
    let values: &PlPrimitiveArray<u8> = downcast(from.values());

    // An element that holds a null byte holds no bytes at all, and reads as null in turn.
    let mut holds_bytes = MaskBuilder::with_capacity(from.len());
    for i in 0..from.len() {
        let range = from.value_range(i);
        polars_ensure!(
            range.len() <= BINVIEW_MAX_ROW_BYTE_LEN,
            InvalidOperation:
            "when casting to BinaryView, list lengths must be <= {BINVIEW_MAX_ROW_BYTE_LEN}"
        );
        holds_bytes.push(
            values
                .validity()
                .is_none_or(|validity| range.clone().all(|value| validity.get(value))),
        );
    }

    let validity = match holds_bytes.finish() {
        None => from.validity().map(PlBitmap::from),
        Some(holds_bytes) => Some(super::and_validity(from.validity(), holds_bytes)),
    };

    // The bytes of the elements laid end to end with the range each of them lies in is what an
    // offset-backed binary array is, so the views are read off it.
    let binary = match from.scalar_offsets() {
        Some(range) => PlBinaryArray::new_broadcast(
            values.to_flat_values().into_owned(),
            Buffer::from(vec![range.start as u64, range.end as u64]),
            from.len(),
            validity,
        ),
        None => PlBinaryArray::new(
            values.to_flat_values().into_owned(),
            from.flat_offsets().unwrap().clone(),
            from.len(),
            validity,
        ),
    };

    Ok(super::binary_to::binary_to_binview(&binary))
}

/// Rebuilds `from` over `values`, which hold as many values as its own do.
fn list_with_values(from: &PlListArray, values: Box<dyn PlArray>) -> PlListArray {
    assert_eq!(
        values.len(),
        from.values().len(),
        "the values a list array is rebuilt over hold one value per value of its own",
    );
    let validity = from.validity().map(PlBitmap::from);

    // SAFETY: the offsets are the ones `from` was built with, which hold ranges within values as
    // many as its own — the length just asserted.
    unsafe {
        match from.scalar_offsets() {
            Some(range) => PlListArray::new_broadcast_unchecked(
                values,
                Buffer::from(vec![range.start as u64, range.end as u64]),
                from.len(),
                validity,
            ),
            None => PlListArray::new_unchecked(
                values,
                from.flat_offsets().unwrap().clone(),
                from.len(),
                validity,
            ),
        }
    }
}

/// Rebuilds `from` over `values`, which hold as many values as its own do.
#[cfg(feature = "dtype-array")]
fn fixed_size_list_with_values(
    from: &PlFixedSizeListArray,
    values: Box<dyn PlArray>,
) -> PlFixedSizeListArray {
    let validity = from.validity().map(PlBitmap::from);
    match from.scalar_value_ignore_validity() {
        Some(_) => PlFixedSizeListArray::new_broadcast(values, from.width(), from.len(), validity),
        None => PlFixedSizeListArray::new(values, from.width(), from.len(), validity),
    }
}
