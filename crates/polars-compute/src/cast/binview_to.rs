use std::ptr::copy_nonoverlapping;

use arrow::array::*;
use arrow::bitmap::MutableBitmap;
#[cfg(feature = "dtype-decimal")]
use arrow::compute::decimal::deserialize_decimal;
use arrow::datatypes::{ArrowDataType, Field, TimeUnit};
use arrow::offset::Offset;
use arrow::types::NativeType;
use bytemuck::cast_slice_mut;
use chrono::Datelike;
use num_traits::FromBytes;
use polars_error::{PolarsResult, polars_err};

use super::CastOptionsImpl;
use super::binary_to::Parse;
use super::temporal::EPOCH_DAYS_FROM_CE;

pub(super) const RFC3339: &str = "%Y-%m-%dT%H:%M:%S%.f%:z";

/// Cast [`BinaryViewArray`] to [`DictionaryArray`], also known as packing.
/// # Errors
/// This function errors if the maximum key is smaller than the number of distinct elements
/// in the array.
pub(super) fn binview_to_dictionary<K: DictionaryKey>(
    from: &BinaryViewArray,
) -> PolarsResult<DictionaryArray<K>> {
    let mut array = MutableDictionaryArray::<K, MutableBinaryViewArray<[u8]>>::new();
    array.reserve(from.len());
    array.try_extend(from.iter())?;

    Ok(array.into())
}

pub(super) fn utf8view_to_dictionary<K: DictionaryKey>(
    from: &Utf8ViewArray,
) -> PolarsResult<DictionaryArray<K>> {
    let mut array = MutableDictionaryArray::<K, MutableBinaryViewArray<str>>::new();
    array.reserve(from.len());
    array.try_extend(from.iter())?;

    Ok(array.into())
}

pub(super) fn view_to_binary<O: Offset>(array: &BinaryViewArray) -> BinaryArray<O> {
    let len: usize = Array::len(array);
    let mut mutable = MutableBinaryValuesArray::<O>::with_capacities(len, array.total_bytes_len());
    for slice in array.values_iter() {
        mutable.push(slice)
    }
    let out: BinaryArray<O> = mutable.into();
    out.with_validity(array.validity().cloned())
}

pub fn utf8view_to_utf8<O: Offset>(array: &Utf8ViewArray) -> Utf8Array<O> {
    let array = array.to_binview();
    let out = view_to_binary::<O>(&array);

    let dtype = Utf8Array::<O>::default_dtype();
    unsafe {
        Utf8Array::new_unchecked(
            dtype,
            out.offsets().clone(),
            out.values().clone(),
            out.validity().cloned(),
        )
    }
}

/// Parses a [`Utf8ViewArray`] with text representations of numbers into a
/// [`PrimitiveArray`], making any unparsable value a Null.
pub(super) fn utf8view_to_primitive<T>(
    from: &Utf8ViewArray,
    to: &ArrowDataType,
) -> PrimitiveArray<T>
where
    T: NativeType + Parse,
{
    let iter = from
        .iter()
        .map(|x| x.and_then::<T, _>(|x| T::parse(x.as_bytes())));

    PrimitiveArray::<T>::from_trusted_len_iter(iter).to(to.clone())
}

/// Parses a `&dyn` [`Array`] of UTF-8 encoded string representations of numbers
/// into a [`PrimitiveArray`], making any unparsable value a Null.
pub(super) fn utf8view_to_primitive_dyn<T>(
    from: &dyn Array,
    to: &ArrowDataType,
    options: CastOptionsImpl,
) -> PolarsResult<Box<dyn Array>>
where
    T: NativeType + Parse,
{
    let from = from.as_any().downcast_ref().unwrap();
    if options.partial {
        unimplemented!()
    } else {
        Ok(Box::new(utf8view_to_primitive::<T>(from, to)))
    }
}

#[cfg(feature = "dtype-decimal")]
pub fn binview_to_decimal(
    array: &BinaryViewArray,
    precision: Option<usize>,
    scale: usize,
) -> PrimitiveArray<i128> {
    let precision = precision.map(|p| p as u8);
    PrimitiveArray::<i128>::from_trusted_len_iter(
        array
            .iter()
            .map(|val| val.and_then(|val| deserialize_decimal(val, precision, scale as u8))),
    )
    .to(ArrowDataType::Decimal(
        precision.unwrap_or(38).into(),
        scale,
    ))
}

pub(super) fn utf8view_to_naive_timestamp_dyn(
    from: &dyn Array,
    time_unit: TimeUnit,
) -> PolarsResult<Box<dyn Array>> {
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(utf8view_to_naive_timestamp(from, time_unit)))
}

/// [`super::temporal::utf8view_to_timestamp`] applied for RFC3339 formatting
pub fn utf8view_to_naive_timestamp(
    from: &Utf8ViewArray,
    time_unit: TimeUnit,
) -> PrimitiveArray<i64> {
    super::temporal::utf8view_to_naive_timestamp(from, RFC3339, time_unit)
}

pub(super) fn utf8view_to_date32(from: &Utf8ViewArray) -> PrimitiveArray<i32> {
    let iter = from.iter().map(|x| {
        x.and_then(|x| {
            x.parse::<chrono::NaiveDate>()
                .ok()
                .map(|x| x.num_days_from_ce() - EPOCH_DAYS_FROM_CE)
        })
    });
    PrimitiveArray::<i32>::from_trusted_len_iter(iter).to(ArrowDataType::Date32)
}

pub(super) fn utf8view_to_date32_dyn(from: &dyn Array) -> PolarsResult<Box<dyn Array>> {
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(utf8view_to_date32(from)))
}

/// Casts a [`BinaryViewArray`] containing binary-encoded numbers to a
/// [`PrimitiveArray`], making any uncastable value a Null.
pub(super) fn binview_to_primitive<T>(
    from: &BinaryViewArray,
    to: &ArrowDataType,
    is_little_endian: bool,
) -> PrimitiveArray<T>
where
    T: FromBytes + NativeType,
    for<'a> &'a <T as FromBytes>::Bytes: TryFrom<&'a [u8]>,
{
    let iter = from.iter().map(|x| {
        x.and_then::<T, _>(|x| {
            if is_little_endian {
                Some(<T as FromBytes>::from_le_bytes(x.try_into().ok()?))
            } else {
                Some(<T as FromBytes>::from_be_bytes(x.try_into().ok()?))
            }
        })
    });

    PrimitiveArray::<T>::from_trusted_len_iter(iter).to(to.clone())
}

/// Casts a `&dyn` [`Array`] containing binary-encoded numbers to a
/// [`PrimitiveArray`], making any uncastable value a Null.
/// # Panics
/// Panics if `Array` is not a `BinaryViewArray`
pub fn binview_to_primitive_dyn<T>(
    from: &dyn Array,
    to: &ArrowDataType,
    is_little_endian: bool,
) -> PolarsResult<Box<dyn Array>>
where
    T: FromBytes + NativeType,
    for<'a> &'a <T as FromBytes>::Bytes: TryFrom<&'a [u8]>,
{
    let from = from.as_any().downcast_ref().unwrap();
    Ok(Box::new(binview_to_primitive::<T>(
        from,
        to,
        is_little_endian,
    )))
}

/// Casts a [`BinaryViewArray`] to a [`FixedSizeListArray`], making any un-castable value a Null.
///
/// # Arguments
///
/// * `from`: The array to reinterpret.
/// * `array_width`: The number of items in each `Array`.
pub(super) fn try_binview_to_fixed_size_list<T, const IS_LITTLE_ENDIAN: bool>(
    from: &BinaryViewArray,
    array_width: usize,
) -> PolarsResult<FixedSizeListArray>
where
    T: FromBytes + NativeType,
    for<'a> &'a <T as FromBytes>::Bytes: TryFrom<&'a [u8]>,
{
    let element_size = std::mem::size_of::<T>();
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
    let (out_u8_ptr, out_len_bytes) = {
        let out_u8_slice = cast_slice_mut::<_, u8>(out.as_mut());
        (out_u8_slice.as_mut_ptr(), out_u8_slice.len())
    };
    assert_eq!(out_len_bytes, row_size_bytes * from.len());
    let mut validity = MutableBitmap::from_len_set(from.len());

    for (index, value) in from.iter().enumerate() {
        if let Some(value) = value
            && value.len() == row_size_bytes
        {
            if cfg!(target_endian = "little") && IS_LITTLE_ENDIAN {
                // Fast path, we can just copy the data with no need to
                // reinterpret.
                let write_index = index * row_size_bytes;
                debug_assert!(value.is_empty() || write_index < out_len_bytes);
                debug_assert!(value.is_empty() || (write_index + value.len() - 1 < out_len_bytes));
                // # Safety
                // - The start index is smaller than `out`'s capacity.
                // - The end index is smaller than `out`'s capacity.
                unsafe {
                    copy_nonoverlapping(value.as_ptr(), out_u8_ptr.add(write_index), value.len());
                }
            } else {
                // Slow path, reinterpret items one by one.
                for j in 0..array_width {
                    let jth_range = (j * element_size)..((j + 1) * element_size);
                    debug_assert!(value.get(jth_range.clone()).is_some());
                    // # Safety
                    // We made sure the range is smaller than `value` length.
                    let jth_bytes = unsafe { value.get_unchecked(jth_range) };
                    // # Safety
                    // We just made sure that the slice has length `element_size`
                    let byte_array = unsafe { jth_bytes.try_into().unwrap_unchecked() };
                    let jth_value = if IS_LITTLE_ENDIAN {
                        <T as FromBytes>::from_le_bytes(byte_array)
                    } else {
                        <T as FromBytes>::from_be_bytes(byte_array)
                    };

                    let write_index = array_width * index + j;
                    debug_assert!(write_index < out.len());
                    // # Safety
                    // - The target index is smaller than the vector's pre-allocated capacity.
                    unsafe {
                        *out.get_unchecked_mut(write_index) = jth_value;
                    }
                }
            }
        } else {
            validity.set(index, false);
        };
    }

    FixedSizeListArray::try_new(
        ArrowDataType::FixedSizeList(
            Box::new(Field::new("".into(), T::PRIMITIVE.into(), true)),
            array_width,
        ),
        from.len(),
        Box::new(PrimitiveArray::<T>::from_vec(out)),
        validity.into(),
    )
}

/// Casts a `dyn` [`Array`] to a [`FixedSizeListArray`], making any un-castable value a Null.
///
/// # Arguments
///
/// * `from`: The array to reinterpret.
/// * `array_width`: The number of items in each `Array`.
///
/// # Panics
///    Panics if `from` is not `BinaryViewArray`.
pub fn binview_to_fixed_size_list_dyn<T>(
    from: &dyn Array,
    array_width: usize,
    is_little_endian: bool,
) -> PolarsResult<Box<dyn Array>>
where
    T: FromBytes + NativeType,
    for<'a> &'a <T as FromBytes>::Bytes: TryFrom<&'a [u8]>,
{
    let from = from.as_any().downcast_ref().unwrap();

    let result = if is_little_endian {
        try_binview_to_fixed_size_list::<T, true>(from, array_width)
    } else {
        try_binview_to_fixed_size_list::<T, false>(from, array_width)
    }?;
    Ok(Box::new(result))
}
