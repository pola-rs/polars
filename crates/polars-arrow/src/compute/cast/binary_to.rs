use polars_error::PolarsResult;

use super::CastOptionsImpl;
use crate::array::*;
use crate::datatypes::ArrowDataType;
use crate::offset::{Offset, Offsets};
use crate::types::NativeType;

pub(super) trait Parse {
    fn parse(val: &[u8]) -> Option<Self>
    where
        Self: Sized;
}

macro_rules! impl_parse {
    ($primitive_type:ident) => {
        impl Parse for $primitive_type {
            fn parse(val: &[u8]) -> Option<Self> {
                atoi_simd::parse_skipped(val).ok()
            }
        }
    };
}
impl_parse!(i8);
impl_parse!(i16);
impl_parse!(i32);
impl_parse!(i64);
impl_parse!(u8);
impl_parse!(u16);
impl_parse!(u32);
impl_parse!(u64);

impl Parse for f32 {
    fn parse(val: &[u8]) -> Option<Self>
    where
        Self: Sized,
    {
        fast_float::parse(val).ok()
    }
}
impl Parse for f64 {
    fn parse(val: &[u8]) -> Option<Self>
    where
        Self: Sized,
    {
        fast_float::parse(val).ok()
    }
}

/// Conversion of binary
pub fn binary_to_large_binary(
    from: &BinaryArray<i32>,
    to_data_type: ArrowDataType,
) -> BinaryArray<i64> {
    let values = from.values().clone();
    BinaryArray::<i64>::new(
        to_data_type,
        from.offsets().into(),
        values,
        from.validity().cloned(),
    )
}

/// Conversion of binary
pub fn binary_large_to_binary(
    from: &BinaryArray<i64>,
    to_data_type: ArrowDataType,
) -> PolarsResult<BinaryArray<i32>> {
    let values = from.values().clone();
    let offsets = from.offsets().try_into()?;
    Ok(BinaryArray::<i32>::new(
        to_data_type,
        offsets,
        values,
        from.validity().cloned(),
    ))
}

/// Conversion to utf8
pub fn binary_to_utf8<O: Offset>(
    from: &BinaryArray<O>,
    to_data_type: ArrowDataType,
) -> PolarsResult<Utf8Array<O>> {
    Utf8Array::<O>::try_new(
        to_data_type,
        from.offsets().clone(),
        from.values().clone(),
        from.validity().cloned(),
    )
}

/// Conversion to utf8
/// # Errors
/// This function errors if the values are not valid utf8
pub fn binary_to_large_utf8(
    from: &BinaryArray<i32>,
    to_data_type: ArrowDataType,
) -> PolarsResult<Utf8Array<i64>> {
    let values = from.values().clone();
    let offsets = from.offsets().into();

    Utf8Array::<i64>::try_new(to_data_type, offsets, values, from.validity().cloned())
}

/// Casts a [`BinaryArray`] to a [`PrimitiveArray`], making any uncastable value a Null.
pub(super) fn binary_to_primitive<O: Offset, T>(
    from: &BinaryArray<O>,
    to: &ArrowDataType,
) -> PrimitiveArray<T>
where
    T: NativeType + Parse,
{
    let iter = from.iter().map(|x| x.and_then::<T, _>(|x| T::parse(x)));

    PrimitiveArray::<T>::from_trusted_len_iter(iter).to(to.clone())
}

pub(super) fn binary_to_primitive_dyn<O: Offset, T>(
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
        Ok(Box::new(binary_to_primitive::<O, T>(from, to)))
    }
}

/// Cast [`BinaryArray`] to [`DictionaryArray`], also known as packing.
/// # Errors
/// This function errors if the maximum key is smaller than the number of distinct elements
/// in the array.
pub fn binary_to_dictionary<O: Offset, K: DictionaryKey>(
    from: &BinaryArray<O>,
) -> PolarsResult<DictionaryArray<K>> {
    let mut array = MutableDictionaryArray::<K, MutableBinaryArray<O>>::new();
    array.reserve(from.len());
    array.try_extend(from.iter())?;

    Ok(array.into())
}

pub(super) fn binary_to_dictionary_dyn<O: Offset, K: DictionaryKey>(
    from: &dyn Array,
) -> PolarsResult<Box<dyn Array>> {
    let values = from.as_any().downcast_ref().unwrap();
    binary_to_dictionary::<O, K>(values).map(|x| Box::new(x) as Box<dyn Array>)
}

fn fixed_size_to_offsets<O: Offset>(values_len: usize, fixed_size: usize) -> Offsets<O> {
    let offsets = (0..(values_len + 1))
        .step_by(fixed_size)
        .map(|v| O::from_as_usize(v))
        .collect();
    // SAFETY:
    // * every element is `>= 0`
    // * element at position `i` is >= than element at position `i-1`.
    unsafe { Offsets::new_unchecked(offsets) }
}

/// Conversion of `FixedSizeBinary` to `Binary`.
pub fn fixed_size_binary_binary<O: Offset>(
    from: &FixedSizeBinaryArray,
    to_data_type: ArrowDataType,
) -> BinaryArray<O> {
    let values = from.values().clone();
    let offsets = fixed_size_to_offsets(values.len(), from.size());
    BinaryArray::<O>::new(
        to_data_type,
        offsets.into(),
        values,
        from.validity().cloned(),
    )
}

pub fn fixed_size_binary_to_binview(from: &FixedSizeBinaryArray) -> BinaryViewArray {
    let mutable = MutableBinaryViewArray::from_values_iter(from.values_iter());
    mutable.freeze().with_validity(from.validity().cloned())
}

/// Conversion of binary
pub fn binary_to_list<O: Offset>(
    from: &BinaryArray<O>,
    to_data_type: ArrowDataType,
) -> ListArray<O> {
    let values = from.values().clone();
    let values = PrimitiveArray::new(ArrowDataType::UInt8, values, None);
    ListArray::<O>::new(
        to_data_type,
        from.offsets().clone(),
        values.boxed(),
        from.validity().cloned(),
    )
}
