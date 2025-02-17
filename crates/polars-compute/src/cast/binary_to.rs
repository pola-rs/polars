use std::sync::Arc;

use arrow::array::*;
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;
use arrow::offset::{Offset, Offsets};
use arrow::types::NativeType;
use polars_error::PolarsResult;

use super::CastOptionsImpl;

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

#[cfg(feature = "dtype-i128")]
impl_parse!(i128);

impl Parse for f32 {
    fn parse(val: &[u8]) -> Option<Self>
    where
        Self: Sized,
    {
        fast_float2::parse(val).ok()
    }
}
impl Parse for f64 {
    fn parse(val: &[u8]) -> Option<Self>
    where
        Self: Sized,
    {
        fast_float2::parse(val).ok()
    }
}

/// Conversion of binary
pub fn binary_to_large_binary(
    from: &BinaryArray<i32>,
    to_dtype: ArrowDataType,
) -> BinaryArray<i64> {
    let values = from.values().clone();
    BinaryArray::<i64>::new(
        to_dtype,
        from.offsets().into(),
        values,
        from.validity().cloned(),
    )
}

/// Conversion of binary
pub fn binary_large_to_binary(
    from: &BinaryArray<i64>,
    to_dtype: ArrowDataType,
) -> PolarsResult<BinaryArray<i32>> {
    let values = from.values().clone();
    let offsets = from.offsets().try_into()?;
    Ok(BinaryArray::<i32>::new(
        to_dtype,
        offsets,
        values,
        from.validity().cloned(),
    ))
}

/// Conversion to utf8
pub fn binary_to_utf8<O: Offset>(
    from: &BinaryArray<O>,
    to_dtype: ArrowDataType,
) -> PolarsResult<Utf8Array<O>> {
    Utf8Array::<O>::try_new(
        to_dtype,
        from.offsets().clone(),
        from.values().clone(),
        from.validity().cloned(),
    )
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
    to_dtype: ArrowDataType,
) -> BinaryArray<O> {
    let values = from.values().clone();
    let offsets = fixed_size_to_offsets(values.len(), from.size());
    BinaryArray::<O>::new(to_dtype, offsets.into(), values, from.validity().cloned())
}

pub fn fixed_size_binary_to_binview(from: &FixedSizeBinaryArray) -> BinaryViewArray {
    let datatype = <[u8] as ViewType>::DATA_TYPE;

    // Fast path: all the views are inlineable
    if from.size() <= View::MAX_INLINE_SIZE as usize {
        // @NOTE: There is something with the code-generation of `View::new_inline_unchecked` that
        // prevents it from properly SIMD-ing this loop. It insists on memcpying while it should
        // know that the size is really small. Dispatching over the `from.size()` and making it
        // constant does make loop SIMD, but it does not actually speed anything up and the code it
        // generates is still horrible.
        //
        // This is really slow, and I don't think it has to be.

        // SAFETY: We checked that slice.len() <= View::MAX_INLINE_SIZE before
        let mut views = Vec::new();
        View::extend_with_inlinable_strided(
            &mut views,
            from.values().as_slice(),
            from.size() as u8,
        );
        let views = Buffer::from(views);
        return BinaryViewArray::try_new(datatype, views, Arc::default(), from.validity().cloned())
            .unwrap();
    }

    const MAX_BYTES_PER_BUFFER: usize = u32::MAX as usize;

    let size = from.size();
    let num_bytes = from.len() * size;
    let num_buffers = num_bytes.div_ceil(MAX_BYTES_PER_BUFFER);
    assert!(num_buffers < u32::MAX as usize);

    let num_elements_per_buffer = MAX_BYTES_PER_BUFFER / size;
    // This is NOT equal to MAX_BYTES_PER_BUFFER because of integer division
    let split_point = num_elements_per_buffer * size;

    // This is zero-copy for the buffer since split just increases the data since
    let mut buffer = from.values().clone();
    let mut buffers = Vec::with_capacity(num_buffers);

    if let Some(num_buffers) = num_buffers.checked_sub(1) {
        for _ in 0..num_buffers {
            let slice;
            (slice, buffer) = buffer.split_at(split_point);
            buffers.push(slice);
        }
        buffers.push(buffer);
    }

    let mut iter = from.values_iter();
    let iter = iter.by_ref();
    let mut views = Vec::with_capacity(from.len());
    for buffer_idx in 0..num_buffers {
        views.extend(
            iter.take(num_elements_per_buffer)
                .enumerate()
                .map(|(i, slice)| {
                    // SAFETY: We checked that slice.len() > View::MAX_INLINE_SIZE before
                    unsafe {
                        View::new_noninline_unchecked(slice, buffer_idx as u32, (i * size) as u32)
                    }
                }),
        );
    }
    let views = views.into();

    BinaryViewArray::try_new(datatype, views, buffers.into(), from.validity().cloned()).unwrap()
}

/// Conversion of binary
pub fn binary_to_list<O: Offset>(from: &BinaryArray<O>, to_dtype: ArrowDataType) -> ListArray<O> {
    let values = from.values().clone();
    let values = PrimitiveArray::new(ArrowDataType::UInt8, values, None);
    ListArray::<O>::new(
        to_dtype,
        from.offsets().clone(),
        values.boxed(),
        from.validity().cloned(),
    )
}
