// commit # 20f2bd49fc95e3ebd73ba6aa7fdf8f1451b7dd40
use arrow::array::*;
use arrow::array::{Array, ArrayRef, BooleanArray, PrimitiveArray};
use arrow::datatypes::DataType;
use arrow::datatypes::*;
use arrow::error::{ArrowError, Result};
use arrow::{
    bitmap::Bitmap,
    buffer::{Buffer, MutableBuffer},
    util::bit_util,
};
use std::{mem, sync::Arc};

/// trait for copying filtered null bitmap bits
trait CopyNullBit {
    fn copy_null_bit(&mut self, source_index: usize);
    fn copy_null_bits(&mut self, source_index: usize, count: usize);
    fn null_count(&self) -> usize;
    fn null_buffer(&mut self) -> Buffer;
}

/// no-op null bitmap copy implementation,
/// used when the filtered data array doesn't have a null bitmap
struct NullBitNoop {}

impl NullBitNoop {
    fn new() -> Self {
        NullBitNoop {}
    }
}

impl CopyNullBit for NullBitNoop {
    #[inline]
    fn copy_null_bit(&mut self, _source_index: usize) {
        // do nothing
    }

    #[inline]
    fn copy_null_bits(&mut self, _source_index: usize, _count: usize) {
        // do nothing
    }

    fn null_count(&self) -> usize {
        0
    }

    fn null_buffer(&mut self) -> Buffer {
        Buffer::from([0u8; 0])
    }
}

/// null bitmap copy implementation,
/// used when the filtered data array has a null bitmap
struct NullBitSetter<'a> {
    target_buffer: MutableBuffer,
    source_bytes: &'a [u8],
    target_index: usize,
    null_count: usize,
}

impl<'a> NullBitSetter<'a> {
    fn new(null_bitmap: &'a Bitmap) -> Self {
        let null_bytes = null_bitmap.buffer_ref().data();
        // create null bitmap buffer with same length and initialize null bitmap buffer to 1s
        let null_buffer = MutableBuffer::new(null_bytes.len()).with_bitset(null_bytes.len(), true);
        NullBitSetter {
            source_bytes: null_bytes,
            target_buffer: null_buffer,
            target_index: 0,
            null_count: 0,
        }
    }
}

static BIT_MASK: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// Sets bit at position `i` for `data` to 0
#[inline]
pub fn unset_bit(data: &mut [u8], i: usize) {
    data[i >> 3] ^= BIT_MASK[i & 7];
}

impl<'a> CopyNullBit for NullBitSetter<'a> {
    #[inline]
    fn copy_null_bit(&mut self, source_index: usize) {
        if !bit_util::get_bit(self.source_bytes, source_index) {
            unset_bit(self.target_buffer.data_mut(), self.target_index);
            self.null_count += 1;
        }
        self.target_index += 1;
    }

    #[inline]
    fn copy_null_bits(&mut self, source_index: usize, count: usize) {
        for i in 0..count {
            self.copy_null_bit(source_index + i);
        }
    }

    fn null_count(&self) -> usize {
        self.null_count
    }

    fn null_buffer(&mut self) -> Buffer {
        self.target_buffer.resize(self.target_index).unwrap();
        // use mem::replace to detach self.target_buffer from self so that it can be returned
        let target_buffer = mem::replace(&mut self.target_buffer, MutableBuffer::new(0));
        target_buffer.freeze()
    }
}

fn get_null_bit_setter<'a>(data_array: &'a impl Array) -> Box<dyn CopyNullBit + 'a> {
    if let Some(null_bitmap) = data_array.data_ref().null_bitmap() {
        // only return an actual null bit copy implementation if null_bitmap is set
        Box::new(NullBitSetter::new(null_bitmap))
    } else {
        // otherwise return a no-op copy null bit implementation
        // for improved performance when the filtered array doesn't contain NULLs
        Box::new(NullBitNoop::new())
    }
}

// transmute filter array to u64
// - optimize filtering with highly selective filters by skipping entire batches of 64 filter bits
// - if the data array being filtered doesn't have a null bitmap, no time is wasted to copy a null bitmap
fn filter_array_impl(
    filter_context: &FilterContext,
    data_array: &impl Array,
    array_type: DataType,
    value_size: usize,
) -> Result<ArrayDataBuilder> {
    if filter_context.filter_len > data_array.len() {
        return Err(ArrowError::ComputeError(
            "Filter array cannot be larger than data array".to_string(),
        ));
    }
    let filtered_count = filter_context.filtered_count;
    let filter_mask = &filter_context.filter_mask;
    let filter_u64 = &filter_context.filter_u64;
    let data_bytes = data_array.data_ref().buffers()[0].data();
    let mut target_buffer = MutableBuffer::new(filtered_count * value_size);
    target_buffer.resize(filtered_count * value_size)?;
    let target_bytes = target_buffer.data_mut();
    let mut target_byte_index: usize = 0;
    let mut null_bit_setter = get_null_bit_setter(data_array);
    let null_bit_setter = null_bit_setter.as_mut();
    let all_ones_batch = !0u64;
    let data_array_offset = data_array.offset();

    for (i, filter_batch) in filter_u64.iter().enumerate() {
        // foreach u64 batch
        let filter_batch = *filter_batch;
        if filter_batch == 0 {
            // if batch == 0, all items are filtered out, so skip entire batch
            continue;
        } else if filter_batch == all_ones_batch {
            // if batch == all 1s: copy all 64 values in one go
            let data_index = (i * 64) + data_array_offset;
            null_bit_setter.copy_null_bits(data_index, 64);
            let data_byte_index = data_index * value_size;
            let data_len = value_size * 64;
            target_bytes[target_byte_index..(target_byte_index + data_len)]
                .copy_from_slice(&data_bytes[data_byte_index..(data_byte_index + data_len)]);
            target_byte_index += data_len;
            continue;
        }
        for (j, filter_mask) in filter_mask.iter().enumerate() {
            // foreach bit in batch:
            if (filter_batch & *filter_mask) != 0 {
                let data_index = (i * 64) + j + data_array_offset;
                null_bit_setter.copy_null_bit(data_index);
                // if filter bit == 1: copy data value bytes
                let data_byte_index = data_index * value_size;
                target_bytes[target_byte_index..(target_byte_index + value_size)]
                    .copy_from_slice(&data_bytes[data_byte_index..(data_byte_index + value_size)]);
                target_byte_index += value_size;
            }
        }
    }

    let mut array_data_builder = ArrayDataBuilder::new(array_type)
        .len(filtered_count)
        .add_buffer(target_buffer.freeze());
    if null_bit_setter.null_count() > 0 {
        array_data_builder = array_data_builder
            .null_count(null_bit_setter.null_count())
            .null_bit_buffer(null_bit_setter.null_buffer());
    }

    Ok(array_data_builder)
}

/// FilterContext can be used to improve performance when
/// filtering multiple data arrays with the same filter array.
#[derive(Debug)]
pub struct FilterContext {
    filter_u64: Vec<u64>,
    filter_len: usize,
    filtered_count: usize,
    filter_mask: Vec<u64>,
}

macro_rules! filter_primitive_array {
    ($context:expr, $array:expr, $array_type:ident) => {{
        let input_array = $array.as_any().downcast_ref::<$array_type>().unwrap();
        let output_array = $context.filter_primitive_array(input_array)?;
        Ok(Arc::new(output_array))
    }};
}

macro_rules! filter_primitive_item_list_array {
    ($context:expr, $array:expr, $item_type:ident, $list_type:ident, $list_builder_type:ident) => {{
        let input_array = $array.as_any().downcast_ref::<$list_type>().unwrap();
        let values_builder = PrimitiveBuilder::<$item_type>::new($context.filtered_count);
        let mut builder = $list_builder_type::new(values_builder);
        for i in 0..$context.filter_u64.len() {
            // foreach u64 batch
            let filter_batch = $context.filter_u64[i];
            if filter_batch == 0 {
                // if batch == 0, all items are filtered out, so skip entire batch
                continue;
            }
            for j in 0..64 {
                // foreach bit in batch:
                if (filter_batch & $context.filter_mask[j]) != 0 {
                    let data_index = (i * 64) + j;
                    if input_array.is_null(data_index) {
                        builder.append(false)?;
                    } else {
                        let this_inner_list = input_array.value(data_index);
                        let inner_list = this_inner_list
                            .as_any()
                            .downcast_ref::<PrimitiveArray<$item_type>>()
                            .unwrap();
                        for k in 0..inner_list.len() {
                            if inner_list.is_null(k) {
                                builder.values().append_null()?;
                            } else {
                                builder.values().append_value(inner_list.value(k))?;
                            }
                        }
                        builder.append(true)?;
                    }
                }
            }
        }
        Ok(Arc::new(builder.finish()))
    }};
}

macro_rules! filter_non_primitive_item_list_array {
    ($context:expr, $array:expr, $item_array_type:ident, $item_builder:ident, $list_type:ident, $list_builder_type:ident) => {{
        let input_array = $array.as_any().downcast_ref::<$list_type>().unwrap();
        let values_builder = $item_builder::new($context.filtered_count);
        let mut builder = $list_builder_type::new(values_builder);
        for i in 0..$context.filter_u64.len() {
            // foreach u64 batch
            let filter_batch = $context.filter_u64[i];
            if filter_batch == 0 {
                // if batch == 0, all items are filtered out, so skip entire batch
                continue;
            }
            for j in 0..64 {
                // foreach bit in batch:
                if (filter_batch & $context.filter_mask[j]) != 0 {
                    let data_index = (i * 64) + j;
                    if input_array.is_null(data_index) {
                        builder.append(false)?;
                    } else {
                        let this_inner_list = input_array.value(data_index);
                        let inner_list = this_inner_list
                            .as_any()
                            .downcast_ref::<$item_array_type>()
                            .unwrap();
                        for k in 0..inner_list.len() {
                            if inner_list.is_null(k) {
                                builder.values().append_null()?;
                            } else {
                                builder.values().append_value(inner_list.value(k))?;
                            }
                        }
                        builder.append(true)?;
                    }
                }
            }
        }
        Ok(Arc::new(builder.finish()))
    }};
}

impl FilterContext {
    /// Returns a new instance of FilterContext
    pub fn new(filter_array: &BooleanArray) -> Result<Self> {
        if filter_array.offset() > 0 {
            return Err(ArrowError::ComputeError(
                "Filter array cannot have offset > 0".to_string(),
            ));
        }
        let filter_mask: Vec<u64> = (0..64).map(|x| 1u64 << x).collect();
        let filter_bytes = filter_array.data_ref().buffers()[0].data();
        let filtered_count = bit_util::count_set_bits_offset(filter_bytes, 0, filter_array.len());

        // transmute filter_bytes to &[u64]
        let mut u64_buffer = MutableBuffer::new(filter_bytes.len());
        // add to the resulting len so is is a multiple of the size of u64
        let pad_addional_len = (8 - filter_bytes.len() % 8) % 8;
        u64_buffer.write_bytes(filter_bytes, pad_addional_len)?;
        let mut filter_u64 = u64_buffer.typed_data_mut::<u64>().to_owned();

        // mask of any bits outside of the given len
        if filter_array.len() % 64 != 0 {
            let last_idx = filter_u64.len() - 1;
            let mask = u64::MAX >> (64 - filter_array.len() % 64);
            filter_u64[last_idx] &= mask;
        }

        Ok(FilterContext {
            filter_u64,
            filter_len: filter_array.len(),
            filtered_count,
            filter_mask,
        })
    }

    /// Returns a new array, containing only the elements matching the filter
    pub fn filter(&self, array: &dyn Array) -> Result<ArrayRef> {
        match array.data_type() {
            DataType::UInt8 => filter_primitive_array!(self, array, UInt8Array),
            DataType::UInt16 => filter_primitive_array!(self, array, UInt16Array),
            DataType::UInt32 => filter_primitive_array!(self, array, UInt32Array),
            DataType::UInt64 => filter_primitive_array!(self, array, UInt64Array),
            DataType::Int8 => filter_primitive_array!(self, array, Int8Array),
            DataType::Int16 => filter_primitive_array!(self, array, Int16Array),
            DataType::Int32 => filter_primitive_array!(self, array, Int32Array),
            DataType::Int64 => filter_primitive_array!(self, array, Int64Array),
            DataType::Float32 => filter_primitive_array!(self, array, Float32Array),
            DataType::Float64 => filter_primitive_array!(self, array, Float64Array),
            DataType::Boolean => {
                let input_array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                let mut builder = BooleanArray::builder(self.filtered_count);
                for i in 0..self.filter_u64.len() {
                    // foreach u64 batch
                    let filter_batch = self.filter_u64[i];
                    if filter_batch == 0 {
                        // if batch == 0, all items are filtered out, so skip entire batch
                        continue;
                    }
                    for j in 0..64 {
                        // foreach bit in batch:
                        if (filter_batch & self.filter_mask[j]) != 0 {
                            let data_index = (i * 64) + j;
                            if input_array.is_null(data_index) {
                                builder.append_null()?;
                            } else {
                                builder.append_value(input_array.value(data_index))?;
                            }
                        }
                    }
                }
                Ok(Arc::new(builder.finish()))
            }
            DataType::Date32(_) => filter_primitive_array!(self, array, Date32Array),
            DataType::Date64(_) => filter_primitive_array!(self, array, Date64Array),
            DataType::Time32(TimeUnit::Second) => {
                filter_primitive_array!(self, array, Time32SecondArray)
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                filter_primitive_array!(self, array, Time32MillisecondArray)
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                filter_primitive_array!(self, array, Time64MicrosecondArray)
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                filter_primitive_array!(self, array, Time64NanosecondArray)
            }
            DataType::Duration(TimeUnit::Second) => {
                filter_primitive_array!(self, array, DurationSecondArray)
            }
            DataType::Duration(TimeUnit::Millisecond) => {
                filter_primitive_array!(self, array, DurationMillisecondArray)
            }
            DataType::Duration(TimeUnit::Microsecond) => {
                filter_primitive_array!(self, array, DurationMicrosecondArray)
            }
            DataType::Duration(TimeUnit::Nanosecond) => {
                filter_primitive_array!(self, array, DurationNanosecondArray)
            }
            DataType::Timestamp(TimeUnit::Second, _) => {
                filter_primitive_array!(self, array, TimestampSecondArray)
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                filter_primitive_array!(self, array, TimestampMillisecondArray)
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                filter_primitive_array!(self, array, TimestampMicrosecondArray)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                filter_primitive_array!(self, array, TimestampNanosecondArray)
            }
            DataType::List(dt) => match &**dt {
                DataType::UInt8 => filter_primitive_item_list_array!(
                    self,
                    array,
                    UInt8Type,
                    ListArray,
                    ListBuilder
                ),
                DataType::UInt16 => filter_primitive_item_list_array!(
                    self,
                    array,
                    UInt16Type,
                    ListArray,
                    ListBuilder
                ),
                DataType::UInt32 => filter_primitive_item_list_array!(
                    self,
                    array,
                    UInt32Type,
                    ListArray,
                    ListBuilder
                ),
                DataType::UInt64 => filter_primitive_item_list_array!(
                    self,
                    array,
                    UInt64Type,
                    ListArray,
                    ListBuilder
                ),
                DataType::Int8 => {
                    filter_primitive_item_list_array!(self, array, Int8Type, ListArray, ListBuilder)
                }
                DataType::Int16 => filter_primitive_item_list_array!(
                    self,
                    array,
                    Int16Type,
                    ListArray,
                    ListBuilder
                ),
                DataType::Int32 => filter_primitive_item_list_array!(
                    self,
                    array,
                    Int32Type,
                    ListArray,
                    ListBuilder
                ),
                DataType::Int64 => filter_primitive_item_list_array!(
                    self,
                    array,
                    Int64Type,
                    ListArray,
                    ListBuilder
                ),
                DataType::Float32 => filter_primitive_item_list_array!(
                    self,
                    array,
                    Float32Type,
                    ListArray,
                    ListBuilder
                ),
                DataType::Float64 => filter_primitive_item_list_array!(
                    self,
                    array,
                    Float64Type,
                    ListArray,
                    ListBuilder
                ),
                DataType::Boolean => filter_primitive_item_list_array!(
                    self,
                    array,
                    BooleanType,
                    ListArray,
                    ListBuilder
                ),
                DataType::Date32(_) => filter_primitive_item_list_array!(
                    self,
                    array,
                    Date32Type,
                    ListArray,
                    ListBuilder
                ),
                DataType::Date64(_) => filter_primitive_item_list_array!(
                    self,
                    array,
                    Date64Type,
                    ListArray,
                    ListBuilder
                ),
                DataType::Time32(TimeUnit::Second) => filter_primitive_item_list_array!(
                    self,
                    array,
                    Time32SecondType,
                    ListArray,
                    ListBuilder
                ),
                DataType::Time32(TimeUnit::Millisecond) => filter_primitive_item_list_array!(
                    self,
                    array,
                    Time32MillisecondType,
                    ListArray,
                    ListBuilder
                ),
                DataType::Time64(TimeUnit::Microsecond) => filter_primitive_item_list_array!(
                    self,
                    array,
                    Time64MicrosecondType,
                    ListArray,
                    ListBuilder
                ),
                DataType::Time64(TimeUnit::Nanosecond) => filter_primitive_item_list_array!(
                    self,
                    array,
                    Time64NanosecondType,
                    ListArray,
                    ListBuilder
                ),
                DataType::Duration(TimeUnit::Second) => filter_primitive_item_list_array!(
                    self,
                    array,
                    DurationSecondType,
                    ListArray,
                    ListBuilder
                ),
                DataType::Duration(TimeUnit::Millisecond) => filter_primitive_item_list_array!(
                    self,
                    array,
                    DurationMillisecondType,
                    ListArray,
                    ListBuilder
                ),
                DataType::Duration(TimeUnit::Microsecond) => filter_primitive_item_list_array!(
                    self,
                    array,
                    DurationMicrosecondType,
                    ListArray,
                    ListBuilder
                ),
                DataType::Duration(TimeUnit::Nanosecond) => filter_primitive_item_list_array!(
                    self,
                    array,
                    DurationNanosecondType,
                    ListArray,
                    ListBuilder
                ),
                DataType::Timestamp(TimeUnit::Second, _) => filter_primitive_item_list_array!(
                    self,
                    array,
                    TimestampSecondType,
                    ListArray,
                    ListBuilder
                ),
                DataType::Timestamp(TimeUnit::Millisecond, _) => filter_primitive_item_list_array!(
                    self,
                    array,
                    TimestampMillisecondType,
                    ListArray,
                    ListBuilder
                ),
                DataType::Timestamp(TimeUnit::Microsecond, _) => filter_primitive_item_list_array!(
                    self,
                    array,
                    TimestampMicrosecondType,
                    ListArray,
                    ListBuilder
                ),
                DataType::Timestamp(TimeUnit::Nanosecond, _) => filter_primitive_item_list_array!(
                    self,
                    array,
                    TimestampNanosecondType,
                    ListArray,
                    ListBuilder
                ),
                DataType::Binary => filter_non_primitive_item_list_array!(
                    self,
                    array,
                    BinaryArray,
                    BinaryBuilder,
                    ListArray,
                    ListBuilder
                ),
                DataType::LargeBinary => filter_non_primitive_item_list_array!(
                    self,
                    array,
                    LargeBinaryArray,
                    LargeBinaryBuilder,
                    ListArray,
                    ListBuilder
                ),
                DataType::Utf8 => filter_non_primitive_item_list_array!(
                    self,
                    array,
                    StringArray,
                    StringBuilder,
                    ListArray,
                    ListBuilder
                ),
                DataType::LargeUtf8 => filter_non_primitive_item_list_array!(
                    self,
                    array,
                    LargeStringArray,
                    LargeStringBuilder,
                    ListArray,
                    ListBuilder
                ),
                other => Err(ArrowError::ComputeError(format!(
                    "filter not supported for List({:?})",
                    other
                ))),
            },
            DataType::LargeList(dt) => match &**dt {
                DataType::UInt8 => filter_primitive_item_list_array!(
                    self,
                    array,
                    UInt8Type,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::UInt16 => filter_primitive_item_list_array!(
                    self,
                    array,
                    UInt16Type,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::UInt32 => filter_primitive_item_list_array!(
                    self,
                    array,
                    UInt32Type,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::UInt64 => filter_primitive_item_list_array!(
                    self,
                    array,
                    UInt64Type,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Int8 => filter_primitive_item_list_array!(
                    self,
                    array,
                    Int8Type,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Int16 => filter_primitive_item_list_array!(
                    self,
                    array,
                    Int16Type,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Int32 => filter_primitive_item_list_array!(
                    self,
                    array,
                    Int32Type,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Int64 => filter_primitive_item_list_array!(
                    self,
                    array,
                    Int64Type,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Float32 => filter_primitive_item_list_array!(
                    self,
                    array,
                    Float32Type,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Float64 => filter_primitive_item_list_array!(
                    self,
                    array,
                    Float64Type,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Boolean => filter_primitive_item_list_array!(
                    self,
                    array,
                    BooleanType,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Date32(_) => filter_primitive_item_list_array!(
                    self,
                    array,
                    Date32Type,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Date64(_) => filter_primitive_item_list_array!(
                    self,
                    array,
                    Date64Type,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Time32(TimeUnit::Second) => filter_primitive_item_list_array!(
                    self,
                    array,
                    Time32SecondType,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Time32(TimeUnit::Millisecond) => filter_primitive_item_list_array!(
                    self,
                    array,
                    Time32MillisecondType,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Time64(TimeUnit::Microsecond) => filter_primitive_item_list_array!(
                    self,
                    array,
                    Time64MicrosecondType,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Time64(TimeUnit::Nanosecond) => filter_primitive_item_list_array!(
                    self,
                    array,
                    Time64NanosecondType,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Duration(TimeUnit::Second) => filter_primitive_item_list_array!(
                    self,
                    array,
                    DurationSecondType,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Duration(TimeUnit::Millisecond) => filter_primitive_item_list_array!(
                    self,
                    array,
                    DurationMillisecondType,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Duration(TimeUnit::Microsecond) => filter_primitive_item_list_array!(
                    self,
                    array,
                    DurationMicrosecondType,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Duration(TimeUnit::Nanosecond) => filter_primitive_item_list_array!(
                    self,
                    array,
                    DurationNanosecondType,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Timestamp(TimeUnit::Second, _) => filter_primitive_item_list_array!(
                    self,
                    array,
                    TimestampSecondType,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Timestamp(TimeUnit::Millisecond, _) => filter_primitive_item_list_array!(
                    self,
                    array,
                    TimestampMillisecondType,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Timestamp(TimeUnit::Microsecond, _) => filter_primitive_item_list_array!(
                    self,
                    array,
                    TimestampMicrosecondType,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Timestamp(TimeUnit::Nanosecond, _) => filter_primitive_item_list_array!(
                    self,
                    array,
                    TimestampNanosecondType,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Binary => filter_non_primitive_item_list_array!(
                    self,
                    array,
                    BinaryArray,
                    BinaryBuilder,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::LargeBinary => filter_non_primitive_item_list_array!(
                    self,
                    array,
                    LargeBinaryArray,
                    LargeBinaryBuilder,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::Utf8 => filter_non_primitive_item_list_array!(
                    self,
                    array,
                    StringArray,
                    StringBuilder,
                    LargeListArray,
                    LargeListBuilder
                ),
                DataType::LargeUtf8 => filter_non_primitive_item_list_array!(
                    self,
                    array,
                    LargeStringArray,
                    LargeStringBuilder,
                    LargeListArray,
                    LargeListBuilder
                ),
                other => Err(ArrowError::ComputeError(format!(
                    "filter not supported for LargeList({:?})",
                    other
                ))),
            },
            other => Err(ArrowError::ComputeError(format!(
                "filter not supported for {:?}",
                other
            ))),
        }
    }

    /// Returns a new PrimitiveArray<T> containing only those values from the array passed as the data_array parameter,
    /// selected by the BooleanArray passed as the filter_array parameter
    pub fn filter_primitive_array<T>(
        &self,
        data_array: &PrimitiveArray<T>,
    ) -> Result<PrimitiveArray<T>>
    where
        T: ArrowNumericType,
    {
        let array_type = T::get_data_type();
        let value_size = mem::size_of::<T::Native>();
        let array_data_builder = filter_array_impl(self, data_array, array_type, value_size)?;
        let data = array_data_builder.build();
        Ok(PrimitiveArray::<T>::from(data))
    }
}

/// Returns a new array, containing only the elements matching the filter.
pub fn filter(array: &dyn Array, filter: &BooleanArray) -> Result<ArrayRef> {
    FilterContext::new(filter)?.filter(array)
}

/// Returns a new PrimitiveArray<T> containing only those values from the array passed as the data_array parameter,
/// selected by the BooleanArray passed as the filter_array parameter
pub fn filter_primitive_array<T>(
    data_array: &PrimitiveArray<T>,
    filter_array: &BooleanArray,
) -> Result<PrimitiveArray<T>>
where
    T: ArrowNumericType,
{
    FilterContext::new(filter_array)?.filter_primitive_array(data_array)
}
