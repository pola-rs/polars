//! The typed heart of every Series column.
use crate::chunked_array::builder::{
    aligned_vec_to_primitive_array, build_with_existing_null_bitmap_and_slice, get_bitmap,
    PrimitiveChunkedBuilder, Utf8ChunkedBuilder,
};
use crate::prelude::*;
use crate::utils::Xob;
use arrow::{
    array::{
        ArrayRef, BooleanArray, Date64Array, Float32Array, Float64Array, Int16Array, Int32Array,
        Int64Array, Int8Array, PrimitiveArray, PrimitiveBuilder, StringArray, StringBuilder,
        Time64NanosecondArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
    },
    buffer::Buffer,
    compute,
    datatypes::{ArrowPrimitiveType, DateUnit, Field, TimeUnit},
};
use itertools::Itertools;
use std::cmp::Ordering;
use std::fmt::{Debug, Formatter};
use std::iter::{Copied, Map};
use std::marker::PhantomData;
use std::sync::Arc;

pub mod aggregate;
pub mod apply;
#[macro_use]
pub mod arithmetic;
pub mod builder;
pub mod cast;
pub mod chunkops;
pub mod comparison;
pub mod iterator;
pub mod take;
pub mod temporal;
pub mod unique;
pub mod upstream_traits;
use arrow::array::{
    Date32Array, DurationMicrosecondArray, DurationMillisecondArray, DurationNanosecondArray,
    DurationSecondArray, IntervalDayTimeArray, IntervalYearMonthArray, Time32MillisecondArray,
    Time32SecondArray, Time64MicrosecondArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray,
};
use std::mem;

/// Get a 'hash' of the chunks in order to compare chunk sizes quickly.
fn create_chunk_id(chunks: &Vec<ArrayRef>) -> Vec<usize> {
    let mut chunk_id = Vec::with_capacity(chunks.len());
    for a in chunks {
        chunk_id.push(a.len())
    }
    chunk_id
}

/// # ChunkedArray
///
/// Every Series contains a `ChunkedArray<T>`. Unlike Series, ChunkedArray's are typed. This allows
/// us to apply closures to the data and collect the results to a `ChunkedArray` of te same type `T`.
/// Below we use an apply to use the cosine function to the values of a `ChunkedArray`.
///
/// ```rust
/// # use polars::prelude::*;
/// fn apply_cosine(ca: &Float32Chunked) -> Float32Chunked {
///     ca.apply(|v| v.cos())
/// }
/// ```
///
/// If we would like to cast the result we could use a Rust Iterator instead of an `apply` method.
/// Note that Iterators are slightly slower as the null values aren't ignored implicitly.
///
/// ```rust
/// # use polars::prelude::*;
/// fn apply_cosine_and_cast(ca: &Float32Chunked) -> Float64Chunked {
///     ca.into_iter()
///         .map(|opt_v| {
///         opt_v.map(|v| v.cos() as f64)
///     }).collect()
/// }
/// ```
///
/// Another option is to first cast and then use an apply.
///
/// ```rust
/// # use polars::prelude::*;
/// fn apply_cosine_and_cast(ca: &Float32Chunked) -> Float64Chunked {
///     ca.cast::<Float64Type>()
///         .unwrap()
///         .apply(|v| v.cos())
/// }
/// ```
///
/// ## Conversion between Series and ChunkedArray's
/// Conversion from a `Series` to a `ChunkedArray` is effortless.
///
/// ```rust
/// # use polars::prelude::*;
/// fn to_chunked_array(series: &Series) -> Result<&Int32Chunked>{
///     series.i32()
/// }
///
/// fn to_series(ca: Int32Chunked) -> Series {
///     ca.into_series()
/// }
/// ```
///
/// # Iterators
///
/// `ChunkedArrays` fully support Rust native [Iterator](https://doc.rust-lang.org/std/iter/trait.Iterator.html)
/// and [DoubleEndedIterator](https://doc.rust-lang.org/std/iter/trait.DoubleEndedIterator.html) traits, thereby
/// giving access to all the excelent methods available for [Iterators](https://doc.rust-lang.org/std/iter/trait.Iterator.html).
///
/// ```rust
/// # use polars::prelude::*;
///
/// fn iter_forward(ca: &Float32Chunked) {
///     ca.into_iter()
///         .for_each(|opt_v| println!("{:?}", opt_v))
/// }
///
/// fn iter_backward(ca: &Float32Chunked) {
///     ca.into_iter()
///         .rev()
///         .for_each(|opt_v| println!("{:?}", opt_v))
/// }
/// ```
///
/// # Memory layout
///
/// `ChunkedArray`'s use [Apache Arrow](https://github.com/apache/arrow) as backend for the memory layout.
/// Arrows memory is immutable which makes it possible to make mutliple zero copy (sub)-views from a single array.
///
/// To be able to append data, Polars uses chunks to append new memory locations, hence the `ChunkedArray<T>` data structure.
/// Appends are cheap, because it will not lead to a full reallocation of the whole array (as could be the case with a Rust Vec).
///
/// However, multiple chunks in a `ChunkArray` will slow down the Iterators, arithmetic and other operations.
/// When multiplying two `ChunkArray'`s with different chunk sizes they cannot utilize [SIMD](https://en.wikipedia.org/wiki/SIMD) for instance.
/// However, when chunk size don't match, Iterators will be used to do the operation (instead of arrows upstream implementation, which may utilize SIMD) and
/// the result will be a single chunked array.
///
/// **The key takeaway is that by applying operations on a `ChunkArray` of multiple chunks, the results will converge to
/// a `ChunkArray` of a single chunk!** It is recommended to leave them as is. If you want to have predictable performance
/// (no unexpected re-allocation of memory), it is adviced to call the [rechunk](chunked_array/chunkops/trait.ChunkOps.html) after
/// multiple append operations.
pub struct ChunkedArray<T> {
    pub(crate) field: Arc<Field>,
    // For now settle with dynamic generics until we are more confident about the api
    pub(crate) chunks: Vec<ArrayRef>,
    // chunk lengths
    chunk_id: Vec<usize>,
    phantom: PhantomData<T>,
}

impl<T> ChunkedArray<T>
where
    T: PolarsDataType,
{
    /// Get data type of ChunkedArray.
    pub fn dtype(&self) -> &ArrowDataType {
        self.field.data_type()
    }

    /// Create a new ChunkedArray from existing chunks.
    pub fn new_from_chunks(name: &str, chunks: Vec<ArrayRef>) -> Self {
        let field = Arc::new(Field::new(name, T::get_data_type(), true));
        let chunk_id = create_chunk_id(&chunks);
        ChunkedArray {
            field,
            chunks,
            chunk_id,
            phantom: PhantomData,
        }
    }

    /// Get the null count and the buffer of bits representing null values
    pub fn null_bits(&self) -> Vec<(usize, Option<Buffer>)> {
        self.chunks
            .iter()
            .map(|arr| get_bitmap(arr.as_ref()))
            .collect()
    }

    /// Wrap as Series
    pub fn into_series(self) -> Series {
        Series::from_chunked_array(self)
    }

    pub fn unpack_series_matching_type(&self, series: &Series) -> Result<&ChunkedArray<T>> {
        macro_rules! unpack {
            ($variant:ident) => {{
                if let Series::$variant(ca) = series {
                    let ca = unsafe { mem::transmute::<_, &ChunkedArray<T>>(ca) };
                    Ok(ca)
                } else {
                    Err(PolarsError::DataTypeMisMatch)
                }
            }};
        }
        match T::get_data_type() {
            ArrowDataType::Utf8 => unpack!(Utf8),
            ArrowDataType::Boolean => unpack!(Bool),
            ArrowDataType::UInt8 => unpack!(UInt8),
            ArrowDataType::UInt16 => unpack!(UInt16),
            ArrowDataType::UInt32 => unpack!(UInt32),
            ArrowDataType::UInt64 => unpack!(UInt64),
            ArrowDataType::Int8 => unpack!(Int8),
            ArrowDataType::Int16 => unpack!(Int16),
            ArrowDataType::Int32 => unpack!(Int32),
            ArrowDataType::Int64 => unpack!(Int64),
            ArrowDataType::Float32 => unpack!(Float32),
            ArrowDataType::Float64 => unpack!(Float64),
            ArrowDataType::Date32(DateUnit::Day) => unpack!(Date32),
            ArrowDataType::Date64(DateUnit::Millisecond) => unpack!(Date64),
            ArrowDataType::Time32(TimeUnit::Millisecond) => unpack!(Time32Millisecond),
            ArrowDataType::Time32(TimeUnit::Second) => unpack!(Time32Second),
            ArrowDataType::Time64(TimeUnit::Nanosecond) => unpack!(Time64Nanosecond),
            ArrowDataType::Time64(TimeUnit::Microsecond) => unpack!(Time64Microsecond),
            ArrowDataType::Interval(IntervalUnit::DayTime) => unpack!(IntervalDayTime),
            ArrowDataType::Interval(IntervalUnit::YearMonth) => unpack!(IntervalYearMonth),
            ArrowDataType::Duration(TimeUnit::Nanosecond) => unpack!(DurationNanosecond),
            ArrowDataType::Duration(TimeUnit::Microsecond) => unpack!(DurationMicrosecond),
            ArrowDataType::Duration(TimeUnit::Millisecond) => unpack!(DurationMillisecond),
            ArrowDataType::Duration(TimeUnit::Second) => unpack!(DurationSecond),
            ArrowDataType::Timestamp(TimeUnit::Nanosecond, _) => unpack!(TimestampNanosecond),
            ArrowDataType::Timestamp(TimeUnit::Microsecond, _) => unpack!(TimestampMicrosecond),
            ArrowDataType::Timestamp(TimeUnit::Millisecond, _) => unpack!(Time32Millisecond),
            ArrowDataType::Timestamp(TimeUnit::Second, _) => unpack!(TimestampSecond),
            _ => unimplemented!(),
        }
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsDataType,
    ChunkedArray<T>: ChunkOps,
{
    /// Get the index of the chunk and the index of the value in that chunk
    #[inline]
    pub(crate) fn index_to_chunked_index(&self, index: usize) -> (usize, usize) {
        let mut index_remainder = index;
        let mut current_chunk_idx = 0;

        for chunk in &self.chunks {
            let chunk_len = chunk.len();
            if chunk_len - 1 >= index_remainder {
                break;
            } else {
                index_remainder -= chunk_len;
                current_chunk_idx += 1;
            }
        }
        (current_chunk_idx, index_remainder)
    }

    pub fn chunk_id(&self) -> &Vec<usize> {
        &self.chunk_id
    }

    /// A reference to the chunks
    pub fn chunks(&self) -> &Vec<ArrayRef> {
        &self.chunks
    }

    /// Returns true if contains a single chunk and has no null values
    pub fn is_optimal_aligned(&self) -> bool {
        self.chunks.len() == 0 && self.null_count() == 0
    }

    /// Count the null values.
    pub fn null_count(&self) -> usize {
        self.chunks.iter().map(|arr| arr.null_count()).sum()
    }

    /// Get a mask of the null values.
    pub fn is_null(&self) -> BooleanChunked {
        if self.null_count() == 0 {
            return BooleanChunked::full("is_null", false, self.len());
        }
        let chunks = self
            .chunks
            .iter()
            .map(|arr| {
                let mut builder = PrimitiveBuilder::<BooleanType>::new(arr.len());
                for i in 0..arr.len() {
                    builder
                        .append_value(arr.is_null(i))
                        .expect("could not append");
                }
                let chunk: ArrayRef = Arc::new(builder.finish());
                chunk
            })
            .collect_vec();
        BooleanChunked::new_from_chunks("is_null", chunks)
    }

    /// Downcast generic `ChunkedArray<T>` to u8.
    pub fn u8(self) -> Result<UInt8Chunked> {
        match T::get_data_type() {
            ArrowDataType::UInt8 => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to u16.
    pub fn u16(self) -> Result<UInt16Chunked> {
        match T::get_data_type() {
            ArrowDataType::UInt16 => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to u32.
    pub fn u32(self) -> Result<UInt32Chunked> {
        match T::get_data_type() {
            ArrowDataType::UInt32 => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to u64.
    pub fn u64(self) -> Result<UInt64Chunked> {
        match T::get_data_type() {
            ArrowDataType::UInt64 => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to i8.
    pub fn i8(self) -> Result<Int8Chunked> {
        match T::get_data_type() {
            ArrowDataType::Int8 => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to i16.
    pub fn i16(self) -> Result<Int16Chunked> {
        match T::get_data_type() {
            ArrowDataType::Int16 => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to i32.
    pub fn i32(self) -> Result<Int32Chunked> {
        match T::get_data_type() {
            ArrowDataType::Int32 => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to i64.
    pub fn i64(self) -> Result<Int64Chunked> {
        match T::get_data_type() {
            ArrowDataType::Int64 => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to f32.
    pub fn f32(self) -> Result<Float32Chunked> {
        match T::get_data_type() {
            ArrowDataType::Float32 => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to f64.
    pub fn f64(self) -> Result<Float64Chunked> {
        match T::get_data_type() {
            ArrowDataType::Float64 => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to bool.
    pub fn bool(self) -> Result<BooleanChunked> {
        match T::get_data_type() {
            ArrowDataType::Boolean => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to UTF-8 encoded string.
    pub fn utf8(self) -> Result<Utf8Chunked> {
        match T::get_data_type() {
            ArrowDataType::Utf8 => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to date32.
    pub fn date32(self) -> Result<Date32Chunked> {
        match T::get_data_type() {
            ArrowDataType::Date32(DateUnit::Day) => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to date32.
    pub fn date64(self) -> Result<Date64Chunked> {
        match T::get_data_type() {
            ArrowDataType::Date64(DateUnit::Millisecond) => unsafe {
                Ok(std::mem::transmute(self))
            },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }
    // TODO: insert types

    /// Downcast generic `ChunkedArray<T>` to time32 with milliseconds unit.
    pub fn time32_second(self) -> Result<Time32SecondChunked> {
        match T::get_data_type() {
            ArrowDataType::Time32(TimeUnit::Second) => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to time32 with milliseconds unit.
    pub fn time32_millisecond(self) -> Result<Time32MillisecondChunked> {
        match T::get_data_type() {
            ArrowDataType::Time32(TimeUnit::Millisecond) => unsafe {
                Ok(std::mem::transmute(self))
            },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to time64 with nanoseconds unit.
    pub fn time64_nanosecond(self) -> Result<Time64NanosecondChunked> {
        match T::get_data_type() {
            ArrowDataType::Time64(TimeUnit::Nanosecond) => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to time64 with microseconds unit.
    pub fn time64_microsecond(self) -> Result<Time64MicrosecondChunked> {
        match T::get_data_type() {
            ArrowDataType::Time64(TimeUnit::Microsecond) => unsafe {
                Ok(std::mem::transmute(self))
            },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to duration with nanoseconds unit.
    pub fn duration_nanosecond(self) -> Result<DurationNanosecondChunked> {
        match T::get_data_type() {
            ArrowDataType::Duration(TimeUnit::Nanosecond) => unsafe {
                Ok(std::mem::transmute(self))
            },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to duration with microseconds unit.
    pub fn duration_microsecond(self) -> Result<DurationMicrosecondChunked> {
        match T::get_data_type() {
            ArrowDataType::Duration(TimeUnit::Microsecond) => unsafe {
                Ok(std::mem::transmute(self))
            },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to duration with seconds unit.
    pub fn duration_second(self) -> Result<DurationSecondChunked> {
        match T::get_data_type() {
            ArrowDataType::Duration(TimeUnit::Second) => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to timestamp with nanoseconds unit.
    pub fn timestamp_nanosecond(self) -> Result<TimestampNanosecondChunked> {
        match T::get_data_type() {
            ArrowDataType::Timestamp(TimeUnit::Nanosecond, _) => unsafe {
                Ok(std::mem::transmute(self))
            },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to timestamp with microseconds unit.
    pub fn timestamp_microsecond(self) -> Result<TimestampMicrosecondChunked> {
        match T::get_data_type() {
            ArrowDataType::Timestamp(TimeUnit::Microsecond, _) => unsafe {
                Ok(std::mem::transmute(self))
            },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to timestamp with milliseconds unit.
    pub fn timestamp_millisecond(self) -> Result<TimestampMillisecondChunked> {
        match T::get_data_type() {
            ArrowDataType::Timestamp(TimeUnit::Millisecond, _) => unsafe {
                Ok(std::mem::transmute(self))
            },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast generic `ChunkedArray<T>` to timestamp with seconds unit.
    pub fn timestamp_second(self) -> Result<TimestampSecondChunked> {
        match T::get_data_type() {
            ArrowDataType::Timestamp(TimeUnit::Second, _) => unsafe {
                Ok(std::mem::transmute(self))
            },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Take a view of top n elements
    pub fn limit(&self, num_elements: usize) -> Result<Self> {
        self.slice(0, num_elements)
    }

    /// Filter values in the ChunkedArray with a boolean mask.
    ///
    /// ```rust
    /// # use polars::prelude::*;
    /// let array = Int32Chunked::new_from_slice("array", &[1, 2, 3]);
    /// let mask = BooleanChunked::new_from_slice("mask", &[true, false, true]);
    ///
    /// let filtered = array.filter(&mask).unwrap();
    /// assert_eq!(Vec::from(&filtered), [Some(1), Some(3)])
    /// ```
    pub fn filter(&self, filter: &BooleanChunked) -> Result<Self> {
        let opt = self.optional_rechunk(filter)?;
        let left = match &opt {
            Some(a) => a,
            None => self,
        };
        let chunks = left
            .chunks
            .iter()
            .zip(&filter.downcast_chunks())
            .map(|(arr, &fil)| compute::filter(&*(arr.clone()), fil))
            .collect::<std::result::Result<Vec<_>, arrow::error::ArrowError>>();

        match chunks {
            Ok(chunks) => Ok(self.copy_with_chunks(chunks)),
            Err(e) => Err(PolarsError::ArrowError(e)),
        }
    }

    /// Append arrow array in place.
    ///
    /// ```rust
    /// # use polars::prelude::*;
    /// let mut array = Int32Chunked::new_from_slice("array", &[1, 2]);
    /// let array_2 = Int32Chunked::new_from_slice("2nd", &[3]);
    ///
    /// array.append(&array_2);
    /// assert_eq!(Vec::from(&array), [Some(1), Some(2), Some(3)])
    /// ```
    pub fn append_array(&mut self, other: ArrayRef) -> Result<()> {
        if other.data_type() == self.field.data_type() {
            self.chunks.push(other);
            Ok(())
        } else {
            Err(PolarsError::DataTypeMisMatch)
        }
    }

    /// Combined length of all the chunks.
    pub fn len(&self) -> usize {
        self.chunks.iter().fold(0, |acc, arr| acc + arr.len())
    }

    /// Get a single value. Beware this is slow.
    pub fn get(&self, index: usize) -> AnyType {
        let (chunk_idx, idx) = self.index_to_chunked_index(index);
        let arr = &self.chunks[chunk_idx];

        if arr.is_null(idx) {
            return AnyType::Null;
        }

        macro_rules! downcast_and_pack {
            ($casttype:ident, $variant:ident) => {{
                let arr = arr
                    .as_any()
                    .downcast_ref::<$casttype>()
                    .expect("could not downcast one of the chunks");
                let v = arr.value(idx);
                AnyType::$variant(v)
            }};
        }
        macro_rules! downcast {
            ($casttype:ident) => {{
                let arr = arr
                    .as_any()
                    .downcast_ref::<$casttype>()
                    .expect("could not downcast one of the chunks");
                arr.value(idx)
            }};
        }
        // TODO: insert types
        match T::get_data_type() {
            ArrowDataType::Utf8 => downcast_and_pack!(StringArray, Utf8),
            ArrowDataType::Boolean => downcast_and_pack!(BooleanArray, Boolean),
            ArrowDataType::UInt8 => downcast_and_pack!(UInt8Array, UInt8),
            ArrowDataType::UInt16 => downcast_and_pack!(UInt16Array, UInt16),
            ArrowDataType::UInt32 => downcast_and_pack!(UInt32Array, UInt32),
            ArrowDataType::UInt64 => downcast_and_pack!(UInt64Array, UInt64),
            ArrowDataType::Int8 => downcast_and_pack!(Int8Array, Int8),
            ArrowDataType::Int16 => downcast_and_pack!(Int16Array, Int16),
            ArrowDataType::Int32 => downcast_and_pack!(Int32Array, Int32),
            ArrowDataType::Int64 => downcast_and_pack!(Int64Array, Int64),
            ArrowDataType::Float32 => downcast_and_pack!(Float32Array, Float32),
            ArrowDataType::Float64 => downcast_and_pack!(Float64Array, Float64),
            ArrowDataType::Date32(DateUnit::Day) => downcast_and_pack!(Date32Array, Date32),
            ArrowDataType::Date64(DateUnit::Millisecond) => downcast_and_pack!(Date64Array, Date64),
            ArrowDataType::Time32(TimeUnit::Millisecond) => {
                let v = downcast!(Time32MillisecondArray);
                AnyType::Time32(v, TimeUnit::Millisecond)
            }
            ArrowDataType::Time32(TimeUnit::Second) => {
                let v = downcast!(Time32SecondArray);
                AnyType::Time32(v, TimeUnit::Second)
            }
            ArrowDataType::Time64(TimeUnit::Nanosecond) => {
                let v = downcast!(Time64NanosecondArray);
                AnyType::Time64(v, TimeUnit::Nanosecond)
            }
            ArrowDataType::Time64(TimeUnit::Microsecond) => {
                let v = downcast!(Time64MicrosecondArray);
                AnyType::Time64(v, TimeUnit::Microsecond)
            }
            ArrowDataType::Interval(IntervalUnit::DayTime) => {
                downcast_and_pack!(IntervalDayTimeArray, IntervalDayTime)
            }
            ArrowDataType::Interval(IntervalUnit::YearMonth) => {
                downcast_and_pack!(IntervalYearMonthArray, IntervalYearMonth)
            }
            ArrowDataType::Duration(TimeUnit::Nanosecond) => {
                let v = downcast!(DurationNanosecondArray);
                AnyType::Duration(v, TimeUnit::Nanosecond)
            }
            ArrowDataType::Duration(TimeUnit::Microsecond) => {
                let v = downcast!(DurationMicrosecondArray);
                AnyType::Duration(v, TimeUnit::Microsecond)
            }
            ArrowDataType::Duration(TimeUnit::Millisecond) => {
                let v = downcast!(DurationMillisecondArray);
                AnyType::Duration(v, TimeUnit::Millisecond)
            }
            ArrowDataType::Duration(TimeUnit::Second) => {
                let v = downcast!(DurationSecondArray);
                AnyType::Duration(v, TimeUnit::Second)
            }
            ArrowDataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let v = downcast!(TimestampNanosecondArray);
                AnyType::TimeStamp(v, TimeUnit::Nanosecond)
            }
            ArrowDataType::Timestamp(TimeUnit::Microsecond, _) => {
                let v = downcast!(TimestampMicrosecondArray);
                AnyType::TimeStamp(v, TimeUnit::Microsecond)
            }
            ArrowDataType::Timestamp(TimeUnit::Millisecond, _) => {
                let v = downcast!(TimestampMillisecondArray);
                AnyType::TimeStamp(v, TimeUnit::Millisecond)
            }
            ArrowDataType::Timestamp(TimeUnit::Second, _) => {
                let v = downcast!(TimestampSecondArray);
                AnyType::TimeStamp(v, TimeUnit::Second)
            }
            _ => unimplemented!(),
        }
    }

    /// Slice the array. The chunks are reallocated the underlying data slices are zero copy.
    pub fn slice(&self, offset: usize, length: usize) -> Result<Self> {
        if offset + length > self.len() {
            return Err(PolarsError::OutOfBounds);
        }
        let mut remaining_length = length;
        let mut remaining_offset = offset;
        let mut new_chunks = vec![];

        for chunk in &self.chunks {
            let chunk_len = chunk.len();
            if remaining_offset >= chunk_len {
                remaining_offset -= chunk_len;
                continue;
            }
            let take_len;
            if remaining_length + remaining_offset > chunk_len {
                take_len = chunk_len - remaining_offset;
            } else {
                take_len = remaining_length;
            }

            new_chunks.push(chunk.slice(remaining_offset, take_len));
            remaining_length -= take_len;
            remaining_offset = 0;
            if remaining_length == 0 {
                break;
            }
        }
        Ok(self.copy_with_chunks(new_chunks))
    }

    /// Get the head of the ChunkedArray
    pub fn head(&self, length: Option<usize>) -> Self {
        let res_ca = match length {
            Some(len) => self.slice(0, std::cmp::min(len, self.len())),
            None => self.slice(0, std::cmp::min(10, self.len())),
        };
        res_ca.unwrap()
    }

    /// Get the tail of the ChunkedArray
    pub fn tail(&self, length: Option<usize>) -> Self {
        let len = match length {
            Some(len) => std::cmp::min(len, self.len()),
            None => std::cmp::min(10, self.len()),
        };
        self.slice(self.len() - len, len).unwrap()
    }

    /// Append in place.
    pub fn append(&mut self, other: &Self)
    where
        Self: std::marker::Sized,
    {
        self.chunks.extend(other.chunks.clone())
    }
}

impl Utf8Chunked {
    pub fn new_utf8_from_slice<S: AsRef<str>>(name: &str, v: &[S]) -> Self {
        let mut builder = StringBuilder::new(v.len());
        v.into_iter().for_each(|val| {
            builder
                .append_value(val.as_ref())
                .expect("Could not append value");
        });

        let field = Arc::new(Field::new(name, ArrowDataType::Utf8, true));

        ChunkedArray {
            field,
            chunks: vec![Arc::new(builder.finish())],
            chunk_id: vec![v.len()],
            phantom: PhantomData,
        }
    }

    pub fn new_utf8_from_opt_slice<S: AsRef<str>>(name: &str, opt_v: &[Option<S>]) -> Self {
        let mut builder = Utf8ChunkedBuilder::new(name, opt_v.len());

        opt_v.iter().for_each(|opt| match opt {
            Some(v) => builder.append_value(v.as_ref()).expect("append value"),
            None => builder.append_null().expect("append null"),
        });
        builder.finish()
    }
}

impl<T> ChunkedArray<T>
where
    T: datatypes::PolarsDataType,
    ChunkedArray<T>: ChunkOps,
{
    /// Name of the ChunkedArray.
    pub fn name(&self) -> &str {
        self.field.name()
    }

    /// Get a reference to the field.
    pub fn ref_field(&self) -> &Field {
        &self.field
    }

    /// Rename this ChunkedArray.
    pub fn rename(&mut self, name: &str) {
        self.field = Arc::new(Field::new(
            name,
            self.field.data_type().clone(),
            self.field.is_nullable(),
        ))
    }

    /// Create a new ChunkedArray from self, where the chunks are replaced.
    fn copy_with_chunks(&self, chunks: Vec<ArrayRef>) -> Self {
        let chunk_id = create_chunk_id(&chunks);
        ChunkedArray {
            field: self.field.clone(),
            chunks,
            chunk_id,
            phantom: PhantomData,
        }
    }

    /// Recompute the chunk_id / chunk_lengths.
    fn set_chunk_id(&mut self) {
        self.chunk_id = create_chunk_id(&self.chunks)
    }
}

impl<T> ChunkedArray<T>
where
    T: ArrowPrimitiveType,
{
    pub fn new_from_slice(name: &str, v: &[T::Native]) -> Self {
        let mut builder = PrimitiveChunkedBuilder::<T>::new(name, v.len());
        v.iter()
            .for_each(|&v| builder.append_value(v).expect("append"));
        builder.finish()
    }

    pub fn new_from_opt_slice(name: &str, opt_v: &[Option<T::Native>]) -> Self {
        let mut builder = PrimitiveChunkedBuilder::<T>::new(name, opt_v.len());
        opt_v
            .iter()
            .for_each(|&opt| builder.append_option(opt).expect("append"));
        builder.finish()
    }

    /// Nullify values in slice with an existing null bitmap
    pub fn new_with_null_bitmap(
        name: &str,
        values: &[T::Native],
        buffer: Option<Buffer>,
        null_count: usize,
    ) -> Self {
        let len = values.len();
        let arr = Arc::new(build_with_existing_null_bitmap_and_slice::<T>(
            buffer, null_count, values,
        ));
        ChunkedArray {
            field: Arc::new(Field::new(name, T::get_data_type(), true)),
            chunks: vec![arr],
            chunk_id: vec![len],
            phantom: PhantomData,
        }
    }

    /// Nullify values in slice with an existing null bitmap
    pub fn new_from_owned_with_null_bitmap(
        name: &str,
        values: AlignedVec<T::Native>,
        buffer: Option<Buffer>,
        null_count: usize,
    ) -> Self {
        let len = values.0.len();
        let arr = Arc::new(aligned_vec_to_primitive_array::<T>(
            values, buffer, null_count,
        ));
        ChunkedArray {
            field: Arc::new(Field::new(name, T::get_data_type(), true)),
            chunks: vec![arr],
            chunk_id: vec![len],
            phantom: PhantomData,
        }
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    /// Contiguous slice
    pub fn cont_slice(&self) -> Result<&[T::Native]> {
        if self.chunks.len() == 1 && self.chunks[0].null_count() == 0 {
            Ok(self.downcast_chunks()[0].value_slice(0, self.len()))
        } else {
            Err(PolarsError::NoSlice)
        }
    }

    /// Get slices of the underlying arrow data.
    /// NOTE: null values should be taken into account by the user of these slices as they are handled
    /// separately
    pub fn data_views(&self) -> Vec<&[T::Native]> {
        self.downcast_chunks()
            .iter()
            .map(|arr| arr.value_slice(0, arr.len()))
            .collect()
    }

    /// Rechunk and return a ptr to the start of the array
    pub fn as_single_ptr(&mut self) -> usize {
        let mut ca = self.rechunk(None).expect("should not fail");
        mem::swap(&mut ca, self);
        let a = self.data_views()[0];
        let ptr = a.as_ptr();
        ptr as usize
    }

    /// If [cont_slice](#method.cont_slice) is successful a closure is mapped over the elements.
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn multiply(ca: &UInt32Chunked) -> Result<Series> {
    ///     let mapped = ca.map(|v| v * 2)?;
    ///     Ok(mapped.collect())
    /// }
    /// ```
    pub fn map<B, F>(&self, f: F) -> Result<Map<Copied<std::slice::Iter<T::Native>>, F>>
    where
        F: Fn(T::Native) -> B,
    {
        let slice = self.cont_slice()?;
        Ok(slice.iter().copied().map(f))
    }

    /// If [cont_slice](#method.cont_slice) fails we can fallback on an iterator with null checks
    /// and map a closure over the elements.
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// use itertools::Itertools;
    /// fn multiply(ca: &UInt32Chunked) -> Series {
    ///     let mapped_result = ca.map(|v| v * 2);
    ///
    ///     if let Ok(mapped) = mapped_result {
    ///         mapped.collect()
    ///     } else {
    ///         ca
    ///         .map_null_checks(|opt_v| opt_v.map(|v |v * 2)).collect()
    ///     }
    /// }
    /// ```
    pub fn map_null_checks<B, F>(&self, f: F) -> Map<NumericChunkIterDispatch<T>, F>
    where
        F: Fn(Option<T::Native>) -> B,
    {
        self.into_iter().map(f)
    }

    /// If [cont_slice](#method.cont_slice) is successful a closure can be applied as aggregation
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn compute_sum(ca: &UInt32Chunked) -> Result<u32> {
    ///     ca.fold(0, |acc, value| acc + value)
    /// }
    /// ```
    pub fn fold<F, B>(&self, init: B, f: F) -> Result<B>
    where
        F: Fn(B, T::Native) -> B,
    {
        let slice = self.cont_slice()?;
        Ok(slice.iter().copied().fold(init, f))
    }

    /// If [cont_slice](#method.cont_slice) fails we can fallback on an iterator with null checks
    /// and a closure for aggregation
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn compute_sum(ca: &UInt32Chunked) -> u32 {
    ///     match ca.fold(0, |acc, value| acc + value) {
    ///         // faster sum without null checks was successful
    ///         Ok(sum) => sum,
    ///         // Null values or multiple chunks in ChunkedArray, we need to do more bounds checking
    ///         Err(_) => ca.fold_null_checks(0, |acc, opt_value| {
    ///             match opt_value {
    ///                 Some(v) => acc + v,
    ///                 None => acc
    ///             }
    ///         })
    ///     }
    /// }
    /// ```
    pub fn fold_null_checks<F, B>(&self, init: B, f: F) -> B
    where
        F: Fn(B, Option<T::Native>) -> B,
    {
        self.into_iter().fold(init, f)
    }
}

impl<T> Debug for ChunkedArray<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("{:?}", self.chunks))
    }
}

impl<T> Clone for ChunkedArray<T> {
    fn clone(&self) -> Self {
        ChunkedArray {
            field: self.field.clone(),
            chunks: self.chunks.clone(),
            chunk_id: self.chunk_id.clone(),
            phantom: PhantomData,
        }
    }
}

pub trait Downcast<T> {
    fn downcast_chunks(&self) -> Vec<&T>;
}

impl<T> Downcast<PrimitiveArray<T>> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn downcast_chunks(&self) -> Vec<&PrimitiveArray<T>> {
        self.chunks
            .iter()
            .map(|arr| {
                arr.as_any()
                    .downcast_ref::<PrimitiveArray<T>>()
                    .expect("could not downcast one of the chunks")
            })
            .collect::<Vec<_>>()
    }
}

impl Downcast<StringArray> for Utf8Chunked {
    fn downcast_chunks(&self) -> Vec<&StringArray> {
        self.chunks
            .iter()
            .map(|arr| {
                arr.as_any()
                    .downcast_ref()
                    .expect("could not downcast one of the chunks")
            })
            .collect::<Vec<_>>()
    }
}

impl Downcast<BooleanArray> for BooleanChunked {
    fn downcast_chunks(&self) -> Vec<&BooleanArray> {
        self.chunks
            .iter()
            .map(|arr| {
                arr.as_any()
                    .downcast_ref()
                    .expect("could not downcast one of the chunks")
            })
            .collect::<Vec<_>>()
    }
}

impl<T> AsRef<ChunkedArray<T>> for ChunkedArray<T> {
    fn as_ref(&self) -> &ChunkedArray<T> {
        self
    }
}

pub trait ChunkSort<T> {
    fn sort(&self, reverse: bool) -> ChunkedArray<T>;

    fn sort_in_place(&mut self, reverse: bool);

    fn argsort(&self, reverse: bool) -> Vec<usize>;
}

fn sort_partial<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
    match (a, b) {
        (Some(a), Some(b)) => a.partial_cmp(b).expect("could not compare"),
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

impl<T> ChunkSort<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: std::cmp::PartialOrd,
{
    fn sort(&self, reverse: bool) -> ChunkedArray<T> {
        if reverse {
            self.into_iter()
                .sorted_by(|a, b| sort_partial(b, a))
                .collect()
        } else {
            self.into_iter()
                .sorted_by(|a, b| sort_partial(a, b))
                .collect()
        }
    }

    fn sort_in_place(&mut self, reverse: bool) {
        let sorted = self.sort(reverse);
        self.chunks = sorted.chunks;
    }

    fn argsort(&self, reverse: bool) -> Vec<usize> {
        if reverse {
            self.into_iter()
                .enumerate()
                .sorted_by(|(_idx_a, a), (_idx_b, b)| sort_partial(b, a))
                .map(|(idx, _v)| idx)
                .collect::<AlignedVec<usize>>()
                .0
        } else {
            self.into_iter()
                .enumerate()
                .sorted_by(|(_idx_a, a), (_idx_b, b)| sort_partial(a, b))
                .map(|(idx, _v)| idx)
                .collect::<AlignedVec<usize>>()
                .0
        }
    }
}

macro_rules! argsort {
    ($self:ident, $closure:expr) => {{
        $self
            .into_iter()
            .enumerate()
            .sorted_by($closure)
            .map(|(idx, _v)| idx)
            .collect::<AlignedVec<usize>>()
            .0
    }};
}

macro_rules! sort {
    ($self:ident, $reverse:ident) => {{
        if $reverse {
            $self.into_iter().sorted_by(|a, b| b.cmp(a)).collect()
        } else {
            $self.into_iter().sorted_by(|a, b| a.cmp(b)).collect()
        }
    }};
}

impl ChunkSort<Utf8Type> for Utf8Chunked {
    fn sort(&self, reverse: bool) -> Utf8Chunked {
        sort!(self, reverse)
    }

    fn sort_in_place(&mut self, reverse: bool) {
        let sorted = self.sort(reverse);
        self.chunks = sorted.chunks;
    }

    fn argsort(&self, reverse: bool) -> Vec<usize> {
        if reverse {
            argsort!(self, |(_idx_a, a), (_idx_b, b)| b.cmp(a))
        } else {
            argsort!(self, |(_idx_a, a), (_idx_b, b)| a.cmp(b))
        }
    }
}

impl ChunkSort<BooleanType> for BooleanChunked {
    fn sort(&self, reverse: bool) -> BooleanChunked {
        sort!(self, reverse)
    }

    fn sort_in_place(&mut self, reverse: bool) {
        let sorted = self.sort(reverse);
        self.chunks = sorted.chunks;
    }

    fn argsort(&self, reverse: bool) -> Vec<usize> {
        if reverse {
            argsort!(self, |(_idx_a, a), (_idx_b, b)| b.cmp(a))
        } else {
            argsort!(self, |(_idx_a, a), (_idx_b, b)| a.cmp(b))
        }
    }
}

/// Fill a ChunkedArray with one value.
pub trait ChunkFull<T> {
    /// Create a ChunkedArray with a single value.
    fn full(name: &str, value: T, length: usize) -> Self
    where
        Self: std::marker::Sized;
}

impl<T> ChunkFull<T::Native> for ChunkedArray<T>
where
    T: ArrowPrimitiveType,
{
    fn full(name: &str, value: T::Native, length: usize) -> Self
    where
        T::Native: Copy,
    {
        let mut builder = PrimitiveChunkedBuilder::new(name, length);

        for _ in 0..length {
            builder.append_value(value).expect("could not append?")
        }
        builder.finish()
    }
}

impl<'a> ChunkFull<&'a str> for Utf8Chunked {
    fn full(name: &str, value: &'a str, length: usize) -> Self {
        let mut builder = Utf8ChunkedBuilder::new(name, length);

        for _ in 0..length {
            builder.append_value(value).expect("could not append?")
        }
        builder.finish()
    }
}

pub trait Reverse<T> {
    fn reverse(&self) -> ChunkedArray<T>;
}

impl<T> Reverse<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkOps,
{
    fn reverse(&self) -> ChunkedArray<T> {
        if let Ok(slice) = self.cont_slice() {
            let ca: Xob<ChunkedArray<T>> = slice.iter().rev().copied().collect();
            let mut ca = ca.into_inner();
            ca.rename(self.name());
            ca
        } else {
            self.take((0..self.len()).rev(), None)
                .expect("implementation error, should not fail")
        }
    }
}

macro_rules! impl_reverse {
    ($arrow_type:ident, $ca_type:ident) => {
        impl Reverse<$arrow_type> for $ca_type {
            fn reverse(&self) -> Self {
                self.take((0..self.len()).rev(), None)
                    .expect("implementation error, should not fail")
            }
        }
    };
}

impl_reverse!(BooleanType, BooleanChunked);
impl_reverse!(Utf8Type, Utf8Chunked);

#[cfg(test)]
pub(crate) mod test {
    use crate::prelude::*;

    pub(crate) fn get_chunked_array() -> Int32Chunked {
        ChunkedArray::new_from_slice("a", &[1, 2, 3])
    }

    #[test]
    fn test_sort() {
        let a = Int32Chunked::new_from_slice("a", &[1, 9, 3, 2]);
        let b = a
            .sort(false)
            .into_iter()
            .map(|opt| opt.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(b, [1, 2, 3, 9]);
        let a = Utf8Chunked::new_utf8_from_slice("a", &["b", "a", "c"]);
        let a = a.sort(false);
        let b = a.into_iter().collect::<Vec<_>>();
        assert_eq!(b, [Some("a"), Some("b"), Some("c")]);
    }

    #[test]
    fn arithmetic() {
        let s1 = get_chunked_array();
        println!("{:?}", s1.chunks);
        let s2 = &s1.clone();
        let s1 = &s1;
        println!("{:?}", s1 + s2);
        println!("{:?}", s1 - s2);
        println!("{:?}", s1 * s2);
    }

    #[test]
    fn iter() {
        let s1 = get_chunked_array();
        // sum
        assert_eq!(s1.into_iter().fold(0, |acc, val| { acc + val.unwrap() }), 6)
    }

    #[test]
    fn limit() {
        let a = get_chunked_array();
        let b = a.limit(2).unwrap();
        println!("{:?}", b);
        assert_eq!(b.len(), 2)
    }

    #[test]
    fn filter() {
        let a = get_chunked_array();
        let b = a
            .filter(&BooleanChunked::new_from_slice(
                "filter",
                &[true, false, false],
            ))
            .unwrap();
        assert_eq!(b.len(), 1);
        assert_eq!(b.into_iter().next(), Some(Some(1)));
    }

    #[test]
    fn aggregates_numeric() {
        let a = get_chunked_array();
        assert_eq!(a.max(), Some(3));
        assert_eq!(a.min(), Some(1));
        assert_eq!(a.sum(), Some(6))
    }

    #[test]
    fn take() {
        let a = get_chunked_array();
        let new = a.take([0u32, 1].as_ref().as_take_iter(), None).unwrap();
        assert_eq!(new.len(), 2)
    }

    #[test]
    fn get() {
        let mut a = get_chunked_array();
        assert_eq!(AnyType::Int32(2), a.get(1));
        // check if chunks indexes are properly determined
        a.append_array(a.chunks[0].clone()).unwrap();
        assert_eq!(AnyType::Int32(1), a.get(3));
    }

    #[test]
    fn cast() {
        let a = get_chunked_array();
        let b = a.cast::<Int64Type>().unwrap();
        assert_eq!(b.field.data_type(), &ArrowDataType::Int64)
    }

    fn assert_slice_equal<T>(ca: &ChunkedArray<T>, eq: &[T::Native])
    where
        ChunkedArray<T>: ChunkOps,
        T: PolarsNumericType,
    {
        assert_eq!(
            ca.into_iter().map(|opt| opt.unwrap()).collect::<Vec<_>>(),
            eq
        )
    }

    #[test]
    fn slice() {
        let mut first = UInt32Chunked::new_from_slice("first", &[0, 1, 2]);
        let second = UInt32Chunked::new_from_slice("second", &[3, 4, 5]);
        first.append(&second);
        assert_slice_equal(&first.slice(0, 3).unwrap(), &[0, 1, 2]);
        assert_slice_equal(&first.slice(0, 4).unwrap(), &[0, 1, 2, 3]);
        assert_slice_equal(&first.slice(1, 4).unwrap(), &[1, 2, 3, 4]);
        assert_slice_equal(&first.slice(3, 2).unwrap(), &[3, 4]);
        assert_slice_equal(&first.slice(3, 3).unwrap(), &[3, 4, 5]);
        assert!(first.slice(3, 4).is_err());
    }

    #[test]
    fn sorting() {
        let s = UInt32Chunked::new_from_slice("", &[9, 2, 4]);
        let sorted = s.sort(false);
        assert_slice_equal(&sorted, &[2, 4, 9]);
        let sorted = s.sort(true);
        assert_slice_equal(&sorted, &[9, 4, 2]);

        let s: Utf8Chunked = ["b", "a", "z"].iter().collect();
        let sorted = s.sort(false);
        assert_eq!(
            sorted.into_iter().collect::<Vec<_>>(),
            &[Some("a"), Some("b"), Some("z")]
        );
        let sorted = s.sort(true);
        assert_eq!(
            sorted.into_iter().collect::<Vec<_>>(),
            &[Some("z"), Some("b"), Some("a")]
        );
    }

    #[test]
    fn reverse() {
        let s = UInt32Chunked::new_from_slice("", &[1, 2, 3]);
        // path with continuous slice
        assert_slice_equal(&s.reverse(), &[3, 2, 1]);
        // path with options
        let s = UInt32Chunked::new_from_opt_slice("", &[Some(1), None, Some(3)]);
        assert_eq!(Vec::from(&s.reverse()), &[Some(3), None, Some(1)]);
        let s = BooleanChunked::new_from_slice("", &[true, false]);
        assert_eq!(Vec::from(&s.reverse()), &[Some(false), Some(true)]);

        let s = Utf8Chunked::new_utf8_from_slice("", &["a", "b", "c"]);
        assert_eq!(Vec::from(&s.reverse()), &[Some("c"), Some("b"), Some("a")]);

        let s = Utf8Chunked::new_utf8_from_opt_slice("", &[Some("a"), None, Some("c")]);
        assert_eq!(Vec::from(&s.reverse()), &[Some("c"), None, Some("a")]);
    }
}
