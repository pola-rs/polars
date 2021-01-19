//! The typed heart of every Series column.
use crate::chunked_array::builder::{
    aligned_vec_to_primitive_array, build_with_existing_null_bitmap_and_slice, get_bitmap,
};
use crate::prelude::*;
use arrow::{
    array::{
        ArrayRef, BooleanArray, Date64Array, Float32Array, Float64Array, Int16Array, Int32Array,
        Int64Array, Int8Array, LargeStringArray, PrimitiveArray, Time64NanosecondArray,
        UInt16Array, UInt32Array, UInt64Array, UInt8Array,
    },
    buffer::Buffer,
    datatypes::TimeUnit,
};
use itertools::Itertools;
use std::convert::TryFrom;
use std::iter::{Copied, Map};
use std::marker::PhantomData;
use std::sync::Arc;

pub mod ops;
#[macro_use]
pub mod arithmetic;
pub mod boolean;
pub mod builder;
pub mod cast;
pub mod comparison;
pub mod float;
pub mod iterator;
pub mod kernels;
#[cfg(feature = "ndarray")]
#[doc(cfg(feature = "ndarray"))]
mod ndarray;

#[cfg(feature = "object")]
#[doc(cfg(feature = "object"))]
pub mod object;
#[cfg(feature = "random")]
#[doc(cfg(feature = "random"))]
mod random;
#[cfg(feature = "strings")]
#[doc(cfg(feature = "strings"))]
pub mod strings;
#[cfg(feature = "temporal")]
#[doc(cfg(feature = "temporal"))]
pub mod temporal;
pub mod upstream_traits;

#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectArray;
use arrow::array::{
    Array, ArrayDataRef, BooleanBuilder, Date32Array, DurationMillisecondArray,
    DurationNanosecondArray, LargeListArray,
};

use ahash::AHashMap;
use arrow::util::bit_util::{get_bit, round_upto_power_of_2};
use polars_arrow::array::ValueSize;
use std::mem;
use std::ops::{Deref, DerefMut};

/// Get a 'hash' of the chunks in order to compare chunk sizes quickly.
fn create_chunk_id(chunks: &[ArrayRef]) -> Vec<usize> {
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
/// # use polars_core::prelude::*;
/// fn apply_cosine(ca: &Float32Chunked) -> Float32Chunked {
///     ca.apply(|v| v.cos())
/// }
/// ```
///
/// If we would like to cast the result we could use a Rust Iterator instead of an `apply` method.
/// Note that Iterators are slightly slower as the null values aren't ignored implicitly.
///
/// ```rust
/// # use polars_core::prelude::*;
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
/// # use polars_core::prelude::*;
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
/// # use polars_core::prelude::*;
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
/// # use polars_core::prelude::*;
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
    pub(crate) chunks: Vec<ArrayRef>,
    // chunk lengths
    chunk_id: Vec<usize>,
    phantom: PhantomData<T>,
    /// maps categorical u32 indexes to String values
    pub(crate) categorical_map: Option<Arc<AHashMap<u32, String>>>,
}

impl<T> ChunkedArray<T> {
    /// Get Arrow ArrayData
    pub fn array_data(&self) -> Vec<ArrayDataRef> {
        self.chunks.iter().map(|arr| arr.data()).collect()
    }

    /// Get the index of the first non null value in this ChunkedArray.
    pub fn first_non_null(&self) -> Option<usize> {
        if self.null_count() == self.len() {
            None
        } else if self.null_count() == 0 {
            Some(0)
        } else {
            let mut offset = 0;
            for (idx, (null_count, null_bit_buffer)) in self.null_bits().iter().enumerate() {
                if *null_count == 0 {
                    return Some(offset);
                } else {
                    let arr = &self.chunks[idx];
                    let null_bit_buffer = null_bit_buffer.as_ref().unwrap();
                    let bit_end = arr.offset() + arr.len();

                    let byte_start = std::cmp::min(round_upto_power_of_2(arr.offset(), 8), bit_end);
                    let data = null_bit_buffer.as_slice();

                    for i in arr.offset()..byte_start {
                        if get_bit(data, i) {
                            return Some(offset + i);
                        }
                    }
                    offset += arr.len()
                }
            }
            None
        }
    }

    /// Get the null count and the buffer of bits representing null values
    pub fn null_bits(&self) -> Vec<(usize, Option<Buffer>)> {
        self.chunks
            .iter()
            .map(|arr| get_bitmap(arr.as_ref()))
            .collect()
    }

    /// Series to ChunkedArray<T>
    pub fn unpack_series_matching_type(&self, series: &Series) -> Result<&ChunkedArray<T>> {
        let series_trait = &**series;
        if self.dtype() == series.dtype() {
            let ca =
                unsafe { &*(series_trait as *const dyn SeriesTrait as *const ChunkedArray<T>) };
            Ok(ca)
        } else {
            Err(PolarsError::DataTypeMisMatch(
                format!("cannot unpack series {:?} into matching type", series).into(),
            ))
        }
    }

    /// Combined length of all the chunks.
    pub fn len(&self) -> usize {
        self.chunks.iter().fold(0, |acc, arr| acc + arr.len())
    }

    /// Check if ChunkedArray is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Unique id representing the number of chunks
    pub fn chunk_id(&self) -> &Vec<usize> {
        &self.chunk_id
    }

    /// A reference to the chunks
    pub fn chunks(&self) -> &Vec<ArrayRef> {
        &self.chunks
    }

    /// Returns true if contains a single chunk and has no null values
    pub fn is_optimal_aligned(&self) -> bool {
        self.chunks.len() == 1 && self.null_count() == 0
    }

    /// Count the null values.
    pub fn null_count(&self) -> usize {
        self.chunks.iter().map(|arr| arr.null_count()).sum()
    }

    /// Take a view of top n elements
    pub fn limit(&self, num_elements: usize) -> Result<Self> {
        self.slice(0, num_elements)
    }

    /// Append arrow array in place.
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut array = Int32Chunked::new_from_slice("array", &[1, 2]);
    /// let array_2 = Int32Chunked::new_from_slice("2nd", &[3]);
    ///
    /// array.append(&array_2);
    /// assert_eq!(Vec::from(&array), [Some(1), Some(2), Some(3)])
    /// ```
    pub fn append_array(&mut self, other: ArrayRef) -> Result<()> {
        if matches!(self.dtype(), DataType::Categorical) {
            return Err(PolarsError::InvalidOperation(
                "append_array not supported for categorical type".into(),
            ));
        }
        if self.field.data_type() == other.data_type() {
            self.chunks.push(other);
            self.chunk_id = create_chunk_id(&self.chunks);
            Ok(())
        } else {
            Err(PolarsError::DataTypeMisMatch(
                format!(
                    "cannot append array of type {:?} in array of type {:?}",
                    other.data_type(),
                    self.dtype()
                )
                .into(),
            ))
        }
    }

    /// Create a new ChunkedArray from self, where the chunks are replaced.
    fn copy_with_chunks(&self, chunks: Vec<ArrayRef>) -> Self {
        let chunk_id = create_chunk_id(&chunks);
        ChunkedArray {
            field: self.field.clone(),
            chunks,
            chunk_id,
            phantom: PhantomData,
            categorical_map: self.categorical_map.clone(),
        }
    }

    /// Slice the array. The chunks are reallocated the underlying data slices are zero copy.
    pub fn slice(&self, offset: usize, length: usize) -> Result<Self> {
        if offset + length > self.len() {
            return Err(PolarsError::OutOfBounds("offset and length was larger than the size of the ChunkedArray during slice operation".into()));
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

    /// Get a mask of the null values.
    pub fn is_null(&self) -> BooleanChunked {
        if self.null_count() == 0 {
            return BooleanChunked::full("is_null", false, self.len());
        }
        let chunks = self
            .chunks
            .iter()
            .map(|arr| {
                let mut builder = BooleanBuilder::new(arr.len());
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

    /// Get a mask of the null values.
    pub fn is_not_null(&self) -> BooleanChunked {
        if self.null_count() == 0 {
            return BooleanChunked::full("is_not_null", true, self.len());
        }
        let chunks = self
            .chunks
            .iter()
            .map(|arr| {
                let mut builder = BooleanBuilder::new(arr.len());
                for i in 0..arr.len() {
                    builder
                        .append_value(arr.is_valid(i))
                        .expect("could not append");
                }
                let chunk: ArrayRef = Arc::new(builder.finish());
                chunk
            })
            .collect_vec();
        BooleanChunked::new_from_chunks("is_not_null", chunks)
    }

    /// Get data type of ChunkedArray.
    pub fn dtype(&self) -> &DataType {
        self.field.data_type()
    }

    /// Get the index of the chunk and the index of the value in that chunk
    #[inline]
    pub(crate) fn index_to_chunked_index(&self, index: usize) -> (usize, usize) {
        if self.chunks.len() == 1 {
            return (0, index);
        }
        let mut index_remainder = index;
        let mut current_chunk_idx = 0;

        for chunk in &self.chunks {
            let chunk_len = chunk.len();
            if chunk_len > index_remainder {
                break;
            } else {
                index_remainder -= chunk_len;
                current_chunk_idx += 1;
            }
        }
        (current_chunk_idx, index_remainder)
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
        if matches!(self.dtype(), DataType::Categorical) {
            assert!(Arc::ptr_eq(
                self.categorical_map.as_ref().unwrap(),
                other.categorical_map.as_ref().unwrap()
            ));
        }

        // replace an empty array
        if self.chunks.len() == 1 && self.is_empty() {
            self.chunks = other.chunks.clone();
        } else {
            self.chunks.extend(other.chunks.clone())
        }
        self.chunk_id = create_chunk_id(&self.chunks);
    }

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
        self.field = Arc::new(Field::new(name, self.field.data_type().clone()))
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsDataType,
{
    /// Create a new ChunkedArray from existing chunks.
    pub fn new_from_chunks(name: &str, chunks: Vec<ArrayRef>) -> Self {
        // prevent List<Null> if the inner list type is known.
        let datatype = if matches!(T::get_dtype(), DataType::List(_)) {
            if let Some(arr) = chunks.get(0) {
                arr.data_type().into()
            } else {
                T::get_dtype()
            }
        } else {
            T::get_dtype()
        };
        let field = Arc::new(Field::new(name, datatype));
        let chunk_id = create_chunk_id(&chunks);
        ChunkedArray {
            field,
            chunks,
            chunk_id,
            phantom: PhantomData,
            categorical_map: None,
        }
    }

    fn arr_to_any_value(&self, arr: &dyn Array, idx: usize) -> AnyValue {
        if arr.is_null(idx) {
            return AnyValue::Null;
        }

        macro_rules! downcast_and_pack {
            ($casttype:ident, $variant:ident) => {{
                let arr = unsafe { &*(arr as *const dyn Array as *const $casttype) };
                let v = arr.value(idx);
                AnyValue::$variant(v)
            }};
        }
        macro_rules! downcast {
            ($casttype:ident) => {{
                let arr = unsafe { &*(arr as *const dyn Array as *const $casttype) };
                arr.value(idx)
            }};
        }
        // TODO: insert types
        match T::get_dtype() {
            DataType::Utf8 => downcast_and_pack!(LargeStringArray, Utf8),
            DataType::Boolean => downcast_and_pack!(BooleanArray, Boolean),
            DataType::UInt8 => downcast_and_pack!(UInt8Array, UInt8),
            DataType::UInt16 => downcast_and_pack!(UInt16Array, UInt16),
            DataType::UInt32 => downcast_and_pack!(UInt32Array, UInt32),
            DataType::UInt64 => downcast_and_pack!(UInt64Array, UInt64),
            DataType::Int8 => downcast_and_pack!(Int8Array, Int8),
            DataType::Int16 => downcast_and_pack!(Int16Array, Int16),
            DataType::Int32 => downcast_and_pack!(Int32Array, Int32),
            DataType::Int64 => downcast_and_pack!(Int64Array, Int64),
            DataType::Float32 => downcast_and_pack!(Float32Array, Float32),
            DataType::Float64 => downcast_and_pack!(Float64Array, Float64),
            DataType::Date32 => downcast_and_pack!(Date32Array, Date32),
            DataType::Date64 => downcast_and_pack!(Date64Array, Date64),
            DataType::Time64(TimeUnit::Nanosecond) => {
                let v = downcast!(Time64NanosecondArray);
                AnyValue::Time64(v, TimeUnit::Nanosecond)
            }
            DataType::Duration(TimeUnit::Nanosecond) => {
                let v = downcast!(DurationNanosecondArray);
                AnyValue::Duration(v, TimeUnit::Nanosecond)
            }
            DataType::Duration(TimeUnit::Millisecond) => {
                let v = downcast!(DurationMillisecondArray);
                AnyValue::Duration(v, TimeUnit::Millisecond)
            }
            DataType::List(_) => {
                let v = downcast!(LargeListArray);
                let s = Series::try_from(("", v));
                AnyValue::List(s.unwrap())
            }
            #[cfg(feature = "object")]
            DataType::Object => AnyValue::Object(&"object"),
            DataType::Categorical => {
                let v = downcast!(UInt32Array);
                AnyValue::Utf8(
                    &self
                        .categorical_map
                        .as_ref()
                        .expect("should be set")
                        .get(&v)
                        .unwrap(),
                )
            }
            _ => unimplemented!(),
        }
    }

    /// Get a single value. Beware this is slow.
    /// If you need to use this slightly performant, cast Categorical to UInt32
    pub(crate) unsafe fn get_any_value_unchecked(&self, index: usize) -> AnyValue {
        let (chunk_idx, idx) = self.index_to_chunked_index(index);
        debug_assert!(chunk_idx < self.chunks.len());
        let arr = &**self.chunks.get_unchecked(chunk_idx);
        debug_assert!(idx < arr.len());
        self.arr_to_any_value(arr, idx)
    }

    /// Get a single value. Beware this is slow.
    /// If you need to use this slightly performant, cast Categorical to UInt32
    pub(crate) fn get_any_value(&self, index: usize) -> AnyValue {
        let (chunk_idx, idx) = self.index_to_chunked_index(index);
        let arr = &*self.chunks[chunk_idx];
        assert!(idx < arr.len());
        self.arr_to_any_value(arr, idx)
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsPrimitiveType,
{
    /// Create a new ChunkedArray by taking ownership of the AlignedVec. This operation is zero copy.
    pub fn new_from_aligned_vec(name: &str, v: AlignedVec<T::Native>) -> Self {
        let arr = aligned_vec_to_primitive_array::<T>(v, None, Some(0));
        Self::new_from_chunks(name, vec![Arc::new(arr)])
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
            field: Arc::new(Field::new(name, T::get_dtype())),
            chunks: vec![arr],
            chunk_id: vec![len],
            phantom: PhantomData,
            categorical_map: None,
        }
    }

    /// Nullify values in slice with an existing null bitmap
    pub fn new_from_owned_with_null_bitmap(
        name: &str,
        values: AlignedVec<T::Native>,
        buffer: Option<Buffer>,
        null_count: usize,
    ) -> Self {
        let len = values.len();
        let arr = Arc::new(aligned_vec_to_primitive_array::<T>(
            values,
            buffer,
            Some(null_count),
        ));
        ChunkedArray {
            field: Arc::new(Field::new(name, T::get_dtype())),
            chunks: vec![arr],
            chunk_id: vec![len],
            phantom: PhantomData,
            categorical_map: None,
        }
    }
}

pub(crate) trait AsSinglePtr {
    /// Rechunk and return a ptr to the start of the array
    fn as_single_ptr(&mut self) -> Result<usize> {
        Err(PolarsError::InvalidOperation(
            "operation as_single_ptr not supported for this dtype".into(),
        ))
    }
}

impl<T> AsSinglePtr for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn as_single_ptr(&mut self) -> Result<usize> {
        let mut ca = self.rechunk().expect("should not fail");
        mem::swap(&mut ca, self);
        let a = self.data_views()[0];
        let ptr = a.as_ptr();
        Ok(ptr as usize)
    }
}

impl AsSinglePtr for BooleanChunked {}
impl AsSinglePtr for ListChunked {}
impl AsSinglePtr for Utf8Chunked {}
impl AsSinglePtr for CategoricalChunked {}
#[cfg(feature = "object")]
impl<T> AsSinglePtr for ObjectChunked<T> {}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    /// Contiguous slice
    pub fn cont_slice(&self) -> Result<&[T::Native]> {
        if self.chunks.len() == 1 && self.chunks[0].null_count() == 0 {
            Ok(self.downcast_chunks()[0].values())
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
            .map(|arr| arr.values())
            .collect()
    }

    /// If [cont_slice](#method.cont_slice) is successful a closure is mapped over the elements.
    ///
    /// # Example
    ///
    /// ```
    /// use polars_core::prelude::*;
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
    /// use polars_core::prelude::*;
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
    pub fn map_null_checks<'a, B, F>(
        &'a self,
        f: F,
    ) -> Map<Box<dyn PolarsIterator<Item = Option<T::Native>> + 'a>, F>
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
    /// use polars_core::prelude::*;
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
    /// use polars_core::prelude::*;
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

impl ListChunked {
    pub(crate) fn get_inner_dtype(&self) -> &ArrowDataType {
        match self.dtype() {
            DataType::List(dt) => dt,
            _ => panic!("should not happen"),
        }
    }
}

impl<T> Clone for ChunkedArray<T> {
    fn clone(&self) -> Self {
        ChunkedArray {
            field: self.field.clone(),
            chunks: self.chunks.clone(),
            chunk_id: self.chunk_id.clone(),
            phantom: PhantomData,
            categorical_map: self.categorical_map.clone(),
        }
    }
}

pub trait Downcast<T> {
    fn downcast_chunks(&self) -> Vec<&T>;
}

impl<T> Downcast<PrimitiveArray<T>> for ChunkedArray<T>
where
    T: PolarsPrimitiveType,
{
    fn downcast_chunks(&self) -> Vec<&PrimitiveArray<T>> {
        self.chunks
            .iter()
            .map(|arr| {
                let arr = &**arr;
                unsafe { &*(arr as *const dyn Array as *const PrimitiveArray<T>) }
            })
            .collect::<Vec<_>>()
    }
}

impl Downcast<BooleanArray> for BooleanChunked {
    fn downcast_chunks(&self) -> Vec<&BooleanArray> {
        self.chunks
            .iter()
            .map(|arr| {
                let arr = &**arr;
                unsafe { &*(arr as *const dyn Array as *const BooleanArray) }
            })
            .collect::<Vec<_>>()
    }
}

impl Downcast<LargeStringArray> for Utf8Chunked {
    fn downcast_chunks(&self) -> Vec<&LargeStringArray> {
        self.chunks
            .iter()
            .map(|arr| {
                let arr = &**arr;
                unsafe { &*(arr as *const dyn Array as *const LargeStringArray) }
            })
            .collect::<Vec<_>>()
    }
}

impl Downcast<LargeListArray> for ListChunked {
    fn downcast_chunks(&self) -> Vec<&LargeListArray> {
        self.chunks
            .iter()
            .map(|arr| {
                let arr = &**arr;
                unsafe { &*(arr as *const dyn Array as *const LargeListArray) }
            })
            .collect::<Vec<_>>()
    }
}

#[cfg(feature = "object")]
impl<T> Downcast<ObjectArray<T>> for ObjectChunked<T>
where
    T: 'static + std::fmt::Debug + Clone + Send + Sync + Default,
{
    fn downcast_chunks(&self) -> Vec<&ObjectArray<T>> {
        self.chunks
            .iter()
            .map(|arr| {
                let arr = &**arr;
                unsafe { &*(arr as *const dyn Array as *const ObjectArray<T>) }
            })
            .collect::<Vec<_>>()
    }
}

impl<T> AsRef<ChunkedArray<T>> for ChunkedArray<T> {
    fn as_ref(&self) -> &ChunkedArray<T> {
        self
    }
}

pub struct NoNull<T>(pub T);

impl Deref for CategoricalChunked {
    type Target = UInt32Chunked;

    fn deref(&self) -> &Self::Target {
        let ptr = self as *const CategoricalChunked;
        let ptr = ptr as *const UInt32Chunked;
        unsafe { &*ptr }
    }
}

impl DerefMut for CategoricalChunked {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let ptr = self as *mut CategoricalChunked;
        let ptr = ptr as *mut UInt32Chunked;
        unsafe { &mut *ptr }
    }
}

impl From<UInt32Chunked> for CategoricalChunked {
    fn from(ca: UInt32Chunked) -> Self {
        ca.cast().unwrap()
    }
}

impl CategoricalChunked {
    fn set_state<T>(mut self, other: &ChunkedArray<T>) -> Self {
        self.categorical_map = other.categorical_map.clone();
        self
    }
}

impl ValueSize for ListChunked {
    fn get_values_size(&self) -> usize {
        self.chunks
            .iter()
            .fold(0usize, |acc, arr| acc + arr.get_values_size())
    }
}

impl ValueSize for Utf8Chunked {
    fn get_values_size(&self) -> usize {
        self.chunks
            .iter()
            .fold(0usize, |acc, arr| acc + arr.get_values_size())
    }
}

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
        let a = Utf8Chunked::new_from_slice("a", &["b", "a", "c"]);
        let a = a.sort(false);
        let b = a.into_iter().collect::<Vec<_>>();
        assert_eq!(b, [Some("a"), Some("b"), Some("c")]);
    }

    #[test]
    fn arithmetic() {
        let s1 = get_chunked_array();
        println!("{:?}", s1.chunks);
        let s2 = &s1;
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
        let new = a.take([0u32, 1].as_ref().as_take_iter(), None);
        assert_eq!(new.len(), 2)
    }

    #[test]
    fn get() {
        let mut a = get_chunked_array();
        assert_eq!(AnyValue::Int32(2), a.get_any_value(1));
        // check if chunks indexes are properly determined
        a.append_array(a.chunks[0].clone()).unwrap();
        assert_eq!(AnyValue::Int32(1), a.get_any_value(3));
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
        let s: Utf8Chunked = [Some("b"), None, Some("z")].iter().collect();
        let sorted = s.sort(false);
        assert_eq!(
            sorted.into_iter().collect::<Vec<_>>(),
            &[None, Some("b"), Some("z")]
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

        let s = Utf8Chunked::new_from_slice("", &["a", "b", "c"]);
        assert_eq!(Vec::from(&s.reverse()), &[Some("c"), Some("b"), Some("a")]);

        let s = Utf8Chunked::new_from_opt_slice("", &[Some("a"), None, Some("c")]);
        assert_eq!(Vec::from(&s.reverse()), &[Some("c"), None, Some("a")]);
    }

    #[test]
    fn test_null_sized_chunks() {
        let mut s = Float64Chunked::new_from_slice("s", &Vec::<f64>::new());
        s.append(&Float64Chunked::new_from_slice("s2", &[1., 2., 3.]));
        dbg!(&s);

        let s = Float64Chunked::new_from_slice("s", &Vec::<f64>::new());
        dbg!(&s.into_iter().next());
    }

    #[test]
    fn test_iter_categorical() {
        let ca =
            Utf8Chunked::new_from_opt_slice("", &[Some("foo"), None, Some("bar"), Some("ham")]);
        let ca = ca.cast::<CategoricalType>().unwrap();
        let v: Vec<_> = ca.into_iter().collect();
        assert_eq!(v, &[Some(0), None, Some(1), Some(2)]);
    }
}
