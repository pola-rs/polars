//! The typed heart of every Series column.
use crate::prelude::*;
use arrow::{array::*, bitmap::Bitmap, datatypes::TimeUnit};
use itertools::Itertools;
use polars_arrow::prelude::ValueSize;
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
mod ndarray;

#[cfg(feature = "object")]
#[cfg_attr(docsrs, doc(cfg(feature = "object")))]
pub mod object;
#[cfg(feature = "random")]
#[cfg_attr(docsrs, doc(cfg(feature = "random")))]
mod random;
#[cfg(feature = "strings")]
#[cfg_attr(docsrs, doc(cfg(feature = "strings")))]
pub mod strings;
#[cfg(feature = "temporal")]
#[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
pub mod temporal;
mod trusted_len;
pub mod upstream_traits;
use arrow::array::Array;

use crate::chunked_array::builder::categorical::RevMapping;
use crate::utils::{slice_offsets, CustomIterTools};
use std::mem;
use std::ops::{Deref, DerefMut};

pub type ChunkIdIter<'a> = std::iter::Map<std::slice::Iter<'a, ArrayRef>, fn(&ArrayRef) -> usize>;

/// # ChunkedArray
///
/// Every Series contains a `ChunkedArray<T>`. Unlike Series, ChunkedArray's are typed. This allows
/// us to apply closures to the data and collect the results to a `ChunkedArray` of the same type `T`.
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
/// giving access to all the excellent methods available for [Iterators](https://doc.rust-lang.org/std/iter/trait.Iterator.html).
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
/// Arrows memory is immutable which makes it possible to make multiple zero copy (sub)-views from a single array.
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
/// (no unexpected re-allocation of memory), it is advised to call the [rechunk](chunked_array/chunkops/trait.ChunkOps.html) after
/// multiple append operations.
pub struct ChunkedArray<T> {
    pub(crate) field: Arc<Field>,
    pub(crate) chunks: Vec<ArrayRef>,
    phantom: PhantomData<T>,
    /// maps categorical u32 indexes to String values
    pub(crate) categorical_map: Option<Arc<RevMapping>>,
    // first bit: sorted
    // second_bit: sorted reverse
    pub(crate) bit_settings: u8,
}

impl<T> ChunkedArray<T> {
    #[cfg(feature = "asof_join")]
    pub(crate) fn is_sorted(&self) -> bool {
        self.bit_settings & 1 != 0
    }

    #[cfg(feature = "asof_join")]
    pub(crate) fn is_sorted_reverse(&self) -> bool {
        self.bit_settings & 1 << 1 != 0
    }

    pub(crate) fn set_sorted(&mut self, reverse: bool) {
        if reverse {
            self.bit_settings |= 1 << 1
        } else {
            self.bit_settings |= 1
        }
    }
    /// Get a reference to the mapping of categorical types to the string values.
    pub fn get_categorical_map(&self) -> Option<&Arc<RevMapping>> {
        self.categorical_map.as_ref()
    }

    /// Get the index of the first non null value in this ChunkedArray.
    pub fn first_non_null(&self) -> Option<usize> {
        let mut offset = 0;
        for (_, null_bitmap) in self.null_bits() {
            if let Some(null_bitmap) = null_bitmap {
                for (idx, is_valid) in null_bitmap.iter().enumerate() {
                    if is_valid {
                        return Some(offset + idx);
                    }
                }
                offset += null_bitmap.len()
            } else {
                return Some(offset);
            }
        }
        None
    }

    /// Get the buffer of bits representing null values
    pub fn null_bits(&self) -> impl Iterator<Item = (usize, &Option<Bitmap>)> + '_ {
        self.chunks
            .iter()
            .map(|arr| (arr.null_count(), arr.validity()))
    }

    /// Shrink the capacity of this array to fit it's length.
    pub fn shrink_to_fit(&mut self) {
        self.chunks = vec![arrow::compute::concat::concatenate(
            self.chunks.iter().map(|a| &**a).collect_vec().as_slice(),
        )
        .unwrap()
        .into()];
    }

    /// Unpack a Series to the same physical type.
    ///
    /// # Safety
    ///
    /// This is unsafe as the dtype may be uncorrect and
    /// is assumed to be correct in other safe code.
    pub(crate) unsafe fn unpack_series_matching_physical_type(
        &self,
        series: &Series,
    ) -> Result<&ChunkedArray<T>> {
        let series_trait = &**series;
        if self.dtype() == series.dtype() {
            let ca = &*(series_trait as *const dyn SeriesTrait as *const ChunkedArray<T>);
            Ok(ca)
        } else {
            use DataType::*;
            match (self.dtype(), series.dtype()) {
                (Int64, Date64) | (Int32, Date32) | (Int64, Duration(_)) | (Int64, Time64(_)) => {
                    let ca = &*(series_trait as *const dyn SeriesTrait as *const ChunkedArray<T>);
                    Ok(ca)
                }
                _ => Err(PolarsError::DataTypeMisMatch(
                    format!(
                        "cannot unpack series {:?} into matching type {:?}",
                        series,
                        self.dtype()
                    )
                    .into(),
                )),
            }
        }
    }

    /// Series to ChunkedArray<T>
    pub fn unpack_series_matching_type(&self, series: &Series) -> Result<&ChunkedArray<T>> {
        if self.dtype() == series.dtype() {
            // Safety
            // dtype will be correct.
            unsafe { self.unpack_series_matching_physical_type(series) }
        } else {
            Err(PolarsError::DataTypeMisMatch(
                format!(
                    "cannot unpack series {:?} into matching type {:?}",
                    series,
                    self.dtype()
                )
                .into(),
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
    pub fn chunk_id(&self) -> ChunkIdIter {
        self.chunks.iter().map(|chunk| chunk.len())
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
    #[inline]
    pub fn null_count(&self) -> usize {
        self.chunks.iter().map(|arr| arr.null_count()).sum()
    }

    /// Take a view of top n elements
    pub fn limit(&self, num_elements: usize) -> Self {
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
        ChunkedArray {
            field: self.field.clone(),
            chunks,
            phantom: PhantomData,
            categorical_map: self.categorical_map.clone(),
            ..Default::default()
        }
    }

    /// Slice the array. The chunks are reallocated the underlying data slices are zero copy.
    ///
    /// When offset is negative it will be counted from the end of the array.
    /// This method will never error,
    /// and will slice the best match when offset, or length is out of bounds
    pub fn slice(&self, offset: i64, length: usize) -> Self {
        let (raw_offset, slice_len) = slice_offsets(offset, length, self.len());

        let mut remaining_length = slice_len;
        let mut remaining_offset = raw_offset;
        let mut new_chunks = vec![];

        for chunk in &self.chunks {
            let chunk_len = chunk.len();
            if remaining_offset > 0 && remaining_offset >= chunk_len {
                remaining_offset -= chunk_len;
                continue;
            }
            let take_len;
            if remaining_length + remaining_offset > chunk_len {
                take_len = chunk_len - remaining_offset;
            } else {
                take_len = remaining_length;
            }

            new_chunks.push(chunk.slice(remaining_offset, take_len).into());
            remaining_length -= take_len;
            remaining_offset = 0;
            if remaining_length == 0 {
                break;
            }
        }
        self.copy_with_chunks(new_chunks)
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
                let bitmap = arr
                    .validity()
                    .as_ref()
                    .map(|bitmap| !bitmap)
                    .unwrap_or_else(|| Bitmap::new_zeroed(arr.len()));
                Arc::new(BooleanArray::from_data(bitmap, None)) as ArrayRef
            })
            .collect_vec();
        BooleanChunked::new_from_chunks("is_null", chunks)
    }

    /// Get a mask of the valid values.
    pub fn is_not_null(&self) -> BooleanChunked {
        if self.null_count() == 0 {
            return BooleanChunked::full("is_not_null", true, self.len());
        }
        let chunks = self
            .chunks
            .iter()
            .map(|arr| {
                let bitmap = arr
                    .validity()
                    .clone()
                    .unwrap_or_else(|| !(&Bitmap::new_zeroed(arr.len())));
                Arc::new(BooleanArray::from_data(bitmap, None)) as ArrayRef
            })
            .collect_vec();
        BooleanChunked::new_from_chunks("is_not_null", chunks)
    }

    /// Get data type of ChunkedArray.
    pub fn dtype(&self) -> &DataType {
        self.field.data_type()
    }

    /// Get the head of the ChunkedArray
    pub fn head(&self, length: Option<usize>) -> Self {
        match length {
            Some(len) => self.slice(0, std::cmp::min(len, self.len())),
            None => self.slice(0, std::cmp::min(10, self.len())),
        }
    }

    /// Get the tail of the ChunkedArray
    pub fn tail(&self, length: Option<usize>) -> Self {
        let len = match length {
            Some(len) => std::cmp::min(len, self.len()),
            None => std::cmp::min(10, self.len()),
        };
        self.slice(-(len as i64), len)
    }

    /// Append in place.
    pub fn append(&mut self, other: &Self)
    where
        Self: std::marker::Sized,
    {
        if let (Some(rev_map_l), Some(rev_map_r)) = (
            self.categorical_map.as_ref(),
            other.categorical_map.as_ref(),
        ) {
            // first assertion checks if the global string cache is equal,
            // the second checks if we append a slice from this array to self
            if !rev_map_l.same_src(rev_map_r) && !Arc::ptr_eq(rev_map_l, rev_map_r) {
                panic!("Appending categorical data can only be done if they are made under the same global string cache. \
                Consider using a global string cache.")
            }
        }

        // replace an empty array
        if self.chunks.len() == 1 && self.is_empty() {
            self.chunks = other.chunks.clone();
        } else {
            self.chunks.extend_from_slice(&other.chunks);
        }
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
    ChunkedArray<T>: ChunkOps,
{
    /// Should be used to match the chunk_id of another ChunkedArray.
    /// # Panics
    /// It is the callers responsibility to ensure that this ChunkedArray has a single chunk.
    pub(crate) fn match_chunks<I>(&self, chunk_id: I) -> Self
    where
        I: Iterator<Item = usize>,
    {
        debug_assert!(self.chunks.len() == 1);
        // Takes a ChunkedArray containing a single chunk
        let slice = |ca: &Self| {
            let array = &ca.chunks[0];

            let mut offset = 0;
            let chunks = chunk_id
                .map(|len| {
                    let out = array.slice(offset, len).into();
                    offset += len;
                    out
                })
                .collect();

            Self::new_from_chunks(self.name(), chunks)
        };

        if self.chunks.len() != 1 {
            let out = self.rechunk();
            slice(&out)
        } else {
            slice(self)
        }
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
        ChunkedArray {
            field,
            chunks,
            phantom: PhantomData,
            categorical_map: None,
            bit_settings: 0,
        }
    }

    #[inline]
    unsafe fn arr_to_any_value(&self, arr: &dyn Array, idx: usize) -> AnyValue {
        if arr.is_null(idx) {
            return AnyValue::Null;
        }

        macro_rules! downcast_and_pack {
            ($casttype:ident, $variant:ident) => {{
                let arr = &*(arr as *const dyn Array as *const $casttype);
                let v = arr.value(idx);
                AnyValue::$variant(v)
            }};
        }
        macro_rules! downcast {
            ($casttype:ident) => {{
                let arr = &*(arr as *const dyn Array as *const $casttype);
                arr.value_unchecked(idx)
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
            DataType::Date32 => downcast_and_pack!(Int32Array, Date32),
            DataType::Date64 => downcast_and_pack!(Int64Array, Date64),
            DataType::Time64(TimeUnit::Nanosecond) => {
                let v = downcast!(Int64Array);
                AnyValue::Time64(v, TimeUnit::Nanosecond)
            }
            DataType::Duration(TimeUnit::Nanosecond) => {
                let v = downcast!(Int64Array);
                AnyValue::Duration(v, TimeUnit::Nanosecond)
            }
            DataType::Duration(TimeUnit::Millisecond) => {
                let v = downcast!(Int64Array);
                AnyValue::Duration(v, TimeUnit::Millisecond)
            }
            DataType::List(_) => {
                let v: ArrayRef = downcast!(LargeListArray).into();
                let s = Series::try_from(("", v));
                AnyValue::List(s.unwrap())
            }
            DataType::Categorical => {
                let v = downcast!(UInt32Array);
                AnyValue::Utf8(self.categorical_map.as_ref().expect("should be set").get(v))
            }
            #[cfg(feature = "object")]
            DataType::Object(_) => panic!("should not be here"),
            _ => unimplemented!(),
        }
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsPrimitiveType,
{
    /// Create a new ChunkedArray by taking ownership of the AlignedVec. This operation is zero copy.
    pub fn new_from_aligned_vec(name: &str, v: AlignedVec<T::Native>) -> Self {
        let arr = to_array::<T>(v, None);
        Self::new_from_chunks(name, vec![arr])
    }

    /// Nullify values in slice with an existing null bitmap
    pub fn new_from_owned_with_null_bitmap(
        name: &str,
        values: AlignedVec<T::Native>,
        buffer: Option<Bitmap>,
    ) -> Self {
        let arr = to_array::<T>(values, buffer);
        ChunkedArray {
            field: Arc::new(Field::new(name, T::get_dtype())),
            chunks: vec![arr],
            phantom: PhantomData,
            categorical_map: None,
            ..Default::default()
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
        let mut ca = self.rechunk();
        mem::swap(&mut ca, self);
        let a = self.data_views().next().unwrap();
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
            Ok(self.downcast_iter().next().map(|arr| arr.values()).unwrap())
        } else {
            Err(PolarsError::NoSlice)
        }
    }

    /// Get slices of the underlying arrow data.
    /// NOTE: null values should be taken into account by the user of these slices as they are handled
    /// separately
    pub fn data_views(&self) -> impl Iterator<Item = &[T::Native]> + DoubleEndedIterator {
        self.downcast_iter().map(|arr| arr.values().as_slice())
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn into_no_null_iter(
        &self,
    ) -> impl Iterator<Item = T::Native>
           + '_
           + Send
           + Sync
           + ExactSizeIterator
           + DoubleEndedIterator
           + TrustedLen {
        // .copied was significantly slower in benchmark, next call did not inline?
        #[allow(clippy::map_clone)]
        self.data_views()
            .flatten()
            .map(|v| *v)
            .trust_my_length(self.len())
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

impl<T> Clone for ChunkedArray<T> {
    fn clone(&self) -> Self {
        ChunkedArray {
            field: self.field.clone(),
            chunks: self.chunks.clone(),
            phantom: PhantomData,
            categorical_map: self.categorical_map.clone(),
            ..Default::default()
        }
    }
}

impl<T> AsRef<ChunkedArray<T>> for ChunkedArray<T> {
    fn as_ref(&self) -> &ChunkedArray<T> {
        self
    }
}

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

impl ListChunked {
    /// Get the inner data type of the list.
    pub fn inner_dtype(&self) -> DataType {
        match self.dtype() {
            DataType::List(dt) => dt.into(),
            _ => unreachable!(),
        }
    }
}

pub(crate) fn to_primitive<T: PolarsPrimitiveType>(
    values: AlignedVec<T::Native>,
    validity: Option<Bitmap>,
) -> PrimitiveArray<T::Native> {
    PrimitiveArray::from_data(T::get_dtype().to_arrow(), values.into(), validity)
}

pub(crate) fn to_array<T: PolarsPrimitiveType>(
    values: AlignedVec<T::Native>,
    validity: Option<Bitmap>,
) -> ArrayRef {
    Arc::new(to_primitive::<T>(values, validity))
}

impl<T: PolarsNumericType> From<PrimitiveArray<T::Native>> for ChunkedArray<T> {
    fn from(a: PrimitiveArray<T::Native>) -> Self {
        ChunkedArray::new_from_chunks("", vec![Arc::new(a)])
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::prelude::*;
    use crate::reset_string_cache;

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
        let b = a.limit(2);
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
        let new = a.take([0usize, 1].iter().copied().into()).unwrap();
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
        assert_slice_equal(&first.slice(0, 3), &[0, 1, 2]);
        assert_slice_equal(&first.slice(0, 4), &[0, 1, 2, 3]);
        assert_slice_equal(&first.slice(1, 4), &[1, 2, 3, 4]);
        assert_slice_equal(&first.slice(3, 2), &[3, 4]);
        assert_slice_equal(&first.slice(3, 3), &[3, 4, 5]);
        assert_slice_equal(&first.slice(-3, 3), &[3, 4, 5]);
        assert_slice_equal(&first.slice(-6, 6), &[0, 1, 2, 3, 4, 5]);

        assert_eq!(first.slice(-7, 2).len(), 2);
        assert_eq!(first.slice(-3, 4).len(), 3);
        assert_eq!(first.slice(3, 4).len(), 3);
        assert_eq!(first.slice(10, 4).len(), 0);
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
        let s: Utf8Chunked = [Some("b"), None, Some("z")].iter().copied().collect();
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
        use crate::SINGLE_LOCK;
        let _lock = SINGLE_LOCK.lock();
        reset_string_cache();
        let ca =
            Utf8Chunked::new_from_opt_slice("", &[Some("foo"), None, Some("bar"), Some("ham")]);
        let ca = ca.cast::<CategoricalType>().unwrap();
        let v: Vec<_> = ca.into_iter().collect();
        assert_eq!(v, &[Some(0), None, Some(1), Some(2)]);
    }

    #[test]
    #[ignore]
    fn test_shrink_to_fit() {
        let mut builder = Utf8ChunkedBuilder::new("foo", 2048, 100 * 2048);
        builder.append_value("foo");
        let mut arr = builder.finish();
        let before = arr
            .chunks()
            .iter()
            .map(|arr| arrow::compute::aggregate::estimated_bytes_size(arr.as_ref()))
            .sum::<usize>();
        arr.shrink_to_fit();
        let after = arr
            .chunks()
            .iter()
            .map(|arr| arrow::compute::aggregate::estimated_bytes_size(arr.as_ref()))
            .sum::<usize>();
        assert!(before > after);
    }
}
