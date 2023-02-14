//! The typed heart of every Series column.
use std::iter::Map;
use std::marker::PhantomData;
use std::sync::Arc;

use arrow::array::*;
use arrow::bitmap::Bitmap;
use polars_arrow::prelude::ValueSize;

use crate::prelude::*;

pub mod ops;
#[macro_use]
pub mod arithmetic;
pub mod builder;
pub mod cast;
pub mod comparison;
pub mod float;
pub mod iterator;
pub mod kernels;
#[cfg(feature = "ndarray")]
mod ndarray;

mod bitwise;
#[cfg(feature = "object")]
mod drop;
mod from;
pub(crate) mod list;
pub(crate) mod logical;
#[cfg(feature = "object")]
pub mod object;
#[cfg(feature = "random")]
mod random;
#[cfg(any(
    feature = "temporal",
    feature = "dtype-datetime",
    feature = "dtype-date"
))]
pub mod temporal;
mod trusted_len;
pub mod upstream_traits;

use std::mem;
use std::slice::Iter;

use bitflags::bitflags;
use polars_arrow::prelude::*;

use crate::series::IsSorted;
use crate::utils::{first_non_null, last_non_null, CustomIterTools};

#[cfg(not(feature = "dtype-categorical"))]
pub struct RevMapping {}

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
///     ca.apply_cast_numeric(|v| v.cos() as f64)
/// }
/// ```
///
/// ## Conversion between Series and ChunkedArray's
/// Conversion from a `Series` to a `ChunkedArray` is effortless.
///
/// ```rust
/// # use polars_core::prelude::*;
/// fn to_chunked_array(series: &Series) -> PolarsResult<&Int32Chunked>{
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
/// However, multiple chunks in a `ChunkArray` will slow down many operations that need random access because we have an extra indirection
/// and indexes need to be mapped to the proper chunk. Arithmetic may also be slowed down by this.
/// When multiplying two `ChunkArray'`s with different chunk sizes they cannot utilize [SIMD](https://en.wikipedia.org/wiki/SIMD) for instance.
///
/// If you want to have predictable performance
/// (no unexpected re-allocation of memory), it is advised to call the [ChunkedArray::rechunk] after
/// multiple append operations.
///
/// See also [`ChunkedArray::extend`] for appends within a chunk.
pub struct ChunkedArray<T: PolarsDataType> {
    pub(crate) field: Arc<Field>,
    pub(crate) chunks: Vec<ArrayRef>,
    phantom: PhantomData<T>,
    pub(crate) bit_settings: Settings,
    length: IdxSize,
}

bitflags! {
    #[derive(Default)]
    pub(crate) struct Settings: u8 {
    const SORTED_ASC = 0x01;
    const SORTED_DSC = 0x02;
    const FAST_EXPLODE_LIST = 0x04;
}}

impl<T: PolarsDataType> ChunkedArray<T> {
    pub(crate) fn is_sorted_flag(&self) -> bool {
        self.bit_settings.contains(Settings::SORTED_ASC)
    }

    pub(crate) fn is_sorted_reverse_flag(&self) -> bool {
        self.bit_settings.contains(Settings::SORTED_DSC)
    }

    pub fn is_sorted_flag2(&self) -> IsSorted {
        if self.is_sorted_flag() {
            IsSorted::Ascending
        } else if self.is_sorted_reverse_flag() {
            IsSorted::Descending
        } else {
            IsSorted::Not
        }
    }

    /// Set the 'sorted' bit meta info.
    pub fn set_sorted_flag(&mut self, sorted: IsSorted) {
        match sorted {
            IsSorted::Not => {
                self.bit_settings
                    .remove(Settings::SORTED_ASC | Settings::SORTED_DSC);
            }
            IsSorted::Ascending => {
                // // unset reverse sorted
                self.bit_settings.remove(Settings::SORTED_DSC);
                // set sorted
                self.bit_settings.insert(Settings::SORTED_ASC)
            }
            IsSorted::Descending => {
                // unset sorted
                self.bit_settings.remove(Settings::SORTED_ASC);
                // set reverse sorted
                self.bit_settings.insert(Settings::SORTED_DSC)
            }
        }
    }

    /// Get the index of the first non null value in this ChunkedArray.
    pub fn first_non_null(&self) -> Option<usize> {
        if self.is_empty() {
            None
        } else {
            first_non_null(self.iter_validities())
        }
    }

    /// Get the index of the last non null value in this ChunkedArray.
    pub fn last_non_null(&self) -> Option<usize> {
        last_non_null(self.iter_validities(), self.length as usize)
    }

    /// Get the buffer of bits representing null values
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn iter_validities(&self) -> Map<Iter<'_, ArrayRef>, fn(&ArrayRef) -> Option<&Bitmap>> {
        fn to_validity(arr: &ArrayRef) -> Option<&Bitmap> {
            arr.validity()
        }
        self.chunks.iter().map(to_validity)
    }

    #[inline]
    /// Return if any the chunks in this `[ChunkedArray]` have a validity bitmap.
    /// no bitmap means no null values.
    pub fn has_validity(&self) -> bool {
        self.iter_validities().any(|valid| valid.is_some())
    }

    /// Shrink the capacity of this array to fit its length.
    pub fn shrink_to_fit(&mut self) {
        self.chunks = vec![arrow::compute::concatenate::concatenate(
            self.chunks
                .iter()
                .map(|a| &**a)
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .unwrap()];
    }

    /// Unpack a Series to the same physical type.
    ///
    /// # Safety
    ///
    /// This is unsafe as the dtype may be incorrect and
    /// is assumed to be correct in other safe code.
    pub(crate) unsafe fn unpack_series_matching_physical_type(
        &self,
        series: &Series,
    ) -> &ChunkedArray<T> {
        let series_trait = &**series;
        if self.dtype() == series.dtype() {
            &*(series_trait as *const dyn SeriesTrait as *const ChunkedArray<T>)
        } else {
            use DataType::*;
            match (self.dtype(), series.dtype()) {
                (Int64, Datetime(_, _)) | (Int64, Duration(_)) | (Int32, Date) => {
                    &*(series_trait as *const dyn SeriesTrait as *const ChunkedArray<T>)
                }
                _ => panic!(
                    "cannot unpack series {:?} into matching type {:?}",
                    series,
                    self.dtype()
                ),
            }
        }
    }

    /// Series to ChunkedArray<T>
    pub fn unpack_series_matching_type(&self, series: &Series) -> PolarsResult<&ChunkedArray<T>> {
        if self.dtype() == series.dtype() {
            // Safety
            // dtype will be correct.
            Ok(unsafe { self.unpack_series_matching_physical_type(series) })
        } else {
            Err(PolarsError::SchemaMisMatch(
                format!(
                    "cannot unpack series {:?} into matching type {:?}",
                    series,
                    self.dtype()
                )
                .into(),
            ))
        }
    }

    /// Unique id representing the number of chunks
    pub fn chunk_id(&self) -> ChunkIdIter {
        self.chunks.iter().map(|chunk| chunk.len())
    }

    /// A reference to the chunks
    #[inline]
    pub fn chunks(&self) -> &Vec<ArrayRef> {
        &self.chunks
    }

    /// A mutable reference to the chunks
    ///
    /// # Safety
    /// The caller must ensure to not change the `DataType` or `length` of any of the chunks.
    #[inline]
    pub unsafe fn chunks_mut(&mut self) -> &mut Vec<ArrayRef> {
        &mut self.chunks
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

    /// Create a new ChunkedArray from self, where the chunks are replaced.
    fn copy_with_chunks(&self, chunks: Vec<ArrayRef>, keep_sorted: bool) -> Self {
        let mut out = ChunkedArray {
            field: self.field.clone(),
            chunks,
            phantom: PhantomData,
            bit_settings: self.bit_settings,
            length: 0,
        };
        out.compute_len();
        if !keep_sorted {
            out.set_sorted_flag(IsSorted::Not);
        }
        out
    }

    /// Get a mask of the null values.
    pub fn is_null(&self) -> BooleanChunked {
        if !self.has_validity() {
            return BooleanChunked::full(self.name(), false, self.len());
        }
        let chunks = self
            .chunks
            .iter()
            .map(|arr| {
                let bitmap = arr
                    .validity()
                    .map(|bitmap| !bitmap)
                    .unwrap_or_else(|| Bitmap::new_zeroed(arr.len()));
                Box::new(BooleanArray::from_data_default(bitmap, None)) as ArrayRef
            })
            .collect::<Vec<_>>();
        unsafe { BooleanChunked::from_chunks(self.name(), chunks) }
    }

    /// Get a mask of the valid values.
    pub fn is_not_null(&self) -> BooleanChunked {
        if !self.has_validity() {
            return BooleanChunked::full(self.name(), true, self.len());
        }
        let chunks = self
            .chunks
            .iter()
            .map(|arr| {
                let bitmap = arr
                    .validity()
                    .cloned()
                    .unwrap_or_else(|| !(&Bitmap::new_zeroed(arr.len())));
                Box::new(BooleanArray::from_data_default(bitmap, None)) as ArrayRef
            })
            .collect::<Vec<_>>();
        unsafe { BooleanChunked::from_chunks(self.name(), chunks) }
    }

    pub(crate) fn coalesce_nulls(&self, other: &[ArrayRef]) -> Self {
        assert_eq!(self.chunks.len(), other.len());
        let chunks = self
            .chunks
            .iter()
            .zip(other)
            .map(|(a, b)| {
                assert_eq!(a.len(), b.len());
                let validity = match (a.validity(), b.validity()) {
                    (None, Some(b)) => Some(b.clone()),
                    (Some(a), Some(b)) => Some(a & b),
                    (Some(a), None) => Some(a.clone()),
                    (None, None) => None,
                };

                a.with_validity(validity)
            })
            .collect();
        self.copy_with_chunks(chunks, true)
    }

    /// Get data type of ChunkedArray.
    pub fn dtype(&self) -> &DataType {
        self.field.data_type()
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
                    // safety:
                    // within bounds
                    debug_assert!((offset + len) <= array.len());
                    let out = unsafe { array.sliced_unchecked(offset, len) };
                    offset += len;
                    out
                })
                .collect();

            unsafe { Self::from_chunks(self.name(), chunks) }
        };

        if self.chunks.len() != 1 {
            let out = self.rechunk();
            slice(&out)
        } else {
            slice(self)
        }
    }
}

pub(crate) trait AsSinglePtr {
    /// Rechunk and return a ptr to the start of the array
    fn as_single_ptr(&mut self) -> PolarsResult<usize> {
        Err(PolarsError::InvalidOperation(
            "operation as_single_ptr not supported for this dtype".into(),
        ))
    }
}

impl<T> AsSinglePtr for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn as_single_ptr(&mut self) -> PolarsResult<usize> {
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
#[cfg(feature = "dtype-binary")]
impl AsSinglePtr for BinaryChunked {}
#[cfg(feature = "object")]
impl<T: PolarsObject> AsSinglePtr for ObjectChunked<T> {}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    /// Contiguous slice
    pub fn cont_slice(&self) -> PolarsResult<&[T::Native]> {
        if self.chunks.len() == 1 && self.chunks[0].null_count() == 0 {
            Ok(self.downcast_iter().next().map(|arr| arr.values()).unwrap())
        } else {
            Err(PolarsError::ComputeError("no_slice".into()))
        }
    }
    /// Contiguous mutable slice
    pub(crate) fn cont_slice_mut(&mut self) -> Option<&mut [T::Native]> {
        if self.chunks.len() == 1 && self.chunks[0].null_count() == 0 {
            // Safety, we will not swap the PrimitiveArray.
            let arr = unsafe { self.downcast_iter_mut().next().unwrap() };
            arr.get_mut_values()
        } else {
            None
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
        // we know the iterators len
        unsafe {
            self.data_views()
                .flatten()
                .map(|v| *v)
                .trust_my_length(self.len())
        }
    }
}

impl<T: PolarsDataType> Clone for ChunkedArray<T> {
    fn clone(&self) -> Self {
        ChunkedArray {
            field: self.field.clone(),
            chunks: self.chunks.clone(),
            phantom: PhantomData,
            bit_settings: self.bit_settings,
            length: self.length,
        }
    }
}

impl<T: PolarsDataType> AsRef<ChunkedArray<T>> for ChunkedArray<T> {
    fn as_ref(&self) -> &ChunkedArray<T> {
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

#[cfg(feature = "dtype-binary")]
impl ValueSize for BinaryChunked {
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
            DataType::List(dt) => *dt.clone(),
            _ => unreachable!(),
        }
    }

    pub fn set_inner_dtype(&mut self, dtype: DataType) {
        assert_eq!(dtype.to_physical(), self.inner_dtype().to_physical());
        let field = Arc::make_mut(&mut self.field);
        field.coerce(DataType::List(Box::new(dtype)));
    }
}

pub(crate) fn to_primitive<T: PolarsNumericType>(
    values: Vec<T::Native>,
    validity: Option<Bitmap>,
) -> PrimitiveArray<T::Native> {
    PrimitiveArray::new(T::get_dtype().to_arrow(), values.into(), validity)
}

pub(crate) fn to_array<T: PolarsNumericType>(
    values: Vec<T::Native>,
    validity: Option<Bitmap>,
) -> ArrayRef {
    Box::new(to_primitive::<T>(values, validity))
}

impl<T: PolarsNumericType> From<PrimitiveArray<T::Native>> for ChunkedArray<T> {
    fn from(a: PrimitiveArray<T::Native>) -> Self {
        unsafe { ChunkedArray::from_chunks("", vec![Box::new(a)]) }
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::prelude::*;

    pub(crate) fn get_chunked_array() -> Int32Chunked {
        ChunkedArray::new("a", &[1, 2, 3])
    }

    #[test]
    fn test_sort() {
        let a = Int32Chunked::new("a", &[1, 9, 3, 2]);
        let b = a
            .sort(false)
            .into_iter()
            .map(|opt| opt.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(b, [1, 2, 3, 9]);
        let a = Utf8Chunked::new("a", &["b", "a", "c"]);
        let a = a.sort(false);
        let b = a.into_iter().collect::<Vec<_>>();
        assert_eq!(b, [Some("a"), Some("b"), Some("c")]);
        assert_eq!(a.is_sorted_flag(), true);
    }

    #[test]
    fn arithmetic() {
        let a = &Int32Chunked::new("a", &[1, 100, 6, 40]);
        let b = &Int32Chunked::new("b", &[-1, 2, 3, 4]);

        // Not really asserting anything here but shill making sure the code is exercised
        // This (and more) is properly tested from the integration test suite and Python bindings.
        println!("{:?}", a + b);
        println!("{:?}", a - b);
        println!("{:?}", a * b);
        println!("{:?}", a / b);
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
            .filter(&BooleanChunked::new("filter", &[true, false, false]))
            .unwrap();
        assert_eq!(b.len(), 1);
        assert_eq!(b.into_iter().next(), Some(Some(1)));
    }

    #[test]
    fn aggregates() {
        let a = &Int32Chunked::new("a", &[1, 100, 10, 9]);
        assert_eq!(a.max(), Some(100));
        assert_eq!(a.min(), Some(1));
        assert_eq!(a.sum(), Some(120))
    }

    #[test]
    fn take() {
        let a = get_chunked_array();
        let new = a.take([0usize, 1].iter().copied().into()).unwrap();
        assert_eq!(new.len(), 2)
    }

    #[test]
    fn cast() {
        let a = get_chunked_array();
        let b = a.cast(&DataType::Int64).unwrap();
        assert_eq!(b.dtype(), &ArrowDataType::Int64)
    }

    fn assert_slice_equal<T>(ca: &ChunkedArray<T>, eq: &[T::Native])
    where
        T: PolarsNumericType,
    {
        assert_eq!(
            ca.into_iter().map(|opt| opt.unwrap()).collect::<Vec<_>>(),
            eq
        )
    }

    #[test]
    fn slice() {
        let mut first = UInt32Chunked::new("first", &[0, 1, 2]);
        let second = UInt32Chunked::new("second", &[3, 4, 5]);
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
        let s = UInt32Chunked::new("", &[9, 2, 4]);
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
        let s = UInt32Chunked::new("", &[1, 2, 3]);
        // path with continuous slice
        assert_slice_equal(&s.reverse(), &[3, 2, 1]);
        // path with options
        let s = UInt32Chunked::new("", &[Some(1), None, Some(3)]);
        assert_eq!(Vec::from(&s.reverse()), &[Some(3), None, Some(1)]);
        let s = BooleanChunked::new("", &[true, false]);
        assert_eq!(Vec::from(&s.reverse()), &[Some(false), Some(true)]);

        let s = Utf8Chunked::new("", &["a", "b", "c"]);
        assert_eq!(Vec::from(&s.reverse()), &[Some("c"), Some("b"), Some("a")]);

        let s = Utf8Chunked::new("", &[Some("a"), None, Some("c")]);
        assert_eq!(Vec::from(&s.reverse()), &[Some("c"), None, Some("a")]);
    }

    #[test]
    #[cfg(feature = "dtype-categorical")]
    fn test_iter_categorical() {
        use crate::{reset_string_cache, SINGLE_LOCK};
        let _lock = SINGLE_LOCK.lock();
        reset_string_cache();
        let ca = Utf8Chunked::new("", &[Some("foo"), None, Some("bar"), Some("ham")]);
        let ca = ca.cast(&DataType::Categorical(None)).unwrap();
        let ca = ca.categorical().unwrap();
        let v: Vec<_> = ca.logical().into_iter().collect();
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
