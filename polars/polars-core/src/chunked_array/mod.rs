//! The typed heart of every Series column.
use crate::prelude::*;
use arrow::{array::*, bitmap::Bitmap};
use polars_arrow::prelude::ValueSize;
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

mod bitwise;
#[cfg(feature = "object")]
mod drop;
pub(crate) mod list;
pub(crate) mod logical;
#[cfg(feature = "object")]
#[cfg_attr(docsrs, doc(cfg(feature = "object")))]
pub mod object;
#[cfg(feature = "random")]
#[cfg_attr(docsrs, doc(cfg(feature = "random")))]
mod random;
#[cfg(feature = "strings")]
#[cfg_attr(docsrs, doc(cfg(feature = "strings")))]
pub mod strings;
#[cfg(any(
    feature = "temporal",
    feature = "dtype-datetime",
    feature = "dtype-date"
))]
#[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
pub mod temporal;
mod trusted_len;
pub mod upstream_traits;

use polars_arrow::prelude::*;

#[cfg(feature = "dtype-categorical")]
use crate::chunked_array::categorical::RevMapping;
use crate::utils::CustomIterTools;
use std::mem;

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
/// However, multiple chunks in a `ChunkArray` will slow down many operations that need random access because we have an extra indirection
/// and indexes need to be mapped to the proper chunk. Arithmetic may also be slowed down by this.
/// When multiplying two `ChunkArray'`s with different chunk sizes they cannot utilize [SIMD](https://en.wikipedia.org/wiki/SIMD) for instance.
///
/// If you want to have predictable performance
/// (no unexpected re-allocation of memory), it is advised to call the [rechunk](chunked_array/chunkops/trait.ChunkOps.html) after
/// multiple append operations.
///
/// See also [`ChunkedArray::extend`] for appends within a chunk.
pub struct ChunkedArray<T> {
    pub(crate) field: Arc<Field>,
    pub(crate) chunks: Vec<ArrayRef>,
    phantom: PhantomData<T>,
    /// maps categorical u32 indexes to String values
    pub(crate) categorical_map: Option<Arc<RevMapping>>,
    /// first bit: sorted
    /// second_bit: sorted reverse
    /// third bit dtype list: fast_explode
    ///     - unset: unknown or not all arrays have at least one value
    ///     - set: all list arrays are filled (this allows for cheap explode)
    pub(crate) bit_settings: u8,
}

impl<T> ChunkedArray<T> {
    pub(crate) fn is_sorted(&self) -> bool {
        self.bit_settings & 1 != 0
    }

    pub(crate) fn is_sorted_reverse(&self) -> bool {
        self.bit_settings & 1 << 1 != 0
    }

    /// Set the 'sorted' bit meta info.
    pub(crate) fn set_sorted(&mut self, reverse: bool) {
        if reverse {
            self.bit_settings |= 1 << 1
        } else {
            self.bit_settings |= 1
        }
    }

    /// Get the index of the first non null value in this ChunkedArray.
    pub fn first_non_null(&self) -> Option<usize> {
        let mut offset = 0;
        for validity in self.iter_validities() {
            if let Some(validity) = validity {
                for (idx, is_valid) in validity.iter().enumerate() {
                    if is_valid {
                        return Some(offset + idx);
                    }
                }
                offset += validity.len()
            } else {
                return Some(offset);
            }
        }
        None
    }

    /// Get the buffer of bits representing null values
    #[inline]
    pub fn iter_validities(&self) -> impl Iterator<Item = Option<&Bitmap>> + '_ {
        self.chunks.iter().map(|arr| arr.validity())
    }

    #[inline]
    /// Return if any the chunks in this `[ChunkedArray]` have a validity bitmap.
    /// no bitmap means no null values.
    pub fn has_validity(&self) -> bool {
        self.iter_validities().any(|valid| valid.is_some())
    }

    /// Shrink the capacity of this array to fit it's length.
    pub fn shrink_to_fit(&mut self) {
        self.chunks = vec![arrow::compute::concatenate::concatenate(
            self.chunks
                .iter()
                .map(|a| &**a)
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .unwrap()
        .into()];
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
    pub fn unpack_series_matching_type(&self, series: &Series) -> Result<&ChunkedArray<T>> {
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
    pub fn chunks(&self) -> &Vec<ArrayRef> {
        &self.chunks
    }

    /// Returns true if contains a single chunk and has no null values
    pub fn is_optimal_aligned(&self) -> bool {
        self.chunks.len() == 1 && !self.has_validity()
    }

    /// Count the null values.
    #[inline]
    pub fn null_count(&self) -> usize {
        self.chunks.iter().map(|arr| arr.null_count()).sum()
    }

    /// Append arrow array in place.
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut array = Int32Chunked::new("array", &[1, 2]);
    /// let array_2 = Int32Chunked::new("2nd", &[3]);
    ///
    /// array.append(&array_2);
    /// assert_eq!(Vec::from(&array), [Some(1), Some(2), Some(3)])
    /// ```
    pub fn append_array(&mut self, other: ArrayRef) -> Result<()> {
        if self.field.data_type() == other.data_type() {
            self.chunks.push(other);
            Ok(())
        } else {
            Err(PolarsError::SchemaMisMatch(
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
            bit_settings: self.bit_settings,
        }
    }

    /// Get a mask of the null values.
    pub fn is_null(&self) -> BooleanChunked {
        if !self.has_validity() {
            return BooleanChunked::full("is_null", false, self.len());
        }
        let chunks = self
            .chunks
            .iter()
            .map(|arr| {
                let bitmap = arr
                    .validity()
                    .map(|bitmap| !bitmap)
                    .unwrap_or_else(|| Bitmap::new_zeroed(arr.len()));
                Arc::new(BooleanArray::from_data_default(bitmap, None)) as ArrayRef
            })
            .collect::<Vec<_>>();
        BooleanChunked::from_chunks(self.name(), chunks)
    }

    /// Get a mask of the valid values.
    pub fn is_not_null(&self) -> BooleanChunked {
        if !self.has_validity() {
            return BooleanChunked::full("is_not_null", true, self.len());
        }
        let chunks = self
            .chunks
            .iter()
            .map(|arr| {
                let bitmap = arr
                    .validity()
                    .cloned()
                    .unwrap_or_else(|| !(&Bitmap::new_zeroed(arr.len())));
                Arc::new(BooleanArray::from_data_default(bitmap, None)) as ArrayRef
            })
            .collect::<Vec<_>>();
        BooleanChunked::from_chunks(self.name(), chunks)
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

            Self::from_chunks(self.name(), chunks)
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
    pub fn from_chunks(name: &str, chunks: Vec<ArrayRef>) -> Self {
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
}

// A hack to save compiler bloat for null arrays
impl Int32Chunked {
    pub(crate) fn new_null(name: &str, len: usize) -> Self {
        let arr = Arc::from(arrow::array::new_null_array(ArrowDataType::Null, len));
        let field = Arc::new(Field::new(name, DataType::Null));
        let chunks = vec![arr as ArrayRef];
        ChunkedArray {
            field,
            chunks,
            phantom: PhantomData,
            categorical_map: None,
            bit_settings: 0,
        }
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    /// Create a new ChunkedArray by taking ownership of the Vec. This operation is zero copy.
    pub fn from_vec(name: &str, v: Vec<T::Native>) -> Self {
        let arr = to_array::<T>(v, None);
        Self::from_chunks(name, vec![arr])
    }

    /// Nullify values in slice with an existing null bitmap
    pub fn new_from_owned_with_null_bitmap(
        name: &str,
        values: Vec<T::Native>,
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
            Err(PolarsError::ComputeError("cannot take slice".into()))
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

impl<T> Clone for ChunkedArray<T> {
    fn clone(&self) -> Self {
        ChunkedArray {
            field: self.field.clone(),
            chunks: self.chunks.clone(),
            phantom: PhantomData,
            categorical_map: self.categorical_map.clone(),
            bit_settings: self.bit_settings,
        }
    }
}

impl<T> AsRef<ChunkedArray<T>> for ChunkedArray<T> {
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

impl ListChunked {
    /// Get the inner data type of the list.
    pub fn inner_dtype(&self) -> DataType {
        match self.dtype() {
            DataType::List(dt) => *dt.clone(),
            _ => unreachable!(),
        }
    }

    pub(crate) fn with_inner_type(&mut self, dtype: DataType) {
        assert_eq!(dtype.to_physical(), self.inner_dtype());
        let field = Arc::make_mut(&mut self.field);
        field.coerce(DataType::List(Box::new(dtype)));
    }
}

pub(crate) fn to_primitive<T: PolarsNumericType>(
    values: Vec<T::Native>,
    validity: Option<Bitmap>,
) -> PrimitiveArray<T::Native> {
    PrimitiveArray::from_data(T::get_dtype().to_arrow(), values.into(), validity)
}

pub(crate) fn to_array<T: PolarsNumericType>(
    values: Vec<T::Native>,
    validity: Option<Bitmap>,
) -> ArrayRef {
    Arc::new(to_primitive::<T>(values, validity))
}

impl<T: PolarsNumericType> From<PrimitiveArray<T::Native>> for ChunkedArray<T> {
    fn from(a: PrimitiveArray<T::Native>) -> Self {
        ChunkedArray::from_chunks("", vec![Arc::new(a)])
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
            .filter(&BooleanChunked::new("filter", &[true, false, false]))
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
        let b = a.cast(&DataType::Int64).unwrap();
        assert_eq!(b.dtype(), &ArrowDataType::Int64)
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
    fn test_null_sized_chunks() {
        let mut s = Float64Chunked::new("s", &Vec::<f64>::new());
        s.append(&Float64Chunked::new("s2", &[1., 2., 3.]));
        dbg!(&s);

        let s = Float64Chunked::new("s", &Vec::<f64>::new());
        dbg!(&s.into_iter().next());
    }

    #[test]
    #[cfg(feature = "dtype-categorical")]
    fn test_iter_categorical() {
        use crate::reset_string_cache;
        use crate::SINGLE_LOCK;
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
