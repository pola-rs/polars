//! The typed heart of every Series column.
use crate::chunked_array::builder::{PrimitiveChunkedBuilder, Utf8ChunkedBuilder};
use crate::prelude::*;
use arrow::{
    array::{
        ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, PrimitiveArray,
        PrimitiveBuilder, StringArray, StringBuilder, UInt32Array,
    },
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
mod arithmetic;
pub mod builder;
pub mod cast;
pub(crate) mod chunkops;
pub mod comparison;
pub mod iterator;
pub mod take;
pub mod unique;

/// Get a 'hash' of the chunks in order to compare chunk sizes quickly.
fn create_chunk_id(chunks: &Vec<ArrayRef>) -> Vec<usize> {
    let mut chunk_id = Vec::with_capacity(chunks.len());
    for a in chunks {
        chunk_id.push(a.len())
    }
    chunk_id
}

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
}

impl<T> ChunkedArray<T>
where
    T: PolarsDataType,
    ChunkedArray<T>: ChunkOps,
{
    /// Get the index of the chunk and the index of the value in that chunk
    pub(crate) fn index_to_chunked_index(&self, index: usize) -> (usize, usize) {
        let mut index_remainder = index;
        let mut current_chunk_idx = 0;

        for chunk in &self.chunks {
            if chunk.len() - 1 >= index_remainder {
                break;
            } else {
                index_remainder -= chunk.len();
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

    /// Count the null values.
    pub fn null_count(&self) -> usize {
        self.chunks.iter().map(|arr| arr.null_count()).sum()
    }

    /// Get a mask of the null values.
    pub fn is_null(&self) -> BooleanChunked {
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

    /// Downcast
    pub fn u32(self) -> Result<UInt32Chunked> {
        match T::get_data_type() {
            ArrowDataType::UInt32 => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast
    pub fn i32(self) -> Result<Int32Chunked> {
        match T::get_data_type() {
            ArrowDataType::Int32 => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast
    pub fn i64(self) -> Result<Int64Chunked> {
        match T::get_data_type() {
            ArrowDataType::Int64 => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast
    pub fn f32(self) -> Result<Float32Chunked> {
        match T::get_data_type() {
            ArrowDataType::Float32 => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast
    pub fn f64(self) -> Result<Float64Chunked> {
        match T::get_data_type() {
            ArrowDataType::Float64 => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast
    pub fn bool(self) -> Result<BooleanChunked> {
        match T::get_data_type() {
            ArrowDataType::Boolean => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast
    pub fn utf8(self) -> Result<Utf8Chunked> {
        match T::get_data_type() {
            ArrowDataType::Utf8 => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast
    pub fn date32(self) -> Result<Date32Chunked> {
        match T::get_data_type() {
            ArrowDataType::Date32(DateUnit::Day) => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast
    pub fn date64(self) -> Result<Date64Chunked> {
        match T::get_data_type() {
            ArrowDataType::Date64(DateUnit::Millisecond) => unsafe {
                Ok(std::mem::transmute(self))
            },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast
    pub fn time64ns(self) -> Result<Time64NsChunked> {
        match T::get_data_type() {
            ArrowDataType::Time64(TimeUnit::Nanosecond) => unsafe { Ok(std::mem::transmute(self)) },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Downcast
    pub fn duration_ns(self) -> Result<DurationNsChunked> {
        match T::get_data_type() {
            ArrowDataType::Duration(TimeUnit::Nanosecond) => unsafe {
                Ok(std::mem::transmute(self))
            },
            _ => Err(PolarsError::DataTypeMisMatch),
        }
    }

    /// Take a view of top n elements
    pub fn limit(&self, num_elements: usize) -> Result<Self> {
        self.slice(0, num_elements)
    }

    /// Chunk sizes should match or rhs should have one chunk
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
        match T::get_data_type() {
            ArrowDataType::Boolean => downcast_and_pack!(BooleanArray, Bool),
            ArrowDataType::UInt32 => downcast_and_pack!(UInt32Array, U32),
            ArrowDataType::Int32 => downcast_and_pack!(Int32Array, I32),
            ArrowDataType::Int64 => downcast_and_pack!(Int64Array, I64),
            ArrowDataType::Float32 => downcast_and_pack!(Float32Array, F32),
            ArrowDataType::Float64 => downcast_and_pack!(Float64Array, F64),
            ArrowDataType::Utf8 => downcast_and_pack!(StringArray, Str),
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

    /// Recompute the chunk_id / chunk_lenghts.
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
    pub fn map_null_checks<'a, B, F>(
        &'a self,
        f: F,
    ) -> Map<Box<dyn ExactSizeIterator<Item = Option<T::Native>> + 'a>, F>
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
    fn sort(&self) -> ChunkedArray<T>;

    fn sort_in_place(&mut self);

    fn argsort(&self) -> Vec<usize>;
}

impl<T> ChunkSort<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: std::cmp::PartialOrd,
{
    fn sort(&self) -> ChunkedArray<T> {
        self.into_iter()
            .sorted_by(|a, b| match (a, b) {
                (Some(a), Some(b)) => a.partial_cmp(b).expect("could not compare"),
                (None, Some(_)) => Ordering::Less,
                (Some(_), None) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            })
            .collect()
    }

    fn sort_in_place(&mut self) {
        let sorted = self.sort();
        self.chunks = sorted.chunks;
    }

    fn argsort(&self) -> Vec<usize> {
        self.into_iter()
            .enumerate()
            .sorted_by(|(_idx_a, a), (_idx_b, b)| match (a, b) {
                (Some(a), Some(b)) => a.partial_cmp(b).expect("could not compare"),
                (None, Some(_)) => Ordering::Less,
                (Some(_), None) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            })
            .map(|(idx, _v)| idx)
            .collect()
    }
}

macro_rules! argsort {
    ($self:ident) => {{
        $self
            .into_iter()
            .enumerate()
            .sorted_by(|(_idx_a, a), (_idx_b, b)| a.cmp(b))
            .map(|(idx, _v)| idx)
            .collect()
    }};
}

impl ChunkSort<Utf8Type> for Utf8Chunked {
    fn sort(&self) -> Utf8Chunked {
        self.into_iter().sorted_by(|a, b| a.cmp(b)).collect()
    }

    fn sort_in_place(&mut self) {
        let sorted = self.sort();
        self.chunks = sorted.chunks;
    }

    fn argsort(&self) -> Vec<usize> {
        argsort!(self)
    }
}

impl ChunkSort<BooleanType> for BooleanChunked {
    fn sort(&self) -> BooleanChunked {
        self.into_iter().sorted_by(|a, b| a.cmp(b)).collect()
    }

    fn sort_in_place(&mut self) {
        let sorted = self.sort();
        self.chunks = sorted.chunks;
    }

    fn argsort(&self) -> Vec<usize> {
        argsort!(self)
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
            .sort()
            .into_iter()
            .map(|opt| opt.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(b, [1, 2, 3, 9]);
        let a = Utf8Chunked::new_utf8_from_slice("a", &["b", "a", "c"]);
        let a = a.sort();
        let b = a.into_iter().collect::<Vec<_>>();
        assert_eq!(b, ["a", "b", "c"]);
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
        assert_eq!(AnyType::I32(2), a.get(1));
        // check if chunks indexes are properly determined
        a.append_array(a.chunks[0].clone()).unwrap();
        assert_eq!(AnyType::I32(1), a.get(3));
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
}
