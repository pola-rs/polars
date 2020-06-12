use self::aggregate::Agg;
use crate::datatypes::{AnyType, ArrowDataType, BooleanChunked, UInt32Chunked};
use crate::{
    datatypes,
    error::{PolarsError, Result},
};
use arrow::array::{
    ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
    StringBuilder,
};
use arrow::compute::TakeOptions;
use arrow::{
    array::{PrimitiveArray, PrimitiveBuilder},
    compute,
    datatypes::{ArrowNumericType, ArrowPrimitiveType, Field},
};
use iterator::ChunkIterator;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;

pub mod aggregate;
mod arithmetic;
pub mod comparison;
pub mod iterator;

/// Operations that are possible without knowing underlying type.
/// These operations will not fail due to non matching types.
pub trait SeriesOps {
    fn limit(&self, num_elements: usize) -> Result<Self>
    where
        Self: std::marker::Sized;
    fn filter(&self, filter: &BooleanChunked) -> Result<Self>
    where
        Self: std::marker::Sized;
    fn take(&self, indices: &UInt32Chunked, options: Option<TakeOptions>) -> Result<Self>
    where
        Self: std::marker::Sized;
    fn append_array(&mut self, other: ArrayRef) -> Result<()>;

    fn len(&self) -> usize;

    fn get(&self, index: usize) -> AnyType;
}

fn create_chunk_id(chunks: &Vec<ArrayRef>) -> String {
    let mut chunk_id = String::new();
    for a in chunks {
        chunk_id.push_str(&format!("{}-", a.len()))
    }
    chunk_id
}

pub struct ChunkedArray<T> {
    pub(crate) field: Field,
    // For now settle with dynamic generics until we are more confident about the api
    pub(crate) chunks: Vec<ArrayRef>,
    /// len_chunk0-len_chunk1-len_chunk2 etc.
    chunk_id: String,
    phantom: PhantomData<T>,
}

impl<T> ChunkedArray<T> {
    pub(crate) fn index_to_chunked_index(&self, index: usize) -> (usize, usize) {
        let mut index_remainder = index;
        let mut current_chunk_idx = 0;

        for chunk in &self.chunks {
            if chunk.len() - 1 > index_remainder {
                break;
            } else {
                index_remainder -= chunk.len();
                current_chunk_idx += 1;
            }
        }
        (current_chunk_idx, index_remainder)
    }
}

impl<T> SeriesOps for ChunkedArray<T>
where
    T: datatypes::PolarsDataType,
    ChunkedArray<T>: ChunkOps,
{
    fn limit(&self, num_elements: usize) -> Result<Self> {
        if num_elements >= self.len() {
            Ok(self.copy_with_chunks(self.chunks.clone()))
        } else {
            let mut new_chunks = Vec::with_capacity(self.chunks.len());
            let mut remaining_elements = num_elements as i64;

            let mut c = 0;
            while remaining_elements > 0 {
                let chunk = &self.chunks[c];
                new_chunks.push(compute::limit(chunk, remaining_elements as usize)?);
                remaining_elements -= chunk.len() as i64;
                c += 1;
            }
            Ok(self.copy_with_chunks(new_chunks))
        }
    }

    /// Chunk sizes should match or rhs should have one chunk
    fn filter(&self, filter: &BooleanChunked) -> Result<Self> {
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

    fn take(
        &self,
        indices: &ChunkedArray<datatypes::UInt32Type>,
        options: Option<TakeOptions>,
    ) -> Result<Self> {
        let taken = self
            .chunks
            .iter()
            .zip(indices.downcast_chunks())
            .map(|(arr, idx)| compute::take(&arr, idx, options.clone()))
            .collect::<std::result::Result<Vec<_>, arrow::error::ArrowError>>();

        match taken {
            Ok(chunks) => Ok(self.copy_with_chunks(chunks.clone())),
            Err(e) => Err(PolarsError::ArrowError(e)),
        }
    }

    fn append_array(&mut self, other: ArrayRef) -> Result<()> {
        if other.data_type() == self.field.data_type() {
            self.chunks.push(other);
            Ok(())
        } else {
            Err(PolarsError::DataTypeMisMatch)
        }
    }

    fn len(&self) -> usize {
        self.chunks.iter().fold(0, |acc, arr| acc + arr.len())
    }

    fn get(&self, index: usize) -> AnyType {
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
            ArrowDataType::Int32 => downcast_and_pack!(Int32Array, I32),
            ArrowDataType::Int64 => downcast_and_pack!(Int64Array, I64),
            ArrowDataType::Float32 => downcast_and_pack!(Float32Array, F32),
            ArrowDataType::Float64 => downcast_and_pack!(Float64Array, F64),
            ArrowDataType::Utf8 => downcast_and_pack!(StringArray, Str),
            _ => unimplemented!(),
        }
    }
}

impl ChunkedArray<datatypes::Utf8Type> {
    pub fn new_utf8_from_slice<S: AsRef<str>>(name: &str, v: &[S]) -> Self {
        let mut builder = StringBuilder::new(v.len());
        v.into_iter().for_each(|val| {
            builder
                .append_value(val.as_ref())
                .expect("Could not append value");
        });

        let field = Field::new(name, ArrowDataType::Utf8, true);

        ChunkedArray {
            field,
            chunks: vec![Arc::new(builder.finish())],
            chunk_id: format!("{}-", v.len()).to_string(),
            phantom: PhantomData,
        }
    }
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

impl<T> ChunkedArray<T>
where
    T: datatypes::PolarsDataType,
    ChunkedArray<T>: ChunkOps,
{
    pub fn name(&self) -> &str {
        self.field.name()
    }

    /// used by Series macro
    pub fn ref_field(&self) -> &Field {
        &self.field
    }

    pub fn new_from_chunks(name: &str, chunks: Vec<ArrayRef>) -> Self {
        let field = Field::new(name, T::get_data_type(), true);
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
    T: ArrowPrimitiveType,
{
    pub fn new_from_slice(name: &str, v: &[T::Native]) -> Self {
        let mut builder = PrimitiveBuilder::<T>::new(v.len());
        v.into_iter().for_each(|&val| {
            builder.append_value(val).expect("Could not append value");
        });

        let field = Field::new(name, T::get_data_type(), true);

        ChunkedArray {
            field,
            chunks: vec![Arc::new(builder.finish())],
            chunk_id: format!("{}-", v.len()).to_string(),
            phantom: PhantomData,
        }
    }

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

impl<T> ChunkedArray<T> {
    fn copy_with_chunks(&self, chunks: Vec<ArrayRef>) -> Self {
        let chunk_id = create_chunk_id(&chunks);
        ChunkedArray {
            field: self.field.clone(),
            chunks,
            chunk_id,
            phantom: PhantomData,
        }
    }

    fn set_chunk_id(&mut self) {
        self.chunk_id = create_chunk_id(&self.chunks)
    }

    pub fn cast<N>(&self) -> Result<ChunkedArray<N>>
    where
        N: ArrowPrimitiveType,
    {
        let chunks = self
            .chunks
            .iter()
            .map(|arr| compute::cast(arr, &N::get_data_type()))
            .collect::<arrow::error::Result<Vec<_>>>()?;

        Ok(ChunkedArray::<N>::new_from_chunks(
            self.field.name(),
            chunks,
        ))
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

pub trait ChunkOps {
    fn rechunk(&mut self);
    fn optional_rechunk<A>(&self, rhs: &ChunkedArray<A>) -> Result<Option<Self>>
    where
        Self: std::marker::Sized;
}

macro_rules! optional_rechunk {
    ($self:tt, $rhs:tt) => {
        if $self.chunk_id != $rhs.chunk_id {
            // we can rechunk ourselves to match
            if $rhs.chunks.len() == 1 {
                let mut new = $self.clone();
                new.rechunk();
                Ok(Some(new))
            } else {
                Err(PolarsError::ChunkMisMatch)
            }
        } else {
            Ok(None)
        }
    };
}

impl<T> ChunkOps for ChunkedArray<T>
where
    T: ArrowPrimitiveType,
{
    fn rechunk(&mut self) {
        if self.chunks.len() > 1 {
            let mut builder = PrimitiveBuilder::<T>::new(self.len());
            self.iter().for_each(|val| {
                builder.append_option(val).expect("Could not append value");
            });
            self.chunks = vec![Arc::new(builder.finish())];
            self.set_chunk_id()
        }
    }

    fn optional_rechunk<A>(&self, rhs: &ChunkedArray<A>) -> Result<Option<Self>> {
        optional_rechunk!(self, rhs)
    }
}

impl ChunkOps for ChunkedArray<datatypes::Utf8Type> {
    fn rechunk(&mut self) {
        if self.chunks.len() > 1 {
            let mut builder = StringBuilder::new(self.len());
            self.iter()
                .for_each(|val| builder.append_value(val).expect("Could not append value"));
            self.chunks = vec![Arc::new(builder.finish())];
            self.set_chunk_id()
        }
    }

    fn optional_rechunk<A>(&self, rhs: &ChunkedArray<A>) -> Result<Option<Self>> {
        optional_rechunk!(self, rhs)
    }
}

impl<T> AsRef<ChunkedArray<T>> for ChunkedArray<T> {
    fn as_ref(&self) -> &ChunkedArray<T> {
        self
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::datatypes::Int32Chunked;

    fn get_array() -> Int32Chunked {
        ChunkedArray::new_from_slice("a", &[1, 2, 3])
    }

    #[test]
    fn arithmetic() {
        let s1 = get_array();
        println!("{:?}", s1.chunks);
        let s2 = &s1.clone();
        let s1 = &s1;
        println!("{:?}", s1 + s2);
        println!("{:?}", s1 - s2);
        println!("{:?}", s1 * s2);
    }

    #[test]
    fn iter() {
        let s1 = get_array();
        let mut a = s1.iter();
        s1.iter().for_each(|a| println!("iterator: {:?}", a));
        // sum
        assert_eq!(s1.iter().fold(0, |acc, val| { acc + val.unwrap() }), 6)
    }

    #[test]
    fn limit() {
        let a = get_array();
        let b = a.limit(2).unwrap();
        println!("{:?}", b);
        assert_eq!(b.len(), 2)
    }

    #[test]
    fn filter() {
        let a = get_array();
        let b = a
            .filter(&BooleanChunked::new_from_slice(
                "filter",
                &[true, false, false],
            ))
            .unwrap();
        assert_eq!(b.len(), 1);
        assert_eq!(b.iter().next(), Some(Some(1)));
    }

    #[test]
    fn aggregates_numeric() {
        let a = get_array();
        assert_eq!(a.max(), Some(3));
        assert_eq!(a.min(), Some(1));
        assert_eq!(a.sum(), Some(6))
    }

    #[test]
    fn take() {
        let a = get_array();
        let new = a
            .take(
                &ChunkedArray::<datatypes::UInt32Type>::new_from_slice("idx", &[0, 1]),
                None,
            )
            .unwrap();
        assert_eq!(new.len(), 2)
    }

    #[test]
    fn get() {
        let mut a = get_array();
        assert_eq!(AnyType::I32(2), a.get(1));
        // check if chunks indexes are properly determined
        a.append_array(a.chunks[0].clone());
        assert_eq!(AnyType::I32(1), a.get(3));
    }

    #[test]
    fn cast() {
        let a = get_array();
        let b = a.cast::<datatypes::Int64Type>().unwrap();
        assert_eq!(b.field.data_type(), &datatypes::ArrowDataType::Int64)
    }
}
