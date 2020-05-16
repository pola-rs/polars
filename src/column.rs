use crate::error::PolarsError;
use crate::error::Result;
use arrow::array::{Array, ArrayRef, BooleanArray};
use arrow::compute::TakeOptions;
use arrow::datatypes::DataType;
use arrow::error::ArrowError;
use arrow::ipc::SparseMatrixCompressedAxis::Column;
use arrow::{
    array,
    array::{PrimitiveArray, PrimitiveArrayOps, PrimitiveBuilder},
    compute, datatypes,
    datatypes::{ArrowNumericType, ArrowPrimitiveType, Field},
};
use num::Zero;
use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;
use std::sync::Arc;

fn create_chunk_id(chunks: &Vec<ArrayRef>) -> String {
    let mut chunk_id = String::new();
    for a in chunks {
        chunk_id.push_str(&format!("{}-", a.len()))
    }
    chunk_id
}

macro_rules! apply_operator {
    ($lhs:ident, $rhs:ident, $operator:expr) => {
        $lhs.downcast_chunks()
            .iter()
            .zip($rhs.downcast_chunks())
            .map(|(left, right)| $operator(left, right))
            .collect::<std::result::Result<Vec<_>, arrow::error::ArrowError>>()
    };
}

struct ChunkedArray<T> {
    field: Field,
    // For now settle with dynamic generics until we are more confident about the api
    chunks: Vec<ArrayRef>,
    /// len_chunk0-len_chunk1-len_chunk2 etc.
    chunk_id: String,
    phantom: PhantomData<T>,
}

impl<T> ChunkedArray<T>
where
    T: ArrowPrimitiveType,
{
    fn new(name: &str, v: &[T::Native]) -> Self {
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

    fn new_from_chunks(name: &str, chunks: Vec<ArrayRef>) -> Self {
        let field = Field::new(name, T::get_data_type(), true);
        let chunk_id = create_chunk_id(&chunks);
        ChunkedArray {
            field,
            chunks,
            chunk_id,
            phantom: PhantomData,
        }
    }

    fn iter(&self) -> ChunkIter<T> {
        let arrays = self
            .chunks
            .iter()
            .map(|a| {
                a.as_any()
                    .downcast_ref::<PrimitiveArray<T>>()
                    .expect("could not downcast")
            })
            .collect::<Vec<_>>();

        ChunkIter {
            arrays,
            chunk_i: 0,
            array_i: 0,
            out_of_bounds: false,
        }
    }
    fn rechunk(&mut self) {
        let mut builder = PrimitiveBuilder::<T>::new(self.len());
        self.iter().for_each(|val| {
            builder.append_option(val).expect("Could not append value");
        });
        self.chunks = vec![Arc::new(builder.finish())];
        self.set_chunk_id()
    }

    fn optional_rechunk<A>(&self, rhs: &ChunkedArray<A>) -> Result<Option<Self>> {
        if self.chunk_id != rhs.chunk_id {
            // we can rechunk ourselves to match
            if rhs.chunks.len() == 1 {
                let mut new = self.clone();
                new.rechunk();
                Ok(Some(new))
            } else {
                Err(PolarsError::ChunkMismatch)
            }
        } else {
            Ok(None)
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

    /// Chunk sizes should match or rhs should have one chunk
    fn filter(&self, filter: &ChunkedArray<datatypes::BooleanType>) -> Result<Self> {
        let opt = self.optional_rechunk(filter)?;
        let left = match &opt {
            Some(a) => a,
            None => self,
        };
        let chunks = left
            .downcast_chunks()
            .iter()
            .zip(&filter.downcast_chunks())
            .map(|(&arr, &fil)| compute::filter(arr, fil))
            .collect::<std::result::Result<Vec<_>, arrow::error::ArrowError>>();

        match chunks {
            Ok(chunks) => Ok(self.copy_with_array(chunks)),
            Err(e) => Err(PolarsError::ArrowError(e)),
        }
    }

    fn limit(&self, num_elements: usize) -> Result<Self> {
        if num_elements >= self.len() {
            Ok(self.copy_with_array(self.chunks.clone()))
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
            Ok(self.copy_with_array(new_chunks))
        }
    }

    fn sum(&self) -> Option<T::Native>
    where
        T: ArrowNumericType,
        T::Native: std::ops::Add<Output = T::Native>,
    {
        self.downcast_chunks()
            .iter()
            .map(|&a| compute::sum(a))
            .fold(None, |acc, v| match v {
                Some(v) => match acc {
                    None => Some(v),
                    Some(acc) => Some(acc + v),
                },
                None => acc,
            })
    }
}

impl<T> ChunkedArray<T>
where
    T: ArrowNumericType,
{
    fn comparison(
        &self,
        rhs: &ChunkedArray<T>,
        operator: impl Fn(&PrimitiveArray<T>, &PrimitiveArray<T>) -> arrow::error::Result<BooleanArray>,
    ) -> Result<ChunkedArray<datatypes::BooleanType>> {
        let opt = self.optional_rechunk(rhs)?;
        let left = match &opt {
            Some(a) => a,
            None => self,
        };

        let chunks_res = left
            .downcast_chunks()
            .iter()
            .zip(rhs.downcast_chunks())
            .map(|(left, right)| operator(left, right))
            .collect::<std::result::Result<Vec<_>, arrow::error::ArrowError>>();

        let chunks_res = chunks_res.map(|chunks| {
            chunks
                .into_iter()
                .map(|arr| Arc::new(arr) as ArrayRef)
                .collect()
        });

        match chunks_res {
            Ok(chunks) => Ok(ChunkedArray::new_from_chunks("", chunks)),
            Err(e) => Err(PolarsError::ArrowError(e)),
        }
    }

    fn eq(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>> {
        self.comparison(rhs, compute::eq)
    }

    fn neq(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>> {
        self.comparison(rhs, compute::neq)
    }

    fn gt(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>> {
        self.comparison(rhs, compute::gt)
    }

    fn gt_eq(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>> {
        self.comparison(rhs, compute::gt_eq)
    }

    fn lt(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>> {
        self.comparison(rhs, compute::lt)
    }

    fn lt_eq(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>> {
        self.comparison(rhs, compute::lt_eq)
    }
}

impl<T> ChunkedArray<T>
where
    T: ArrowNumericType,
    T::Native: std::cmp::Ord,
{
    fn max(&self) -> Option<T::Native> {
        self.downcast_chunks()
            .iter()
            .filter_map(|&a| compute::max(a))
            .max()
    }

    fn min(&self) -> Option<T::Native> {
        self.downcast_chunks()
            .iter()
            .filter_map(|&a| compute::min(a))
            .min()
    }
}

impl<T> ChunkedArray<T> {
    fn len(&self) -> usize {
        self.chunks.iter().fold(0, |acc, arr| acc + arr.len())
    }
    fn copy_with_array(&self, chunks: Vec<ArrayRef>) -> Self {
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
            Ok(chunks) => Ok(self.copy_with_array(chunks.clone())),
            Err(e) => Err(PolarsError::ArrowError(e)),
        }
    }
}

struct ChunkIter<'a, T>
where
    T: ArrowPrimitiveType,
{
    arrays: Vec<&'a PrimitiveArray<T>>,
    chunk_i: usize,
    array_i: usize,
    out_of_bounds: bool,
}

impl<T> ChunkIter<'_, T>
where
    T: ArrowPrimitiveType,
{
    fn set_indexes(&mut self, arr: &PrimitiveArray<T>) {
        self.array_i += 1;
        if self.array_i >= arr.len() {
            // go to next array in the chunks
            self.array_i = 0;
            self.chunk_i += 1;
        }
        if self.chunk_i >= self.arrays.len() {
            self.out_of_bounds = true;
        }
    }
}

impl<T> Iterator for ChunkIter<'_, T>
where
    T: ArrowPrimitiveType,
{
    // nullable, therefore an option
    type Item = Option<T::Native>;

    /// Because arrow types are nullable an option is returned. This is wrapped in another option
    /// to indicate if the iterator returns Some or None.
    fn next(&mut self) -> Option<Self::Item> {
        if self.out_of_bounds {
            return None;
        }

        let arr = unsafe { self.arrays.get_unchecked(self.chunk_i) };
        let data = arr.data();
        let ret;
        if data.is_null(self.array_i) {
            ret = Some(None)
        } else {
            let v = arr.value(self.array_i);
            ret = Some(Some(v))
        }
        self.set_indexes(arr);
        ret
    }
}

macro_rules! operand_on_primitive_arr {
    ($_self:expr, $rhs:tt, $operator:expr, $expect:expr) => {{
        let mut new_chunks = Vec::with_capacity($_self.chunks.len());
        $_self
            .downcast_chunks()
            .iter()
            .zip($rhs.downcast_chunks())
            .for_each(|(left, right)| {
                let res = Arc::new($operator(left, right).expect($expect)) as ArrayRef;
                new_chunks.push(res);
            });
        $_self.copy_with_array(new_chunks)
    }};
}

impl<T> Add for &ChunkedArray<T>
where
    T: ArrowNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = ChunkedArray<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let expect_str = "Could not add, check data types and length";
        operand_on_primitive_arr![self, rhs, compute::add, expect_str]
    }
}

impl<T> Mul for &ChunkedArray<T>
where
    T: ArrowNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = ChunkedArray<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let expect_str = "Could not multiply, check data types and length";
        operand_on_primitive_arr!(self, rhs, compute::multiply, expect_str)
    }
}

impl<T> Sub for &ChunkedArray<T>
where
    T: ArrowNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = ChunkedArray<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let expect_str = "Could not subtract, check data types and length";
        operand_on_primitive_arr![self, rhs, compute::subtract, expect_str]
    }
}

impl<T> Add for ChunkedArray<T>
where
    T: ArrowNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        (&self).add(&rhs)
    }
}

impl<T> Mul for ChunkedArray<T>
where
    T: ArrowNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        (&self).mul(&rhs)
    }
}

impl<T> Sub for ChunkedArray<T>
where
    T: ArrowNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        (&self).sub(&rhs)
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

#[cfg(test)]
mod test {
    use super::*;

    fn get_array() -> ChunkedArray<datatypes::Int32Type> {
        ChunkedArray::<datatypes::Int32Type>::new("a", &[1, 2, 3])
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
        assert_eq!(b.len, 2)
    }

    #[test]
    fn filter() {
        let a = get_array();
        let b = a
            .filter(&ChunkedArray::<datatypes::BooleanType>::new(
                "filter",
                &[true, false, false],
            ))
            .unwrap();
        assert_eq!(b.len, 1);
        assert_eq!(b.iter().next(), Some(Some(1)));
    }

    #[test]
    fn aggregates() {
        let a = get_array();
        assert_eq!(a.max(), Some(3));
        assert_eq!(a.min(), Some(1));
        assert_eq!(a.sum(), Some(6))
    }

    #[test]
    fn take() {
        let a = get_array();
        let new = a.take(&ChunkedArray::new("idx", &[0, 1]), None).unwrap();
        assert_eq!(new.len, 2)
    }
}
