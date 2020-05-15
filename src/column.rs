use crate::error::PolarsError;
use crate::error::Result;
use arrow::array::{Array, ArrayRef, BooleanArray};
use arrow::datatypes::DataType;
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

struct ChunkedArray<T> {
    field: Field,
    // For now settle with dynamic generics until we are more confident about the api
    chunks: Vec<ArrayRef>,
    /// sum of all chunk lengths
    len: usize,
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
            len: v.len(),
            chunk_id: format!("{}-", v.len()).to_string(),
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
        let mut builder = PrimitiveBuilder::<T>::new(self.len);
        self.iter().for_each(|val| {
            builder.append_option(val).expect("Could not append value");
        });
        self.chunks = vec![Arc::new(builder.finish())];
        self.set_chunk_id()
    }

    fn optional_rechunk<A>(&mut self, rhs: &ChunkedArray<A>) -> Result<()> {
        if self.chunk_id != rhs.chunk_id {
            // we can rechunk ourselves to match
            if rhs.chunks.len() == 1 {
                self.rechunk();
                Ok(())
            } else {
                Err(PolarsError::ChunkMismatch)
            }
        } else {
            Ok(())
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
    fn filter(&mut self, filter: &ChunkedArray<datatypes::BooleanType>) -> Result<Self> {
        self.optional_rechunk(filter)?;

        let chunks = self
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
}

impl<T> ChunkedArray<T> {
    fn copy_with_array(&self, arr: Vec<ArrayRef>) -> Self {
        let len = arr.len();
        let chunk_id = create_chunk_id(&arr);
        ChunkedArray {
            field: self.field.clone(),
            chunks: arr,
            len,
            chunk_id,
            phantom: PhantomData,
        }
    }

    fn set_chunk_id(&mut self) {
        self.chunk_id = create_chunk_id(&self.chunks)
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

macro_rules! variant_operand {
    ($_self:expr, $rhs:tt, $data_type:ty, $operand:ident, $expect:expr) => {{
        let mut new_chunks = Vec::with_capacity($_self.chunks.len());
        $_self
            .chunks
            .iter()
            .zip($rhs.chunks.iter())
            .for_each(|(l, r)| {
                let left_any = l.as_any();
                let right_any = r.as_any();
                let left = left_any
                    .downcast_ref::<PrimitiveArray<$data_type>>()
                    .unwrap();
                let right = right_any
                    .downcast_ref::<PrimitiveArray<$data_type>>()
                    .unwrap();
                let res =
                    Arc::new(arrow::compute::$operand(left, right).expect($expect)) as ArrayRef;
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
        variant_operand![self, rhs, T, add, expect_str]
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
        variant_operand!(self, rhs, T, multiply, expect_str)
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
        variant_operand![self, rhs, T, subtract, expect_str]
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
            len: self.len,
            chunk_id: self.chunk_id.clone(),
            phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn get_array() -> ChunkedArray<datatypes::Int32Type> {
        ChunkedArray::new::<datatypes::Int32Type>("a", &[1, 2, 3])
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
    }
}
