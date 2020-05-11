use crate::error::PolarsError;
use crate::error::Result;
use arrow::array::{Array, ArrayRef};
use arrow::datatypes::DataType;
use arrow::{
    array,
    array::{PrimitiveArray, PrimitiveBuilder},
    compute, datatypes,
    datatypes::{ArrowNumericType, ArrowPrimitiveType, Field, Int8Type},
};
use num::Zero;
use std::fmt::{Debug, Formatter};
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

#[derive(Clone)]
struct ChunkedArray {
    field: Field,
    // For now settle with dynamic generics until we are more confident about the api
    chunks: Vec<ArrayRef>,
    /// sum of all chunk lengths
    len: usize,
    /// sum of all chunk nulls
    null_counts: usize,
}

impl ChunkedArray {
    fn new<T>(name: &str, v: &[T::Native]) -> Self
    where
        T: ArrowPrimitiveType,
    {
        let mut builder = PrimitiveBuilder::<T>::new(v.len());
        v.into_iter().for_each(|&val| {
            builder.append_value(val).expect("Could not append value");
        });

        let field = Field::new(name, T::get_data_type(), true);

        ChunkedArray {
            field,
            chunks: vec![Arc::new(builder.finish())],
            len: v.len(),
            null_counts: 0,
        }
    }

    fn copy_with_array(&self, arr: Vec<ArrayRef>) -> Self {
        ChunkedArray {
            field: self.field.clone(),
            chunks: arr,
            len: self.len,
            null_counts: self.null_counts,
        }
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

impl Add for &ChunkedArray {
    type Output = ChunkedArray;

    fn add(self, rhs: Self) -> Self::Output {
        let expect_str = "Could not add, check data types and length";
        let result = match self.field.data_type() {
            DataType::Int32 => variant_operand![self, rhs, datatypes::Int32Type, add, expect_str],
            DataType::Float32 => {
                variant_operand![self, rhs, datatypes::Float32Type, add, expect_str]
            }
            _ => return unimplemented!(),
        };
        result
    }
}

impl Mul for &ChunkedArray {
    type Output = ChunkedArray;

    fn mul(self, rhs: Self) -> Self::Output {
        let expect_str = "Could not multiply, check data types and length";
        let result = match self.field.data_type() {
            DataType::Int32 => {
                variant_operand![self, rhs, datatypes::Int32Type, multiply, expect_str]
            }
            DataType::Float32 => {
                variant_operand![self, rhs, datatypes::Float32Type, multiply, expect_str]
            }
            _ => return unimplemented!(),
        };
        result
    }
}

impl Sub for &ChunkedArray {
    type Output = ChunkedArray;

    fn sub(self, rhs: Self) -> Self::Output {
        let expect_str = "Could not subtract, check data types and length";
        let result = match self.field.data_type() {
            DataType::Int32 => {
                variant_operand![self, rhs, datatypes::Int32Type, subtract, expect_str]
            }
            DataType::Float32 => {
                variant_operand![self, rhs, datatypes::Float32Type, subtract, expect_str]
            }
            _ => return unimplemented!(),
        };
        result
    }
}

impl Add for ChunkedArray {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        (&self).add(&rhs)
    }
}

impl Mul for ChunkedArray {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        (&self).mul(&rhs)
    }
}

impl Sub for ChunkedArray {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        (&self).sub(&rhs)
    }
}

impl Debug for ChunkedArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("{:?}", self.chunks))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn arithmetic() {
        let s1 = ChunkedArray::new::<datatypes::Int32Type>("a", &[1, 2, 3]);
        println!("{:?}", s1.chunks);
        let s2 = s1.clone();
        println!("{:?}", &s1 + &s2);
        println!("{:?}", &s1 - &s2);
        println!("{:?}", &s1 * &s2);
    }
}
