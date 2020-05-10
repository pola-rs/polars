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
struct Column {
    field: Field,
    // For now settle with dynamic generics until we are more confident about the api
    data_chunks: Vec<ArrayRef>,
}

impl Column {
    fn new<T>(name: &str, v: &[T::Native]) -> Self
    where
        T: ArrowPrimitiveType,
    {
        let mut builder = PrimitiveBuilder::<T>::new(v.len());
        v.into_iter().for_each(|&val| {
            builder.append_value(val);
        });

        let field = Field::new(name, T::get_data_type(), true);

        Column {
            field,
            data_chunks: vec![Arc::new(builder.finish())],
        }
    }

    fn replace_array(self, arr: Vec<ArrayRef>) -> Self {
        Column {
            field: self.field,
            data_chunks: arr,
        }
    }
}

macro_rules! variant_operand {
    ($_self:expr, $rhs:expr, $data_type:ty, $operand:ident, $expect:expr) => {{
        let mut new_chunks = Vec::with_capacity($_self.data_chunks.len());
        $_self
            .data_chunks
            .iter()
            .zip($rhs.data_chunks.iter())
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
        $_self.replace_array(new_chunks)
    }};
}

impl Add for Column {
    type Output = Self;

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

impl Mul for Column {
    type Output = Self;

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

impl Sub for Column {
    type Output = Self;

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

impl Add for &Column {
    type Output = Column;

    fn add(self, rhs: Self) -> Self::Output {
        let lhs = self.clone();
        lhs.add(rhs.clone())
    }
}

impl Mul for &Column {
    type Output = Column;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhs = self.clone();
        lhs.mul(rhs.clone())
    }
}

impl Sub for &Column {
    type Output = Column;

    fn sub(self, rhs: Self) -> Self::Output {
        let lhs = self.clone();
        lhs.sub(rhs.clone())
    }
}

impl Debug for Column {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("{:?}", self.data_chunks))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn arithmetic() {
        let s1 = Column::new::<datatypes::Int32Type>("a", &[1, 2, 3]);
        println!("{:?}", s1.data_chunks);
        let s2 = s1.clone();
        println!("{:?}", &s1 + &s2);
        println!("{:?}", &s1 - &s2);
        println!("{:?}", &s1 * &s2);
    }
}
