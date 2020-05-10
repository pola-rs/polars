use crate::error::PolarsError;
use crate::{arithmetic::Arithmetic, error::Result};
use arrow::array::{Array, ArrayRef};
use arrow::datatypes::DataType;
use arrow::{
    array,
    array::{PrimitiveArray, PrimitiveBuilder},
    compute,
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

impl Add for Column {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let result = match self.field.data_type() {
            DataType::Int8 => {
                let mut new_chunks = Vec::with_capacity(self.data_chunks.len());
                self.data_chunks
                    .iter()
                    .zip(rhs.data_chunks.iter())
                    .for_each(|(l, r)| {
                        let left_any = l.as_any();
                        let right_any = r.as_any();
                        let left = left_any.downcast_ref::<PrimitiveArray<Int8Type>>().unwrap();
                        let right = right_any
                            .downcast_ref::<PrimitiveArray<Int8Type>>()
                            .unwrap();
                        let res = Arc::new(compute::add(left, right).unwrap()) as ArrayRef;
                        new_chunks.push(res);
                    });
                self.replace_array(new_chunks)
            }
            _ => return unimplemented!(),
        };
        result
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
    fn add() {
        let s1 = Column::new::<Int8Type>("a", &[1, 2, 3]);
        println!("{:?}", s1.data_chunks);
        let s2 = s1.clone();
        println!("{:?}", (s1 + s2));
    }
}
