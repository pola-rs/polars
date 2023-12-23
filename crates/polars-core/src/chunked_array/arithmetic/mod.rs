//! Implementations of arithmetic operations on ChunkedArray's.
#[cfg(feature = "dtype-decimal")]
mod decimal;
mod numeric;

use std::ops::{Add, Div, Mul, Rem, Sub};

use arrow::array::PrimitiveArray;
use arrow::compute::arithmetics::basic;
use arrow::compute::arity_assign;
use arrow::compute::utils::combine_validities_and;
use arrow::types::NativeType;
use num_traits::{Num, NumCast, ToPrimitive, Zero};
pub(super) use numeric::arithmetic_helper;

use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::align_chunks_binary_owned;

pub trait ArrayArithmetics
where
    Self: NativeType,
{
    fn add(lhs: &PrimitiveArray<Self>, rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self>;
    fn sub(lhs: &PrimitiveArray<Self>, rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self>;
    fn mul(lhs: &PrimitiveArray<Self>, rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self>;
    fn div(lhs: &PrimitiveArray<Self>, rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self>;
    fn div_scalar(lhs: &PrimitiveArray<Self>, rhs: &Self) -> PrimitiveArray<Self>;
    fn rem(lhs: &PrimitiveArray<Self>, rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self>;
    fn rem_scalar(lhs: &PrimitiveArray<Self>, rhs: &Self) -> PrimitiveArray<Self>;
}

macro_rules! native_array_arithmetics {
    ($ty: ty) => {
        impl ArrayArithmetics for $ty
        {
            fn add(lhs: &PrimitiveArray<Self>, rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self> {
                basic::add(lhs, rhs)
            }
            fn sub(lhs: &PrimitiveArray<Self>, rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self> {
                basic::sub(lhs, rhs)
            }
            fn mul(lhs: &PrimitiveArray<Self>, rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self> {
                basic::mul(lhs, rhs)
            }
            fn div(lhs: &PrimitiveArray<Self>, rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self> {
                basic::div(lhs, rhs)
            }
            fn div_scalar(lhs: &PrimitiveArray<Self>, rhs: &Self) -> PrimitiveArray<Self> {
                basic::div_scalar(lhs, rhs)
            }
            fn rem(lhs: &PrimitiveArray<Self>, rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self> {
                basic::rem(lhs, rhs)
            }
            fn rem_scalar(lhs: &PrimitiveArray<Self>, rhs: &Self) -> PrimitiveArray<Self> {
                basic::rem_scalar(lhs, rhs)
            }
        }
    };
    ($($ty:ty),*) => {
        $(native_array_arithmetics!($ty);)*
    }
}

native_array_arithmetics!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

fn concat_binary_arrs(l: &[u8], r: &[u8], buf: &mut Vec<u8>) {
    buf.clear();

    buf.extend_from_slice(l);
    buf.extend_from_slice(r);
}

impl Add for &StringChunked {
    type Output = StringChunked;

    fn add(self, rhs: Self) -> Self::Output {
        unsafe { (self.as_binary() + rhs.as_binary()).to_utf8() }
    }
}

impl Add for StringChunked {
    type Output = StringChunked;

    fn add(self, rhs: Self) -> Self::Output {
        (&self).add(&rhs)
    }
}

impl Add<&str> for &StringChunked {
    type Output = StringChunked;

    fn add(self, rhs: &str) -> Self::Output {
        unsafe { ((&self.as_binary()) + rhs.as_bytes()).to_utf8() }
    }
}

fn concat_binary(a: &BinaryArray<i64>, b: &BinaryArray<i64>) -> BinaryArray<i64> {
    let validity = combine_validities_and(a.validity(), b.validity());
    let mut values = Vec::with_capacity(a.get_values_size() + b.get_values_size());
    let mut offsets = Vec::with_capacity(a.len() + 1);
    let mut offset_so_far = 0i64;
    offsets.push(offset_so_far);

    for (a, b) in a.values_iter().zip(b.values_iter()) {
        values.extend_from_slice(a);
        values.extend_from_slice(b);
        offset_so_far = values.len() as i64;
        offsets.push(offset_so_far)
    }
    unsafe { BinaryArray::from_data_unchecked_default(offsets.into(), values.into(), validity) }
}

impl Add for &BinaryChunked {
    type Output = BinaryChunked;

    fn add(self, rhs: Self) -> Self::Output {
        // broadcasting path rhs
        if rhs.len() == 1 {
            let rhs = rhs.get(0);
            let mut buf = vec![];
            return match rhs {
                Some(rhs) => {
                    self.apply_mut(|s| {
                        concat_binary_arrs(s, rhs, &mut buf);
                        let out = buf.as_slice();
                        // SAFETY: lifetime is bound to the outer scope and the
                        // ref is valid for the lifetime of this closure.
                        unsafe { std::mem::transmute::<_, &'static [u8]>(out) }
                    })
                },
                None => BinaryChunked::full_null(self.name(), self.len()),
            };
        }
        // broadcasting path lhs
        if self.len() == 1 {
            let lhs = self.get(0);
            let mut buf = vec![];
            return match lhs {
                Some(lhs) => rhs.apply_mut(|s| {
                    concat_binary_arrs(lhs, s, &mut buf);
                    let out = buf.as_slice();
                    // SAFETY: lifetime is bound to the outer scope and the
                    // ref is valid for the lifetime of this closure.
                    unsafe { std::mem::transmute::<_, &'static [u8]>(out) }
                }),
                None => BinaryChunked::full_null(self.name(), rhs.len()),
            };
        }

        arity::binary(self, rhs, concat_binary)
    }
}

impl Add for BinaryChunked {
    type Output = BinaryChunked;

    fn add(self, rhs: Self) -> Self::Output {
        (&self).add(&rhs)
    }
}

impl Add<&[u8]> for &BinaryChunked {
    type Output = BinaryChunked;

    fn add(self, rhs: &[u8]) -> Self::Output {
        let arr = BinaryArray::<i64>::from_slice([rhs]);
        let rhs: BinaryChunked = arr.into();
        self.add(&rhs)
    }
}

fn add_boolean(a: &BooleanArray, b: &BooleanArray) -> PrimitiveArray<IdxSize> {
    let validity = combine_validities_and(a.validity(), b.validity());

    let values = a
        .values_iter()
        .zip(b.values_iter())
        .map(|(a, b)| a as IdxSize + b as IdxSize)
        .collect::<Vec<_>>();
    PrimitiveArray::from_data_default(values.into(), validity)
}

impl Add for &BooleanChunked {
    type Output = IdxCa;

    fn add(self, rhs: Self) -> Self::Output {
        // Broadcasting path rhs.
        if rhs.len() == 1 {
            let rhs = rhs.get(0);
            return match rhs {
                Some(rhs) => self.apply_values_generic(|v| v as IdxSize + rhs as IdxSize),
                None => IdxCa::full_null(self.name(), self.len()),
            };
        }
        // Broadcasting path lhs.
        if self.len() == 1 {
            return rhs.add(self);
        }
        arity::binary(self, rhs, add_boolean)
    }
}

impl Add for BooleanChunked {
    type Output = IdxCa;

    fn add(self, rhs: Self) -> Self::Output {
        (&self).add(&rhs)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::prelude::*;

    pub(crate) fn create_two_chunked() -> (Int32Chunked, Int32Chunked) {
        let mut a1 = Int32Chunked::new("a", &[1, 2, 3]);
        let a2 = Int32Chunked::new("a", &[4, 5, 6]);
        let a3 = Int32Chunked::new("a", &[1, 2, 3, 4, 5, 6]);
        a1.append(&a2);
        (a1, a3)
    }

    #[test]
    #[allow(clippy::eq_op)]
    fn test_chunk_mismatch() {
        let (a1, a2) = create_two_chunked();
        // With different chunks.
        let _ = &a1 + &a2;
        let _ = &a1 - &a2;
        let _ = &a1 / &a2;
        let _ = &a1 * &a2;

        // With same chunks.
        let _ = &a1 + &a1;
        let _ = &a1 - &a1;
        let _ = &a1 / &a1;
        let _ = &a1 * &a1;
    }
}
