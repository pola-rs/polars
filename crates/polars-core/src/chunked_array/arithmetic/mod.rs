//! Implementations of arithmetic operations on ChunkedArray's.
#[cfg(feature = "dtype-decimal")]
mod decimal;
mod numeric;

use std::ops::{Add, Div, Mul, Rem, Sub};

use arrow::compute::utils::combine_validities_and;
use num_traits::{Num, NumCast, ToPrimitive};
pub use numeric::ArithmeticChunked;

use crate::prelude::arity::unary_elementwise_values;
use crate::prelude::*;

#[inline]
fn concat_binary_arrs(l: &[u8], r: &[u8], buf: &mut Vec<u8>) {
    buf.clear();

    buf.extend_from_slice(l);
    buf.extend_from_slice(r);
}

impl Add for &StringChunked {
    type Output = StringChunked;

    fn add(self, rhs: Self) -> Self::Output {
        unsafe { (self.as_binary() + rhs.as_binary()).to_string_unchecked() }
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
        unsafe { ((&self.as_binary()) + rhs.as_bytes()).to_string_unchecked() }
    }
}

fn concat_binview(a: &BinaryViewArray, b: &BinaryViewArray) -> BinaryViewArray {
    let validity = combine_validities_and(a.validity(), b.validity());

    let mut mutable = MutableBinaryViewArray::with_capacity(a.len());

    let mut scratch = vec![];
    for (a, b) in a.values_iter().zip(b.values_iter()) {
        concat_binary_arrs(a, b, &mut scratch);
        mutable.push_value(&scratch)
    }

    mutable.freeze().with_validity(validity)
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

        arity::binary(self, rhs, concat_binview)
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
        let arr = BinaryViewArray::from_slice_values([rhs]);
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
                Some(rhs) => unary_elementwise_values(self, |v| v as IdxSize + rhs as IdxSize),
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
        a1.append(&a2).unwrap();
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
