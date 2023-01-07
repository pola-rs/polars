use std::ops::Not;

use arrow::array::{BooleanArray, PrimitiveArray, Utf8Array};
use arrow::compute;
use arrow::compute::comparison;
#[cfg(feature = "dtype-binary")]
use arrow::scalar::BinaryScalar;
use arrow::scalar::{PrimitiveScalar, Scalar, Utf8Scalar};
use num::{NumCast, ToPrimitive};
use polars_arrow::prelude::FromData;

use crate::prelude::*;
use crate::utils::{align_chunks_binary, NoNull};

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    /// First ensure that the chunks of lhs and rhs match and then iterates over the chunks and applies
    /// the comparison operator.
    fn comparison(
        &self,
        rhs: &ChunkedArray<T>,
        f: impl Fn(&PrimitiveArray<T::Native>, &PrimitiveArray<T::Native>) -> BooleanArray,
    ) -> BooleanChunked {
        let chunks = self
            .downcast_iter()
            .zip(rhs.downcast_iter())
            .map(|(left, right)| {
                let arr = f(left, right);
                Box::new(arr) as ArrayRef
            })
            .collect::<Vec<_>>();

        unsafe { ChunkedArray::from_chunks("", chunks) }
    }
}

impl<T> ChunkCompare<&ChunkedArray<T>> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Item = BooleanChunked;

    fn equal(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    self.equal(value)
                } else {
                    self.is_null()
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    rhs.equal(value)
                } else {
                    self.is_null()
                }
            }
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                lhs.comparison(&rhs, |x, y| comparison::eq_and_validity(x, y))
            }
        }
    }

    fn not_equal(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    self.not_equal(value)
                } else {
                    self.is_not_null()
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    rhs.not_equal(value)
                } else {
                    self.is_not_null()
                }
            }
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                lhs.comparison(&rhs, |x, y| comparison::neq_and_validity(x, y))
            }
        }
    }

    fn gt(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    self.gt(value)
                } else {
                    BooleanChunked::full("", false, self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    rhs.lt(value)
                } else {
                    BooleanChunked::full("", false, rhs.len())
                }
            }
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                lhs.comparison(&rhs, |x, y| comparison::gt(x, y))
            }
        }
    }

    fn gt_eq(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    self.gt_eq(value)
                } else {
                    BooleanChunked::full("", false, self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    rhs.lt_eq(value)
                } else {
                    BooleanChunked::full("", false, rhs.len())
                }
            }
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                lhs.comparison(&rhs, |x, y| comparison::gt_eq(x, y))
            }
        }
    }

    fn lt(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    self.lt(value)
                } else {
                    BooleanChunked::full("", false, self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    rhs.gt(value)
                } else {
                    BooleanChunked::full("", false, rhs.len())
                }
            }
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                lhs.comparison(&rhs, |x, y| comparison::lt(x, y))
            }
        }
    }

    fn lt_eq(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    self.lt_eq(value)
                } else {
                    BooleanChunked::full("", false, self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    rhs.gt_eq(value)
                } else {
                    BooleanChunked::full("", false, rhs.len())
                }
            }
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                lhs.comparison(&rhs, |x, y| comparison::lt_eq(x, y))
            }
        }
    }
}

fn compare_bools(
    lhs: &BooleanChunked,
    rhs: &BooleanChunked,
    f: impl Fn(&BooleanArray, &BooleanArray) -> BooleanArray,
) -> BooleanChunked {
    let chunks = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(l, r)| Box::new(f(l, r)) as ArrayRef)
        .collect();

    unsafe { BooleanChunked::from_chunks(lhs.name(), chunks) }
}

impl ChunkCompare<&BooleanChunked> for BooleanChunked {
    type Item = BooleanChunked;

    fn equal(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    match value {
                        true => {
                            if self.null_count() == 0 {
                                self.clone()
                            } else {
                                let chunks = self
                                    .downcast_iter()
                                    .map(|arr| {
                                        if let Some(validity) = arr.validity() {
                                            Box::new(BooleanArray::from_data_default(
                                                arr.values() & validity,
                                                None,
                                            ))
                                                as ArrayRef
                                        } else {
                                            Box::new(arr.clone())
                                        }
                                    })
                                    .collect();
                                unsafe { BooleanChunked::from_chunks("", chunks) }
                            }
                        }
                        false => {
                            if self.null_count() == 0 {
                                self.not()
                            } else {
                                let chunks = self
                                    .downcast_iter()
                                    .map(|arr| {
                                        let bitmap = if let Some(validity) = arr.validity() {
                                            arr.values() ^ validity
                                        } else {
                                            arr.values().not()
                                        };
                                        Box::new(BooleanArray::from_data_default(bitmap, None))
                                            as ArrayRef
                                    })
                                    .collect();
                                unsafe { BooleanChunked::from_chunks("", chunks) }
                            }
                        }
                    }
                } else {
                    self.is_null()
                }
            }
            (1, _) => rhs.equal(self),
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                compare_bools(&lhs, &rhs, |lhs, rhs| comparison::eq_and_validity(lhs, rhs))
            }
        }
    }

    fn not_equal(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    match value {
                        true => {
                            if self.null_count() == 0 {
                                self.not()
                            } else {
                                let chunks = self
                                    .downcast_iter()
                                    .map(|arr| {
                                        let bitmap = if let Some(validity) = arr.validity() {
                                            (arr.values() & validity).not()
                                        } else {
                                            arr.values().not()
                                        };
                                        Box::new(BooleanArray::from_data_default(bitmap, None))
                                            as ArrayRef
                                    })
                                    .collect();
                                unsafe { BooleanChunked::from_chunks("", chunks) }
                            }
                        }
                        false => {
                            if self.null_count() == 0 {
                                self.clone()
                            } else {
                                let chunks = self
                                    .downcast_iter()
                                    .map(|arr| {
                                        let bitmap = if let Some(validity) = arr.validity() {
                                            (arr.values() ^ validity).not()
                                        } else {
                                            arr.values().clone()
                                        };
                                        Box::new(BooleanArray::from_data_default(bitmap, None))
                                            as ArrayRef
                                    })
                                    .collect();
                                unsafe { BooleanChunked::from_chunks("", chunks) }
                            }
                        }
                    }
                } else {
                    self.is_not_null()
                }
            }
            (1, _) => rhs.not_equal(self),
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                compare_bools(&lhs, &rhs, |lhs, rhs| {
                    comparison::neq_and_validity(lhs, rhs)
                })
            }
        }
    }

    fn gt(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    match value {
                        true => BooleanChunked::full("", false, self.len()),
                        false => self.clone(),
                    }
                } else {
                    BooleanChunked::full("", false, self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    match value {
                        true => rhs.not(),
                        false => BooleanChunked::full("", false, rhs.len()),
                    }
                } else {
                    BooleanChunked::full("", false, rhs.len())
                }
            }
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                compare_bools(&lhs, &rhs, |lhs, rhs| comparison::gt(lhs, rhs))
            }
        }
    }

    fn gt_eq(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    match value {
                        true => self.clone(),
                        false => BooleanChunked::full("", true, self.len()),
                    }
                } else {
                    BooleanChunked::full("", false, self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    match value {
                        true => BooleanChunked::full("", true, rhs.len()),
                        false => rhs.not(),
                    }
                } else {
                    BooleanChunked::full("", false, rhs.len())
                }
            }
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                compare_bools(&lhs, &rhs, |lhs, rhs| comparison::gt_eq(lhs, rhs))
            }
        }
    }

    fn lt(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    match value {
                        true => self.not(),
                        false => BooleanChunked::full("", false, self.len()),
                    }
                } else {
                    BooleanChunked::full("", false, self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    match value {
                        true => BooleanChunked::full("", false, rhs.len()),
                        false => rhs.clone(),
                    }
                } else {
                    BooleanChunked::full("", false, rhs.len())
                }
            }
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                compare_bools(&lhs, &rhs, |lhs, rhs| comparison::lt(lhs, rhs))
            }
        }
    }

    fn lt_eq(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    match value {
                        true => BooleanChunked::full("", true, self.len()),
                        false => BooleanChunked::full("", false, self.len()),
                    }
                } else {
                    BooleanChunked::full("", false, self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    match value {
                        true => rhs.clone(),
                        false => BooleanChunked::full("", true, rhs.len()),
                    }
                } else {
                    BooleanChunked::full("", false, rhs.len())
                }
            }
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                compare_bools(&lhs, &rhs, |lhs, rhs| comparison::lt_eq(lhs, rhs))
            }
        }
    }
}

impl Utf8Chunked {
    fn comparison(
        &self,
        rhs: &Utf8Chunked,
        f: impl Fn(&Utf8Array<i64>, &Utf8Array<i64>) -> BooleanArray,
    ) -> BooleanChunked {
        let chunks = self
            .downcast_iter()
            .zip(rhs.downcast_iter())
            .map(|(left, right)| {
                let arr = f(left, right);
                Box::new(arr) as ArrayRef
            })
            .collect();
        unsafe { BooleanChunked::from_chunks("", chunks) }
    }
}

impl ChunkCompare<&Utf8Chunked> for Utf8Chunked {
    type Item = BooleanChunked;

    fn equal(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.equal(value)
            } else {
                self.is_null()
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.equal(value)
            } else {
                self.is_null()
            }
        } else {
            let (lhs, rhs) = align_chunks_binary(self, rhs);
            lhs.comparison(&rhs, comparison::utf8::eq_and_validity)
        }
    }

    fn not_equal(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.not_equal(value)
            } else {
                self.is_not_null()
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.not_equal(value)
            } else {
                self.is_not_null()
            }
        } else {
            let (lhs, rhs) = align_chunks_binary(self, rhs);
            lhs.comparison(&rhs, comparison::utf8::neq_and_validity)
        }
    }

    fn gt(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.gt(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.lt(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, |l, r| comparison::gt(l, r))
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, >)
        }
    }

    fn gt_eq(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.gt_eq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.lt_eq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, |l, r| comparison::gt_eq(l, r))
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, >=)
        }
    }

    fn lt(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.lt(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.gt(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, |l, r| comparison::lt(l, r))
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, <)
        }
    }

    fn lt_eq(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.lt_eq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.gt_eq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, |l, r| comparison::lt_eq(l, r))
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, <=)
        }
    }
}

#[cfg(feature = "dtype-binary")]
impl BinaryChunked {
    fn comparison(
        &self,
        rhs: &BinaryChunked,
        f: impl Fn(&BinaryArray<i64>, &BinaryArray<i64>) -> BooleanArray,
    ) -> BooleanChunked {
        let chunks = self
            .downcast_iter()
            .zip(rhs.downcast_iter())
            .map(|(left, right)| {
                let arr = f(left, right);
                Box::new(arr) as ArrayRef
            })
            .collect();
        unsafe { BooleanChunked::from_chunks("", chunks) }
    }
}

#[cfg(feature = "dtype-binary")]
impl ChunkCompare<&BinaryChunked> for BinaryChunked {
    type Item = BooleanChunked;

    fn equal(&self, rhs: &BinaryChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.equal(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.equal(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else {
            let (lhs, rhs) = align_chunks_binary(self, rhs);
            lhs.comparison(&rhs, comparison::binary::eq_and_validity)
        }
    }

    fn not_equal(&self, rhs: &BinaryChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.not_equal(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.not_equal(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else {
            let (lhs, rhs) = align_chunks_binary(self, rhs);
            lhs.comparison(&rhs, comparison::binary::neq_and_validity)
        }
    }

    fn gt(&self, rhs: &BinaryChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.gt(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.lt(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, |l, r| comparison::gt(l, r))
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, >)
        }
    }

    fn gt_eq(&self, rhs: &BinaryChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.gt_eq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.lt_eq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, |l, r| comparison::gt_eq(l, r))
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, >=)
        }
    }

    fn lt(&self, rhs: &BinaryChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.lt(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.gt(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, |l, r| comparison::lt(l, r))
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, <)
        }
    }

    fn lt_eq(&self, rhs: &BinaryChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.lt_eq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.gt_eq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, |l, r| comparison::lt_eq(l, r))
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, <=)
        }
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn primitive_compare_scalar<Rhs: ToPrimitive>(
        &self,
        rhs: Rhs,
        f: impl Fn(&PrimitiveArray<T::Native>, &dyn Scalar) -> BooleanArray,
    ) -> BooleanChunked {
        let rhs: T::Native =
            NumCast::from(rhs).expect("could not cast to underlying chunkedarray type");
        let scalar = PrimitiveScalar::new(T::get_dtype().to_arrow(), Some(rhs));
        self.apply_kernel_cast(&|arr| Box::new(f(arr, &scalar)))
    }
}

impl<T, Rhs> ChunkCompare<Rhs> for ChunkedArray<T>
where
    T: PolarsNumericType,
    Rhs: ToPrimitive,
{
    type Item = BooleanChunked;
    fn equal(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, |l, rhs| comparison::eq_scalar_and_validity(l, rhs))
    }

    fn not_equal(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, |l, rhs| comparison::neq_scalar_and_validity(l, rhs))
    }

    fn gt(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, |l, rhs| comparison::gt_scalar(l, rhs))
    }

    fn gt_eq(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, |l, rhs| comparison::gt_eq_scalar(l, rhs))
    }

    fn lt(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, |l, rhs| comparison::lt_scalar(l, rhs))
    }

    fn lt_eq(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, |l, rhs| comparison::lt_eq_scalar(l, rhs))
    }
}

impl Utf8Chunked {
    fn utf8_compare_scalar(
        &self,
        rhs: &str,
        f: impl Fn(&Utf8Array<i64>, &dyn Scalar) -> BooleanArray,
    ) -> BooleanChunked {
        let scalar = Utf8Scalar::<i64>::new(Some(rhs));
        self.apply_kernel_cast(&|arr| Box::new(f(arr, &scalar)))
    }
}

impl ChunkCompare<&str> for Utf8Chunked {
    type Item = BooleanChunked;
    fn equal(&self, rhs: &str) -> BooleanChunked {
        self.utf8_compare_scalar(rhs, |l, rhs| comparison::eq_scalar_and_validity(l, rhs))
    }
    fn not_equal(&self, rhs: &str) -> BooleanChunked {
        self.utf8_compare_scalar(rhs, |l, rhs| comparison::neq_scalar_and_validity(l, rhs))
    }

    fn gt(&self, rhs: &str) -> BooleanChunked {
        self.utf8_compare_scalar(rhs, |l, rhs| comparison::gt_scalar(l, rhs))
    }

    fn gt_eq(&self, rhs: &str) -> BooleanChunked {
        self.utf8_compare_scalar(rhs, |l, rhs| comparison::gt_eq_scalar(l, rhs))
    }

    fn lt(&self, rhs: &str) -> BooleanChunked {
        self.utf8_compare_scalar(rhs, |l, rhs| comparison::lt_scalar(l, rhs))
    }

    fn lt_eq(&self, rhs: &str) -> BooleanChunked {
        self.utf8_compare_scalar(rhs, |l, rhs| comparison::lt_eq_scalar(l, rhs))
    }
}

#[cfg(feature = "dtype-binary")]
impl BinaryChunked {
    fn binary_compare_scalar(
        &self,
        rhs: &[u8],
        f: impl Fn(&BinaryArray<i64>, &dyn Scalar) -> BooleanArray,
    ) -> BooleanChunked {
        let scalar = BinaryScalar::<i64>::new(Some(rhs));
        self.apply_kernel_cast(&|arr| Box::new(f(arr, &scalar)))
    }
}

#[cfg(feature = "dtype-binary")]
impl ChunkCompare<&[u8]> for BinaryChunked {
    type Item = BooleanChunked;
    fn equal(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::eq_scalar_and_validity(l, rhs))
    }
    fn not_equal(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::neq_scalar_and_validity(l, rhs))
    }

    fn gt(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::gt_scalar(l, rhs))
    }

    fn gt_eq(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::gt_eq_scalar(l, rhs))
    }

    fn lt(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::lt_scalar(l, rhs))
    }

    fn lt_eq(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::lt_eq_scalar(l, rhs))
    }
}

impl ChunkCompare<&ListChunked> for ListChunked {
    type Item = BooleanChunked;
    fn equal(&self, rhs: &ListChunked) -> BooleanChunked {
        self.amortized_iter()
            .zip(rhs.amortized_iter())
            .map(|(left, right)| match (left, right) {
                (None, None) => true,
                (Some(l), Some(r)) => l.as_ref().series_equal_missing(r.as_ref()),
                _ => false,
            })
            .collect_trusted()
    }

    fn not_equal(&self, rhs: &ListChunked) -> BooleanChunked {
        self.amortized_iter()
            .zip(rhs.amortized_iter())
            .map(|(left, right)| {
                let out = match (left, right) {
                    (None, None) => true,
                    (Some(l), Some(r)) => l.as_ref().series_equal_missing(r.as_ref()),
                    _ => false,
                };
                !out
            })
            .collect_trusted()
    }

    // following are not implemented because gt, lt comparison of series don't make sense
    fn gt(&self, _rhs: &ListChunked) -> BooleanChunked {
        unimplemented!()
    }

    fn gt_eq(&self, _rhs: &ListChunked) -> BooleanChunked {
        unimplemented!()
    }

    fn lt(&self, _rhs: &ListChunked) -> BooleanChunked {
        unimplemented!()
    }

    fn lt_eq(&self, _rhs: &ListChunked) -> BooleanChunked {
        unimplemented!()
    }
}

impl Not for &BooleanChunked {
    type Output = BooleanChunked;

    fn not(self) -> Self::Output {
        let chunks = self
            .downcast_iter()
            .map(|a| {
                let arr = compute::boolean::not(a);
                Box::new(arr) as ArrayRef
            })
            .collect::<Vec<_>>();
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
    }
}

impl Not for BooleanChunked {
    type Output = BooleanChunked;

    fn not(self) -> Self::Output {
        (&self).not()
    }
}

impl BooleanChunked {
    /// Check if all values are `true`
    pub fn all(&self) -> bool {
        self.downcast_iter().all(compute::boolean::all)
    }

    /// Check if any value is `true`
    pub fn any(&self) -> bool {
        self.downcast_iter().any(compute::boolean::any)
    }
}

// private
pub(crate) trait ChunkEqualElement {
    /// Only meant for physical types.
    /// Check if element in self is equal to element in other, assumes same dtypes
    ///
    /// # Safety
    ///
    /// No type checks.
    unsafe fn equal_element(&self, _idx_self: usize, _idx_other: usize, _other: &Series) -> bool {
        unimplemented!()
    }
}

impl<T> ChunkEqualElement for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
        let ca_other = other.as_ref().as_ref();
        debug_assert!(self.dtype() == other.dtype());
        let ca_other = &*(ca_other as *const ChunkedArray<T>);
        // Should be get and not get_unchecked, because there could be nulls
        self.get(idx_self) == ca_other.get(idx_other)
    }
}

impl ChunkEqualElement for BooleanChunked {
    unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
        let ca_other = other.as_ref().as_ref();
        debug_assert!(self.dtype() == other.dtype());
        let ca_other = &*(ca_other as *const BooleanChunked);
        self.get(idx_self) == ca_other.get(idx_other)
    }
}

impl ChunkEqualElement for Utf8Chunked {
    unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
        let ca_other = other.as_ref().as_ref();
        debug_assert!(self.dtype() == other.dtype());
        let ca_other = &*(ca_other as *const Utf8Chunked);
        self.get(idx_self) == ca_other.get(idx_other)
    }
}

#[cfg(feature = "dtype-binary")]
impl ChunkEqualElement for BinaryChunked {
    unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
        let ca_other = other.as_ref().as_ref();
        debug_assert!(self.dtype() == other.dtype());
        let ca_other = &*(ca_other as *const BinaryChunked);
        self.get(idx_self) == ca_other.get(idx_other)
    }
}

impl ChunkEqualElement for ListChunked {}

#[cfg(feature = "dtype-struct")]
impl ChunkCompare<&StructChunked> for StructChunked {
    type Item = BooleanChunked;
    fn equal(&self, rhs: &StructChunked) -> BooleanChunked {
        use std::ops::BitAnd;
        if self.len() != rhs.len() || self.fields().len() != rhs.fields().len() {
            BooleanChunked::full("", false, self.len())
        } else {
            self.fields()
                .iter()
                .zip(rhs.fields().iter())
                .map(|(l, r)| l.equal(r).unwrap())
                .reduce(|lhs, rhs| lhs.bitand(rhs))
                .unwrap()
        }
    }

    fn not_equal(&self, rhs: &StructChunked) -> BooleanChunked {
        use std::ops::BitOr;
        if self.len() != rhs.len() || self.fields().len() != rhs.fields().len() {
            BooleanChunked::full("", false, self.len())
        } else {
            self.fields()
                .iter()
                .zip(rhs.fields().iter())
                .map(|(l, r)| l.not_equal(r).unwrap())
                .reduce(|lhs, rhs| lhs.bitor(rhs))
                .unwrap()
        }
    }

    // following are not implemented because gt, lt comparison of series don't make sense
    fn gt(&self, _rhs: &StructChunked) -> BooleanChunked {
        unimplemented!()
    }

    fn gt_eq(&self, _rhs: &StructChunked) -> BooleanChunked {
        unimplemented!()
    }

    fn lt(&self, _rhs: &StructChunked) -> BooleanChunked {
        unimplemented!()
    }

    fn lt_eq(&self, _rhs: &StructChunked) -> BooleanChunked {
        unimplemented!()
    }
}

#[cfg(test)]
mod test {
    use std::iter::repeat;

    use super::super::arithmetic::test::create_two_chunked;
    use super::super::test::get_chunked_array;
    use crate::prelude::*;

    #[test]
    fn test_bitwise_ops() {
        let a = BooleanChunked::new("a", &[true, false, false]);
        let b = BooleanChunked::new("b", &[Some(true), Some(true), None]);
        assert_eq!(Vec::from(&a | &b), &[Some(true), Some(true), None]);
        assert_eq!(Vec::from(&a & &b), &[Some(true), Some(false), Some(false)]);
        assert_eq!(Vec::from(!b), &[Some(false), Some(false), None]);
    }

    #[test]
    fn test_compare_chunk_diff() {
        let (a1, a2) = create_two_chunked();

        assert_eq!(
            a1.equal(&a2).into_iter().collect::<Vec<_>>(),
            repeat(Some(true)).take(6).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.equal(&a1).into_iter().collect::<Vec<_>>(),
            repeat(Some(true)).take(6).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.not_equal(&a2).into_iter().collect::<Vec<_>>(),
            repeat(Some(false)).take(6).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.not_equal(&a1).into_iter().collect::<Vec<_>>(),
            repeat(Some(false)).take(6).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.gt(&a2).into_iter().collect::<Vec<_>>(),
            repeat(Some(false)).take(6).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.gt(&a1).into_iter().collect::<Vec<_>>(),
            repeat(Some(false)).take(6).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.gt_eq(&a2).into_iter().collect::<Vec<_>>(),
            repeat(Some(true)).take(6).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.gt_eq(&a1).into_iter().collect::<Vec<_>>(),
            repeat(Some(true)).take(6).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.lt_eq(&a2).into_iter().collect::<Vec<_>>(),
            repeat(Some(true)).take(6).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.lt_eq(&a1).into_iter().collect::<Vec<_>>(),
            repeat(Some(true)).take(6).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.lt(&a2).into_iter().collect::<Vec<_>>(),
            repeat(Some(false)).take(6).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.lt(&a1).into_iter().collect::<Vec<_>>(),
            repeat(Some(false)).take(6).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_equal_chunks() {
        let a1 = get_chunked_array();
        let a2 = get_chunked_array();

        assert_eq!(
            a1.equal(&a2).into_iter().collect::<Vec<_>>(),
            repeat(Some(true)).take(3).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.equal(&a1).into_iter().collect::<Vec<_>>(),
            repeat(Some(true)).take(3).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.not_equal(&a2).into_iter().collect::<Vec<_>>(),
            repeat(Some(false)).take(3).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.not_equal(&a1).into_iter().collect::<Vec<_>>(),
            repeat(Some(false)).take(3).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.gt(&a2).into_iter().collect::<Vec<_>>(),
            repeat(Some(false)).take(3).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.gt(&a1).into_iter().collect::<Vec<_>>(),
            repeat(Some(false)).take(3).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.gt_eq(&a2).into_iter().collect::<Vec<_>>(),
            repeat(Some(true)).take(3).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.gt_eq(&a1).into_iter().collect::<Vec<_>>(),
            repeat(Some(true)).take(3).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.lt_eq(&a2).into_iter().collect::<Vec<_>>(),
            repeat(Some(true)).take(3).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.lt_eq(&a1).into_iter().collect::<Vec<_>>(),
            repeat(Some(true)).take(3).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.lt(&a2).into_iter().collect::<Vec<_>>(),
            repeat(Some(false)).take(3).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.lt(&a1).into_iter().collect::<Vec<_>>(),
            repeat(Some(false)).take(3).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_null_handling() {
        // assert we comply with arrows way of handling null data
        // we check comparison on two arrays with one chunk and verify it is equal to a differently
        // chunked array comparison.

        // two same chunked arrays
        let a1: Int32Chunked = (&[Some(1), None, Some(3)]).iter().copied().collect();
        let a2: Int32Chunked = (&[Some(1), Some(2), Some(3)]).iter().copied().collect();

        let mut a2_2chunks: Int32Chunked = (&[Some(1), Some(2)]).iter().copied().collect();
        a2_2chunks.append(&(&[Some(3)]).iter().copied().collect());

        assert_eq!(
            a1.equal(&a2).into_iter().collect::<Vec<_>>(),
            a1.equal(&a2_2chunks).into_iter().collect::<Vec<_>>()
        );

        assert_eq!(
            a1.not_equal(&a2).into_iter().collect::<Vec<_>>(),
            a1.not_equal(&a2_2chunks).into_iter().collect::<Vec<_>>()
        );
        assert_eq!(
            a1.not_equal(&a2).into_iter().collect::<Vec<_>>(),
            a2_2chunks.not_equal(&a1).into_iter().collect::<Vec<_>>()
        );

        assert_eq!(
            a1.gt(&a2).into_iter().collect::<Vec<_>>(),
            a1.gt(&a2_2chunks).into_iter().collect::<Vec<_>>()
        );
        assert_eq!(
            a1.gt(&a2).into_iter().collect::<Vec<_>>(),
            a2_2chunks.gt(&a1).into_iter().collect::<Vec<_>>()
        );

        assert_eq!(
            a1.gt_eq(&a2).into_iter().collect::<Vec<_>>(),
            a1.gt_eq(&a2_2chunks).into_iter().collect::<Vec<_>>()
        );
        assert_eq!(
            a1.gt_eq(&a2).into_iter().collect::<Vec<_>>(),
            a2_2chunks.gt_eq(&a1).into_iter().collect::<Vec<_>>()
        );

        assert_eq!(
            a1.lt_eq(&a2).into_iter().collect::<Vec<_>>(),
            a1.lt_eq(&a2_2chunks).into_iter().collect::<Vec<_>>()
        );
        assert_eq!(
            a1.lt_eq(&a2).into_iter().collect::<Vec<_>>(),
            a2_2chunks.lt_eq(&a1).into_iter().collect::<Vec<_>>()
        );

        assert_eq!(
            a1.lt(&a2).into_iter().collect::<Vec<_>>(),
            a1.lt(&a2_2chunks).into_iter().collect::<Vec<_>>()
        );
        assert_eq!(
            a1.lt(&a2).into_iter().collect::<Vec<_>>(),
            a2_2chunks.lt(&a1).into_iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_left_right() {
        // This failed with arrow comparisons.
        // sliced
        let a1: Int32Chunked = (&[Some(1), Some(2)]).iter().copied().collect();
        let a1 = a1.slice(1, 1);
        let a2: Int32Chunked = (&[Some(2)]).iter().copied().collect();
        assert_eq!(a1.equal(&a2).sum(), a2.equal(&a1).sum());
        assert_eq!(a1.not_equal(&a2).sum(), a2.not_equal(&a1).sum());
        assert_eq!(a1.gt(&a2).sum(), a2.gt(&a1).sum());
        assert_eq!(a1.lt(&a2).sum(), a2.lt(&a1).sum());
        assert_eq!(a1.lt_eq(&a2).sum(), a2.lt_eq(&a1).sum());
        assert_eq!(a1.gt_eq(&a2).sum(), a2.gt_eq(&a1).sum());

        let a1: Utf8Chunked = (&["a", "b"]).iter().copied().collect();
        let a1 = a1.slice(1, 1);
        let a2: Utf8Chunked = (&["b"]).iter().copied().collect();
        assert_eq!(a1.equal(&a2).sum(), a2.equal(&a1).sum());
        assert_eq!(a1.not_equal(&a2).sum(), a2.not_equal(&a1).sum());
        assert_eq!(a1.gt(&a2).sum(), a2.gt(&a1).sum());
        assert_eq!(a1.lt(&a2).sum(), a2.lt(&a1).sum());
        assert_eq!(a1.lt_eq(&a2).sum(), a2.lt_eq(&a1).sum());
        assert_eq!(a1.gt_eq(&a2).sum(), a2.gt_eq(&a1).sum());
    }

    #[test]
    fn test_kleene() {
        let a = BooleanChunked::new("", &[Some(true), Some(false), None]);
        let trues = BooleanChunked::from_slice("", &[true, true, true]);
        let falses = BooleanChunked::from_slice("", &[false, false, false]);

        let c = &a | &trues;
        assert_eq!(Vec::from(&c), &[Some(true), Some(true), Some(true)]);

        let c = &a | &falses;
        assert_eq!(Vec::from(&c), &[Some(true), Some(false), None])
    }

    #[test]
    fn test_broadcasting_bools() {
        let a = BooleanChunked::from_slice("", &[true, false, true]);
        let true_ = BooleanChunked::from_slice("", &[true]);
        let false_ = BooleanChunked::from_slice("", &[false]);

        let out = a.equal(&true_);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(true)]);
        let out = true_.equal(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(true)]);
        let out = a.equal(&false_);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(false)]);
        let out = false_.equal(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(false)]);

        let out = a.not_equal(&true_);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(false)]);
        let out = true_.not_equal(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(false)]);
        let out = a.not_equal(&false_);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(true)]);
        let out = false_.not_equal(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(true)]);

        let out = a.gt(&true_);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(false)]);
        let out = true_.gt(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(false)]);
        let out = a.gt(&false_);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(true)]);
        let out = false_.gt(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(false)]);

        let out = a.gt_eq(&true_);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(true)]);
        let out = true_.gt_eq(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(true)]);
        let out = a.gt_eq(&false_);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(true)]);
        let out = false_.gt_eq(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(false)]);

        let out = a.lt(&true_);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(false)]);
        let out = true_.lt(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(false)]);
        let out = a.lt(&false_);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(false)]);
        let out = false_.lt(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(true)]);

        let out = a.lt_eq(&true_);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(true)]);
        let out = true_.lt_eq(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(true)]);
        let out = a.lt_eq(&false_);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(false)]);
        let out = false_.lt_eq(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(true)]);

        let a = BooleanChunked::from_slice_options("", &[Some(true), Some(false), None]);
        let all_true = BooleanChunked::from_slice("", &[true, true, true]);
        let all_false = BooleanChunked::from_slice("", &[false, false, false]);
        let out = a.equal(&true_);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(false)]);
        let out = a.not_equal(&true_);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(true)]);

        let out = a.equal(&all_true);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(false)]);
        let out = a.not_equal(&all_true);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(true)]);
        let out = a.equal(&false_);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(false)]);
        let out = a.not_equal(&false_);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(true)]);
        let out = a.equal(&all_false);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(false)]);
        let out = a.not_equal(&all_false);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(true)]);
    }

    #[test]
    fn test_broadcasting_numeric() {
        let a = Int32Chunked::from_slice("", &[1, 2, 3]);
        let one = Int32Chunked::from_slice("", &[1]);
        let three = Int32Chunked::from_slice("", &[3]);

        let out = a.equal(&one);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(false)]);
        let out = one.equal(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(false)]);
        let out = a.equal(&three);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(true)]);
        let out = three.equal(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(true)]);

        let out = a.not_equal(&one);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(true)]);
        let out = one.not_equal(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(true)]);
        let out = a.not_equal(&three);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(false)]);
        let out = three.not_equal(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(false)]);

        let out = a.gt(&one);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(true)]);
        let out = one.gt(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(false)]);
        let out = a.gt(&three);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(false)]);
        let out = three.gt(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(false)]);

        let out = a.lt(&one);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(false)]);
        let out = one.lt(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(true)]);
        let out = a.lt(&three);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(false)]);
        let out = three.lt(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(false)]);

        let out = a.gt_eq(&one);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(true)]);
        let out = one.gt_eq(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(false)]);
        let out = a.gt_eq(&three);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(true)]);
        let out = three.gt_eq(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(true)]);

        let out = a.lt_eq(&one);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(false)]);
        let out = one.lt_eq(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(true)]);
        let out = a.lt_eq(&three);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(true)]);
        let out = three.lt_eq(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(true)]);
    }
}
