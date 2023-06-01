mod scalar;

use std::ops::Not;

use arrow::array::{BooleanArray, PrimitiveArray, Utf8Array};
use arrow::bitmap::MutableBitmap;
use arrow::compute;
use arrow::compute::comparison;
use arrow::scalar::{BinaryScalar, PrimitiveScalar, Scalar, Utf8Scalar};
use num_traits::{NumCast, ToPrimitive};
use polars_arrow::kernels::rolling::compare_fn_nan_max;
use polars_arrow::prelude::FromData;

use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::align_chunks_binary;

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

    // also includes validity  in comparison
    pub fn not_equal_and_validity(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        let (lhs, rhs) = align_chunks_binary(self, rhs);
        lhs.comparison(&rhs, |x, y| comparison::neq_and_validity(x, y))
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
                    BooleanChunked::full_null("", self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    rhs.equal(value)
                } else {
                    BooleanChunked::full_null("", rhs.len())
                }
            }
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                lhs.comparison(&rhs, |x, y| comparison::eq(x, y))
            }
        }
    }

    fn equal_missing(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    self.equal_missing(value)
                } else {
                    self.is_null()
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    rhs.equal_missing(value)
                } else {
                    rhs.is_null()
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
                    BooleanChunked::full_null("", self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    rhs.not_equal(value)
                } else {
                    BooleanChunked::full_null("", rhs.len())
                }
            }
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                lhs.comparison(&rhs, |x, y| comparison::neq(x, y))
            }
        }
    }

    fn not_equal_missing(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    self.not_equal_missing(value)
                } else {
                    self.is_not_null()
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    rhs.not_equal_missing(value)
                } else {
                    rhs.is_not_null()
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
                    BooleanChunked::full_null("", self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    rhs.lt(value)
                } else {
                    BooleanChunked::full_null("", rhs.len())
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
                    BooleanChunked::full_null("", self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    rhs.lt_eq(value)
                } else {
                    BooleanChunked::full_null("", rhs.len())
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
                    BooleanChunked::full_null("", self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    rhs.gt(value)
                } else {
                    BooleanChunked::full_null("", rhs.len())
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
                    BooleanChunked::full_null("", self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    rhs.gt_eq(value)
                } else {
                    BooleanChunked::full_null("", rhs.len())
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
                    if value {
                        self.clone()
                    } else {
                        self.not()
                    }
                } else {
                    BooleanChunked::full_null("", self.len())
                }
            }
            (1, _) => rhs.equal(self),
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                compare_bools(&lhs, &rhs, |lhs, rhs| comparison::eq(lhs, rhs))
            }
        }
    }

    fn equal_missing(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    if value {
                        if self.null_count() == 0 {
                            self.clone()
                        } else {
                            self.apply_kernel(&|arr| {
                                if let Some(validity) = arr.validity() {
                                    Box::new(BooleanArray::from_data_default(
                                        arr.values() & validity,
                                        None,
                                    )) as ArrayRef
                                } else {
                                    Box::new(arr.clone())
                                }
                            })
                        }
                    } else {
                        self.apply_kernel(&|arr| {
                            let bitmap = if let Some(validity) = arr.validity() {
                                arr.values() ^ validity
                            } else {
                                arr.values().not()
                            };
                            Box::new(BooleanArray::from_data_default(bitmap, None)) as ArrayRef
                        })
                    }
                } else {
                    self.is_null()
                }
            }
            (1, _) => rhs.equal_missing(self),
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
                    if value {
                        self.not()
                    } else {
                        self.clone()
                    }
                } else {
                    BooleanChunked::full_null("", self.len())
                }
            }
            (1, _) => rhs.not_equal(self),
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                compare_bools(&lhs, &rhs, |lhs, rhs| comparison::neq(lhs, rhs))
            }
        }
    }

    fn not_equal_missing(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    if value {
                        self.not()
                    } else {
                        self.clone()
                    }
                } else {
                    self.is_not_null()
                }
            }
            (1, _) => rhs.not_equal_missing(self),
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
                    BooleanChunked::full_null("", self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    match value {
                        true => rhs.not(),
                        false => BooleanChunked::full("", false, rhs.len()),
                    }
                } else {
                    BooleanChunked::full_null("", rhs.len())
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
                    BooleanChunked::full_null("", self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    match value {
                        true => BooleanChunked::full("", true, rhs.len()),
                        false => rhs.not(),
                    }
                } else {
                    BooleanChunked::full_null("", rhs.len())
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
                    BooleanChunked::full_null("", self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    match value {
                        true => BooleanChunked::full("", false, rhs.len()),
                        false => rhs.clone(),
                    }
                } else {
                    BooleanChunked::full_null("", rhs.len())
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
                    BooleanChunked::full_null("", self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    match value {
                        true => rhs.clone(),
                        false => BooleanChunked::full("", true, rhs.len()),
                    }
                } else {
                    BooleanChunked::full_null("", rhs.len())
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

impl ChunkCompare<&Utf8Chunked> for Utf8Chunked {
    type Item = BooleanChunked;

    fn equal(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        self.as_binary().equal(&rhs.as_binary())
    }

    fn equal_missing(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        self.as_binary().equal_missing(&rhs.as_binary())
    }

    fn not_equal(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        self.as_binary().not_equal(&rhs.as_binary())
    }
    fn not_equal_missing(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        self.as_binary().not_equal_missing(&rhs.as_binary())
    }

    fn gt(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        self.as_binary().gt(&rhs.as_binary())
    }

    fn gt_eq(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        self.as_binary().gt_eq(&rhs.as_binary())
    }

    fn lt(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        self.as_binary().lt(&rhs.as_binary())
    }

    fn lt_eq(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        self.as_binary().lt_eq(&rhs.as_binary())
    }
}

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

impl ChunkCompare<&BinaryChunked> for BinaryChunked {
    type Item = BooleanChunked;

    fn equal(&self, rhs: &BinaryChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.equal(value)
            } else {
                BooleanChunked::full_null("", self.len())
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.equal(value)
            } else {
                BooleanChunked::full_null("", rhs.len())
            }
        } else {
            let (lhs, rhs) = align_chunks_binary(self, rhs);
            lhs.comparison(&rhs, comparison::binary::eq)
        }
    }

    fn equal_missing(&self, rhs: &BinaryChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.equal_missing(value)
            } else {
                self.is_null()
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.equal_missing(value)
            } else {
                rhs.is_null()
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
                BooleanChunked::full_null("", self.len())
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.not_equal(value)
            } else {
                BooleanChunked::full_null("", rhs.len())
            }
        } else {
            let (lhs, rhs) = align_chunks_binary(self, rhs);
            lhs.comparison(&rhs, comparison::binary::neq)
        }
    }

    fn not_equal_missing(&self, rhs: &BinaryChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.not_equal_missing(value)
            } else {
                self.is_not_null()
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.not_equal_missing(value)
            } else {
                rhs.is_not_null()
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
                BooleanChunked::full_null("", self.len())
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.lt(value)
            } else {
                BooleanChunked::full_null("", self.len())
            }
        } else {
            let (lhs, rhs) = align_chunks_binary(self, rhs);
            lhs.comparison(&rhs, |l, r| comparison::gt(l, r))
        }
    }

    fn gt_eq(&self, rhs: &BinaryChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.gt_eq(value)
            } else {
                BooleanChunked::full_null("", self.len())
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.lt_eq(value)
            } else {
                BooleanChunked::full_null("", self.len())
            }
        } else {
            let (lhs, rhs) = align_chunks_binary(self, rhs);
            lhs.comparison(&rhs, |l, r| comparison::gt_eq(l, r))
        }
    }

    fn lt(&self, rhs: &BinaryChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.lt(value)
            } else {
                BooleanChunked::full_null("", self.len())
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.gt(value)
            } else {
                BooleanChunked::full_null("", self.len())
            }
        } else {
            let (lhs, rhs) = align_chunks_binary(self, rhs);
            lhs.comparison(&rhs, |l, r| comparison::lt(l, r))
        }
    }

    fn lt_eq(&self, rhs: &BinaryChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.lt_eq(value)
            } else {
                BooleanChunked::full_null("", self.len())
            }
        } else if self.len() == 1 {
            if let Some(value) = self.get(0) {
                rhs.gt_eq(value)
            } else {
                BooleanChunked::full_null("", self.len())
            }
        } else {
            let (lhs, rhs) = align_chunks_binary(self, rhs);
            lhs.comparison(&rhs, |l, r| comparison::lt_eq(l, r))
        }
    }
}

impl ChunkCompare<&str> for Utf8Chunked {
    type Item = BooleanChunked;
    fn equal(&self, rhs: &str) -> BooleanChunked {
        self.utf8_compare_scalar(rhs, |l, rhs| comparison::eq_scalar(l, rhs))
    }

    fn equal_missing(&self, rhs: &str) -> BooleanChunked {
        self.utf8_compare_scalar(rhs, |l, rhs| comparison::eq_scalar_and_validity(l, rhs))
    }

    fn not_equal(&self, rhs: &str) -> BooleanChunked {
        self.utf8_compare_scalar(rhs, |l, rhs| comparison::neq_scalar(l, rhs))
    }

    fn not_equal_missing(&self, rhs: &str) -> BooleanChunked {
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

impl ChunkCompare<&ListChunked> for ListChunked {
    type Item = BooleanChunked;
    fn equal(&self, rhs: &ListChunked) -> BooleanChunked {
        self.amortized_iter()
            .zip(rhs.amortized_iter())
            .map(|(left, right)| match (left, right) {
                (Some(l), Some(r)) => Some(l.as_ref().series_equal_missing(r.as_ref())),
                _ => None,
            })
            .collect_trusted()
    }

    fn equal_missing(&self, rhs: &ListChunked) -> BooleanChunked {
        self.amortized_iter()
            .zip(rhs.amortized_iter())
            .map(|(left, right)| match (left, right) {
                (Some(l), Some(r)) => l.as_ref().series_equal_missing(r.as_ref()),
                (None, None) => true,
                _ => false,
            })
            .collect_trusted()
    }

    fn not_equal(&self, rhs: &ListChunked) -> BooleanChunked {
        self.amortized_iter()
            .zip(rhs.amortized_iter())
            .map(|(left, right)| match (left, right) {
                (Some(l), Some(r)) => Some(!l.as_ref().series_equal_missing(r.as_ref())),
                _ => None,
            })
            .collect_trusted()
    }

    fn not_equal_missing(&self, rhs: &ListChunked) -> BooleanChunked {
        self.amortized_iter()
            .zip(rhs.amortized_iter())
            .map(|(left, right)| match (left, right) {
                (Some(l), Some(r)) => !l.as_ref().series_equal_missing(r.as_ref()),
                (None, None) => false,
                _ => true,
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

    fn equal_missing(&self, rhs: &StructChunked) -> BooleanChunked {
        use std::ops::BitAnd;
        if self.len() != rhs.len() || self.fields().len() != rhs.fields().len() {
            BooleanChunked::full("", false, self.len())
        } else {
            self.fields()
                .iter()
                .zip(rhs.fields().iter())
                .map(|(l, r)| l.equal_missing(r).unwrap())
                .reduce(|lhs, rhs| lhs.bitand(rhs))
                .unwrap()
        }
    }

    fn not_equal(&self, rhs: &StructChunked) -> BooleanChunked {
        use std::ops::BitOr;
        if self.len() != rhs.len() || self.fields().len() != rhs.fields().len() {
            BooleanChunked::full("", true, self.len())
        } else {
            self.fields()
                .iter()
                .zip(rhs.fields().iter())
                .map(|(l, r)| l.not_equal(r).unwrap())
                .reduce(|lhs, rhs| lhs.bitor(rhs))
                .unwrap()
        }
    }

    fn not_equal_missing(&self, rhs: &StructChunked) -> BooleanChunked {
        use std::ops::BitOr;
        if self.len() != rhs.len() || self.fields().len() != rhs.fields().len() {
            BooleanChunked::full("", true, self.len())
        } else {
            self.fields()
                .iter()
                .zip(rhs.fields().iter())
                .map(|(l, r)| l.not_equal_missing(r).unwrap())
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

#[cfg(feature = "dtype-array")]
impl ChunkCompare<&ArrayChunked> for ArrayChunked {
    type Item = BooleanChunked;
    fn equal(&self, rhs: &ArrayChunked) -> BooleanChunked {
        let (a, b) = align_chunks_binary(self, rhs);
        let chunks = a
            .downcast_iter()
            .zip(b.downcast_iter())
            .map(|(a, b)| {
                Box::new(polars_arrow::kernels::comparison::fixed_size_list_eq(a, b)) as ArrayRef
            })
            .collect::<Vec<_>>();
        unsafe { BooleanChunked::from_chunks(self.name(), chunks) }
    }

    fn equal_missing(&self, rhs: &ArrayChunked) -> BooleanChunked {
        // TODO!: maybe do something else here
        self.equal(rhs)
    }

    fn not_equal(&self, rhs: &ArrayChunked) -> BooleanChunked {
        let (a, b) = align_chunks_binary(self, rhs);
        let chunks = a
            .downcast_iter()
            .zip(b.downcast_iter())
            .map(|(a, b)| {
                Box::new(polars_arrow::kernels::comparison::fixed_size_list_neq(a, b)) as ArrayRef
            })
            .collect::<Vec<_>>();
        unsafe { BooleanChunked::from_chunks(self.name(), chunks) }
    }

    fn not_equal_missing(&self, rhs: &ArrayChunked) -> Self::Item {
        // TODO!: maybe do something else here
        self.not_equal(rhs)
    }

    // following are not implemented because gt, lt comparison of series don't make sense
    fn gt(&self, _rhs: &ArrayChunked) -> BooleanChunked {
        unimplemented!()
    }

    fn gt_eq(&self, _rhs: &ArrayChunked) -> BooleanChunked {
        unimplemented!()
    }

    fn lt(&self, _rhs: &ArrayChunked) -> BooleanChunked {
        unimplemented!()
    }

    fn lt_eq(&self, _rhs: &ArrayChunked) -> BooleanChunked {
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

impl ChunkEqualElement for BinaryChunked {
    unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
        let ca_other = other.as_ref().as_ref();
        debug_assert!(self.dtype() == other.dtype());
        let ca_other = &*(ca_other as *const BinaryChunked);
        self.get(idx_self) == ca_other.get(idx_other)
    }
}

impl ChunkEqualElement for ListChunked {}
#[cfg(feature = "dtype-array")]
impl ChunkEqualElement for ArrayChunked {}

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
        let a1: Int32Chunked = [Some(1), None, Some(3)].iter().copied().collect();
        let a2: Int32Chunked = [Some(1), Some(2), Some(3)].iter().copied().collect();

        let mut a2_2chunks: Int32Chunked = [Some(1), Some(2)].iter().copied().collect();
        a2_2chunks.append(&[Some(3)].iter().copied().collect());

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
        let a1: Int32Chunked = [Some(1), Some(2)].iter().copied().collect();
        let a1 = a1.slice(1, 1);
        let a2: Int32Chunked = [Some(2)].iter().copied().collect();
        assert_eq!(a1.equal(&a2).sum(), a2.equal(&a1).sum());
        assert_eq!(a1.not_equal(&a2).sum(), a2.not_equal(&a1).sum());
        assert_eq!(a1.gt(&a2).sum(), a2.gt(&a1).sum());
        assert_eq!(a1.lt(&a2).sum(), a2.lt(&a1).sum());
        assert_eq!(a1.lt_eq(&a2).sum(), a2.lt_eq(&a1).sum());
        assert_eq!(a1.gt_eq(&a2).sum(), a2.gt_eq(&a1).sum());

        let a1: Utf8Chunked = ["a", "b"].iter().copied().collect();
        let a1 = a1.slice(1, 1);
        let a2: Utf8Chunked = ["b"].iter().copied().collect();
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
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), None]);
        let out = a.not_equal(&true_);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), None]);

        let out = a.equal(&all_true);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), None]);
        let out = a.not_equal(&all_true);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), None]);
        let out = a.equal(&false_);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), None]);
        let out = a.not_equal(&false_);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), None]);
        let out = a.equal(&all_false);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), None]);
        let out = a.not_equal(&all_false);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), None]);
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
