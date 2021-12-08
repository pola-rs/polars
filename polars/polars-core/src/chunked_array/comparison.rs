use crate::utils::align_chunks_binary;
use crate::{prelude::*, utils::NoNull};
use arrow::scalar::{PrimitiveScalar, Scalar, Utf8Scalar};
use arrow::{
    array::{ArrayRef, BooleanArray, PrimitiveArray, Utf8Array},
    compute,
    compute::comparison,
};
use num::{NumCast, ToPrimitive};
use std::ops::Not;
use std::sync::Arc;

type LargeStringArray = Utf8Array<i64>;

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
                Arc::new(arr) as ArrayRef
            })
            .collect::<Vec<_>>();

        ChunkedArray::new_from_chunks("", chunks)
    }
}

macro_rules! impl_eq_missing {
    ($self:ident, $rhs:ident) => {{
        match ($self.has_validity(), $rhs.has_validity()) {
            (false, false) => $self
                .into_no_null_iter()
                .zip($rhs.into_no_null_iter())
                .map(|(opt_a, opt_b)| opt_a == opt_b)
                .collect(),
            (_, _) => $self
                .into_iter()
                .zip($rhs)
                .map(|(opt_a, opt_b)| opt_a == opt_b)
                .collect(),
        }
    }};
}

impl<T> ChunkCompare<&ChunkedArray<T>> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn eq_missing(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        impl_eq_missing!(self, rhs)
    }

    fn equal(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    self.equal(value)
                } else {
                    BooleanChunked::full("", false, self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    rhs.equal(value)
                } else {
                    BooleanChunked::full("", false, rhs.len())
                }
            }
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                lhs.comparison(&rhs, |x, y| comparison::eq(x, y))
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
                    BooleanChunked::full("", false, self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    rhs.not_equal(value)
                } else {
                    BooleanChunked::full("", false, rhs.len())
                }
            }
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                lhs.comparison(&rhs, |x, y| comparison::neq(x, y))
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
        .map(|(l, r)| Arc::new(f(l, r)) as ArrayRef)
        .collect();

    BooleanChunked::new_from_chunks(lhs.name(), chunks)
}

impl ChunkCompare<&BooleanChunked> for BooleanChunked {
    fn eq_missing(&self, rhs: &BooleanChunked) -> BooleanChunked {
        impl_eq_missing!(self, rhs)
    }

    fn equal(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    match value {
                        true => self.clone(),
                        false => self.not(),
                    }
                } else {
                    BooleanChunked::full("", false, self.len())
                }
            }
            (1, _) => {
                if let Some(value) = self.get(0) {
                    match value {
                        true => rhs.clone(),
                        false => rhs.not(),
                    }
                } else {
                    BooleanChunked::full("", false, rhs.len())
                }
            }
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                compare_bools(&lhs, &rhs, |lhs, rhs| comparison::eq(lhs, rhs))
            }
        }
    }

    fn not_equal(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        match (self.len(), rhs.len()) {
            (_, 1) => {
                if let Some(value) = rhs.get(0) {
                    match value {
                        true => self.not(),
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
                        false => rhs.clone(),
                    }
                } else {
                    BooleanChunked::full("", false, rhs.len())
                }
            }
            _ => {
                // same length
                let (lhs, rhs) = align_chunks_binary(self, rhs);
                compare_bools(&lhs, &rhs, |lhs, rhs| comparison::neq(lhs, rhs))
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
            .chunks
            .iter()
            .zip(&rhs.chunks)
            .map(|(left, right)| {
                let left = left
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .expect("could not downcast one of the chunks");
                let right = right
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .expect("could not downcast one of the chunks");
                let arr = f(left, right);
                Arc::new(arr) as ArrayRef
            })
            .collect::<Vec<_>>();

        ChunkedArray::new_from_chunks("", chunks)
    }
}

impl ChunkCompare<&Utf8Chunked> for Utf8Chunked {
    fn eq_missing(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        impl_eq_missing!(self, rhs)
    }

    fn equal(&self, rhs: &Utf8Chunked) -> BooleanChunked {
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
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, |l, r| comparison::eq(l, r))
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, ==)
        }
    }

    fn not_equal(&self, rhs: &Utf8Chunked) -> BooleanChunked {
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
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, |l, r| comparison::neq(l, r))
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, !=)
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
        self.apply_kernel_cast(|arr| Arc::new(f(arr, &scalar)))
    }
}

impl<T, Rhs> ChunkCompare<Rhs> for ChunkedArray<T>
where
    T: PolarsNumericType,
    Rhs: ToPrimitive,
{
    fn eq_missing(&self, rhs: Rhs) -> BooleanChunked {
        self.equal(rhs)
    }

    fn equal(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, |l, rhs| comparison::eq_scalar(l, rhs))
    }

    fn not_equal(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, |l, rhs| comparison::neq_scalar(l, rhs))
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
        self.apply_kernel_cast(|arr| Arc::new(f(arr, &scalar)))
    }
}

impl ChunkCompare<&str> for Utf8Chunked {
    fn eq_missing(&self, rhs: &str) -> BooleanChunked {
        self.equal(rhs)
    }

    fn equal(&self, rhs: &str) -> BooleanChunked {
        self.utf8_compare_scalar(rhs, |l, rhs| comparison::eq_scalar(l, rhs))
    }
    fn not_equal(&self, rhs: &str) -> BooleanChunked {
        self.utf8_compare_scalar(rhs, |l, rhs| comparison::neq_scalar(l, rhs))
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

macro_rules! impl_cmp_list {
    ($self:ident, $rhs:ident, $cmp_method:ident) => {{
        match ($self.has_validity(), $rhs.has_validity()) {
            (false, false) => $self
                .into_no_null_iter()
                .zip($rhs.into_no_null_iter())
                .map(|(left, right)| left.$cmp_method(&right))
                .collect_trusted(),
            (false, _) => $self
                .into_no_null_iter()
                .zip($rhs.into_iter())
                .map(|(left, opt_right)| opt_right.map(|right| left.$cmp_method(&right)))
                .collect_trusted(),
            (_, false) => $self
                .into_iter()
                .zip($rhs.into_no_null_iter())
                .map(|(opt_left, right)| opt_left.map(|left| left.$cmp_method(&right)))
                .collect_trusted(),
            (_, _) => $self
                .into_iter()
                .zip($rhs.into_iter())
                .map(|(opt_left, opt_right)| match (opt_left, opt_right) {
                    (None, None) => None,
                    (None, Some(_)) => None,
                    (Some(_), None) => None,
                    (Some(left), Some(right)) => Some(left.$cmp_method(&right)),
                })
                .collect(),
        }
    }};
}

impl ChunkCompare<&ListChunked> for ListChunked {
    fn eq_missing(&self, rhs: &ListChunked) -> BooleanChunked {
        impl_cmp_list!(self, rhs, series_equal_missing)
    }

    fn equal(&self, rhs: &ListChunked) -> BooleanChunked {
        impl_cmp_list!(self, rhs, series_equal)
    }

    fn not_equal(&self, rhs: &ListChunked) -> BooleanChunked {
        self.equal(rhs).not()
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
                Arc::new(arr) as ArrayRef
            })
            .collect::<Vec<_>>();
        ChunkedArray::new_from_chunks(self.name(), chunks)
    }
}

impl Not for BooleanChunked {
    type Output = BooleanChunked;

    fn not(self) -> Self::Output {
        (&self).not()
    }
}

impl BooleanChunked {
    /// Check if all values are true
    pub fn all_true(&self) -> bool {
        match self.sum() {
            None => false,
            Some(n) => (n as usize) == self.len(),
        }
    }
}

impl BooleanChunked {
    /// Check if all values are false
    pub fn all_false(&self) -> bool {
        match self.sum() {
            None => false,
            Some(n) => (n as usize) == 0,
        }
    }
}

// private
pub(crate) trait ChunkEqualElement {
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

impl ChunkEqualElement for ListChunked {}
#[cfg(feature = "dtype-categorical")]
impl ChunkEqualElement for CategoricalChunked {
    unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
        let ca_other = other.as_ref().as_ref();
        debug_assert!(self.dtype() == other.dtype());
        let ca_other = &*(ca_other as *const CategoricalChunked);
        self.get(idx_self) == ca_other.get(idx_other)
    }
}

#[cfg(test)]
mod test {
    use super::super::{arithmetic::test::create_two_chunked, test::get_chunked_array};
    use crate::prelude::*;
    use itertools::Itertools;
    use std::iter::repeat;

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
            a1.equal(&a2).into_iter().collect_vec(),
            repeat(Some(true)).take(6).collect_vec()
        );
        assert_eq!(
            a2.equal(&a1).into_iter().collect_vec(),
            repeat(Some(true)).take(6).collect_vec()
        );
        assert_eq!(
            a1.not_equal(&a2).into_iter().collect_vec(),
            repeat(Some(false)).take(6).collect_vec()
        );
        assert_eq!(
            a2.not_equal(&a1).into_iter().collect_vec(),
            repeat(Some(false)).take(6).collect_vec()
        );
        assert_eq!(
            a1.gt(&a2).into_iter().collect_vec(),
            repeat(Some(false)).take(6).collect_vec()
        );
        assert_eq!(
            a2.gt(&a1).into_iter().collect_vec(),
            repeat(Some(false)).take(6).collect_vec()
        );
        assert_eq!(
            a1.gt_eq(&a2).into_iter().collect_vec(),
            repeat(Some(true)).take(6).collect_vec()
        );
        assert_eq!(
            a2.gt_eq(&a1).into_iter().collect_vec(),
            repeat(Some(true)).take(6).collect_vec()
        );
        assert_eq!(
            a1.lt_eq(&a2).into_iter().collect_vec(),
            repeat(Some(true)).take(6).collect_vec()
        );
        assert_eq!(
            a2.lt_eq(&a1).into_iter().collect_vec(),
            repeat(Some(true)).take(6).collect_vec()
        );
        assert_eq!(
            a1.lt(&a2).into_iter().collect_vec(),
            repeat(Some(false)).take(6).collect_vec()
        );
        assert_eq!(
            a2.lt(&a1).into_iter().collect_vec(),
            repeat(Some(false)).take(6).collect_vec()
        );
    }

    #[test]
    fn test_equal_chunks() {
        let a1 = get_chunked_array();
        let a2 = get_chunked_array();

        assert_eq!(
            a1.equal(&a2).into_iter().collect_vec(),
            repeat(Some(true)).take(3).collect_vec()
        );
        assert_eq!(
            a2.equal(&a1).into_iter().collect_vec(),
            repeat(Some(true)).take(3).collect_vec()
        );
        assert_eq!(
            a1.not_equal(&a2).into_iter().collect_vec(),
            repeat(Some(false)).take(3).collect_vec()
        );
        assert_eq!(
            a2.not_equal(&a1).into_iter().collect_vec(),
            repeat(Some(false)).take(3).collect_vec()
        );
        assert_eq!(
            a1.gt(&a2).into_iter().collect_vec(),
            repeat(Some(false)).take(3).collect_vec()
        );
        assert_eq!(
            a2.gt(&a1).into_iter().collect_vec(),
            repeat(Some(false)).take(3).collect_vec()
        );
        assert_eq!(
            a1.gt_eq(&a2).into_iter().collect_vec(),
            repeat(Some(true)).take(3).collect_vec()
        );
        assert_eq!(
            a2.gt_eq(&a1).into_iter().collect_vec(),
            repeat(Some(true)).take(3).collect_vec()
        );
        assert_eq!(
            a1.lt_eq(&a2).into_iter().collect_vec(),
            repeat(Some(true)).take(3).collect_vec()
        );
        assert_eq!(
            a2.lt_eq(&a1).into_iter().collect_vec(),
            repeat(Some(true)).take(3).collect_vec()
        );
        assert_eq!(
            a1.lt(&a2).into_iter().collect_vec(),
            repeat(Some(false)).take(3).collect_vec()
        );
        assert_eq!(
            a2.lt(&a1).into_iter().collect_vec(),
            repeat(Some(false)).take(3).collect_vec()
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
            a1.equal(&a2).into_iter().collect_vec(),
            a1.equal(&a2_2chunks).into_iter().collect_vec()
        );

        assert_eq!(
            a1.not_equal(&a2).into_iter().collect_vec(),
            a1.not_equal(&a2_2chunks).into_iter().collect_vec()
        );
        assert_eq!(
            a1.not_equal(&a2).into_iter().collect_vec(),
            a2_2chunks.not_equal(&a1).into_iter().collect_vec()
        );

        assert_eq!(
            a1.gt(&a2).into_iter().collect_vec(),
            a1.gt(&a2_2chunks).into_iter().collect_vec()
        );
        assert_eq!(
            a1.gt(&a2).into_iter().collect_vec(),
            a2_2chunks.gt(&a1).into_iter().collect_vec()
        );

        assert_eq!(
            a1.gt_eq(&a2).into_iter().collect_vec(),
            a1.gt_eq(&a2_2chunks).into_iter().collect_vec()
        );
        assert_eq!(
            a1.gt_eq(&a2).into_iter().collect_vec(),
            a2_2chunks.gt_eq(&a1).into_iter().collect_vec()
        );

        assert_eq!(
            a1.lt_eq(&a2).into_iter().collect_vec(),
            a1.lt_eq(&a2_2chunks).into_iter().collect_vec()
        );
        assert_eq!(
            a1.lt_eq(&a2).into_iter().collect_vec(),
            a2_2chunks.lt_eq(&a1).into_iter().collect_vec()
        );

        assert_eq!(
            a1.lt(&a2).into_iter().collect_vec(),
            a1.lt(&a2_2chunks).into_iter().collect_vec()
        );
        assert_eq!(
            a1.lt(&a2).into_iter().collect_vec(),
            a2_2chunks.lt(&a1).into_iter().collect_vec()
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
        let trues = BooleanChunked::new_from_slice("", &[true, true, true]);
        let falses = BooleanChunked::new_from_slice("", &[false, false, false]);

        let c = &a | &trues;
        assert_eq!(Vec::from(&c), &[Some(true), Some(true), Some(true)]);

        let c = &a | &falses;
        assert_eq!(Vec::from(&c), &[Some(true), Some(false), None])
    }

    #[test]
    fn test_broadcasting_bools() {
        let a = BooleanChunked::new_from_slice("", &[true, false, true]);
        let true_ = BooleanChunked::new_from_slice("", &[true]);
        let false_ = BooleanChunked::new_from_slice("", &[false]);

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
    }

    #[test]
    fn test_broadcasting_numeric() {
        let a = Int32Chunked::new_from_slice("", &[1, 2, 3]);
        let one = Int32Chunked::new_from_slice("", &[1]);
        let three = Int32Chunked::new_from_slice("", &[3]);

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
