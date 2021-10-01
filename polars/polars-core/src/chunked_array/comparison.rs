use crate::{prelude::*, utils::NoNull};
use arrow::scalar::Utf8Scalar;
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
        operator: impl Fn(
            &PrimitiveArray<T::Native>,
            &PrimitiveArray<T::Native>,
        ) -> arrow::error::Result<BooleanArray>,
    ) -> Result<BooleanChunked> {
        let chunks = self
            .downcast_iter()
            .zip(rhs.downcast_iter())
            .map(|(left, right)| {
                let arr_res = operator(left, right);
                let arr = match arr_res {
                    Ok(arr) => arr,
                    Err(e) => return Err(PolarsError::ArrowError(e)),
                };
                Ok(Arc::new(arr) as ArrayRef)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(ChunkedArray::new_from_chunks("", chunks))
    }
}

macro_rules! impl_eq_missing {
    ($self:ident, $rhs:ident) => {{
        match ($self.null_count(), $rhs.null_count()) {
            (0, 0) => $self
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

    fn eq(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.eq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            // should not fail if arrays are equal
            self.comparison(rhs, |x, y| {
                comparison::compare(x, y, comparison::Operator::Eq)
            })
            .expect("should not fail.")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, ==)
        }
    }

    fn neq(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.neq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, |x, y| {
                comparison::compare(x, y, comparison::Operator::Neq)
            })
            .expect("should not fail.")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, !=)
        }
    }

    fn gt(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.gt(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, |x, y| {
                comparison::compare(x, y, comparison::Operator::Gt)
            })
            .expect("should not fail.")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, >)
        }
    }

    fn gt_eq(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.gt_eq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, |x, y| {
                comparison::compare(x, y, comparison::Operator::GtEq)
            })
            .expect("should not fail.")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, >=)
        }
    }

    fn lt(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.lt(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, |x, y| {
                comparison::compare(x, y, comparison::Operator::Lt)
            })
            .expect("should not fail.")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, <)
        }
    }

    fn lt_eq(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.lt_eq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, |x, y| {
                comparison::compare(x, y, comparison::Operator::LtEq)
            })
            .expect("should not fail.")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, <=)
        }
    }
}

impl ChunkCompare<&BooleanChunked> for BooleanChunked {
    fn eq_missing(&self, rhs: &BooleanChunked) -> BooleanChunked {
        impl_eq_missing!(self, rhs)
    }

    fn eq(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                match value {
                    true => self.clone(),
                    false => self.not(),
                }
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, ==)
        }
    }

    fn neq(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                match value {
                    true => self.not(),
                    false => self.clone(),
                }
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, !=)
        }
    }

    fn gt(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                match value {
                    true => BooleanChunked::full("", false, self.len()),
                    false => self.clone(),
                }
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, >)
        }
    }

    fn gt_eq(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                match value {
                    true => self.clone(),
                    false => BooleanChunked::full("", true, self.len()),
                }
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, >=)
        }
    }

    fn lt(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                match value {
                    true => self.not(),
                    false => BooleanChunked::full("", false, self.len()),
                }
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, <)
        }
    }

    fn lt_eq(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                match value {
                    true => BooleanChunked::full("", true, self.len()),
                    false => self.not(),
                }
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, <=)
        }
    }
}

impl Utf8Chunked {
    fn comparison(
        &self,
        rhs: &Utf8Chunked,
        operator: comparison::Operator,
    ) -> Result<BooleanChunked> {
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
                let arr_res = comparison::compare(left, right, operator);
                let arr = match arr_res {
                    Ok(arr) => arr,
                    Err(e) => return Err(PolarsError::ArrowError(e)),
                };
                Ok(Arc::new(arr) as ArrayRef)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(ChunkedArray::new_from_chunks("", chunks))
    }
}

impl ChunkCompare<&Utf8Chunked> for Utf8Chunked {
    fn eq_missing(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        impl_eq_missing!(self, rhs)
    }

    fn eq(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.eq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, comparison::Operator::Eq)
                .expect("should not fail")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, ==)
        }
    }

    fn neq(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.neq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, comparison::Operator::Neq)
                .expect("should not fail")
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
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, comparison::Operator::Gt)
                .expect("should not fail")
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
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, comparison::Operator::GtEq)
                .expect("should not fail")
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
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, comparison::Operator::Lt)
                .expect("should not fail")
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
        }
        // same length
        else if self.chunk_id().zip(rhs.chunk_id()).all(|(l, r)| l == r) {
            self.comparison(rhs, comparison::Operator::LtEq)
                .expect("should not fail")
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
        op: comparison::Operator,
    ) -> BooleanChunked {
        let rhs: T::Native =
            NumCast::from(rhs).expect("could not cast to underlying chunkedarray type");
        self.apply_kernel_cast(|arr| {
            Arc::new(comparison::primitive_compare_scalar(
                arr,
                &Some(rhs).into(),
                op,
            ))
        })
    }
}

impl<T, Rhs> ChunkCompare<Rhs> for ChunkedArray<T>
where
    T: PolarsNumericType,
    Rhs: ToPrimitive,
{
    fn eq_missing(&self, rhs: Rhs) -> BooleanChunked {
        self.eq(rhs)
    }

    fn eq(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, comparison::Operator::Eq)
    }

    fn neq(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, comparison::Operator::Neq)
    }

    fn gt(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, comparison::Operator::Gt)
    }

    fn gt_eq(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, comparison::Operator::GtEq)
    }

    fn lt(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, comparison::Operator::Lt)
    }

    fn lt_eq(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, comparison::Operator::LtEq)
    }
}

impl Utf8Chunked {
    fn utf8_compare_scalar(&self, rhs: &str, op: comparison::Operator) -> BooleanChunked {
        self.apply_kernel_cast(|arr| {
            Arc::new(comparison::utf8_compare_scalar(
                arr,
                &Utf8Scalar::<i64>::new(Some(rhs)),
                op,
            ))
        })
    }
}

impl ChunkCompare<&str> for Utf8Chunked {
    fn eq_missing(&self, rhs: &str) -> BooleanChunked {
        self.eq(rhs)
    }

    fn eq(&self, rhs: &str) -> BooleanChunked {
        self.utf8_compare_scalar(rhs, comparison::Operator::Eq)
    }
    fn neq(&self, rhs: &str) -> BooleanChunked {
        self.utf8_compare_scalar(rhs, comparison::Operator::Neq)
    }

    fn gt(&self, rhs: &str) -> BooleanChunked {
        self.utf8_compare_scalar(rhs, comparison::Operator::Gt)
    }

    fn gt_eq(&self, rhs: &str) -> BooleanChunked {
        self.utf8_compare_scalar(rhs, comparison::Operator::GtEq)
    }

    fn lt(&self, rhs: &str) -> BooleanChunked {
        self.utf8_compare_scalar(rhs, comparison::Operator::Lt)
    }

    fn lt_eq(&self, rhs: &str) -> BooleanChunked {
        self.utf8_compare_scalar(rhs, comparison::Operator::LtEq)
    }
}

macro_rules! impl_cmp_list {
    ($self:ident, $rhs:ident, $cmp_method:ident) => {{
        match ($self.null_count(), $rhs.null_count()) {
            (0, 0) => $self
                .into_no_null_iter()
                .zip($rhs.into_no_null_iter())
                .map(|(left, right)| left.$cmp_method(&right))
                .collect(),
            (0, _) => $self
                .into_no_null_iter()
                .zip($rhs.into_iter())
                .map(|(left, opt_right)| opt_right.map(|right| left.$cmp_method(&right)))
                .collect(),
            (_, 0) => $self
                .into_iter()
                .zip($rhs.into_no_null_iter())
                .map(|(opt_left, right)| opt_left.map(|left| left.$cmp_method(&right)))
                .collect(),
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

    fn eq(&self, rhs: &ListChunked) -> BooleanChunked {
        impl_cmp_list!(self, rhs, series_equal)
    }

    fn neq(&self, rhs: &ListChunked) -> BooleanChunked {
        self.eq(rhs).not()
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
        let a = BooleanChunked::new_from_slice("a", &[true, false, false]);
        let b = BooleanChunked::new_from_opt_slice("b", &[Some(true), Some(true), None]);
        assert_eq!(Vec::from(&a | &b), &[Some(true), Some(true), None]);
        assert_eq!(Vec::from(&a & &b), &[Some(true), Some(false), Some(false)]);
        assert_eq!(Vec::from(!b), &[Some(false), Some(false), None]);
    }

    #[test]
    fn test_compare_chunk_diff() {
        let (a1, a2) = create_two_chunked();

        assert_eq!(
            a1.eq(&a2).into_iter().collect_vec(),
            repeat(Some(true)).take(6).collect_vec()
        );
        assert_eq!(
            a2.eq(&a1).into_iter().collect_vec(),
            repeat(Some(true)).take(6).collect_vec()
        );
        assert_eq!(
            a1.neq(&a2).into_iter().collect_vec(),
            repeat(Some(false)).take(6).collect_vec()
        );
        assert_eq!(
            a2.neq(&a1).into_iter().collect_vec(),
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
            a1.eq(&a2).into_iter().collect_vec(),
            repeat(Some(true)).take(3).collect_vec()
        );
        assert_eq!(
            a2.eq(&a1).into_iter().collect_vec(),
            repeat(Some(true)).take(3).collect_vec()
        );
        assert_eq!(
            a1.neq(&a2).into_iter().collect_vec(),
            repeat(Some(false)).take(3).collect_vec()
        );
        assert_eq!(
            a2.neq(&a1).into_iter().collect_vec(),
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
            a1.eq(&a2).into_iter().collect_vec(),
            a1.eq(&a2_2chunks).into_iter().collect_vec()
        );

        assert_eq!(
            a1.neq(&a2).into_iter().collect_vec(),
            a1.neq(&a2_2chunks).into_iter().collect_vec()
        );
        assert_eq!(
            a1.neq(&a2).into_iter().collect_vec(),
            a2_2chunks.neq(&a1).into_iter().collect_vec()
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
        assert_eq!(a1.eq(&a2).sum(), a2.eq(&a1).sum());
        assert_eq!(a1.neq(&a2).sum(), a2.neq(&a1).sum());
        assert_eq!(a1.gt(&a2).sum(), a2.gt(&a1).sum());
        assert_eq!(a1.lt(&a2).sum(), a2.lt(&a1).sum());
        assert_eq!(a1.lt_eq(&a2).sum(), a2.lt_eq(&a1).sum());
        assert_eq!(a1.gt_eq(&a2).sum(), a2.gt_eq(&a1).sum());

        let a1: Utf8Chunked = (&["a", "b"]).iter().copied().collect();
        let a1 = a1.slice(1, 1);
        let a2: Utf8Chunked = (&["b"]).iter().copied().collect();
        assert_eq!(a1.eq(&a2).sum(), a2.eq(&a1).sum());
        assert_eq!(a1.neq(&a2).sum(), a2.neq(&a1).sum());
        assert_eq!(a1.gt(&a2).sum(), a2.gt(&a1).sum());
        assert_eq!(a1.lt(&a2).sum(), a2.lt(&a1).sum());
        assert_eq!(a1.lt_eq(&a2).sum(), a2.lt_eq(&a1).sum());
        assert_eq!(a1.gt_eq(&a2).sum(), a2.gt_eq(&a1).sum());
    }

    #[test]
    fn test_kleene() {
        let a = BooleanChunked::new_from_opt_slice("", &[Some(true), Some(false), None]);
        let trues = BooleanChunked::new_from_slice("", &[true, true, true]);
        let falses = BooleanChunked::new_from_slice("", &[false, false, false]);

        let c = &a | &trues;
        assert_eq!(Vec::from(&c), &[Some(true), Some(true), Some(true)]);

        let c = &a | &falses;
        assert_eq!(Vec::from(&c), &[Some(true), Some(false), None])
    }
}
