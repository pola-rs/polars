//! Comparison functions for [`BinaryArray`]
use crate::compute::comparison::{finish_eq_validities, finish_neq_validities};
use crate::{
    array::{BinaryArray, BooleanArray},
    bitmap::Bitmap,
    datatypes::DataType,
    offset::Offset,
};

use super::super::utils::combine_validities;

/// Evaluate `op(lhs, rhs)` for [`BinaryArray`]s using a specified
/// comparison function.
fn compare_op<O, F>(lhs: &BinaryArray<O>, rhs: &BinaryArray<O>, op: F) -> BooleanArray
where
    O: Offset,
    F: Fn(&[u8], &[u8]) -> bool,
{
    assert_eq!(lhs.len(), rhs.len());

    let validity = combine_validities(lhs.validity(), rhs.validity());

    let values = lhs
        .values_iter()
        .zip(rhs.values_iter())
        .map(|(lhs, rhs)| op(lhs, rhs));
    let values = Bitmap::from_trusted_len_iter(values);

    BooleanArray::new(DataType::Boolean, values, validity)
}

/// Evaluate `op(lhs, rhs)` for [`BinaryArray`] and scalar using
/// a specified comparison function.
fn compare_op_scalar<O, F>(lhs: &BinaryArray<O>, rhs: &[u8], op: F) -> BooleanArray
where
    O: Offset,
    F: Fn(&[u8], &[u8]) -> bool,
{
    let validity = lhs.validity().cloned();

    let values = lhs.values_iter().map(|lhs| op(lhs, rhs));
    let values = Bitmap::from_trusted_len_iter(values);

    BooleanArray::new(DataType::Boolean, values, validity)
}

/// Perform `lhs == rhs` operation on [`BinaryArray`].
/// # Panic
/// iff the arrays do not have the same length.
pub fn eq<O: Offset>(lhs: &BinaryArray<O>, rhs: &BinaryArray<O>) -> BooleanArray {
    compare_op(lhs, rhs, |a, b| a == b)
}

/// Perform `lhs == rhs` operation on [`BinaryArray`] and include validities in comparison.
/// # Panic
/// iff the arrays do not have the same length.
pub fn eq_and_validity<O: Offset>(lhs: &BinaryArray<O>, rhs: &BinaryArray<O>) -> BooleanArray {
    let validity_lhs = lhs.validity().cloned();
    let validity_rhs = rhs.validity().cloned();
    let lhs = lhs.clone().with_validity(None);
    let rhs = rhs.clone().with_validity(None);
    let out = compare_op(&lhs, &rhs, |a, b| a == b);

    finish_eq_validities(out, validity_lhs, validity_rhs)
}

/// Perform `lhs == rhs` operation on [`BinaryArray`] and a scalar.
pub fn eq_scalar<O: Offset>(lhs: &BinaryArray<O>, rhs: &[u8]) -> BooleanArray {
    compare_op_scalar(lhs, rhs, |a, b| a == b)
}

/// Perform `lhs == rhs` operation on [`BinaryArray`] and a scalar and include validities in comparison.
pub fn eq_scalar_and_validity<O: Offset>(lhs: &BinaryArray<O>, rhs: &[u8]) -> BooleanArray {
    let validity = lhs.validity().cloned();
    let lhs = lhs.clone().with_validity(None);
    let out = compare_op_scalar(&lhs, rhs, |a, b| a == b);

    finish_eq_validities(out, validity, None)
}

/// Perform `lhs != rhs` operation on [`BinaryArray`].
/// # Panic
/// iff the arrays do not have the same length.
pub fn neq<O: Offset>(lhs: &BinaryArray<O>, rhs: &BinaryArray<O>) -> BooleanArray {
    compare_op(lhs, rhs, |a, b| a != b)
}

/// Perform `lhs != rhs` operation on [`BinaryArray`].
/// # Panic
/// iff the arrays do not have the same length and include validities in comparison.
pub fn neq_and_validity<O: Offset>(lhs: &BinaryArray<O>, rhs: &BinaryArray<O>) -> BooleanArray {
    let validity_lhs = lhs.validity().cloned();
    let validity_rhs = rhs.validity().cloned();
    let lhs = lhs.clone().with_validity(None);
    let rhs = rhs.clone().with_validity(None);

    let out = compare_op(&lhs, &rhs, |a, b| a != b);
    finish_neq_validities(out, validity_lhs, validity_rhs)
}

/// Perform `lhs != rhs` operation on [`BinaryArray`] and a scalar.
pub fn neq_scalar<O: Offset>(lhs: &BinaryArray<O>, rhs: &[u8]) -> BooleanArray {
    compare_op_scalar(lhs, rhs, |a, b| a != b)
}

/// Perform `lhs != rhs` operation on [`BinaryArray`] and a scalar and include validities in comparison.
pub fn neq_scalar_and_validity<O: Offset>(lhs: &BinaryArray<O>, rhs: &[u8]) -> BooleanArray {
    let validity = lhs.validity().cloned();
    let lhs = lhs.clone().with_validity(None);
    let out = compare_op_scalar(&lhs, rhs, |a, b| a != b);

    finish_neq_validities(out, validity, None)
}

/// Perform `lhs < rhs` operation on [`BinaryArray`].
pub fn lt<O: Offset>(lhs: &BinaryArray<O>, rhs: &BinaryArray<O>) -> BooleanArray {
    compare_op(lhs, rhs, |a, b| a < b)
}

/// Perform `lhs < rhs` operation on [`BinaryArray`] and a scalar.
pub fn lt_scalar<O: Offset>(lhs: &BinaryArray<O>, rhs: &[u8]) -> BooleanArray {
    compare_op_scalar(lhs, rhs, |a, b| a < b)
}

/// Perform `lhs <= rhs` operation on [`BinaryArray`].
pub fn lt_eq<O: Offset>(lhs: &BinaryArray<O>, rhs: &BinaryArray<O>) -> BooleanArray {
    compare_op(lhs, rhs, |a, b| a <= b)
}

/// Perform `lhs <= rhs` operation on [`BinaryArray`] and a scalar.
pub fn lt_eq_scalar<O: Offset>(lhs: &BinaryArray<O>, rhs: &[u8]) -> BooleanArray {
    compare_op_scalar(lhs, rhs, |a, b| a <= b)
}

/// Perform `lhs > rhs` operation on [`BinaryArray`].
pub fn gt<O: Offset>(lhs: &BinaryArray<O>, rhs: &BinaryArray<O>) -> BooleanArray {
    compare_op(lhs, rhs, |a, b| a > b)
}

/// Perform `lhs > rhs` operation on [`BinaryArray`] and a scalar.
pub fn gt_scalar<O: Offset>(lhs: &BinaryArray<O>, rhs: &[u8]) -> BooleanArray {
    compare_op_scalar(lhs, rhs, |a, b| a > b)
}

/// Perform `lhs >= rhs` operation on [`BinaryArray`].
pub fn gt_eq<O: Offset>(lhs: &BinaryArray<O>, rhs: &BinaryArray<O>) -> BooleanArray {
    compare_op(lhs, rhs, |a, b| a >= b)
}

/// Perform `lhs >= rhs` operation on [`BinaryArray`] and a scalar.
pub fn gt_eq_scalar<O: Offset>(lhs: &BinaryArray<O>, rhs: &[u8]) -> BooleanArray {
    compare_op_scalar(lhs, rhs, |a, b| a >= b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_generic<O: Offset, F: Fn(&BinaryArray<O>, &BinaryArray<O>) -> BooleanArray>(
        lhs: Vec<&[u8]>,
        rhs: Vec<&[u8]>,
        op: F,
        expected: Vec<bool>,
    ) {
        let lhs = BinaryArray::<O>::from_slice(lhs);
        let rhs = BinaryArray::<O>::from_slice(rhs);
        let expected = BooleanArray::from_slice(expected);
        assert_eq!(op(&lhs, &rhs), expected);
    }

    fn test_generic_scalar<O: Offset, F: Fn(&BinaryArray<O>, &[u8]) -> BooleanArray>(
        lhs: Vec<&[u8]>,
        rhs: &[u8],
        op: F,
        expected: Vec<bool>,
    ) {
        let lhs = BinaryArray::<O>::from_slice(lhs);
        let expected = BooleanArray::from_slice(expected);
        assert_eq!(op(&lhs, rhs), expected);
    }

    #[test]
    fn test_gt_eq() {
        test_generic::<i32, _>(
            vec![b"arrow", b"datafusion", b"flight", b"parquet"],
            vec![b"flight", b"flight", b"flight", b"flight"],
            gt_eq,
            vec![false, false, true, true],
        )
    }

    #[test]
    fn test_gt_eq_scalar() {
        test_generic_scalar::<i32, _>(
            vec![b"arrow", b"datafusion", b"flight", b"parquet"],
            b"flight",
            gt_eq_scalar,
            vec![false, false, true, true],
        )
    }

    #[test]
    fn test_eq() {
        test_generic::<i32, _>(
            vec![b"arrow", b"arrow", b"arrow", b"arrow"],
            vec![b"arrow", b"parquet", b"datafusion", b"flight"],
            eq,
            vec![true, false, false, false],
        )
    }

    #[test]
    fn test_eq_scalar() {
        test_generic_scalar::<i32, _>(
            vec![b"arrow", b"parquet", b"datafusion", b"flight"],
            b"arrow",
            eq_scalar,
            vec![true, false, false, false],
        )
    }

    #[test]
    fn test_neq() {
        test_generic::<i32, _>(
            vec![b"arrow", b"arrow", b"arrow", b"arrow"],
            vec![b"arrow", b"parquet", b"datafusion", b"flight"],
            neq,
            vec![false, true, true, true],
        )
    }

    #[test]
    fn test_neq_scalar() {
        test_generic_scalar::<i32, _>(
            vec![b"arrow", b"parquet", b"datafusion", b"flight"],
            b"arrow",
            neq_scalar,
            vec![false, true, true, true],
        )
    }
}
