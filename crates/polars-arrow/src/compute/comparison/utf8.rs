//! Comparison functions for [`Utf8Array`]
use super::super::utils::combine_validities;
use crate::array::{BooleanArray, Utf8Array};
use crate::bitmap::Bitmap;
use crate::compute::comparison::{finish_eq_validities, finish_neq_validities};
use crate::datatypes::ArrowDataType;
use crate::offset::Offset;

/// Evaluate `op(lhs, rhs)` for [`Utf8Array`]s using a specified
/// comparison function.
fn compare_op<O, F>(lhs: &Utf8Array<O>, rhs: &Utf8Array<O>, op: F) -> BooleanArray
where
    O: Offset,
    F: Fn(&str, &str) -> bool,
{
    assert_eq!(lhs.len(), rhs.len());
    let validity = combine_validities(lhs.validity(), rhs.validity());

    let values = lhs
        .values_iter()
        .zip(rhs.values_iter())
        .map(|(lhs, rhs)| op(lhs, rhs));
    let values = Bitmap::from_trusted_len_iter(values);

    BooleanArray::new(ArrowDataType::Boolean, values, validity)
}

/// Evaluate `op(lhs, rhs)` for [`Utf8Array`] and scalar using
/// a specified comparison function.
fn compare_op_scalar<O, F>(lhs: &Utf8Array<O>, rhs: &str, op: F) -> BooleanArray
where
    O: Offset,
    F: Fn(&str, &str) -> bool,
{
    let validity = lhs.validity().cloned();

    let values = lhs.values_iter().map(|lhs| op(lhs, rhs));
    let values = Bitmap::from_trusted_len_iter(values);

    BooleanArray::new(ArrowDataType::Boolean, values, validity)
}

/// Perform `lhs == rhs` operation on [`Utf8Array`].
pub fn eq<O: Offset>(lhs: &Utf8Array<O>, rhs: &Utf8Array<O>) -> BooleanArray {
    compare_op(lhs, rhs, |a, b| a == b)
}

/// Perform `lhs == rhs` operation on [`Utf8Array`] and include validities in comparison.
pub fn eq_and_validity<O: Offset>(lhs: &Utf8Array<O>, rhs: &Utf8Array<O>) -> BooleanArray {
    let validity_lhs = lhs.validity().cloned();
    let validity_rhs = rhs.validity().cloned();
    let lhs = lhs.clone().with_validity(None);
    let rhs = rhs.clone().with_validity(None);
    let out = compare_op(&lhs, &rhs, |a, b| a == b);

    finish_eq_validities(out, validity_lhs, validity_rhs)
}

/// Perform `lhs != rhs` operation on [`Utf8Array`] and include validities in comparison.
pub fn neq_and_validity<O: Offset>(lhs: &Utf8Array<O>, rhs: &Utf8Array<O>) -> BooleanArray {
    let validity_lhs = lhs.validity().cloned();
    let validity_rhs = rhs.validity().cloned();
    let lhs = lhs.clone().with_validity(None);
    let rhs = rhs.clone().with_validity(None);
    let out = compare_op(&lhs, &rhs, |a, b| a != b);

    finish_neq_validities(out, validity_lhs, validity_rhs)
}

/// Perform `lhs == rhs` operation on [`Utf8Array`] and a scalar.
pub fn eq_scalar<O: Offset>(lhs: &Utf8Array<O>, rhs: &str) -> BooleanArray {
    compare_op_scalar(lhs, rhs, |a, b| a == b)
}

/// Perform `lhs == rhs` operation on [`Utf8Array`] and a scalar. Also includes null values in comparison.
pub fn eq_scalar_and_validity<O: Offset>(lhs: &Utf8Array<O>, rhs: &str) -> BooleanArray {
    let validity = lhs.validity().cloned();
    let lhs = lhs.clone().with_validity(None);
    let out = compare_op_scalar(&lhs, rhs, |a, b| a == b);

    finish_eq_validities(out, validity, None)
}

/// Perform `lhs != rhs` operation on [`Utf8Array`] and a scalar. Also includes null values in comparison.
pub fn neq_scalar_and_validity<O: Offset>(lhs: &Utf8Array<O>, rhs: &str) -> BooleanArray {
    let validity = lhs.validity().cloned();
    let lhs = lhs.clone().with_validity(None);
    let out = compare_op_scalar(&lhs, rhs, |a, b| a != b);

    finish_neq_validities(out, validity, None)
}

/// Perform `lhs != rhs` operation on [`Utf8Array`].
pub fn neq<O: Offset>(lhs: &Utf8Array<O>, rhs: &Utf8Array<O>) -> BooleanArray {
    compare_op(lhs, rhs, |a, b| a != b)
}

/// Perform `lhs != rhs` operation on [`Utf8Array`] and a scalar.
pub fn neq_scalar<O: Offset>(lhs: &Utf8Array<O>, rhs: &str) -> BooleanArray {
    compare_op_scalar(lhs, rhs, |a, b| a != b)
}

/// Perform `lhs < rhs` operation on [`Utf8Array`].
pub fn lt<O: Offset>(lhs: &Utf8Array<O>, rhs: &Utf8Array<O>) -> BooleanArray {
    compare_op(lhs, rhs, |a, b| a < b)
}

/// Perform `lhs < rhs` operation on [`Utf8Array`] and a scalar.
pub fn lt_scalar<O: Offset>(lhs: &Utf8Array<O>, rhs: &str) -> BooleanArray {
    compare_op_scalar(lhs, rhs, |a, b| a < b)
}

/// Perform `lhs <= rhs` operation on [`Utf8Array`].
pub fn lt_eq<O: Offset>(lhs: &Utf8Array<O>, rhs: &Utf8Array<O>) -> BooleanArray {
    compare_op(lhs, rhs, |a, b| a <= b)
}

/// Perform `lhs <= rhs` operation on [`Utf8Array`] and a scalar.
pub fn lt_eq_scalar<O: Offset>(lhs: &Utf8Array<O>, rhs: &str) -> BooleanArray {
    compare_op_scalar(lhs, rhs, |a, b| a <= b)
}

/// Perform `lhs > rhs` operation on [`Utf8Array`].
pub fn gt<O: Offset>(lhs: &Utf8Array<O>, rhs: &Utf8Array<O>) -> BooleanArray {
    compare_op(lhs, rhs, |a, b| a > b)
}

/// Perform `lhs > rhs` operation on [`Utf8Array`] and a scalar.
pub fn gt_scalar<O: Offset>(lhs: &Utf8Array<O>, rhs: &str) -> BooleanArray {
    compare_op_scalar(lhs, rhs, |a, b| a > b)
}

/// Perform `lhs >= rhs` operation on [`Utf8Array`].
pub fn gt_eq<O: Offset>(lhs: &Utf8Array<O>, rhs: &Utf8Array<O>) -> BooleanArray {
    compare_op(lhs, rhs, |a, b| a >= b)
}

/// Perform `lhs >= rhs` operation on [`Utf8Array`] and a scalar.
pub fn gt_eq_scalar<O: Offset>(lhs: &Utf8Array<O>, rhs: &str) -> BooleanArray {
    compare_op_scalar(lhs, rhs, |a, b| a >= b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_generic<O: Offset, F: Fn(&Utf8Array<O>, &Utf8Array<O>) -> BooleanArray>(
        lhs: Vec<&str>,
        rhs: Vec<&str>,
        op: F,
        expected: Vec<bool>,
    ) {
        let lhs = Utf8Array::<O>::from_slice(lhs);
        let rhs = Utf8Array::<O>::from_slice(rhs);
        let expected = BooleanArray::from_slice(expected);
        assert_eq!(op(&lhs, &rhs), expected);
    }

    fn test_generic_scalar<O: Offset, F: Fn(&Utf8Array<O>, &str) -> BooleanArray>(
        lhs: Vec<&str>,
        rhs: &str,
        op: F,
        expected: Vec<bool>,
    ) {
        let lhs = Utf8Array::<O>::from_slice(lhs);
        let expected = BooleanArray::from_slice(expected);
        assert_eq!(op(&lhs, rhs), expected);
    }

    #[test]
    fn test_gt_eq() {
        test_generic::<i32, _>(
            vec!["arrow", "datafusion", "flight", "parquet"],
            vec!["flight", "flight", "flight", "flight"],
            gt_eq,
            vec![false, false, true, true],
        )
    }

    #[test]
    fn test_gt_eq_scalar() {
        test_generic_scalar::<i32, _>(
            vec!["arrow", "datafusion", "flight", "parquet"],
            "flight",
            gt_eq_scalar,
            vec![false, false, true, true],
        )
    }

    #[test]
    fn test_eq() {
        test_generic::<i32, _>(
            vec!["arrow", "arrow", "arrow", "arrow"],
            vec!["arrow", "parquet", "datafusion", "flight"],
            eq,
            vec![true, false, false, false],
        )
    }

    #[test]
    fn test_eq_scalar() {
        test_generic_scalar::<i32, _>(
            vec!["arrow", "parquet", "datafusion", "flight"],
            "arrow",
            eq_scalar,
            vec![true, false, false, false],
        )
    }

    #[test]
    fn test_neq() {
        test_generic::<i32, _>(
            vec!["arrow", "arrow", "arrow", "arrow"],
            vec!["arrow", "parquet", "datafusion", "flight"],
            neq,
            vec![false, true, true, true],
        )
    }

    #[test]
    fn test_neq_scalar() {
        test_generic_scalar::<i32, _>(
            vec!["arrow", "parquet", "datafusion", "flight"],
            "arrow",
            neq_scalar,
            vec![false, true, true, true],
        )
    }

    /*
    test_utf8!(
        test_utf8_array_lt,
        vec!["arrow", "datafusion", "flight", "parquet"],
        vec!["flight", "flight", "flight", "flight"],
        lt_utf8,
        vec![true, true, false, false]
    );
    test_utf8_scalar!(
        test_utf8_array_lt_scalar,
        vec!["arrow", "datafusion", "flight", "parquet"],
        "flight",
        lt_utf8_scalar,
        vec![true, true, false, false]
    );

    test_utf8!(
        test_utf8_array_lt_eq,
        vec!["arrow", "datafusion", "flight", "parquet"],
        vec!["flight", "flight", "flight", "flight"],
        lt_eq_utf8,
        vec![true, true, true, false]
    );
    test_utf8_scalar!(
        test_utf8_array_lt_eq_scalar,
        vec!["arrow", "datafusion", "flight", "parquet"],
        "flight",
        lt_eq_utf8_scalar,
        vec![true, true, true, false]
    );

    test_utf8!(
        test_utf8_array_gt,
        vec!["arrow", "datafusion", "flight", "parquet"],
        vec!["flight", "flight", "flight", "flight"],
        gt_utf8,
        vec![false, false, false, true]
    );
    test_utf8_scalar!(
        test_utf8_array_gt_scalar,
        vec!["arrow", "datafusion", "flight", "parquet"],
        "flight",
        gt_utf8_scalar,
        vec![false, false, false, true]
    );

    test_utf8!(
        test_utf8_array_gt_eq,
        vec!["arrow", "datafusion", "flight", "parquet"],
        vec!["flight", "flight", "flight", "flight"],
        gt_eq_utf8,
        vec![false, false, true, true]
    );
    test_utf8_scalar!(
        test_utf8_array_gt_eq_scalar,
        vec!["arrow", "datafusion", "flight", "parquet"],
        "flight",
        gt_eq_utf8_scalar,
        vec![false, false, true, true]
    );
    */
}
