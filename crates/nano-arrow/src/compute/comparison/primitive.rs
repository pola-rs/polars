//! Comparison functions for [`PrimitiveArray`]
use crate::compute::comparison::{finish_eq_validities, finish_neq_validities};
use crate::{
    array::{BooleanArray, PrimitiveArray},
    bitmap::MutableBitmap,
    datatypes::DataType,
    types::NativeType,
};

use super::super::utils::combine_validities;
use super::simd::{Simd8, Simd8Lanes, Simd8PartialEq, Simd8PartialOrd};

pub(crate) fn compare_values_op<T, F>(lhs: &[T], rhs: &[T], op: F) -> MutableBitmap
where
    T: NativeType + Simd8,
    F: Fn(T::Simd, T::Simd) -> u8,
{
    assert_eq!(lhs.len(), rhs.len());

    let lhs_chunks_iter = lhs.chunks_exact(8);
    let lhs_remainder = lhs_chunks_iter.remainder();
    let rhs_chunks_iter = rhs.chunks_exact(8);
    let rhs_remainder = rhs_chunks_iter.remainder();

    let mut values = Vec::with_capacity((lhs.len() + 7) / 8);
    let iterator = lhs_chunks_iter.zip(rhs_chunks_iter).map(|(lhs, rhs)| {
        let lhs = T::Simd::from_chunk(lhs);
        let rhs = T::Simd::from_chunk(rhs);
        op(lhs, rhs)
    });
    values.extend(iterator);

    if !lhs_remainder.is_empty() {
        let lhs = T::Simd::from_incomplete_chunk(lhs_remainder, T::default());
        let rhs = T::Simd::from_incomplete_chunk(rhs_remainder, T::default());
        values.push(op(lhs, rhs))
    };
    MutableBitmap::from_vec(values, lhs.len())
}

pub(crate) fn compare_values_op_scalar<T, F>(lhs: &[T], rhs: T, op: F) -> MutableBitmap
where
    T: NativeType + Simd8,
    F: Fn(T::Simd, T::Simd) -> u8,
{
    let rhs = T::Simd::from_chunk(&[rhs; 8]);

    let lhs_chunks_iter = lhs.chunks_exact(8);
    let lhs_remainder = lhs_chunks_iter.remainder();

    let mut values = Vec::with_capacity((lhs.len() + 7) / 8);
    let iterator = lhs_chunks_iter.map(|lhs| {
        let lhs = T::Simd::from_chunk(lhs);
        op(lhs, rhs)
    });
    values.extend(iterator);

    if !lhs_remainder.is_empty() {
        let lhs = T::Simd::from_incomplete_chunk(lhs_remainder, T::default());
        values.push(op(lhs, rhs))
    };

    MutableBitmap::from_vec(values, lhs.len())
}

/// Evaluate `op(lhs, rhs)` for [`PrimitiveArray`]s using a specified
/// comparison function.
fn compare_op<T, F>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>, op: F) -> BooleanArray
where
    T: NativeType + Simd8,
    F: Fn(T::Simd, T::Simd) -> u8,
{
    let validity = combine_validities(lhs.validity(), rhs.validity());

    let values = compare_values_op(lhs.values(), rhs.values(), op);

    BooleanArray::new(DataType::Boolean, values.into(), validity)
}

/// Evaluate `op(left, right)` for [`PrimitiveArray`] and scalar using
/// a specified comparison function.
pub fn compare_op_scalar<T, F>(lhs: &PrimitiveArray<T>, rhs: T, op: F) -> BooleanArray
where
    T: NativeType + Simd8,
    F: Fn(T::Simd, T::Simd) -> u8,
{
    let validity = lhs.validity().cloned();

    let values = compare_values_op_scalar(lhs.values(), rhs, op);

    BooleanArray::new(DataType::Boolean, values.into(), validity)
}

/// Perform `lhs == rhs` operation on two arrays.
pub fn eq<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> BooleanArray
where
    T: NativeType + Simd8,
    T::Simd: Simd8PartialEq,
{
    compare_op(lhs, rhs, |a, b| a.eq(b))
}

/// Perform `lhs == rhs` operation on two arrays and include validities in comparison.
pub fn eq_and_validity<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> BooleanArray
where
    T: NativeType + Simd8,
    T::Simd: Simd8PartialEq,
{
    let validity_lhs = lhs.validity().cloned();
    let validity_rhs = rhs.validity().cloned();
    let lhs = lhs.clone().with_validity(None);
    let rhs = rhs.clone().with_validity(None);
    let out = compare_op(&lhs, &rhs, |a, b| a.eq(b));

    finish_eq_validities(out, validity_lhs, validity_rhs)
}

/// Perform `left == right` operation on an array and a scalar value.
pub fn eq_scalar<T>(lhs: &PrimitiveArray<T>, rhs: T) -> BooleanArray
where
    T: NativeType + Simd8,
    T::Simd: Simd8PartialEq,
{
    compare_op_scalar(lhs, rhs, |a, b| a.eq(b))
}

/// Perform `left == right` operation on an array and a scalar value and include validities in comparison.
pub fn eq_scalar_and_validity<T>(lhs: &PrimitiveArray<T>, rhs: T) -> BooleanArray
where
    T: NativeType + Simd8,
    T::Simd: Simd8PartialEq,
{
    let validity = lhs.validity().cloned();
    let lhs = lhs.clone().with_validity(None);
    let out = compare_op_scalar(&lhs, rhs, |a, b| a.eq(b));

    finish_eq_validities(out, validity, None)
}

/// Perform `left != right` operation on two arrays.
pub fn neq<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> BooleanArray
where
    T: NativeType + Simd8,
    T::Simd: Simd8PartialEq,
{
    compare_op(lhs, rhs, |a, b| a.neq(b))
}

/// Perform `left != right` operation on two arrays and include validities in comparison.
pub fn neq_and_validity<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> BooleanArray
where
    T: NativeType + Simd8,
    T::Simd: Simd8PartialEq,
{
    let validity_lhs = lhs.validity().cloned();
    let validity_rhs = rhs.validity().cloned();
    let lhs = lhs.clone().with_validity(None);
    let rhs = rhs.clone().with_validity(None);
    let out = compare_op(&lhs, &rhs, |a, b| a.neq(b));

    finish_neq_validities(out, validity_lhs, validity_rhs)
}

/// Perform `left != right` operation on an array and a scalar value.
pub fn neq_scalar<T>(lhs: &PrimitiveArray<T>, rhs: T) -> BooleanArray
where
    T: NativeType + Simd8,
    T::Simd: Simd8PartialEq,
{
    compare_op_scalar(lhs, rhs, |a, b| a.neq(b))
}

/// Perform `left != right` operation on an array and a scalar value and include validities in comparison.
pub fn neq_scalar_and_validity<T>(lhs: &PrimitiveArray<T>, rhs: T) -> BooleanArray
where
    T: NativeType + Simd8,
    T::Simd: Simd8PartialEq,
{
    let validity = lhs.validity().cloned();
    let lhs = lhs.clone().with_validity(None);
    let out = compare_op_scalar(&lhs, rhs, |a, b| a.neq(b));

    finish_neq_validities(out, validity, None)
}

/// Perform `left < right` operation on two arrays.
pub fn lt<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> BooleanArray
where
    T: NativeType + Simd8,
    T::Simd: Simd8PartialOrd,
{
    compare_op(lhs, rhs, |a, b| a.lt(b))
}

/// Perform `left < right` operation on an array and a scalar value.
pub fn lt_scalar<T>(lhs: &PrimitiveArray<T>, rhs: T) -> BooleanArray
where
    T: NativeType + Simd8,
    T::Simd: Simd8PartialOrd,
{
    compare_op_scalar(lhs, rhs, |a, b| a.lt(b))
}

/// Perform `left <= right` operation on two arrays.
pub fn lt_eq<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> BooleanArray
where
    T: NativeType + Simd8,
    T::Simd: Simd8PartialOrd,
{
    compare_op(lhs, rhs, |a, b| a.lt_eq(b))
}

/// Perform `left <= right` operation on an array and a scalar value.
/// Null values are less than non-null values.
pub fn lt_eq_scalar<T>(lhs: &PrimitiveArray<T>, rhs: T) -> BooleanArray
where
    T: NativeType + Simd8,
    T::Simd: Simd8PartialOrd,
{
    compare_op_scalar(lhs, rhs, |a, b| a.lt_eq(b))
}

/// Perform `left > right` operation on two arrays. Non-null values are greater than null
/// values.
pub fn gt<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> BooleanArray
where
    T: NativeType + Simd8,
    T::Simd: Simd8PartialOrd,
{
    compare_op(lhs, rhs, |a, b| a.gt(b))
}

/// Perform `left > right` operation on an array and a scalar value.
/// Non-null values are greater than null values.
pub fn gt_scalar<T>(lhs: &PrimitiveArray<T>, rhs: T) -> BooleanArray
where
    T: NativeType + Simd8,
    T::Simd: Simd8PartialOrd,
{
    compare_op_scalar(lhs, rhs, |a, b| a.gt(b))
}

/// Perform `left >= right` operation on two arrays. Non-null values are greater than null
/// values.
pub fn gt_eq<T>(lhs: &PrimitiveArray<T>, rhs: &PrimitiveArray<T>) -> BooleanArray
where
    T: NativeType + Simd8,
    T::Simd: Simd8PartialOrd,
{
    compare_op(lhs, rhs, |a, b| a.gt_eq(b))
}

/// Perform `left >= right` operation on an array and a scalar value.
/// Non-null values are greater than null values.
pub fn gt_eq_scalar<T>(lhs: &PrimitiveArray<T>, rhs: T) -> BooleanArray
where
    T: NativeType + Simd8,
    T::Simd: Simd8PartialOrd,
{
    compare_op_scalar(lhs, rhs, |a, b| a.gt_eq(b))
}

// disable wrapping inside literal vectors used for test data and assertions
#[rustfmt::skip::macros(vec)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{Int64Array, Int8Array};

    /// Evaluate `KERNEL` with two vectors as inputs and assert against the expected output.
    /// `A_VEC` and `B_VEC` can be of type `Vec<i64>` or `Vec<Option<i64>>`.
    /// `EXPECTED` can be either `Vec<bool>` or `Vec<Option<bool>>`.
    /// The main reason for this macro is that inputs and outputs align nicely after `cargo fmt`.
    macro_rules! cmp_i64 {
        ($KERNEL:ident, $A_VEC:expr, $B_VEC:expr, $EXPECTED:expr) => {
            let a = Int64Array::from_slice($A_VEC);
            let b = Int64Array::from_slice($B_VEC);
            let c = $KERNEL(&a, &b);
            assert_eq!(BooleanArray::from_slice($EXPECTED), c);
        };
    }

    macro_rules! cmp_i64_options {
        ($KERNEL:ident, $A_VEC:expr, $B_VEC:expr, $EXPECTED:expr) => {
            let a = Int64Array::from($A_VEC);
            let b = Int64Array::from($B_VEC);
            let c = $KERNEL(&a, &b);
            assert_eq!(BooleanArray::from($EXPECTED), c);
        };
    }

    /// Evaluate `KERNEL` with one vectors and one scalar as inputs and assert against the expected output.
    /// `A_VEC` can be of type `Vec<i64>` or `Vec<Option<i64>>`.
    /// `EXPECTED` can be either `Vec<bool>` or `Vec<Option<bool>>`.
    /// The main reason for this macro is that inputs and outputs align nicely after `cargo fmt`.
    macro_rules! cmp_i64_scalar_options {
        ($KERNEL:ident, $A_VEC:expr, $B:literal, $EXPECTED:expr) => {
            let a = Int64Array::from($A_VEC);
            let c = $KERNEL(&a, $B);
            assert_eq!(BooleanArray::from($EXPECTED), c);
        };
    }

    macro_rules! cmp_i64_scalar {
        ($KERNEL:ident, $A_VEC:expr, $B:literal, $EXPECTED:expr) => {
            let a = Int64Array::from_slice($A_VEC);
            let c = $KERNEL(&a, $B);
            assert_eq!(BooleanArray::from_slice($EXPECTED), c);
        };
    }

    #[test]
    fn test_primitive_array_eq() {
        cmp_i64!(
            eq,
            &[8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            &[6, 7, 8, 9, 10, 6, 7, 8, 9, 10],
            vec![false, false, true, false, false, false, false, true, false, false]
        );
    }

    #[test]
    fn test_primitive_array_eq_scalar() {
        cmp_i64_scalar!(
            eq_scalar,
            &[6, 7, 8, 9, 10, 6, 7, 8, 9, 10],
            8,
            vec![false, false, true, false, false, false, false, true, false, false]
        );
    }

    #[test]
    fn test_primitive_array_eq_with_slice() {
        let a = Int64Array::from_slice([6, 7, 8, 8, 10]);
        let mut b = Int64Array::from_slice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        b.slice(5, 5);
        let d = eq(&b, &a);
        assert_eq!(d, BooleanArray::from_slice([true, true, true, false, true]));
    }

    #[test]
    fn test_primitive_array_neq() {
        cmp_i64!(
            neq,
            &[8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            &[6, 7, 8, 9, 10, 6, 7, 8, 9, 10],
            vec![true, true, false, true, true, true, true, false, true, true]
        );
    }

    #[test]
    fn test_primitive_array_neq_scalar() {
        cmp_i64_scalar!(
            neq_scalar,
            &[6, 7, 8, 9, 10, 6, 7, 8, 9, 10],
            8,
            vec![true, true, false, true, true, true, true, false, true, true]
        );
    }

    #[test]
    fn test_primitive_array_lt() {
        cmp_i64!(
            lt,
            &[8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            &[6, 7, 8, 9, 10, 6, 7, 8, 9, 10],
            vec![false, false, false, true, true, false, false, false, true, true]
        );
    }

    #[test]
    fn test_primitive_array_lt_scalar() {
        cmp_i64_scalar!(
            lt_scalar,
            &[6, 7, 8, 9, 10, 6, 7, 8, 9, 10],
            8,
            vec![true, true, false, false, false, true, true, false, false, false]
        );
    }

    #[test]
    fn test_primitive_array_lt_nulls() {
        cmp_i64_options!(
            lt,
            &[None, None, Some(1), Some(1), None, None, Some(2), Some(2),],
            &[None, Some(1), None, Some(1), None, Some(3), None, Some(3),],
            vec![None, None, None, Some(false), None, None, None, Some(true)]
        );
    }

    #[test]
    fn test_primitive_array_lt_scalar_nulls() {
        cmp_i64_scalar_options!(
            lt_scalar,
            &[None, Some(1), Some(2), Some(3), None, Some(1), Some(2), Some(3), Some(2), None],
            2,
            vec![None, Some(true), Some(false), Some(false), None, Some(true), Some(false), Some(false), Some(false), None]
        );
    }

    #[test]
    fn test_primitive_array_lt_eq() {
        cmp_i64!(
            lt_eq,
            &[8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            &[6, 7, 8, 9, 10, 6, 7, 8, 9, 10],
            vec![false, false, true, true, true, false, false, true, true, true]
        );
    }

    #[test]
    fn test_primitive_array_lt_eq_scalar() {
        cmp_i64_scalar!(
            lt_eq_scalar,
            &[6, 7, 8, 9, 10, 6, 7, 8, 9, 10],
            8,
            vec![true, true, true, false, false, true, true, true, false, false]
        );
    }

    #[test]
    fn test_primitive_array_lt_eq_nulls() {
        cmp_i64_options!(
            lt_eq,
            &[
                None,
                None,
                Some(1),
                None,
                None,
                Some(1),
                None,
                None,
                Some(1)
            ],
            &[
                None,
                Some(1),
                Some(0),
                None,
                Some(1),
                Some(2),
                None,
                None,
                Some(3)
            ],
            vec![None, None, Some(false), None, None, Some(true), None, None, Some(true)]
        );
    }

    #[test]
    fn test_primitive_array_lt_eq_scalar_nulls() {
        cmp_i64_scalar_options!(
            lt_eq_scalar,
            &[None, Some(1), Some(2), None, Some(1), Some(2), None, Some(1), Some(2)],
            1,
            vec![None, Some(true), Some(false), None, Some(true), Some(false), None, Some(true), Some(false)]
        );
    }

    #[test]
    fn test_primitive_array_gt() {
        cmp_i64!(
            gt,
            &[8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            &[6, 7, 8, 9, 10, 6, 7, 8, 9, 10],
            vec![true, true, false, false, false, true, true, false, false, false]
        );
    }

    #[test]
    fn test_primitive_array_gt_scalar() {
        cmp_i64_scalar!(
            gt_scalar,
            &[6, 7, 8, 9, 10, 6, 7, 8, 9, 10],
            8,
            vec![false, false, false, true, true, false, false, false, true, true]
        );
    }

    #[test]
    fn test_primitive_array_gt_nulls() {
        cmp_i64_options!(
            gt,
            &[
                None,
                None,
                Some(1),
                None,
                None,
                Some(2),
                None,
                None,
                Some(3)
            ],
            &[
                None,
                Some(1),
                Some(1),
                None,
                Some(1),
                Some(1),
                None,
                Some(1),
                Some(1)
            ],
            vec![None, None, Some(false), None, None, Some(true), None, None, Some(true)]
        );
    }

    #[test]
    fn test_primitive_array_gt_scalar_nulls() {
        cmp_i64_scalar_options!(
            gt_scalar,
            &[None, Some(1), Some(2), None, Some(1), Some(2), None, Some(1), Some(2)],
            1,
            vec![None, Some(false), Some(true), None, Some(false), Some(true), None, Some(false), Some(true)]
        );
    }

    #[test]
    fn test_primitive_array_gt_eq() {
        cmp_i64!(
            gt_eq,
            &[8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            &[6, 7, 8, 9, 10, 6, 7, 8, 9, 10],
            vec![true, true, true, false, false, true, true, true, false, false]
        );
    }

    #[test]
    fn test_primitive_array_gt_eq_scalar() {
        cmp_i64_scalar!(
            gt_eq_scalar,
            &[6, 7, 8, 9, 10, 6, 7, 8, 9, 10],
            8,
            vec![false, false, true, true, true, false, false, true, true, true]
        );
    }

    #[test]
    fn test_primitive_array_gt_eq_nulls() {
        cmp_i64_options!(
            gt_eq,
            vec![None, None, Some(1), None, Some(1), Some(2), None, None, Some(1)],
            vec![None, Some(1), None, None, Some(1), Some(1), None, Some(2), Some(2)],
            vec![None, None, None, None, Some(true), Some(true), None, None, Some(false)]
        );
    }

    #[test]
    fn test_primitive_array_gt_eq_scalar_nulls() {
        cmp_i64_scalar_options!(
            gt_eq_scalar,
            vec![None, Some(1), Some(2), None, Some(2), Some(3), None, Some(3), Some(4)],
            2,
            vec![None, Some(false), Some(true), None, Some(true), Some(true), None, Some(true), Some(true)]
        );
    }

    #[test]
    fn test_primitive_array_compare_slice() {
        let mut a = (0..100).map(Some).collect::<PrimitiveArray<i32>>();
        a.slice(50, 50);
        let mut b = (100..200).map(Some).collect::<PrimitiveArray<i32>>();
        b.slice(50, 50);
        let actual = lt(&a, &b);
        let expected: BooleanArray = (0..50).map(|_| Some(true)).collect();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_primitive_array_compare_scalar_slice() {
        let mut a = (0..100).map(Some).collect::<PrimitiveArray<i32>>();
        a.slice(50, 50);
        let actual = lt_scalar(&a, 200);
        let expected: BooleanArray = (0..50).map(|_| Some(true)).collect();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_length_of_result_buffer() {
        // `item_count` is chosen to not be a multiple of 64.
        const ITEM_COUNT: usize = 130;

        let array_a = Int8Array::from_slice([1; ITEM_COUNT]);
        let array_b = Int8Array::from_slice([2; ITEM_COUNT]);
        let expected = BooleanArray::from_slice([false; ITEM_COUNT]);
        let result = gt_eq(&array_a, &array_b);

        assert_eq!(result, expected)
    }
}
