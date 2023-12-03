//! null-preserving operators such as [`and`], [`or`] and [`not`].
use super::utils::combine_validities_and;
use crate::array::{Array, BooleanArray};
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::datatypes::ArrowDataType;
use crate::scalar::BooleanScalar;

fn assert_lengths(lhs: &BooleanArray, rhs: &BooleanArray) {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "lhs and rhs must have the same length"
    );
}

/// Helper function to implement binary kernels
pub(crate) fn binary_boolean_kernel<F>(
    lhs: &BooleanArray,
    rhs: &BooleanArray,
    op: F,
) -> BooleanArray
where
    F: Fn(&Bitmap, &Bitmap) -> Bitmap,
{
    assert_lengths(lhs, rhs);
    let validity = combine_validities_and(lhs.validity(), rhs.validity());

    let left_buffer = lhs.values();
    let right_buffer = rhs.values();

    let values = op(left_buffer, right_buffer);

    BooleanArray::new(ArrowDataType::Boolean, values, validity)
}

/// Performs `&&` operation on two [`BooleanArray`], combining the validities.
/// # Panics
/// This function panics iff the arrays have different lengths.
/// # Examples
/// ```rust
/// use polars_arrow::array::BooleanArray;
/// use polars_arrow::compute::boolean::and;
///
/// let a = BooleanArray::from(&[Some(false), Some(true), None]);
/// let b = BooleanArray::from(&[Some(true), Some(true), Some(false)]);
/// let and_ab = and(&a, &b);
/// assert_eq!(and_ab, BooleanArray::from(&[Some(false), Some(true), None]));
/// ```
pub fn and(lhs: &BooleanArray, rhs: &BooleanArray) -> BooleanArray {
    if lhs.null_count() == 0 && rhs.null_count() == 0 {
        let left_buffer = lhs.values();
        let right_buffer = rhs.values();

        match (left_buffer.unset_bits(), right_buffer.unset_bits()) {
            // all values are `true` on both sides
            (0, 0) => {
                assert_lengths(lhs, rhs);
                return lhs.clone();
            },
            // all values are `false` on left side
            (l, _) if l == lhs.len() => {
                assert_lengths(lhs, rhs);
                return lhs.clone();
            },
            // all values are `false` on right side
            (_, r) if r == rhs.len() => {
                assert_lengths(lhs, rhs);
                return rhs.clone();
            },
            // ignore the rest
            _ => {},
        }
    }

    binary_boolean_kernel(lhs, rhs, |lhs, rhs| lhs & rhs)
}

/// Performs `||` operation on two [`BooleanArray`], combining the validities.
/// # Panics
/// This function panics iff the arrays have different lengths.
/// # Examples
/// ```rust
/// use polars_arrow::array::BooleanArray;
/// use polars_arrow::compute::boolean::or;
///
/// let a = BooleanArray::from(vec![Some(false), Some(true), None]);
/// let b = BooleanArray::from(vec![Some(true), Some(true), Some(false)]);
/// let or_ab = or(&a, &b);
/// assert_eq!(or_ab, BooleanArray::from(vec![Some(true), Some(true), None]));
/// ```
pub fn or(lhs: &BooleanArray, rhs: &BooleanArray) -> BooleanArray {
    if lhs.null_count() == 0 && rhs.null_count() == 0 {
        let left_buffer = lhs.values();
        let right_buffer = rhs.values();

        match (left_buffer.unset_bits(), right_buffer.unset_bits()) {
            // all values are `true` on left side
            (0, _) => {
                assert_lengths(lhs, rhs);
                return lhs.clone();
            },
            // all values are `true` on right side
            (_, 0) => {
                assert_lengths(lhs, rhs);
                return rhs.clone();
            },
            // all values on lhs and rhs are `false`
            (l, r) if l == lhs.len() && r == rhs.len() => {
                assert_lengths(lhs, rhs);
                return rhs.clone();
            },
            // ignore the rest
            _ => {},
        }
    }

    binary_boolean_kernel(lhs, rhs, |lhs, rhs| lhs | rhs)
}

/// Performs unary `NOT` operation on an arrays. If value is null then the result is also
/// null.
/// # Example
/// ```rust
/// use polars_arrow::array::BooleanArray;
/// use polars_arrow::compute::boolean::not;
///
/// let a = BooleanArray::from(vec![Some(false), Some(true), None]);
/// let not_a = not(&a);
/// assert_eq!(not_a, BooleanArray::from(vec![Some(true), Some(false), None]));
/// ```
pub fn not(array: &BooleanArray) -> BooleanArray {
    let values = !array.values();
    let validity = array.validity().cloned();
    BooleanArray::new(ArrowDataType::Boolean, values, validity)
}

/// Returns a non-null [`BooleanArray`] with whether each value of the array is null.
/// # Example
/// ```rust
/// use polars_arrow::array::BooleanArray;
/// use polars_arrow::compute::boolean::is_null;
/// # fn main() {
/// let a = BooleanArray::from(vec![Some(false), Some(true), None]);
/// let a_is_null = is_null(&a);
/// assert_eq!(a_is_null, BooleanArray::from_slice(vec![false, false, true]));
/// # }
/// ```
pub fn is_null(input: &dyn Array) -> BooleanArray {
    let len = input.len();

    let values = match input.validity() {
        None => MutableBitmap::from_len_zeroed(len).into(),
        Some(buffer) => !buffer,
    };

    BooleanArray::new(ArrowDataType::Boolean, values, None)
}

/// Returns a non-null [`BooleanArray`] with whether each value of the array is not null.
/// # Example
/// ```rust
/// use polars_arrow::array::BooleanArray;
/// use polars_arrow::compute::boolean::is_not_null;
///
/// let a = BooleanArray::from(&vec![Some(false), Some(true), None]);
/// let a_is_not_null = is_not_null(&a);
/// assert_eq!(a_is_not_null, BooleanArray::from_slice(&vec![true, true, false]));
/// ```
pub fn is_not_null(input: &dyn Array) -> BooleanArray {
    let values = match input.validity() {
        None => {
            let mut mutable = MutableBitmap::new();
            mutable.extend_constant(input.len(), true);
            mutable.into()
        },
        Some(buffer) => buffer.clone(),
    };
    BooleanArray::new(ArrowDataType::Boolean, values, None)
}

/// Performs `AND` operation on an array and a scalar value. If either left or right value
/// is null then the result is also null.
/// # Example
/// ```rust
/// use polars_arrow::array::BooleanArray;
/// use polars_arrow::compute::boolean::and_scalar;
/// use polars_arrow::scalar::BooleanScalar;
///
/// let array = BooleanArray::from_slice(&[false, false, true, true]);
/// let scalar = BooleanScalar::new(Some(true));
/// let result = and_scalar(&array, &scalar);
/// assert_eq!(result, BooleanArray::from_slice(&[false, false, true, true]));
///
/// ```
pub fn and_scalar(array: &BooleanArray, scalar: &BooleanScalar) -> BooleanArray {
    match scalar.value() {
        Some(true) => array.clone(),
        Some(false) => {
            let values = Bitmap::new_zeroed(array.len());
            BooleanArray::new(ArrowDataType::Boolean, values, array.validity().cloned())
        },
        None => BooleanArray::new_null(ArrowDataType::Boolean, array.len()),
    }
}

/// Performs `OR` operation on an array and a scalar value. If either left or right value
/// is null then the result is also null.
/// # Example
/// ```rust
/// use polars_arrow::array::BooleanArray;
/// use polars_arrow::compute::boolean::or_scalar;
/// use polars_arrow::scalar::BooleanScalar;
/// # fn main() {
/// let array = BooleanArray::from_slice(&[false, false, true, true]);
/// let scalar = BooleanScalar::new(Some(true));
/// let result = or_scalar(&array, &scalar);
/// assert_eq!(result, BooleanArray::from_slice(&[true, true, true, true]));
/// # }
/// ```
pub fn or_scalar(array: &BooleanArray, scalar: &BooleanScalar) -> BooleanArray {
    match scalar.value() {
        Some(true) => {
            let mut values = MutableBitmap::new();
            values.extend_constant(array.len(), true);
            BooleanArray::new(
                ArrowDataType::Boolean,
                values.into(),
                array.validity().cloned(),
            )
        },
        Some(false) => array.clone(),
        None => BooleanArray::new_null(ArrowDataType::Boolean, array.len()),
    }
}

/// Returns whether any of the values in the array are `true`.
///
/// Null values are ignored.
///
/// # Example
///
/// ```
/// use polars_arrow::array::BooleanArray;
/// use polars_arrow::compute::boolean::any;
///
/// let a = BooleanArray::from(&[Some(true), Some(false)]);
/// let b = BooleanArray::from(&[Some(false), Some(false)]);
/// let c = BooleanArray::from(&[None, Some(false)]);
///
/// assert_eq!(any(&a), true);
/// assert_eq!(any(&b), false);
/// assert_eq!(any(&c), false);
/// ```
pub fn any(array: &BooleanArray) -> bool {
    if array.is_empty() {
        false
    } else if array.null_count() > 0 {
        array.into_iter().any(|v| v == Some(true))
    } else {
        let vals = array.values();
        vals.unset_bits() != vals.len()
    }
}

/// Returns whether all values in the array are `true`.
///
/// Null values are ignored.
///
/// # Example
///
/// ```
/// use polars_arrow::array::BooleanArray;
/// use polars_arrow::compute::boolean::all;
///
/// let a = BooleanArray::from(&[Some(true), Some(true)]);
/// let b = BooleanArray::from(&[Some(false), Some(true)]);
/// let c = BooleanArray::from(&[None, Some(true)]);
///
/// assert_eq!(all(&a), true);
/// assert_eq!(all(&b), false);
/// assert_eq!(all(&c), true);
/// ```
pub fn all(array: &BooleanArray) -> bool {
    if array.is_empty() {
        true
    } else if array.null_count() > 0 {
        !array.into_iter().any(|v| v == Some(false))
    } else {
        let vals = array.values();
        vals.unset_bits() == 0
    }
}
