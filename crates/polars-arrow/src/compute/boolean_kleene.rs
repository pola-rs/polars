//! Boolean operators of [Kleene logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics).
use crate::array::{Array, BooleanArray};
use crate::bitmap::{binary, quaternary, ternary, unary, Bitmap, MutableBitmap};
use crate::datatypes::ArrowDataType;
use crate::scalar::BooleanScalar;

/// Logical 'or' operation on two arrays with [Kleene logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics)
/// # Panics
/// This function panics iff the arrays have a different length
/// # Example
///
/// ```rust
/// use polars_arrow::array::BooleanArray;
/// use polars_arrow::compute::boolean_kleene::or;
///
/// let a = BooleanArray::from(&[Some(true), Some(false), None]);
/// let b = BooleanArray::from(&[None, None, None]);
/// let or_ab = or(&a, &b);
/// assert_eq!(or_ab, BooleanArray::from(&[Some(true), None, None]));
/// ```
pub fn or(lhs: &BooleanArray, rhs: &BooleanArray) -> BooleanArray {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "lhs and rhs must have the same length"
    );

    let lhs_values = lhs.values();
    let rhs_values = rhs.values();

    let lhs_validity = lhs.validity();
    let rhs_validity = rhs.validity();

    let validity = match (lhs_validity, rhs_validity) {
        (Some(lhs_validity), Some(rhs_validity)) => {
            Some(quaternary(
                lhs_values,
                rhs_values,
                lhs_validity,
                rhs_validity,
                // see https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics
                |lhs, rhs, lhs_v, rhs_v| {
                    // A = T
                    (lhs & lhs_v) |
                    // B = T
                    (rhs & rhs_v) |
                    // A = F & B = F
                    (!lhs & lhs_v) & (!rhs & rhs_v)
                },
            ))
        },
        (Some(lhs_validity), None) => {
            // B != U
            Some(ternary(
                lhs_values,
                rhs_values,
                lhs_validity,
                // see https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics
                |lhs, rhs, lhs_v| {
                    // A = T
                    (lhs & lhs_v) |
                    // B = T
                    rhs |
                    // A = F & B = F
                    (!lhs & lhs_v) & !rhs
                },
            ))
        },
        (None, Some(rhs_validity)) => {
            Some(ternary(
                lhs_values,
                rhs_values,
                rhs_validity,
                // see https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics
                |lhs, rhs, rhs_v| {
                    // A = T
                    lhs |
                    // B = T
                    (rhs & rhs_v) |
                    // A = F & B = F
                    !lhs & (!rhs & rhs_v)
                },
            ))
        },
        (None, None) => None,
    };
    BooleanArray::new(ArrowDataType::Boolean, lhs_values | rhs_values, validity)
}

/// Logical 'and' operation on two arrays with [Kleene logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics)
/// # Panics
/// This function panics iff the arrays have a different length
/// # Example
///
/// ```rust
/// use polars_arrow::array::BooleanArray;
/// use polars_arrow::compute::boolean_kleene::and;
///
/// let a = BooleanArray::from(&[Some(true), Some(false), None]);
/// let b = BooleanArray::from(&[None, None, None]);
/// let and_ab = and(&a, &b);
/// assert_eq!(and_ab, BooleanArray::from(&[None, Some(false), None]));
/// ```
pub fn and(lhs: &BooleanArray, rhs: &BooleanArray) -> BooleanArray {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "lhs and rhs must have the same length"
    );

    let lhs_values = lhs.values();
    let rhs_values = rhs.values();

    let lhs_validity = lhs.validity();
    let rhs_validity = rhs.validity();

    let validity = match (lhs_validity, rhs_validity) {
        (Some(lhs_validity), Some(rhs_validity)) => {
            Some(quaternary(
                lhs_values,
                rhs_values,
                lhs_validity,
                rhs_validity,
                // see https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics
                |lhs, rhs, lhs_v, rhs_v| {
                    // B = F
                    (!rhs & rhs_v) |
                    // A = F
                    (!lhs & lhs_v) |
                    // A = T & B = T
                    (lhs & lhs_v) & (rhs & rhs_v)
                },
            ))
        },
        (Some(lhs_validity), None) => {
            Some(ternary(
                lhs_values,
                rhs_values,
                lhs_validity,
                // see https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics
                |lhs, rhs, lhs_v| {
                    // B = F
                    !rhs |
                    // A = F
                    (!lhs & lhs_v) |
                    // A = T & B = T
                    (lhs & lhs_v) & rhs
                },
            ))
        },
        (None, Some(rhs_validity)) => {
            Some(ternary(
                lhs_values,
                rhs_values,
                rhs_validity,
                // see https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics
                |lhs, rhs, rhs_v| {
                    // B = F
                    (!rhs & rhs_v) |
                    // A = F
                    !lhs |
                    // A = T & B = T
                    lhs & (rhs & rhs_v)
                },
            ))
        },
        (None, None) => None,
    };
    BooleanArray::new(ArrowDataType::Boolean, lhs_values & rhs_values, validity)
}

/// Logical 'or' operation on an array and a scalar value with [Kleene logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics)
/// # Example
///
/// ```rust
/// use polars_arrow::array::BooleanArray;
/// use polars_arrow::scalar::BooleanScalar;
/// use polars_arrow::compute::boolean_kleene::or_scalar;
///
/// let array = BooleanArray::from(&[Some(true), Some(false), None]);
/// let scalar = BooleanScalar::new(Some(false));
/// let result = or_scalar(&array, &scalar);
/// assert_eq!(result, BooleanArray::from(&[Some(true), Some(false), None]));
/// ```
pub fn or_scalar(array: &BooleanArray, scalar: &BooleanScalar) -> BooleanArray {
    match scalar.value() {
        Some(true) => {
            let mut values = MutableBitmap::new();
            values.extend_constant(array.len(), true);
            BooleanArray::new(ArrowDataType::Boolean, values.into(), None)
        },
        Some(false) => array.clone(),
        None => {
            let values = array.values();
            let validity = match array.validity() {
                Some(validity) => binary(values, validity, |value, validity| validity & value),
                None => unary(values, |value| value),
            };
            BooleanArray::new(ArrowDataType::Boolean, values.clone(), Some(validity))
        },
    }
}

/// Logical 'and' operation on an array and a scalar value with [Kleene logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics)
/// # Example
///
/// ```rust
/// use polars_arrow::array::BooleanArray;
/// use polars_arrow::scalar::BooleanScalar;
/// use polars_arrow::compute::boolean_kleene::and_scalar;
///
/// let array = BooleanArray::from(&[Some(true), Some(false), None]);
/// let scalar = BooleanScalar::new(None);
/// let result = and_scalar(&array, &scalar);
/// assert_eq!(result, BooleanArray::from(&[None, Some(false), None]));
/// ```
pub fn and_scalar(array: &BooleanArray, scalar: &BooleanScalar) -> BooleanArray {
    match scalar.value() {
        Some(true) => array.clone(),
        Some(false) => {
            let values = Bitmap::new_zeroed(array.len());
            BooleanArray::new(ArrowDataType::Boolean, values, None)
        },
        None => {
            let values = array.values();
            let validity = match array.validity() {
                Some(validity) => binary(values, validity, |value, validity| validity & !value),
                None => unary(values, |value| !value),
            };
            BooleanArray::new(
                ArrowDataType::Boolean,
                array.values().clone(),
                Some(validity),
            )
        },
    }
}

/// Returns whether any of the values in the array are `true`.
///
/// The output is unknown (`None`) if the array contains any null values and
/// no `true` values.
///
/// # Example
///
/// ```
/// use polars_arrow::array::BooleanArray;
/// use polars_arrow::compute::boolean_kleene::any;
///
/// let a = BooleanArray::from(&[Some(true), Some(false)]);
/// let b = BooleanArray::from(&[Some(false), Some(false)]);
/// let c = BooleanArray::from(&[None, Some(false)]);
///
/// assert_eq!(any(&a), Some(true));
/// assert_eq!(any(&b), Some(false));
/// assert_eq!(any(&c), None);
/// ```
pub fn any(array: &BooleanArray) -> Option<bool> {
    if array.is_empty() {
        Some(false)
    } else if array.null_count() > 0 {
        if array.into_iter().any(|v| v == Some(true)) {
            Some(true)
        } else {
            None
        }
    } else {
        let vals = array.values();
        Some(vals.unset_bits() != vals.len())
    }
}

/// Returns whether all values in the array are `true`.
///
/// The output is unknown (`None`) if the array contains any null values and
/// no `false` values.
///
/// # Example
///
/// ```
/// use polars_arrow::array::BooleanArray;
/// use polars_arrow::compute::boolean_kleene::all;
///
/// let a = BooleanArray::from(&[Some(true), Some(true)]);
/// let b = BooleanArray::from(&[Some(false), Some(true)]);
/// let c = BooleanArray::from(&[None, Some(true)]);
///
/// assert_eq!(all(&a), Some(true));
/// assert_eq!(all(&b), Some(false));
/// assert_eq!(all(&c), None);
/// ```
pub fn all(array: &BooleanArray) -> Option<bool> {
    if array.is_empty() {
        Some(true)
    } else if array.null_count() > 0 {
        if array.into_iter().any(|v| v == Some(false)) {
            Some(false)
        } else {
            None
        }
    } else {
        let vals = array.values();
        Some(vals.unset_bits() == 0)
    }
}
