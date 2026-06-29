use arrow::array::{Array, BooleanArray};
use arrow::bitmap::{binary_fold, quaternary, ternary};
use arrow::datatypes::ArrowDataType;

/// Returns whether any of the non-null values in the array are `true`.
///
/// If there are no non-null values, None is returned.
pub fn any(arr: &BooleanArray) -> Option<bool> {
    let null_count = arr.null_count();
    if null_count == arr.len() {
        None
    } else if null_count == 0 {
        Some(arr.values().set_bits() > 0)
    } else {
        Some(arr.values().intersects_with(arr.validity().unwrap()))
    }
}

/// Returns whether all non-null values in the array are `true`.
///
/// If there are no non-null values, None is returned.
pub fn all(arr: &BooleanArray) -> Option<bool> {
    let null_count = arr.null_count();
    if null_count == arr.len() {
        None
    } else if null_count == 0 {
        Some(arr.values().unset_bits() == 0)
    } else {
        let false_found = binary_fold(
            arr.values(),
            arr.validity().unwrap(),
            |lhs, rhs| (!lhs & rhs) != 0,
            false,
            |a, b| a || b,
        );
        Some(!false_found)
    }
}

/// Inverts false to true and vice versa. Nulls remain null.
pub fn not(array: &BooleanArray) -> BooleanArray {
    let values = !array.values();
    let validity = array.validity().cloned();
    BooleanArray::new(ArrowDataType::Boolean, values, validity)
}

/// Logical 'or' operation on two arrays with [Kleene logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics)..
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

/// Logical 'and' operation on two arrays with [Kleene logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics).
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
