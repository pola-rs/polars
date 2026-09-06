use arrow::bitmap::{Bitmap, binary_fold, quaternary, ternary};
use polars_array::{Flat, PlBitmap, PlBooleanArray};

/// The validity mask of `arr`, if it holds one bit per element.
pub(crate) fn flat_validity(arr: &PlBooleanArray) -> Option<&Bitmap> {
    arr.validity().and_then(|validity| validity.flat_bitmap())
}

/// Returns whether any of the non-null values in the array are `true`.
///
/// If there are no non-null values, None is returned.
pub fn any(arr: &PlBooleanArray) -> Option<bool> {
    if arr.null_count() == arr.len() {
        return None;
    }

    // Every element reads the one bit a scalar values buffer holds, and at least one of them is
    // non-null: that bit is the answer, and no buffer is walked at all.
    if let Some(value) = arr.scalar_values() {
        return Some(value);
    }

    let values = arr.flat_values().unwrap();

    match flat_validity(arr) {
        Some(validity) => Some(values.intersects_with(validity)),
        // Either there is no mask, or it marks every element valid: the check above has caught
        // the only other thing a scalar mask can say.
        None => Some(values.set_bits() > 0),
    }
}

/// Returns whether all non-null values in the array are `true`.
///
/// If there are no non-null values, None is returned.
pub fn all(arr: &PlBooleanArray) -> Option<bool> {
    if arr.null_count() == arr.len() {
        return None;
    }

    // As in `any`: the one bit every element shares is the answer.
    if let Some(value) = arr.scalar_values() {
        return Some(value);
    }

    let values = arr.flat_values().unwrap();

    match flat_validity(arr) {
        Some(validity) => {
            let false_found = binary_fold(
                values,
                validity,
                |lhs, rhs| (!lhs & rhs) != 0,
                false,
                |a, b| a || b,
            );
            Some(!false_found)
        },
        None => Some(values.unset_bits() == 0),
    }
}

/// Inverts false to true and vice versa. Nulls remain null.
pub fn not(arr: &PlBooleanArray) -> PlBooleanArray {
    // Inverting the backing bitmap keeps the representation: the single bit a scalar values
    // buffer holds inverts in `O(1)` and still stands for every element.
    let inverted = match arr.scalar_values() {
        Some(value) => PlBooleanArray::new_scalar(!value, arr.len()),
        None => PlBooleanArray::from_values(!arr.flat_values().unwrap()),
    };

    inverted.with_validity(arr.validity().map(PlBitmap::from))
}

/// Logical 'or' operation on two arrays with [Kleene logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics)..
pub fn or(lhs: &Flat<PlBooleanArray>, rhs: &Flat<PlBooleanArray>) -> PlBooleanArray {
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
    PlBooleanArray::new(
        lhs_values | rhs_values,
        lhs.len(),
        validity.map(PlBitmap::from_bitmap),
    )
}

/// Logical 'and' operation on two arrays with [Kleene logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics).
pub fn and(lhs: &Flat<PlBooleanArray>, rhs: &Flat<PlBooleanArray>) -> PlBooleanArray {
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
    PlBooleanArray::new(
        lhs_values & rhs_values,
        lhs.len(),
        validity.map(PlBitmap::from_bitmap),
    )
}
