use arrow::bitmap::{Bitmap, binary_fold, quaternary, ternary};
use arrow::compute::utils::combine_validities_and;
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

/// The value every element of `arr` is known to hold, if its values are one bit and none is null.
fn known_value(arr: &PlBooleanArray) -> Option<bool> {
    (arr.null_count() == 0)
        .then(|| arr.scalar_values())
        .flatten()
}

/// Logical 'or' operation on two arrays with [Kleene logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics)..
pub fn or(lhs: &PlBooleanArray, rhs: &PlBooleanArray) -> PlBooleanArray {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "lhs and rhs must have the same length"
    );

    // A side that is `true` throughout makes the answer `true` throughout, whatever the other
    // side holds — a null included, which `true` absorbs under Kleene logic — and that answer is
    // the single bit it repeats. A side that is `false` throughout leaves the other one as it is,
    // in whatever representation it is in; neither side is written out.
    match (known_value(lhs), known_value(rhs)) {
        (Some(true), _) | (_, Some(true)) => return PlBooleanArray::new_scalar(true, lhs.len()),
        (Some(false), _) => return rhs.clone(),
        (_, Some(false)) => return lhs.clone(),
        (None, None) => {},
    }

    or_flat(&lhs.to_flat(), &rhs.to_flat())
}

/// Logical 'and' operation on two arrays with [Kleene logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics).
pub fn and(lhs: &PlBooleanArray, rhs: &PlBooleanArray) -> PlBooleanArray {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "lhs and rhs must have the same length"
    );

    // The mirror of `or`: `false` is what absorbs a null here, and `true` is what leaves the
    // other side alone.
    match (known_value(lhs), known_value(rhs)) {
        (Some(false), _) | (_, Some(false)) => return PlBooleanArray::new_scalar(false, lhs.len()),
        (Some(true), _) => return rhs.clone(),
        (_, Some(true)) => return lhs.clone(),
        (None, None) => {},
    }

    and_flat(&lhs.to_flat(), &rhs.to_flat())
}

/// Exclusive 'or' operation on two arrays.
pub fn xor(lhs: &PlBooleanArray, rhs: &PlBooleanArray) -> PlBooleanArray {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "lhs and rhs must have the same length"
    );

    // A side that is `false` throughout leaves the other one as it is, and one that is `true`
    // throughout inverts it — which `not` does without writing a scalar values buffer out. Unlike
    // `and` and `or`, neither value absorbs a null: the nulls of the other side carry over.
    match (known_value(lhs), known_value(rhs)) {
        (Some(l), Some(r)) => return PlBooleanArray::new_scalar(l != r, lhs.len()),
        (Some(false), None) => return rhs.clone(),
        (None, Some(false)) => return lhs.clone(),
        (Some(true), None) => return not(rhs),
        (None, Some(true)) => return not(lhs),
        (None, None) => {},
    }

    let lhs = lhs.to_flat();
    let rhs = rhs.to_flat();
    let validity = combine_validities_and(lhs.validity(), rhs.validity());

    PlBooleanArray::new(
        lhs.values() ^ rhs.values(),
        lhs.len(),
        validity.map(PlBitmap::from_bitmap),
    )
}

/// [`or`] for two chunks that each hold one bit per element.
fn or_flat(lhs: &Flat<PlBooleanArray>, rhs: &Flat<PlBooleanArray>) -> PlBooleanArray {
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

/// [`and`] for two chunks that each hold one bit per element.
fn and_flat(lhs: &Flat<PlBooleanArray>, rhs: &Flat<PlBooleanArray>) -> PlBooleanArray {
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
