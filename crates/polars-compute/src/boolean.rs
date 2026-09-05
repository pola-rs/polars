use arrow::bitmap::{Bitmap, binary_fold, quaternary, ternary};
use polars_array::{ArrayRepr, Flat, PlBitmap, PlBooleanArray};

/// The validity mask of `arr`, if it holds one bit per element.
///
/// A scalar mask marks every element null or none of them, so a caller that has already told
/// those two apart has nothing left to read in it.
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
    let values = match arr.values_repr() {
        ArrayRepr::Scalar(value) => return Some(value),
        ArrayRepr::Flat(values) => values,
    };

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
    let values = match arr.values_repr() {
        ArrayRepr::Scalar(value) => return Some(value),
        ArrayRepr::Flat(values) => values,
    };

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
    let inverted = match arr.values_repr() {
        ArrayRepr::Scalar(value) => PlBooleanArray::new_scalar(!value, arr.len()),
        ArrayRepr::Flat(values) => PlBooleanArray::from_values(!values),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `length` copies of `value`, marked by `validity`, in both representations: the scalar one
    /// holds a single bit per buffer, and the flat one a bit per element.
    fn repeated(value: bool, validity: Option<bool>, length: usize) -> [PlBooleanArray; 2] {
        let scalar = PlBooleanArray::new_scalar(value, length)
            .with_validity(validity.map(|valid| PlBitmap::new_scalar(valid, length)));
        let flat = PlBooleanArray::new(
            Bitmap::new_with_value(value, length),
            length,
            (validity.map(|valid| Bitmap::new_with_value(valid, length)))
                .map(PlBitmap::from_bitmap),
        );
        assert_eq!(scalar, flat);
        [scalar, flat]
    }

    #[test]
    fn a_repeated_value_reads_the_same_either_way() {
        for length in [0, 1, 3, 64, 100] {
            for value in [false, true] {
                for validity in [None, Some(true), Some(false)] {
                    let [scalar, flat] = repeated(value, validity, length);

                    // The one value every element repeats is the answer to both questions, and
                    // an array of no elements or of nothing but nulls answers neither.
                    let expected = (length > 0 && validity != Some(false)).then_some(value);
                    assert_eq!(any(&scalar), expected, "any of {scalar:?}");
                    assert_eq!(all(&scalar), expected, "all of {scalar:?}");
                    assert_eq!(any(&flat), expected, "any of {flat:?}");
                    assert_eq!(all(&flat), expected, "all of {flat:?}");

                    assert_eq!(not(&scalar), not(&flat));
                    // Inverting a scalar array inverts the single bit it is backed by.
                    assert!(not(&scalar).values_are_scalar() || length <= 1);
                }
            }
        }
    }

    /// A values buffer that holds a single bit still stands for every element when the mask holds
    /// one bit per element.
    #[test]
    fn a_repeated_value_under_a_flat_mask() {
        let arr = PlBooleanArray::new_scalar(true, 4).with_validity(Some(PlBitmap::from_bitmap(
            Bitmap::from_iter([false, true, true, false]),
        )));
        assert_eq!(any(&arr), Some(true));
        assert_eq!(all(&arr), Some(true));

        // Every element that is left is null, so there is nothing to answer for.
        let none = PlBooleanArray::new_scalar(true, 2).with_validity(Some(PlBitmap::from_bitmap(
            Bitmap::from_iter([false, false]),
        )));
        assert_eq!(any(&none), None);
        assert_eq!(all(&none), None);
    }

    #[test]
    fn null_elements_are_passed_over() {
        //            null   false  true
        let values = Bitmap::from_iter([true, false, true]);
        let validity = Bitmap::from_iter([false, true, true]);
        let arr = PlBooleanArray::new(values, 3, Some(PlBitmap::from_bitmap(validity)));

        assert_eq!(any(&arr), Some(true));
        assert_eq!(all(&arr), Some(false));
        assert_eq!(
            not(&arr),
            PlBooleanArray::from_iter([None, Some(true), Some(false)])
        );
    }
}
