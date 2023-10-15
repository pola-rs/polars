use crate::array::{DictionaryArray, DictionaryKey};

pub(super) fn equal<K: DictionaryKey>(lhs: &DictionaryArray<K>, rhs: &DictionaryArray<K>) -> bool {
    if !(lhs.data_type() == rhs.data_type() && lhs.len() == rhs.len()) {
        return false;
    };

    // if x is not valid and y is but its child is not, the slots are equal.
    lhs.iter().zip(rhs.iter()).all(|(x, y)| match (&x, &y) {
        (None, Some(y)) => !y.is_valid(),
        (Some(x), None) => !x.is_valid(),
        _ => x == y,
    })
}
