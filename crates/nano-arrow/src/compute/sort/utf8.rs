use crate::array::{DictionaryArray, DictionaryKey, PrimitiveArray, Utf8Array};
use crate::offset::Offset;
use crate::types::Index;

use super::common;
use super::SortOptions;

pub(super) fn indices_sorted_unstable_by<I: Index, O: Offset>(
    array: &Utf8Array<O>,
    options: &SortOptions,
    limit: Option<usize>,
) -> PrimitiveArray<I> {
    let get = |idx| unsafe { array.value_unchecked(idx) };
    let cmp = |lhs: &&str, rhs: &&str| lhs.cmp(rhs);
    common::indices_sorted_unstable_by(array.validity(), get, cmp, array.len(), options, limit)
}

pub(super) fn indices_sorted_unstable_by_dictionary<I: Index, K: DictionaryKey, O: Offset>(
    array: &DictionaryArray<K>,
    options: &SortOptions,
    limit: Option<usize>,
) -> PrimitiveArray<I> {
    let keys = array.keys();

    let dict = array
        .values()
        .as_any()
        .downcast_ref::<Utf8Array<O>>()
        .unwrap();

    let get = |index| unsafe {
        // safety: indices_sorted_unstable_by is guaranteed to get items in bounds
        let index = keys.value_unchecked(index);
        // safety: dictionaries are guaranteed to have valid usize keys
        let index = index.as_usize();
        // safety: dictionaries are guaranteed to have keys in bounds
        dict.value_unchecked(index)
    };

    let cmp = |lhs: &&str, rhs: &&str| lhs.cmp(rhs);
    common::indices_sorted_unstable_by(array.validity(), get, cmp, array.len(), options, limit)
}
