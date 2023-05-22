use arrow::array::{BooleanArray, FixedSizeListArray};
use arrow::bitmap::utils::count_zeros;

use crate::utils::combine_validities_and;

fn fixed_size_list_cmp<F>(a: &FixedSizeListArray, b: &FixedSizeListArray, func: F) -> BooleanArray
where
    F: Fn(usize) -> bool,
{
    assert_eq!(a.size(), b.size());
    let mask = arrow::compute::comparison::eq(a.values().as_ref(), b.values().as_ref());
    let mask = combine_validities_and(Some(mask.values()), mask.validity()).unwrap();
    let (slice, offset, _len) = mask.as_slice();
    assert_eq!(offset, 0);

    let width = a.size();
    let iter = (0..a.len()).map(|i| func(count_zeros(slice, i, width)));
    // range is trustedlen
    unsafe { BooleanArray::from_trusted_len_values_iter_unchecked(iter) }
}

pub fn fixed_size_list_eq(a: &FixedSizeListArray, b: &FixedSizeListArray) -> BooleanArray {
    fixed_size_list_cmp(a, b, |count_zeros| count_zeros == 0)
}
pub fn fixed_size_list_neq(a: &FixedSizeListArray, b: &FixedSizeListArray) -> BooleanArray {
    fixed_size_list_cmp(a, b, |count_zeros| count_zeros != 0)
}
