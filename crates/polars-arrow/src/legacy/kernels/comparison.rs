use crate::array::{Array, BooleanArray, FixedSizeListArray};
use crate::bitmap::utils::count_zeros;
use crate::legacy::utils::combine_validities_and;

fn fixed_size_list_cmp<F1, F2>(
    a: &FixedSizeListArray,
    b: &FixedSizeListArray,
    cmp_func: F1,
    func: F2,
) -> BooleanArray
where
    F1: Fn(&dyn Array, &dyn Array) -> BooleanArray,
    F2: Fn(usize) -> bool,
{
    assert_eq!(a.size(), b.size());
    let mask = cmp_func(a.values().as_ref(), b.values().as_ref());
    let mask = combine_validities_and(Some(mask.values()), mask.validity()).unwrap();
    let (slice, offset, _len) = mask.as_slice();
    assert_eq!(offset, 0);

    let width = a.size();
    let iter = (0..a.len()).map(|i| func(count_zeros(slice, i * width, width)));
    // range is trustedlen
    unsafe { BooleanArray::from_trusted_len_values_iter_unchecked(iter) }
}

pub fn fixed_size_list_eq(a: &FixedSizeListArray, b: &FixedSizeListArray) -> BooleanArray {
    fixed_size_list_cmp(a, b, crate::compute::comparison::eq, |count_zeros| {
        count_zeros == 0
    })
}
pub fn fixed_size_list_neq(a: &FixedSizeListArray, b: &FixedSizeListArray) -> BooleanArray {
    fixed_size_list_cmp(a, b, crate::compute::comparison::eq, |count_zeros| {
        count_zeros != 0
    })
}
pub fn fixed_size_list_eq_missing(a: &FixedSizeListArray, b: &FixedSizeListArray) -> BooleanArray {
    fixed_size_list_cmp(
        a,
        b,
        crate::compute::comparison::eq_and_validity,
        |count_zeros| count_zeros == 0,
    )
}
pub fn fixed_size_list_neq_missing(a: &FixedSizeListArray, b: &FixedSizeListArray) -> BooleanArray {
    fixed_size_list_cmp(
        a,
        b,
        crate::compute::comparison::eq_and_validity,
        |count_zeros| count_zeros != 0,
    )
}
