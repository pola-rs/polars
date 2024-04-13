/// Find the index of the first element of `arr` that is greater
/// or equal to `val`.
/// Assumes that `arr` is sorted.
pub fn find_first_ge_index<T>(arr: &[T], val: T) -> usize
where
    T: Ord,
{
    match arr.binary_search(&val) {
        Ok(x) => x,
        Err(x) => x,
    }
}

/// Find the index of the first element of `arr` that is greater
/// than `val`.
/// Assumes that `arr` is sorted.
pub fn find_first_gt_index<T>(arr: &[T], val: T) -> usize
where
    T: Ord,
{
    match arr.binary_search(&val) {
        Ok(x) => x + 1,
        Err(x) => x,
    }
}
