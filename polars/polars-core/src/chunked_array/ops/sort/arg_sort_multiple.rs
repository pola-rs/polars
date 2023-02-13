use polars_arrow::data_types::IsFloat;

use super::*;

pub(crate) fn args_validate<T: PolarsDataType>(
    ca: &ChunkedArray<T>,
    other: &[Series],
    reverse: &[bool],
) -> PolarsResult<()> {
    for s in other {
        assert_eq!(ca.len(), s.len());
    }
    if other.len() != (reverse.len() - 1) {
        return Err(PolarsError::ComputeError(
            format!(
                "The amount of ordering booleans: {} does not match that no. of Series: {}",
                reverse.len(),
                other.len() + 1
            )
            .into(),
        ));
    }

    assert_eq!(other.len(), reverse.len() - 1);
    Ok(())
}

pub(crate) fn arg_sort_multiple_impl<T: PartialOrd + Send + IsFloat + Copy>(
    mut vals: Vec<(IdxSize, T)>,
    other: &[Series],
    reverse: &[bool],
) -> PolarsResult<IdxCa> {
    assert_eq!(reverse.len() - 1, other.len());
    let compare_inner: Vec<_> = other
        .iter()
        .map(|s| s.into_partial_ord_inner())
        .collect_trusted();

    let first_reverse = reverse[0];
    vals.par_sort_by(|tpl_a, tpl_b| {
        match (first_reverse, compare_fn_nan_max(&tpl_a.1, &tpl_b.1)) {
            // if ordering is equal, we check the other arrays until we find a non-equal ordering
            // if we have exhausted all arrays, we keep the equal ordering.
            (_, Ordering::Equal) => {
                let idx_a = tpl_a.0 as usize;
                let idx_b = tpl_b.0 as usize;
                unsafe {
                    ordering_other_columns(&compare_inner, reverse.get_unchecked(1..), idx_a, idx_b)
                }
            }
            (true, Ordering::Less) => Ordering::Greater,
            (true, Ordering::Greater) => Ordering::Less,
            (_, ord) => ord,
        }
    });
    let ca: NoNull<IdxCa> = vals.into_iter().map(|(idx, _v)| idx).collect_trusted();
    // Don't set to sorted. Argsort indices are not sorted.
    Ok(ca.into_inner())
}
