use polars_arrow::data_types::IsFloat;
use polars_utils::iter::EnumerateIdxTrait;

use super::*;
use crate::POOL;

pub(crate) fn args_validate<T: PolarsDataType>(
    ca: &ChunkedArray<T>,
    other: &[Series],
    descending: &[bool],
) -> PolarsResult<()> {
    for s in other {
        assert_eq!(ca.len(), s.len());
    }
    if other.len() != (descending.len() - 1) {
        return Err(PolarsError::ComputeError(
            format!(
                "The amount of ordering booleans: {} does not match that no. of Series: {}",
                descending.len(),
                other.len() + 1
            )
            .into(),
        ));
    }

    assert_eq!(other.len(), descending.len() - 1);
    Ok(())
}

pub(crate) fn arg_sort_multiple_impl<T: PartialOrd + Send + IsFloat + Copy>(
    mut vals: Vec<(IdxSize, T)>,
    other: &[Series],
    descending: &[bool],
) -> PolarsResult<IdxCa> {
    assert_eq!(descending.len() - 1, other.len());
    let compare_inner: Vec<_> = other
        .iter()
        .map(|s| s.into_partial_ord_inner())
        .collect_trusted();

    let first_descending = descending[0];
    POOL.install(|| {
        vals.par_sort_by(|tpl_a, tpl_b| {
            match (first_descending, compare_fn_nan_max(&tpl_a.1, &tpl_b.1)) {
                // if ordering is equal, we check the other arrays until we find a non-equal ordering
                // if we have exhausted all arrays, we keep the equal ordering.
                (_, Ordering::Equal) => {
                    let idx_a = tpl_a.0 as usize;
                    let idx_b = tpl_b.0 as usize;
                    unsafe {
                        ordering_other_columns(
                            &compare_inner,
                            descending.get_unchecked(1..),
                            idx_a,
                            idx_b,
                        )
                    }
                }
                (true, Ordering::Less) => Ordering::Greater,
                (true, Ordering::Greater) => Ordering::Less,
                (_, ord) => ord,
            }
        });
    });
    let ca: NoNull<IdxCa> = vals.into_iter().map(|(idx, _v)| idx).collect_trusted();
    // Don't set to sorted. Argsort indices are not sorted.
    Ok(ca.into_inner())
}

pub(crate) fn argsort_multiple_row_fmt(
    by: &[Series],
    descending: &[bool],
    nulls_last: bool,
    parallel: bool,
) -> PolarsResult<IdxCa> {
    use polars_row::{convert_columns, SortField};

    let mut cols = Vec::with_capacity(by.len());
    let mut fields = Vec::with_capacity(by.len());

    for (by, descending) in by.iter().zip(descending) {
        let by = convert_sort_column_multi_sort(by, true)?;
        let data_type = by.dtype().to_arrow();
        let by = by.rechunk();
        cols.push(by.chunks()[0].clone());
        fields.push(SortField {
            descending: *descending,
            nulls_last,
            data_type,
        })
    }
    let rows_encoded = convert_columns(&cols, fields);
    let mut items: Vec<_> = rows_encoded.iter().enumerate_idx().collect();

    if parallel {
        POOL.install(|| items.par_sort_by(|a, b| a.1.cmp(b.1)));
    } else {
        items.sort_by(|a, b| a.1.cmp(b.1));
    }

    let ca: NoNull<IdxCa> = items.into_iter().map(|tpl| tpl.0).collect();
    Ok(ca.into_inner())
}
