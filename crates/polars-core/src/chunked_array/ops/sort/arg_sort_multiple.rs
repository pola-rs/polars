use compare_inner::NullOrderCmp;
use polars_utils::itertools::Itertools;

use super::*;
use crate::chunked_array::ops::row_encode::_get_rows_encoded;

pub(crate) fn args_validate<T: PolarsDataType>(
    ca: &ChunkedArray<T>,
    other: &[Column],
    param_value: &[bool],
    param_name: &str,
) -> PolarsResult<()> {
    for s in other {
        assert_eq!(ca.len(), s.len());
    }
    polars_ensure!(other.len() == (param_value.len() - 1),
        ComputeError:
        "the length of `{}` ({}) does not match the number of series ({})",
        param_name, param_value.len(), other.len() + 1,
    );
    Ok(())
}

pub(crate) fn arg_sort_multiple_impl<T: NullOrderCmp + Send + Copy>(
    mut vals: Vec<(IdxSize, T)>,
    by: &[Column],
    options: &SortMultipleOptions,
) -> PolarsResult<IdxCa> {
    let nulls_last = &options.nulls_last;
    let descending = &options.descending;

    debug_assert_eq!(descending.len() - 1, by.len());
    debug_assert_eq!(nulls_last.len() - 1, by.len());

    let compare_inner: Vec<_> = by
        .iter()
        .map(|c| c.into_total_ord_inner())
        .collect_trusted();

    let first_descending = descending[0];
    let first_nulls_last = nulls_last[0];

    let compare = |tpl_a: &(_, T), tpl_b: &(_, T)| -> Ordering {
        match (
            first_descending,
            tpl_a
                .1
                .null_order_cmp(&tpl_b.1, first_nulls_last ^ first_descending),
        ) {
            // if ordering is equal, we check the other arrays until we find a non-equal ordering
            // if we have exhausted all arrays, we keep the equal ordering.
            (_, Ordering::Equal) => {
                let idx_a = tpl_a.0 as usize;
                let idx_b = tpl_b.0 as usize;
                unsafe {
                    ordering_other_columns(
                        &compare_inner,
                        descending.get_unchecked(1..),
                        nulls_last.get_unchecked(1..),
                        idx_a,
                        idx_b,
                    )
                }
            },
            (true, Ordering::Less) => Ordering::Greater,
            (true, Ordering::Greater) => Ordering::Less,
            (_, ord) => ord,
        }
    };

    match (options.multithreaded, options.maintain_order) {
        (true, true) => POOL.install(|| {
            vals.par_sort_by(compare);
        }),
        (true, false) => POOL.install(|| {
            vals.par_sort_unstable_by(compare);
        }),
        (false, true) => vals.sort_by(compare),
        (false, false) => vals.sort_unstable_by(compare),
    }

    let ca: NoNull<IdxCa> = vals.into_iter().map(|(idx, _v)| idx).collect_trusted();
    // Don't set to sorted. Argsort indices are not sorted.
    Ok(ca.into_inner())
}

pub(crate) fn argsort_multiple_row_fmt(
    by: &[Column],
    mut descending: Vec<bool>,
    mut nulls_last: Vec<bool>,
    parallel: bool,
) -> PolarsResult<IdxCa> {
    _broadcast_bools(by.len(), &mut descending);
    _broadcast_bools(by.len(), &mut nulls_last);

    let rows_encoded = _get_rows_encoded(by, &descending, &nulls_last)?;
    let mut items: Vec<_> = rows_encoded.iter().enumerate_idx().collect();

    if parallel {
        POOL.install(|| items.par_sort_by_key(|i| i.1));
    } else {
        items.sort_by_key(|i| i.1);
    }

    let ca: NoNull<IdxCa> = items.into_iter().map(|tpl| tpl.0).collect();
    Ok(ca.into_inner())
}
