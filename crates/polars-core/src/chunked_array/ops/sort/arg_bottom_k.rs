use polars_utils::itertools::Itertools;

use super::*;

#[derive(Eq)]
struct CompareRow<'a> {
    idx: IdxSize,
    bytes: &'a [u8],
}

impl PartialEq for CompareRow<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}

impl Ord for CompareRow<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.bytes.cmp(other.bytes)
    }
}

impl PartialOrd for CompareRow<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Return the indices of the bottom k elements.
///
/// Similar to .argsort() then .slice(0, k) but with a more efficient implementation.
pub fn _arg_bottom_k(
    k: usize,
    by_column: &[Series],
    sort_options: &mut SortMultipleOptions,
) -> PolarsResult<NoNull<IdxCa>> {
    let from_n_rows = by_column[0].len();
    _broadcast_bools(by_column.len(), &mut sort_options.descending);
    _broadcast_bools(by_column.len(), &mut sort_options.nulls_last);

    let encoded = _get_rows_encoded(
        by_column,
        &sort_options.descending,
        &sort_options.nulls_last,
    )?;
    let arr = encoded.into_array();
    let mut rows = arr
        .values_iter()
        .enumerate_idx()
        .map(|(idx, bytes)| CompareRow { idx, bytes })
        .collect::<Vec<_>>();

    let sorted = if k >= from_n_rows {
        match (sort_options.multithreaded, sort_options.maintain_order) {
            (true, true) => POOL.install(|| {
                rows.par_sort();
            }),
            (true, false) => POOL.install(|| {
                rows.par_sort_unstable();
            }),
            (false, true) => rows.sort(),
            (false, false) => rows.sort_unstable(),
        }
        &rows
    } else if sort_options.maintain_order {
        // todo: maybe there is some more efficient method, comparable to select_nth_unstable
        if sort_options.multithreaded {
            POOL.install(|| {
                rows.par_sort();
            })
        } else {
            rows.sort();
        }
        &rows[..k]
    } else {
        // todo: possible multi threaded `select_nth_unstable`?
        let (lower, _el, _upper) = rows.select_nth_unstable(k);
        if sort_options.multithreaded {
            POOL.install(|| {
                lower.par_sort_unstable();
            })
        } else {
            lower.sort_unstable();
        }
        &*lower
    };

    let idx: NoNull<IdxCa> = sorted.iter().map(|cmp_row| cmp_row.idx).collect();
    Ok(idx)
}
