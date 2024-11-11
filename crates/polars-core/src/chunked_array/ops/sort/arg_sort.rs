use polars_utils::itertools::Itertools;

use self::row_encode::_get_rows_encoded;
use super::*;

// Reduce monomorphisation.
fn sort_impl<T>(vals: &mut [(IdxSize, T)], options: SortOptions)
where
    T: TotalOrd + Send + Sync,
{
    sort_by_branch(
        vals,
        options.descending,
        |a, b| a.1.tot_cmp(&b.1),
        options.multithreaded,
    );
}
fn reverse_stable_no_nulls<I, J, T>(iters: I, len: usize) -> Vec<u32>
where
    I: IntoIterator<Item = J>,
    J: IntoIterator<Item = T>,
    T: TotalOrd + Send + Sync,
{
    let mut current_start = 0;
    let mut current_end = 1;
    let mut flattened_iter = iters.into_iter().flatten();
    let first_element = flattened_iter.next();
    let mut rev_idx = Vec::with_capacity(len);
    let mut i: IdxSize;
    match first_element {
        Some(value) => {
            let mut previous_element = value;
            while let Some(current_element) = flattened_iter.next() {
                if current_element.tot_cmp(&previous_element) == Ordering::Equal {
                    current_end += 1;
                } else {
                    //rev_idx.extend((current_start..current_end).rev());
                    i = current_end;
                    while i > current_start {
                        i = i - 1;
                        unsafe { rev_idx.push_unchecked(i) };
                    }
                    current_start = current_end;
                    current_end = current_end + 1;
                }
                previous_element = current_element;
            }
            rev_idx.extend((current_start..current_end).rev());
            rev_idx.reverse();
            return rev_idx;
        },
        None => return rev_idx,
    }
}

pub(super) fn arg_sort<I, J, T>(
    name: PlSmallStr,
    iters: I,
    options: SortOptions,
    null_count: usize,
    mut len: usize,
    is_sorted_descending_flag: bool,
    is_sorted_ascending_flag: bool,
    first_element_null: bool,
) -> IdxCa
where
    I: IntoIterator<Item = J>,
    J: IntoIterator<Item = Option<T>>,
    T: TotalOrd + Send + Sync,
{
    let nulls_last = options.nulls_last;
    let null_cap = if nulls_last { null_count } else { len };

    if (options.descending && is_sorted_descending_flag)
        || (!options.descending && is_sorted_ascending_flag)
    {
        if (nulls_last && !first_element_null) || (!nulls_last && first_element_null) {
            return ChunkedArray::with_chunk(
                name,
                IdxArr::from_data_default(
                    Buffer::from((0..(len as IdxSize)).collect::<Vec<IdxSize>>()),
                    None,
                ),
            );
        }
    }

    let mut vals = Vec::with_capacity(len - null_count);
    let mut nulls_idx = Vec::with_capacity(null_cap);
    let mut count: IdxSize = 0;

    for arr_iter in iters {
        let iter = arr_iter.into_iter().filter_map(|v| {
            let i = count;
            count += 1;
            match v {
                Some(v) => Some((i, v)),
                None => {
                    // SAFETY: we allocated enough.
                    unsafe { nulls_idx.push_unchecked(i) };
                    None
                },
            }
        });
        vals.extend(iter);
    }

    let vals = if let Some((limit, desc)) = options.limit {
        let limit = limit as usize;
        // Overwrite output len.
        len = limit;
        let out = if limit >= vals.len() {
            vals.as_mut_slice()
        } else if desc {
            let (lower, _el, _upper) = vals
                .as_mut_slice()
                .select_nth_unstable_by(limit, |a, b| b.1.tot_cmp(&a.1));
            lower
        } else {
            let (lower, _el, _upper) = vals
                .as_mut_slice()
                .select_nth_unstable_by(limit, |a, b| a.1.tot_cmp(&b.1));
            lower
        };

        sort_impl(out, options);
        out
    } else {
        sort_impl(vals.as_mut_slice(), options);
        vals.as_slice()
    };

    let iter = vals.iter().map(|(idx, _v)| idx).copied();
    let idx = if nulls_last {
        let mut idx = Vec::with_capacity(len);
        idx.extend(iter);

        let nulls_idx = if options.limit.is_some() {
            &nulls_idx[..len - idx.len()]
        } else {
            &nulls_idx
        };
        idx.extend_from_slice(nulls_idx);
        idx
    } else if options.limit.is_some() {
        nulls_idx.extend(iter.take(len - nulls_idx.len()));
        nulls_idx
    } else {
        let ptr = nulls_idx.as_ptr() as usize;
        nulls_idx.extend(iter);
        // We had a realloc.
        debug_assert_eq!(nulls_idx.as_ptr() as usize, ptr);
        nulls_idx
    };

    ChunkedArray::with_chunk(name, IdxArr::from_data_default(Buffer::from(idx), None))
}

pub(super) fn arg_sort_no_nulls<I, J, T>(
    name: PlSmallStr,
    iters: I,
    options: SortOptions,
    len: usize,
    is_sorted_descending_flag: bool,
    is_sorted_ascending_flag: bool,
) -> IdxCa
where
    I: IntoIterator<Item = J>,
    J: IntoIterator<Item = T>,
    T: TotalOrd + Send + Sync,
{
    if (options.descending && is_sorted_descending_flag)
        || (!options.descending && is_sorted_ascending_flag)
    {
        return ChunkedArray::with_chunk(
            name,
            IdxArr::from_data_default(
                Buffer::from((0..(len as IdxSize)).collect::<Vec<IdxSize>>()),
                None,
            ),
        );
    } else if (options.descending && is_sorted_ascending_flag)
        || (!options.descending && is_sorted_descending_flag)
    {
        return ChunkedArray::with_chunk(
            name,
            IdxArr::from_data_default(Buffer::from(reverse_stable_no_nulls(iters, len)), None),
        );
    }

    let mut vals = Vec::with_capacity(len);

    let mut count: IdxSize = 0;
    for arr_iter in iters {
        vals.extend(arr_iter.into_iter().map(|v| {
            let idx = count;
            count += 1;
            (idx, v)
        }));
    }

    let vals = if let Some((limit, desc)) = options.limit {
        let limit = limit as usize;
        let out = if limit >= vals.len() {
            vals.as_mut_slice()
        } else if desc {
            let (lower, _el, _upper) = vals
                .as_mut_slice()
                .select_nth_unstable_by(limit, |a, b| b.1.tot_cmp(&a.1));
            lower
        } else {
            let (lower, _el, _upper) = vals
                .as_mut_slice()
                .select_nth_unstable_by(limit, |a, b| a.1.tot_cmp(&b.1));
            lower
        };
        sort_impl(out, options);
        out
    } else {
        sort_impl(vals.as_mut_slice(), options);
        vals.as_slice()
    };

    let iter = vals.iter().map(|(idx, _v)| idx).copied();
    let idx: Vec<_> = iter.collect_trusted();

    ChunkedArray::with_chunk(name, IdxArr::from_data_default(Buffer::from(idx), None))
}

pub(crate) fn arg_sort_row_fmt(
    by: &[Column],
    descending: bool,
    nulls_last: bool,
    parallel: bool,
) -> PolarsResult<IdxCa> {
    let rows_encoded = _get_rows_encoded(by, &[descending], &[nulls_last])?;
    let mut items: Vec<_> = rows_encoded.iter().enumerate_idx().collect();

    if parallel {
        POOL.install(|| items.par_sort_by(|a, b| a.1.cmp(b.1)));
    } else {
        items.sort_by(|a, b| a.1.cmp(b.1));
    }

    let ca: NoNull<IdxCa> = items.into_iter().map(|tpl| tpl.0).collect();
    Ok(ca.into_inner())
}
