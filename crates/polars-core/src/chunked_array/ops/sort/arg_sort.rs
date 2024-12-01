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

pub(super) fn arg_sort<I, J, T>(
    name: PlSmallStr,
    iters: I,
    options: SortOptions,
    null_count: usize,
    mut len: usize,
) -> IdxCa
where
    I: IntoIterator<Item = J>,
    J: IntoIterator<Item = Option<T>>,
    T: TotalOrd + Send + Sync,
{
    let nulls_last = options.nulls_last;

    let mut vals = Vec::with_capacity(len - null_count);

    let null_cap = if nulls_last { null_count } else { len };
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
) -> IdxCa
where
    I: IntoIterator<Item = J>,
    J: IntoIterator<Item = T>,
    T: TotalOrd + Send + Sync,
{
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
