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
// Compute the indexes after reversing a sorted array, maintaining
// the order of equal elements, in linear time. Faster than sort_impl
//  as we avoid allocating extra memory.
pub(super) fn reverse_stable_no_nulls<I, J, T>(iters: I, len: usize) -> Vec<IdxSize>
where
    I: IntoIterator<Item = J>,
    J: IntoIterator<Item = T>,
    T: TotalOrd + Send + Sync,
{
    let mut current_start: IdxSize = 0;
    let mut current_end: IdxSize = 0;
    let mut rev_idx: Vec<IdxSize> = Vec::with_capacity(len);
    let mut i: IdxSize;
    // We traverse the array, comparing consecutive elements.
    // We maintain the start and end indice of elements with same value.
    // When we see a new element we push the previous indices in reverse order.
    // We do a final reverse to get stable reverse index.
    // Example -
    // 1 2 2 3 3 3 4
    // 0 1 2 3 4 5 6
    // We get start and end position of equal values -
    // 0 1-2 3-5 6
    // We insert the indexes of equal elements in reverse
    // 0 2 1 5 4 3 6
    // Then do a final reverse
    // 6 3 4 5 1 2 0
    let mut previous_element: Option<T> = None;
    for arr_iter in iters {
        for current_element in arr_iter {
            match &previous_element {
                None => {
                    //There is atleast one element
                    current_end = 1;
                },
                Some(prev) => {
                    if current_element.tot_cmp(prev) == Ordering::Equal {
                        current_end += 1;
                    } else {
                        // Insert in reverse order
                        i = current_end;
                        while i > current_start {
                            i -= 1;
                            //SAFETY - we allocated enough
                            unsafe { rev_idx.push_unchecked(i) };
                        }
                        current_start = current_end;
                        current_end += 1;
                    }
                },
            }
            previous_element = Some(current_element);
        }
    }
    // If there are no elements this does nothing
    i = current_end;
    while i > current_start {
        i -= 1;
        unsafe { rev_idx.push_unchecked(i) };
    }
    // Final reverse
    rev_idx.reverse();
    rev_idx
}

pub(super) fn arg_sort<I, J, T>(
    name: PlSmallStr,
    iters: I,
    options: SortOptions,
    null_count: usize,
    mut len: usize,
    is_sorted_flag: IsSorted,
    first_element_null: bool,
) -> IdxCa
where
    I: IntoIterator<Item = J>,
    J: IntoIterator<Item = Option<T>>,
    T: TotalOrd + Send + Sync,
{
    let nulls_last = options.nulls_last;
    let null_cap = if nulls_last { null_count } else { len };

    // Fast path
    // Only if array is already sorted in the required ordered and
    // nulls are also in the correct position
    if ((options.descending && is_sorted_flag == IsSorted::Descending)
        || (!options.descending && is_sorted_flag == IsSorted::Ascending))
        && ((nulls_last && !first_element_null) || (!nulls_last && first_element_null))
    {
        len = options
            .limit
            .map_or(len, |limit| std::cmp::min(limit.try_into().unwrap(), len));
        return ChunkedArray::with_chunk(
            name,
            IdxArr::from_data_default(
                Buffer::from((0..(len as IdxSize)).collect::<Vec<IdxSize>>()),
                None,
            ),
        );
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

    let vals = if let Some(limit) = options.limit {
        let limit = limit as usize;
        // Overwrite output len.
        len = limit;
        let out = if limit >= vals.len() {
            vals.as_mut_slice()
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
    is_sorted_flag: IsSorted,
) -> IdxCa
where
    I: IntoIterator<Item = J>,
    J: IntoIterator<Item = T>,
    T: TotalOrd + Send + Sync,
{
    // Fast path
    // 1) If array is already sorted in the required ordered .
    // 2) If array is reverse sorted -> we do a stable reverse.
    if is_sorted_flag != IsSorted::Not {
        let len_final = options
            .limit
            .map_or(len, |limit| std::cmp::min(limit.try_into().unwrap(), len));
        if (options.descending && is_sorted_flag == IsSorted::Descending)
            || (!options.descending && is_sorted_flag == IsSorted::Ascending)
        {
            return ChunkedArray::with_chunk(
                name,
                IdxArr::from_data_default(
                    Buffer::from((0..(len_final as IdxSize)).collect::<Vec<IdxSize>>()),
                    None,
                ),
            );
        } else if (options.descending && is_sorted_flag == IsSorted::Ascending)
            || (!options.descending && is_sorted_flag == IsSorted::Descending)
        {
            let idx = reverse_stable_no_nulls(iters, len);
            let idx = Buffer::from(idx).sliced(0, len_final);
            return ChunkedArray::with_chunk(name, IdxArr::from_data_default(idx, None));
        }
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

    let vals = if let Some(limit) = options.limit {
        let limit = limit as usize;
        let out = if limit >= vals.len() {
            vals.as_mut_slice()
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
#[cfg(test)]
mod test {
    use sort::arg_sort::reverse_stable_no_nulls;

    use crate::prelude::*;

    #[test]
    fn test_reverse_stable_no_nulls() {
        let a = Int32Chunked::new(
            PlSmallStr::from_static("a"),
            &[
                Some(1), // 0
                Some(2), // 1
                Some(2), // 2
                Some(3), // 3
                Some(3), // 4
                Some(3), // 5
                Some(4), // 6
            ],
        );
        let idx = reverse_stable_no_nulls(&a, 7);
        let expected = [6, 3, 4, 5, 1, 2, 0];
        assert_eq!(idx, expected);

        let a = Int32Chunked::new(
            PlSmallStr::from_static("a"),
            &[
                Some(1), // 0
                Some(2), // 1
                Some(3), // 2
                Some(4), // 3
                Some(5), // 4
                Some(6), // 5
                Some(7), // 6
            ],
        );
        let idx = reverse_stable_no_nulls(&a, 7);
        let expected = [6, 5, 4, 3, 2, 1, 0];
        assert_eq!(idx, expected);

        let a = Int32Chunked::new(
            PlSmallStr::from_static("a"),
            &[
                Some(1), // 0
            ],
        );
        let idx = reverse_stable_no_nulls(&a, 1);
        let expected = [0];
        assert_eq!(idx, expected);

        let empty_array: [i32; 0] = [];
        let a = Int32Chunked::new(PlSmallStr::from_static("a"), &empty_array);
        let idx = reverse_stable_no_nulls(&a, 0);
        assert_eq!(idx.len(), 0);
    }
}
