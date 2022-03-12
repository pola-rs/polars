use super::*;

fn default_order<T: PartialOrd>(a: &(IdxSize, T), b: &(IdxSize, T)) -> Ordering {
    a.1.partial_cmp(&b.1).unwrap()
}

fn reverse_order<T: PartialOrd>(a: &(IdxSize, T), b: &(IdxSize, T)) -> Ordering {
    b.1.partial_cmp(&a.1).unwrap()
}

pub(super) fn argsort<I, J, K>(
    name: &str,
    iters: I,
    options: SortOptions,
    null_count: usize,
    len: usize,
) -> IdxCa
where
    I: IntoIterator<Item = J>,
    J: IntoIterator<Item = Option<K>>,
    K: PartialOrd + Send + Sync,
{
    let reverse = options.descending;
    let nulls_last = options.nulls_last;

    let mut vals = Vec::with_capacity(len - null_count);

    // if we sort reverse, the nulls are last
    // and need to be extended to the indices in reverse order
    let null_cap = if reverse || nulls_last {
        null_count
        // if we sort normally, the nulls are first
        // and can be extended with the sorted indices
    } else {
        len
    };
    let mut nulls_idx = Vec::with_capacity(null_cap);
    let mut count: IdxSize = 0;

    for arr_iter in iters {
        let iter = arr_iter.into_iter().filter_map(|v| {
            let i = count;
            count += 1;
            match v {
                Some(v) => Some((i, v)),
                None => {
                    // Safety:
                    // we allocated enough
                    unsafe { nulls_idx.push_unchecked(i) };
                    None
                }
            }
        });
        vals.extend(iter);
    }

    argsort_branch(vals.as_mut_slice(), reverse, default_order, reverse_order);

    let iter = vals.into_iter().map(|(idx, _v)| idx);
    let idx = if reverse || nulls_last {
        let mut idx = Vec::with_capacity(len);
        idx.extend(iter);
        idx.extend(nulls_idx.into_iter().rev());
        idx
    } else {
        let ptr = nulls_idx.as_ptr() as usize;
        nulls_idx.extend(iter);
        // we had a realloc
        debug_assert_eq!(nulls_idx.as_ptr() as usize, ptr);
        nulls_idx
    };

    let arr = IdxArr::from_data_default(Buffer::from(idx), None);
    IdxCa::from_chunks(name, vec![Arc::new(arr)])
}
