use crate::{array::PrimitiveArray, bitmap::Bitmap, types::Index};

use super::SortOptions;

/// # Safety
/// This function guarantees that:
/// * `get` is only called for `0 <= i < limit`
/// * `cmp` is only called from the co-domain of `get`.
#[inline]
fn k_element_sort_inner<I: Index, T, G, F>(
    indices: &mut [I],
    get: G,
    descending: bool,
    limit: usize,
    mut cmp: F,
) where
    G: Fn(usize) -> T,
    F: FnMut(&T, &T) -> std::cmp::Ordering,
{
    if descending {
        let mut compare = |lhs: &I, rhs: &I| {
            let lhs = get(lhs.to_usize());
            let rhs = get(rhs.to_usize());
            cmp(&rhs, &lhs)
        };
        let (before, _, _) = indices.select_nth_unstable_by(limit, &mut compare);
        before.sort_unstable_by(&mut compare);
    } else {
        let mut compare = |lhs: &I, rhs: &I| {
            let lhs = get(lhs.to_usize());
            let rhs = get(rhs.to_usize());
            cmp(&lhs, &rhs)
        };
        let (before, _, _) = indices.select_nth_unstable_by(limit, &mut compare);
        before.sort_unstable_by(&mut compare);
    }
}

/// # Safety
/// This function guarantees that:
/// * `get` is only called for `0 <= i < limit`
/// * `cmp` is only called from the co-domain of `get`.
#[inline]
fn sort_unstable_by<I, T, G, F>(
    indices: &mut [I],
    get: G,
    mut cmp: F,
    descending: bool,
    limit: usize,
) where
    I: Index,
    G: Fn(usize) -> T,
    F: FnMut(&T, &T) -> std::cmp::Ordering,
{
    if limit != indices.len() {
        return k_element_sort_inner(indices, get, descending, limit, cmp);
    }

    if descending {
        indices.sort_unstable_by(|lhs, rhs| {
            let lhs = get(lhs.to_usize());
            let rhs = get(rhs.to_usize());
            cmp(&rhs, &lhs)
        })
    } else {
        indices.sort_unstable_by(|lhs, rhs| {
            let lhs = get(lhs.to_usize());
            let rhs = get(rhs.to_usize());
            cmp(&lhs, &rhs)
        })
    }
}

/// # Safety
/// This function guarantees that:
/// * `get` is only called for `0 <= i < length`
/// * `cmp` is only called from the co-domain of `get`.
#[inline]
pub(super) fn indices_sorted_unstable_by<I, T, G, F>(
    validity: Option<&Bitmap>,
    get: G,
    cmp: F,
    length: usize,
    options: &SortOptions,
    limit: Option<usize>,
) -> PrimitiveArray<I>
where
    I: Index,
    G: Fn(usize) -> T,
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    let descending = options.descending;

    let limit = limit.unwrap_or(length);
    // Safety: without this, we go out of bounds when limit >= length.
    let limit = limit.min(length);

    let indices = if let Some(validity) = validity {
        let mut indices = vec![I::default(); length];
        if options.nulls_first {
            let mut nulls = 0;
            let mut valids = 0;
            validity
                .iter()
                .zip(I::range(0, length).unwrap())
                .for_each(|(is_valid, index)| {
                    if is_valid {
                        indices[validity.unset_bits() + valids] = index;
                        valids += 1;
                    } else {
                        indices[nulls] = index;
                        nulls += 1;
                    }
                });

            if limit > validity.unset_bits() {
                // when limit is larger, we must sort values:

                // Soundness:
                // all indices in `indices` are by construction `< array.len() == values.len()`
                // limit is by construction < indices.len()
                let limit = limit.saturating_sub(validity.unset_bits());
                let indices = &mut indices.as_mut_slice()[validity.unset_bits()..];
                sort_unstable_by(indices, get, cmp, options.descending, limit)
            }
        } else {
            let last_valid_index = length.saturating_sub(validity.unset_bits());
            let mut nulls = 0;
            let mut valids = 0;
            validity
                .iter()
                .zip(I::range(0, length).unwrap())
                .for_each(|(x, index)| {
                    if x {
                        indices[valids] = index;
                        valids += 1;
                    } else {
                        indices[last_valid_index + nulls] = index;
                        nulls += 1;
                    }
                });

            // Soundness:
            // all indices in `indices` are by construction `< array.len() == values.len()`
            // limit is by construction <= values.len()
            let limit = limit.min(last_valid_index);
            let indices = &mut indices.as_mut_slice()[..last_valid_index];
            sort_unstable_by(indices, get, cmp, options.descending, limit);
        }

        indices.truncate(limit);
        indices.shrink_to_fit();

        indices
    } else {
        let mut indices = I::range(0, length).unwrap().collect::<Vec<_>>();

        sort_unstable_by(&mut indices, get, cmp, descending, limit);
        indices.truncate(limit);
        indices.shrink_to_fit();
        indices
    };

    let data_type = I::PRIMITIVE.into();
    PrimitiveArray::<I>::new(data_type, indices.into(), None)
}
