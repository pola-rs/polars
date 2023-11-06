use super::*;

/// Probe the build table and add tuples to the results (inner join)
fn probe_outer<T, F, G, H>(
    probe_hashes: &[Vec<(u64, T)>],
    hash_tbls: &mut [PlHashMap<T, (bool, Vec<IdxSize>)>],
    results: &mut Vec<(Option<IdxSize>, Option<IdxSize>)>,
    n_tables: u64,
    // Function that get index_a, index_b when there is a match and pushes to result
    swap_fn_match: F,
    // Function that get index_a when there is no match and pushes to result
    swap_fn_no_match: G,
    // Function that get index_b from the build table that did not match any in A and pushes to result
    swap_fn_drain: H,
) where
    T: Send + Hash + Eq + Sync + Copy,
    // idx_a, idx_b -> ...
    F: Fn(IdxSize, IdxSize) -> (Option<IdxSize>, Option<IdxSize>),
    // idx_a -> ...
    G: Fn(IdxSize) -> (Option<IdxSize>, Option<IdxSize>),
    // idx_b -> ...
    H: Fn(IdxSize) -> (Option<IdxSize>, Option<IdxSize>),
{
    // needed for the partition shift instead of modulo to make sense
    let mut idx_a = 0;
    for probe_hashes in probe_hashes {
        for (h, key) in probe_hashes {
            let h = *h;
            // probe table that contains the hashed value
            let current_probe_table =
                unsafe { get_hash_tbl_threaded_join_mut_partitioned(h, hash_tbls, n_tables) };

            let entry = current_probe_table
                .raw_entry_mut()
                .from_key_hashed_nocheck(h, key);

            match entry {
                // match and remove
                RawEntryMut::Occupied(mut occupied) => {
                    let (tracker, indexes_b) = occupied.get_mut();
                    *tracker = true;
                    results.extend(indexes_b.iter().map(|&idx_b| swap_fn_match(idx_a, idx_b)))
                },
                // no match
                RawEntryMut::Vacant(_) => results.push(swap_fn_no_match(idx_a)),
            }
            idx_a += 1;
        }
    }

    for hash_tbl in hash_tbls {
        hash_tbl.iter().for_each(|(_k, (tracker, indexes_b))| {
            // remaining joined values from the right table
            if !*tracker {
                results.extend(indexes_b.iter().map(|&idx_b| swap_fn_drain(idx_b)))
            }
        });
    }
}

/// Hash join outer. Both left and right can have no match so Options
pub(super) fn hash_join_tuples_outer<T, I, J>(
    probe: Vec<I>,
    build: Vec<J>,
    swapped: bool,
    validate: JoinValidation,
) -> PolarsResult<Vec<(Option<IdxSize>, Option<IdxSize>)>>
where
    I: IntoIterator<Item = T>,
    J: IntoIterator<Item = T>,
    <J as IntoIterator>::IntoIter: TrustedLen + Send,
    <I as IntoIterator>::IntoIter: TrustedLen + Send,
    T: Hash + Eq + Copy + Sync + Send,
{
    let probe = probe.into_iter().map(|i| i.into_iter()).collect::<Vec<_>>();
    let build = build.into_iter().map(|i| i.into_iter()).collect::<Vec<_>>();
    // This function is partially multi-threaded.
    // Parts that are done in parallel:
    //  - creation of the probe tables
    //  - creation of the hashes

    // during the probe phase values are removed from the tables, that's done single threaded to
    // keep it lock free.

    let size = probe
        .iter()
        .map(|a| a.size_hint().1.unwrap())
        .sum::<usize>()
        + build
            .iter()
            .map(|b| b.size_hint().1.unwrap())
            .sum::<usize>();
    let mut results = Vec::with_capacity(size);

    // prepare hash table
    let mut hash_tbls = if validate.needs_checks() {
        let expected_size = build.iter().map(|i| i.size_hint().0).sum();
        let hash_tbls = prepare_hashed_relation_threaded(build);
        let build_size = hash_tbls.iter().map(|m| m.len()).sum();
        validate.validate_build(build_size, expected_size, swapped)?;
        hash_tbls
    } else {
        prepare_hashed_relation_threaded(build)
    };
    let random_state = hash_tbls[0].hasher().clone();

    // we pre hash the probing values
    let (probe_hashes, _) = create_hash_and_keys_threaded_vectorized(probe, Some(random_state));

    let n_tables = hash_tbls.len() as u64;

    // probe the hash table.
    // Note: indexes from b that are not matched will be None, Some(idx_b)
    // Therefore we remove the matches and the remaining will be joined from the right

    // branch is because we want to only do the swap check once
    if swapped {
        probe_outer(
            &probe_hashes,
            &mut hash_tbls,
            &mut results,
            n_tables,
            |idx_a, idx_b| (Some(idx_b), Some(idx_a)),
            |idx_a| (None, Some(idx_a)),
            |idx_b| (Some(idx_b), None),
        )
    } else {
        probe_outer(
            &probe_hashes,
            &mut hash_tbls,
            &mut results,
            n_tables,
            |idx_a, idx_b| (Some(idx_a), Some(idx_b)),
            |idx_a| (Some(idx_a), None),
            |idx_b| (None, Some(idx_b)),
        )
    }
    Ok(results)
}
