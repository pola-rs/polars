use super::*;

/// Probe the build table and add tuples to the results (inner join)
pub(super) fn probe_inner<T, F>(
    probe: &[T],
    hash_tbls: &[PlHashMap<T, Vec<IdxSize>>],
    results: &mut Vec<(IdxSize, IdxSize)>,
    local_offset: usize,
    n_tables: u64,
    swap_fn: F,
) where
    T: Send + Hash + Eq + Sync + Copy + AsU64,
    F: Fn(IdxSize, IdxSize) -> (IdxSize, IdxSize),
{
    assert!(hash_tbls.len().is_power_of_two());
    probe.iter().enumerate().for_each(|(idx_a, k)| {
        let idx_a = (idx_a + local_offset) as IdxSize;
        // probe table that contains the hashed value
        let current_probe_table =
            unsafe { get_hash_tbl_threaded_join_partitioned(k.as_u64(), hash_tbls, n_tables) };

        let value = current_probe_table.get(k);

        if let Some(indexes_b) = value {
            let tuples = indexes_b.iter().map(|&idx_b| swap_fn(idx_a, idx_b));
            results.extend(tuples);
        }
    });
}

pub(crate) fn create_probe_table<T, IntoSlice>(
    keys: Vec<IntoSlice>,
) -> Vec<PlHashMap<T, Vec<IdxSize>>>
where
    T: Send + Hash + Eq + Sync + Copy + AsU64,
    IntoSlice: AsRef<[T]> + Send + Sync,
{
    let n_partitions = set_partition_size();

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_partitions).into_par_iter().map(|partition_no| {
            let partition_no = partition_no as u64;

            let mut hash_tbl: PlHashMap<T, Vec<IdxSize>> =
                PlHashMap::with_capacity(HASHMAP_INIT_SIZE);

            let n_partitions = n_partitions as u64;
            let mut offset = 0;
            for keys in &keys {
                let keys = keys.as_ref();
                let len = keys.len() as IdxSize;

                let mut cnt = 0;
                keys.iter().for_each(|k| {
                    let idx = cnt + offset;
                    cnt += 1;

                    if this_partition(k.as_u64(), partition_no, n_partitions) {
                        let entry = hash_tbl.entry(*k);

                        match entry {
                            Entry::Vacant(entry) => {
                                entry.insert(vec![idx]);
                            }
                            Entry::Occupied(mut entry) => {
                                let v = entry.get_mut();
                                v.push(idx);
                            }
                        }
                    }
                });
                offset += len;
            }
            hash_tbl
        })
    })
    .collect()
}

pub(super) fn hash_join_tuples_inner<T, IntoSlice>(
    probe: Vec<IntoSlice>,
    build: Vec<IntoSlice>,
    // Because b should be the shorter relation we could need to swap to keep left left and right right.
    swap: bool,
) -> Vec<(IdxSize, IdxSize)>
where
    IntoSlice: AsRef<[T]> + Send + Sync,
    T: Send + Hash + Eq + Sync + Copy + AsU64,
{
    // NOTE: see the left join for more elaborate comments

    // first we hash one relation
    let hash_tbls = create_probe_table(build);

    let n_tables = hash_tbls.len() as u64;
    debug_assert!(n_tables.is_power_of_two());
    let offsets = probe
        .iter()
        .map(|ph| ph.as_ref().len())
        .scan(0, |state, val| {
            let out = *state;
            *state += val;
            Some(out)
        })
        .collect::<Vec<_>>();
    // next we probe the other relation
    // code duplication is because we want to only do the swap check once
    POOL.install(|| {
        probe
            .into_par_iter()
            .zip(offsets)
            .map(|(probe, offset)| {
                let probe = probe.as_ref();
                // local reference
                let hash_tbls = &hash_tbls;
                let mut results = Vec::with_capacity(probe.len());
                let local_offset = offset;

                // branch is to hoist swap out of the inner loop.
                if swap {
                    probe_inner(
                        probe,
                        hash_tbls,
                        &mut results,
                        local_offset,
                        n_tables,
                        |idx_a, idx_b| (idx_b, idx_a),
                    )
                } else {
                    probe_inner(
                        probe,
                        hash_tbls,
                        &mut results,
                        local_offset,
                        n_tables,
                        |idx_a, idx_b| (idx_a, idx_b),
                    )
                }

                results
            })
            .flatten()
            .collect()
    })
}

pub(super) fn hash_join_tuples_left<T, IntoSlice>(
    probe: Vec<IntoSlice>,
    build: Vec<IntoSlice>,
) -> Vec<(IdxSize, Option<IdxSize>)>
where
    IntoSlice: AsRef<[T]> + Send + Sync,
    T: Send + Hash + Eq + Sync + Copy + AsU64,
{
    // first we hash one relation
    let hash_tbls = create_probe_table(build);

    // we determine the offset so that we later know which index to store in the join tuples
    let offsets = probe
        .iter()
        .map(|ph| ph.as_ref().len())
        .scan(0, |state, val| {
            let out = *state;
            *state += val;
            Some(out)
        })
        .collect::<Vec<_>>();

    let n_tables = hash_tbls.len() as u64;
    debug_assert!(n_tables.is_power_of_two());

    // next we probe the other relation
    POOL.install(|| {
        probe
            .into_par_iter()
            .zip(offsets)
            // probes_hashes: Vec<u64> processed by this thread
            // offset: offset index
            .map(|(probe, offset)| {
                // local reference
                let hash_tbls = &hash_tbls;
                let probe = probe.as_ref();

                // assume the result tuples equal lenght of the no. of hashes processed by this thread.
                let mut results = Vec::with_capacity(probe.len());

                probe.iter().enumerate().for_each(|(idx_a, k)| {
                    let idx_a = (idx_a + offset) as IdxSize;
                    // probe table that contains the hashed value
                    let current_probe_table = unsafe {
                        get_hash_tbl_threaded_join_partitioned(k.as_u64(), hash_tbls, n_tables)
                    };

                    // we already hashed, so we don't have to hash again.
                    let value = current_probe_table.get(k);

                    match value {
                        // left and right matches
                        Some(indexes_b) => {
                            results.extend(indexes_b.iter().map(|&idx_b| (idx_a, Some(idx_b))))
                        }
                        // only left values, right = null
                        None => results.push((idx_a, None)),
                    }
                });
                results
            })
            .flatten()
            .collect()
    })
}

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
    assert!(n_tables.is_power_of_two());
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
                }
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
    a: Vec<I>,
    b: Vec<J>,
    swap: bool,
) -> Vec<(Option<IdxSize>, Option<IdxSize>)>
where
    I: Iterator<Item = T> + Send + TrustedLen,
    J: Iterator<Item = T> + Send + TrustedLen,
    T: Hash + Eq + Copy + Sync + Send,
{
    // This function is partially multi-threaded.
    // Parts that are done in parallel:
    //  - creation of the probe tables
    //  - creation of the hashes

    // during the probe phase values are removed from the tables, that's done single threaded to
    // keep it lock free.

    let size = a.iter().map(|a| a.size_hint().0).sum::<usize>()
        + b.iter().map(|b| b.size_hint().0).sum::<usize>();
    let mut results = Vec::with_capacity(size);

    // prepare hash table
    let mut hash_tbls = prepare_hashed_relation_threaded(b);
    let random_state = hash_tbls[0].hasher().clone();

    // we pre hash the probing values
    let (probe_hashes, _) = create_hash_and_keys_threaded_vectorized(a, Some(random_state));

    let n_tables = hash_tbls.len() as u64;

    // probe the hash table.
    // Note: indexes from b that are not matched will be None, Some(idx_b)
    // Therefore we remove the matches and the remaining will be joined from the right

    // branch is because we want to only do the swap check once
    if swap {
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
    results
}
