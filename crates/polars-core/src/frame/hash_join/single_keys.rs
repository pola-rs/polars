use super::*;

pub(crate) fn build_tables<T, IntoSlice>(keys: Vec<IntoSlice>) -> Vec<PlHashMap<T, Vec<IdxSize>>>
where
    T: Send + Hash + Eq + Sync + Copy + AsU64,
    IntoSlice: AsRef<[T]> + Send + Sync,
{
    let n_partitions = _set_partition_size();

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_partitions)
            .into_par_iter()
            .map(|partition_no| {
                let partition_no = partition_no as u64;

                let mut hash_tbl: PlHashMap<T, Vec<IdxSize>> =
                    PlHashMap::with_capacity(HASHMAP_INIT_SIZE);

                let n_partitions = n_partitions as u64;

                keys.iter()
                    .map(|array| array.as_ref().into_iter())
                    .flatten()
                    .enumerate()
                    .for_each(|(idx, key)| {
                        let idx = idx as IdxSize;
                        if this_partition(key.as_u64(), partition_no, n_partitions) {
                            let entry = hash_tbl.entry(*key);

                            match entry {
                                Entry::Vacant(entry) => {
                                    entry.insert(vec![idx as IdxSize]);
                                }
                                Entry::Occupied(mut entry) => {
                                    let v = entry.get_mut();
                                    v.push(idx as IdxSize);
                                }
                            }
                        }
                    });
                hash_tbl
            })
            .collect()
    })
}

// we determine the offset so that we later know which index to store in the join tuples
pub(super) fn probe_to_offsets<T, IntoSlice>(probe: &[IntoSlice]) -> Vec<usize>
where
    IntoSlice: AsRef<[T]> + Send + Sync,
    T: Send + Hash + Eq + Sync + Copy + AsU64,
{
    probe
        .iter()
        .map(|ph| ph.as_ref().len())
        .scan(0, |state, val| {
            let out = *state;
            *state += val;
            Some(out)
        })
        .collect()
}
