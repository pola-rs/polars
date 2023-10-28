use super::*;

pub(crate) fn build_tables<T, I>(keys: Vec<I>) -> Vec<PlHashMap<T, Vec<IdxSize>>>
where
    T: Send + Hash + Eq + Sync + Copy + AsU64,
    I: IntoIterator<Item = T> + Send + Sync + Clone,
    // <I as IntoIterator>::IntoIter: TrustedLen,
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
                    PlHashMap::with_capacity(_HASHMAP_INIT_SIZE);

                let n_partitions = n_partitions as u64;
                let mut offset = 0;
                for keys in &keys {
                    let keys = keys.clone().into_iter();
                    let len = keys.size_hint().1.unwrap() as IdxSize;

                    let mut cnt = 0;
                    keys.for_each(|k| {
                        let idx = cnt + offset;
                        cnt += 1;

                        if this_partition(k.as_u64(), partition_no, n_partitions) {
                            let entry = hash_tbl.entry(k);

                            match entry {
                                Entry::Vacant(entry) => {
                                    entry.insert(vec![idx]);
                                },
                                Entry::Occupied(mut entry) => {
                                    let v = entry.get_mut();
                                    v.push(idx);
                                },
                            }
                        }
                    });
                    offset += len;
                }
                hash_tbl
            })
            .collect()
    })
}

// we determine the offset so that we later know which index to store in the join tuples
pub(super) fn probe_to_offsets<T, I>(probe: &[I]) -> Vec<usize>
where
    I: IntoIterator<Item = T> + Clone,
    // <I as IntoIterator>::IntoIter: TrustedLen,
    T: Send + Hash + Eq + Sync + Copy + AsU64,
{
    probe
        .iter()
        .map(|ph| ph.clone().into_iter().size_hint().1.unwrap())
        .scan(0, |state, val| {
            let out = *state;
            *state += val;
            Some(out)
        })
        .collect()
}
