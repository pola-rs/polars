use polars_utils::sync::SyncPtr;

use super::*;

// FIXME: we should compute the number of threads / partition size we'll use.
// let avail_threads = POOL.current_num_threads();
// let n_threads = (num_keys / MIN_ELEMS_PER_THREAD).clamp(1, avail_threads);
// Use a small element per thread threshold for debugging/testing purposes.
const MIN_ELEMS_PER_THREAD: usize = if cfg!(debug_assertions) { 1 } else { 128 };

pub(crate) fn build_tables<T, I>(keys: Vec<I>) -> Vec<PlHashMap<T, Vec<IdxSize>>>
where
    T: Send + Hash + Eq + Sync + Copy + AsU64,
    I: IntoIterator<Item = T> + Send + Sync + Clone,
{
    // FIXME: change interface to split the input here, instead of taking
    // pre-split input iterators.
    let n_partitions = keys.len();
    let n_threads = n_partitions;
    let num_keys_est: usize = keys
        .iter()
        .map(|k| k.clone().into_iter().size_hint().0)
        .sum();

    // Don't bother parallelizing anything for small inputs.
    if num_keys_est < 2 * MIN_ELEMS_PER_THREAD {
        let mut hm: PlHashMap<T, Vec<IdxSize>> = PlHashMap::new();
        let mut offset = 0;
        for it in keys {
            for k in it {
                hm.entry(k).or_default().push(offset);
                offset += 1;
            }
        }
        return vec![hm];
    }

    POOL.install(|| {
        // Compute the number of elements in each partition for each portion.
        let per_thread_partition_sizes: Vec<Vec<usize>> = keys
            .par_iter()
            .map(|key_portion| {
                let mut partition_sizes = vec![0; n_partitions];
                for key in key_portion.clone() {
                    let h = key.as_u64();
                    let p = hash_to_partition(h, n_partitions);
                    unsafe {
                        *partition_sizes.get_unchecked_mut(p) += 1;
                    }
                }
                partition_sizes
            })
            .collect();

        // Compute output offsets with a cumulative sum.
        let mut per_thread_partition_offsets = vec![0; n_partitions * n_threads + 1];
        let mut partition_offsets = vec![0; n_partitions + 1];
        let mut cum_offset = 0;
        for p in 0..n_partitions {
            partition_offsets[p] = cum_offset;
            for t in 0..n_threads {
                per_thread_partition_offsets[t * n_partitions + p] = cum_offset;
                cum_offset += per_thread_partition_sizes[t][p];
            }
        }
        let num_keys = cum_offset;
        per_thread_partition_offsets[n_threads * n_partitions] = num_keys;
        partition_offsets[n_partitions] = num_keys;

        // FIXME: we wouldn't need this if we changed our interface to split the
        // input in this function, instead of taking a vec of iterators.
        let mut per_thread_input_offsets = vec![0; n_partitions];
        cum_offset = 0;
        for t in 0..n_threads {
            per_thread_input_offsets[t] = cum_offset;
            for p in 0..n_partitions {
                cum_offset += per_thread_partition_sizes[t][p];
            }
        }

        // Scatter values into partitions.
        let mut scatter_keys: Vec<T> = Vec::with_capacity(num_keys);
        let mut scatter_idxs: Vec<IdxSize> = Vec::with_capacity(num_keys);
        let scatter_keys_ptr = unsafe { SyncPtr::new(scatter_keys.as_mut_ptr()) };
        let scatter_idxs_ptr = unsafe { SyncPtr::new(scatter_idxs.as_mut_ptr()) };
        keys.into_par_iter()
            .enumerate()
            .for_each(|(t, key_portion)| {
                let mut partition_offsets =
                    per_thread_partition_offsets[t * n_partitions..(t + 1) * n_partitions].to_vec();
                for (i, key) in key_portion.into_iter().enumerate() {
                    unsafe {
                        let p = hash_to_partition(key.as_u64(), n_partitions);
                        let off = partition_offsets.get_unchecked_mut(p);
                        *scatter_keys_ptr.get().add(*off) = key;
                        *scatter_idxs_ptr.get().add(*off) =
                            (per_thread_input_offsets[t] + i) as IdxSize;
                        *off += 1;
                    }
                }
            });
        unsafe {
            scatter_keys.set_len(num_keys);
            scatter_idxs.set_len(num_keys);
        }

        // Build tables.
        (0..n_partitions)
            .into_par_iter()
            .map(|p| {
                // Resizing the hash map is very, very expensive. That's why we
                // adopt a hybrid strategy: we assume an initially small hash
                // map, which would satisfy a highly skewed relation. If this
                // fills up we immediately reserve enough for a full cardinality
                // data set.
                let partition_range = partition_offsets[p]..partition_offsets[p + 1];
                let full_size = partition_range.len();
                let mut conservative_size = _HASHMAP_INIT_SIZE.max(full_size / 64);
                let mut hm: PlHashMap<T, Vec<IdxSize>> =
                    PlHashMap::with_capacity(conservative_size);

                unsafe {
                    for i in partition_range {
                        if hm.len() == conservative_size {
                            hm.reserve(full_size - conservative_size);
                            conservative_size = 0; // Hack to ensure we never hit this branch again.
                        }

                        let key = *scatter_keys.get_unchecked(i);
                        let idx = *scatter_idxs.get_unchecked(i);
                        match hm.entry(key) {
                            Entry::Occupied(mut o) => {
                                o.get_mut().push(idx as IdxSize);
                            },
                            Entry::Vacant(v) => {
                                v.insert(vec![idx as IdxSize]);
                            },
                        };
                    }
                }

                hm
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
