use std::sync::atomic::{AtomicUsize, Ordering};

use polars_utils::sync::SyncPtr;
use polars_utils::{range_portion, sqrt_approx};

use super::*;

const MIN_ELEMS_PER_THREAD: usize = 128;


pub(crate) fn build_tables_nonnull<T>(keys: &[T]) -> Vec<PlHashMap<T, Vec<IdxSize>>>
where
    T: Send + Hash + Eq + Sync + Copy + AsU64,
{
    // Don't bother parallelizing anything for small inputs.
    if keys.len() < 2*MIN_ELEMS_PER_THREAD {
        let mut hm: PlHashMap<T, Vec<IdxSize>> = PlHashMap::new();
        for (i, k) in keys.iter().enumerate() {
            hm.entry(*k).or_default().push(i as IdxSize);
        }
        return vec![hm];
    }
    
    // Compute the number of threads / partition size we'll use.
    let avail_threads = POOL.current_num_threads();
    let n_threads = (keys.len() / MIN_ELEMS_PER_THREAD).clamp(1, avail_threads);
    let n_partitions = n_threads;
    
    POOL.install(|| {
        // Compute the number of elements in each partition for each portion.
        let per_thread_partition_sizes: Vec<Vec<usize>> = (0..n_threads)
            .into_par_iter()
            .map(|t| {
                let mut partition_sizes = vec![0; n_partitions];
                let key_portion = &keys[range_portion(t, n_threads, 0..keys.len())];
                for key in key_portion {
                    let h = key.as_u64();
                    let p = hash_to_partition(h, n_partitions);
                    unsafe {
                        *partition_sizes
                            .get_unchecked_mut(p) += 1;
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
                per_thread_partition_offsets[t*n_partitions + p] = cum_offset;
                cum_offset += per_thread_partition_sizes[t][p];
            }
        }
        per_thread_partition_offsets[n_threads*n_partitions] = keys.len();
        partition_offsets[n_partitions] = keys.len();
        
        // Scatter values into partitions.
        let mut scatter_keys: Vec<T> = Vec::with_capacity(keys.len());
        let mut scatter_idxs: Vec<IdxSize> = Vec::with_capacity(keys.len());
        let scatter_keys_ptr = unsafe { SyncPtr::new(scatter_keys.as_mut_ptr()) };
        let scatter_idxs_ptr = unsafe { SyncPtr::new(scatter_idxs.as_mut_ptr()) };
        (0..n_threads)
            .into_par_iter()
            .for_each(|t| {
                let scatter_keys_ptr = scatter_keys_ptr;
                let scatter_idxs_ptr = scatter_idxs_ptr;
                let mut partition_offsets = per_thread_partition_offsets[t*n_partitions..(t + 1)*n_partitions].to_vec();
                let key_range = range_portion(t, n_threads, 0..keys.len());
                let key_portion = &keys[key_range.clone()];
                for (i, key) in key_portion.iter().enumerate() {
                    unsafe {
                        let p = hash_to_partition(key.as_u64(), n_partitions);
                        let off = partition_offsets.get_unchecked_mut(p);
                        *scatter_keys_ptr.get().add(*off) = *key;
                        *scatter_idxs_ptr.get().add(*off) = (key_range.start + i) as IdxSize;
                        *off += 1;
                    }
                }
            });
        unsafe {
            scatter_keys.set_len(keys.len());
            scatter_idxs.set_len(keys.len());
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
                let partition_range = partition_offsets[p]..partition_offsets[p+1];
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
                            Entry::Occupied(mut o) => { o.get_mut().push(idx as IdxSize); }
                            Entry::Vacant(v) => { v.insert(vec![idx as IdxSize]); }
                        };
                    }
                }
                
                hm
            })
            .collect()
    })
}

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
