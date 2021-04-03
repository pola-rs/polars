use crate::frame::groupby::populate_multiple_key_hashmap;
use crate::frame::hash_join::n_join_threads;
use crate::prelude::*;
use crate::utils::split_df;
use crate::vector_hasher::{
    df_rows_to_hashes, df_rows_to_hashes_threaded, this_thread, IdBuildHasher, IdxHash,
};
use crate::POOL;
use ahash::RandomState;
use hashbrown::HashMap;
use rayon::prelude::*;

fn create_build_table(
    hashes: &[UInt64Chunked],
    keys: &DataFrame,
) -> Vec<HashMap<IdxHash, Vec<u32>, IdBuildHasher>> {
    // POOL.install(|| {
    //     hashes.into_par_iter()
    // })
    // let mut tbl = HashMap::with_capacity_and_hasher(hashes, IdBuildHasher::default());
    let n_threads = hashes.len();
    let size = hashes.iter().fold(0, |acc, v| acc + v.len());

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_threads).into_par_iter().map(|thread_no| {
            let thread_no = thread_no as u64;
            // TODO:: benchmark size
            let mut hash_tbl: HashMap<IdxHash, Vec<u32>, IdBuildHasher> =
                HashMap::with_capacity_and_hasher(size / (5 * n_threads), IdBuildHasher::default());

            let n_threads = n_threads as u64;
            let mut offset = 0;
            for hashes in hashes {
                for hashes in hashes.data_views() {
                    let len = hashes.len();
                    let mut idx = 0;
                    hashes.iter().for_each(|h| {
                        // partition hashes by thread no.
                        // So only a part of the hashes go to this hashmap
                        if this_thread(*h, thread_no, n_threads) {
                            let idx = idx + offset;
                            populate_multiple_key_hashmap(
                                &mut hash_tbl,
                                idx,
                                *h,
                                keys,
                                vec![idx],
                                |v| v.push(idx),
                            )
                        }
                        idx += 1;
                    });

                    offset += len as u32;
                }
            }
            hash_tbl
        })
    })
    .collect()
}

fn inner_join_multiple_keys(left: &DataFrame, right: &DataFrame, swap: bool) {
    // we assume that the left DataFrame is the shorter relation.
    // left will be used for the build phase.

    let n_threads = n_join_threads();
    let left_dfs = split_df(&left, n_threads).unwrap();
    let right_dfs = split_df(&right, n_threads).unwrap();

    let (build_hashes, random_state) = df_rows_to_hashes_threaded(&left_dfs, None);
    let (probe_hashes, random_state) = df_rows_to_hashes_threaded(&right_dfs, Some(random_state));

    todo!()

    // let n_tables = hash_tbls.len() as u64;
    // let offsets = probe_hashes
    //     .iter()
    //     .map(|ph| ph.len())
    //     .scan(0, |state, val| {
    //         let out = *state;
    //         *state += val;
    //         Some(out)
    //     })
    //     .collect::<Vec<_>>();
    // // next we probe the other relation
    // // code duplication is because we want to only do the swap check once
    // POOL.install(|| {
    //     probe_hashes
    //         .into_par_iter()
    //         .zip(offsets)
    //         .map(|(probe_hashes, offset)| {
    //             // local reference
    //             let hash_tbls = &hash_tbls;
    //             let mut results =
    //                 Vec::with_capacity(probe_hashes.len() / POOL.current_num_threads());
    //             let local_offset = offset;
    //             // code duplication is to hoist swap out of the inner loop.
    //             if swap {
    //                 probe_hashes.iter().enumerate().for_each(|(idx_a, (h, k))| {
    //                     let idx_a = (idx_a + local_offset) as u32;
    //                     // probe table that contains the hashed value
    //                     let current_probe_table = unsafe { get_hash_tbl(*h, hash_tbls, n_tables) };
    //
    //                     let entry = current_probe_table
    //                         .raw_entry()
    //                         .from_key_hashed_nocheck(*h, k);
    //
    //                     if let Some((_, indexes_b)) = entry {
    //                         let tuples = indexes_b.iter().map(|&idx_b| (idx_b, idx_a));
    //                         results.extend(tuples);
    //                     }
    //                 });
    //             } else {
    //                 probe_hashes.iter().enumerate().for_each(|(idx_a, (h, k))| {
    //                     let idx_a = (idx_a + local_offset) as u32;
    //                     // probe table that contains the hashed value
    //                     let current_probe_table = unsafe { get_hash_tbl(*h, hash_tbls, n_tables) };
    //
    //                     let entry = current_probe_table
    //                         .raw_entry()
    //                         .from_key_hashed_nocheck(*h, k);
    //
    //                     if let Some((_, indexes_b)) = entry {
    //                         let tuples = indexes_b.iter().map(|&idx_b| (idx_a, idx_b));
    //                         results.extend(tuples);
    //                     }
    //                 });
    //             }
    //
    //             results
    //         })
    //         .flatten()
    //         .collect()
}
