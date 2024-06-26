use std::cmp::Reverse;
use std::collections::BinaryHeap;

use polars_core::frame::DataFrame;
use polars_core::POOL;
use polars_utils::priority::Priority;
use polars_utils::sync::SyncPtr;

use crate::morsel::Morsel;

/// Amount of morsels we need to consider spawning a thread during linearization.
const MORSELS_PER_THREAD: usize = 256;

/// Given a Vec<Morsel> for each pipe, it will output a vec of the contained dataframes.
/// If the morsels are ordered by their sequence ids within each vec, and no
/// sequence ID occurs in multiple vecs, the output will follow the same order globally.
pub fn linearize(mut morsels_per_pipe: Vec<Vec<Morsel>>) -> Vec<DataFrame> {
    let num_morsels: usize = morsels_per_pipe.iter().map(|p| p.len()).sum();
    if num_morsels == 0 {
        return vec![];
    }

    let n_threads = num_morsels
        .div_ceil(MORSELS_PER_THREAD)
        .min(POOL.current_num_threads()) as u64;

    // Partitioning based on sequence number.
    let max_seq = morsels_per_pipe
        .iter()
        .flat_map(|p| p.iter().map(|m| m.seq().to_u64()))
        .max()
        .unwrap();
    let seqs_per_thread = (max_seq + 1).div_ceil(n_threads);

    let morsels_per_p = &morsels_per_pipe;
    let mut dataframes: Vec<DataFrame> = Vec::with_capacity(num_morsels);
    let dataframes_ptr = unsafe { SyncPtr::new(dataframes.as_mut_ptr()) };
    rayon::scope(|s| {
        let mut out_offset = 0;
        let mut stop_idx_per_pipe = vec![0; morsels_per_p.len()];
        for t in 0..n_threads {
            // This thread will handle all morsels with sequence id
            // [t * seqs_per_thread, (t + 1) * seqs_per_threads).
            // Compute per pipe the slice that covers this range, re-using
            // the stop indices from the previous thread as our starting indices.
            let this_thread_out_offset = out_offset;
            let partition_max_seq = (t + 1) * seqs_per_thread;
            let cur_idx_per_pipe = stop_idx_per_pipe;
            stop_idx_per_pipe = Vec::with_capacity(morsels_per_p.len());
            for p in 0..morsels_per_p.len() {
                let stop_idx =
                    morsels_per_p[p].partition_point(|m| m.seq().to_u64() < partition_max_seq);
                assert!(stop_idx >= cur_idx_per_pipe[p]);
                out_offset += stop_idx - cur_idx_per_pipe[p];
                stop_idx_per_pipe.push(stop_idx);
            }

            {
                let stop_idx_per_pipe = stop_idx_per_pipe.clone();
                s.spawn(move |_| unsafe {
                    fill_partition(
                        morsels_per_p,
                        cur_idx_per_pipe,
                        &stop_idx_per_pipe,
                        dataframes_ptr.get().add(this_thread_out_offset),
                    )
                });
            }
        }
    });

    // SAFETY: all partitions were handled, so dataframes is full filled and
    // morsels_per_pipe fully consumed.
    unsafe {
        for morsels in morsels_per_pipe.iter_mut() {
            morsels.set_len(0);
        }
        dataframes.set_len(num_morsels);
    }
    dataframes
}

unsafe fn fill_partition(
    morsels_per_pipe: &[Vec<Morsel>],
    mut cur_idx_per_pipe: Vec<usize>,
    stop_idx_per_pipe: &[usize],
    mut out_ptr: *mut DataFrame,
) {
    // K-way merge, initialize priority queue with one element per pipe.
    let mut kmerge = BinaryHeap::with_capacity(morsels_per_pipe.len());
    for (p, morsels) in morsels_per_pipe.iter().enumerate() {
        if cur_idx_per_pipe[p] != stop_idx_per_pipe[p] {
            let seq = morsels[cur_idx_per_pipe[p]].seq();
            kmerge.push(Priority(Reverse(seq), p));
        }
    }

    // While the merge queue isn't empty, keep copying elements into the output.
    unsafe {
        while let Some(Priority(Reverse(mut seq), p)) = kmerge.pop() {
            // Write the next morsel from this pipe to the output.
            let morsels = &morsels_per_pipe[p];
            let cur_idx = &mut cur_idx_per_pipe[p];
            core::ptr::copy_nonoverlapping(morsels[*cur_idx].df(), out_ptr, 1);
            out_ptr = out_ptr.add(1);
            *cur_idx += 1;

            // Handle next element from this pipe.
            while *cur_idx != stop_idx_per_pipe[p] {
                let new_seq = morsels[*cur_idx].seq();
                if new_seq <= seq.successor() {
                    // New sequence number is the same, or a direct successor, can output immediately.
                    core::ptr::copy_nonoverlapping(morsels[*cur_idx].df(), out_ptr, 1);
                    out_ptr = out_ptr.add(1);
                    *cur_idx += 1;
                    seq = new_seq;
                } else {
                    kmerge.push(Priority(Reverse(new_seq), p));
                    break;
                }
            }
        }
    }
}
