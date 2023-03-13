use rayon::prelude::*;
use rayon::ThreadPool;

use crate::IdxSize;

/// This is a perfect sort particularly useful for an arg_sort of an arg_sort
/// The second arg_sort sorts indices from `0` to `len` so can be just assigned to the
/// new index location.
///
/// Besides that we know that all indices are unique and thus not alias so we can parallelize.
///
/// This sort does not sort in place and will allocate.
///
/// - The right indices are used for sorting
/// - The left indices are placed at the location right points to.
///
/// # Safety
/// The caller must ensure that the right indexes fo `&[(_, IdxSize)]` are integers ranging from `0..idx.len`
#[cfg(not(target_family = "wasm"))]
pub unsafe fn perfect_sort(pool: &ThreadPool, idx: &[(IdxSize, IdxSize)], out: &mut Vec<IdxSize>) {
    let chunk_size = std::cmp::max(
        idx.len() / pool.current_num_threads(),
        pool.current_num_threads(),
    );

    out.reserve(idx.len());
    let ptr = out.as_mut_ptr() as *const IdxSize as usize;

    pool.install(|| {
        idx.par_chunks(chunk_size).for_each(|indices| {
            let ptr = ptr as *mut IdxSize;
            for (idx_val, idx_location) in indices {
                // Safety:
                // idx_location is in bounds by invariant of this function
                // and we ensured we have at least `idx.len()` capacity
                *ptr.add(*idx_location as usize) = *idx_val;
            }
        });
    });
    // Safety:
    // all elements are written
    out.set_len(idx.len());
}

// wasm alternative with different signature
#[cfg(target_family = "wasm")]
pub unsafe fn perfect_sort(
    pool: &crate::wasm::Pool,
    idx: &[(IdxSize, IdxSize)],
    out: &mut Vec<IdxSize>,
) {
    let chunk_size = std::cmp::max(
        idx.len() / pool.current_num_threads(),
        pool.current_num_threads(),
    );

    out.reserve(idx.len());
    let ptr = out.as_mut_ptr() as *const IdxSize as usize;

    pool.install(|| {
        idx.par_chunks(chunk_size).for_each(|indices| {
            let ptr = ptr as *mut IdxSize;
            for (idx_val, idx_location) in indices {
                // Safety:
                // idx_location is in bounds by invariant of this function
                // and we ensured we have at least `idx.len()` capacity
                *ptr.add(*idx_location as usize) = *idx_val;
            }
        });
    });
    // Safety:
    // all elements are written
    out.set_len(idx.len());
}
