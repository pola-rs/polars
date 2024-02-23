use std::mem::MaybeUninit;

use num_traits::FromPrimitive;
use rayon::prelude::*;
use rayon::ThreadPool;

use crate::total_ord::TotalOrd;
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
/// The caller must ensure that the right indexes for `&[(_, IdxSize)]` are integers ranging from `0..idx.len`
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
                // SAFETY:
                // idx_location is in bounds by invariant of this function
                // and we ensured we have at least `idx.len()` capacity
                *ptr.add(*idx_location as usize) = *idx_val;
            }
        });
    });
    // SAFETY:
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
                // SAFETY:
                // idx_location is in bounds by invariant of this function
                // and we ensured we have at least `idx.len()` capacity
                *ptr.add(*idx_location as usize) = *idx_val;
            }
        });
    });
    // SAFETY:
    // all elements are written
    out.set_len(idx.len());
}

unsafe fn assume_init_mut<T>(slice: &mut [MaybeUninit<T>]) -> &mut [T] {
    &mut *(slice as *mut [MaybeUninit<T>] as *mut [T])
}

pub fn arg_sort_ascending<'a, T: TotalOrd + Copy + 'a, Idx, I: IntoIterator<Item = T>>(
    v: I,
    scratch: &'a mut Vec<u8>,
    n: usize,
) -> &'a mut [Idx]
where
    Idx: FromPrimitive + Copy,
{
    // Needed to be able to write back to back in the same buffer.
    debug_assert_eq!(std::mem::align_of::<T>(), std::mem::align_of::<(T, Idx)>());
    let size = std::mem::size_of::<(T, Idx)>();
    let upper_bound = size * n + size;
    scratch.reserve(upper_bound);
    let scratch_slice = unsafe {
        let cap_slice = scratch.spare_capacity_mut();
        let (_, scratch_slice, _) = cap_slice.align_to_mut::<MaybeUninit<(T, Idx)>>();
        &mut scratch_slice[..n]
    };

    for ((i, v), dst) in v.into_iter().enumerate().zip(scratch_slice.iter_mut()) {
        *dst = MaybeUninit::new((v, Idx::from_usize(i).unwrap()));
    }
    debug_assert_eq!(n, scratch_slice.len());

    let scratch_slice = unsafe { assume_init_mut(scratch_slice) };
    scratch_slice.sort_by(|key1, key2| key1.0.tot_cmp(&key2.0));

    // now we write the indexes in the same array.
    // So from <T, Idxsize> to <IdxSize>
    unsafe {
        let src = scratch_slice.as_ptr();

        let (_, scratch_slice_aligned_to_idx, _) = scratch_slice.align_to_mut::<Idx>();

        let dst = scratch_slice_aligned_to_idx.as_mut_ptr();

        for i in 0..n {
            dst.add(i).write((*src.add(i)).1);
        }

        &mut scratch_slice_aligned_to_idx[..n]
    }
}
