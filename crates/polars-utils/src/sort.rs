use std::mem::MaybeUninit;
use rayon::prelude::*;
use rayon::ThreadPool;
use crate::float::IsFloat;
use crate::ord::compare_fn_nan_max;

use crate::IdxSize;
use crate::iter::EnumerateIdxTrait;

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

/// used a lot, ensure there is a single impl
pub fn sort_slice_ascending<T: IsFloat + PartialOrd>(v: &mut [T]) {
    v.sort_unstable_by(|a, b| compare_fn_nan_max(a, b))
}
pub fn sort_slice_descending<T: IsFloat + PartialOrd>(v: &mut [T]) {
    v.sort_unstable_by(|a, b| compare_fn_nan_max(b, a))
}

unsafe fn assume_init_mut<T>(slice: &mut [MaybeUninit<T>]) -> &mut [T] {
    &mut *(slice as *mut [MaybeUninit<T>] as *mut [T])
}

pub fn arg_sort_ascending<'a, T: IsFloat + PartialOrd + Copy + 'a>(v: &[T], scratch: &'a mut Vec<u8>) -> &'a mut [IdxSize] {
    // Needed to be able to write back to back in the same buffer.
    debug_assert_eq!(std::mem::align_of::<T>(), std::mem::align_of::<(T, IdxSize)>());
    let n = v.len();
    let size = std::mem::size_of::<(T, IdxSize)>();
    let upper_bound = size * v.len() + size;
    scratch.reserve(upper_bound);
    let scratch_slice = unsafe {
        let cap_slice = scratch.spare_capacity_mut();
        let (_, scratch_slice, _) = cap_slice.align_to_mut::<MaybeUninit<(T, IdxSize)>>();
        &mut scratch_slice[..n]
    };

    for ((i, v), dst) in v.iter().enumerate_idx().zip(scratch_slice.iter_mut()) {
        *dst = MaybeUninit::new((*v, i));
    }
    debug_assert_eq!(v.len(), scratch_slice.len());

    let scratch_slice = unsafe {
        assume_init_mut(scratch_slice)
    };
    scratch_slice.sort_by(|key1,  key2| compare_fn_nan_max(&key1.0, &key2.0));

    // now we write the indexes in the same array.
    // So from <T, Idxsize> to <IdxSize>
    unsafe {
        let src = scratch_slice.as_ptr();

        let (_, scratch_slice_aligned_to_idx, _) = scratch_slice.align_to_mut::<IdxSize>();

        let dst = scratch_slice_aligned_to_idx.as_mut_ptr();

        for i  in 0..n {
            dst.add(i).write((*src.add(i)).1);
        }

        &mut scratch_slice_aligned_to_idx[..n]
    }
}

#[cfg(test)]
mod test{
    use super::*;
    #[test]
    fn test_argsort_ascending()  {
        let array = &[3, 1, 9, 23, 2];

        let scratch = &mut vec![];
        let out = arg_sort_ascending(array, scratch);

        dbg!(out);

    }

}