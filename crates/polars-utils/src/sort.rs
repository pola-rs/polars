use std::cmp::Ordering;
use std::mem::MaybeUninit;

use num_traits::FromPrimitive;

use crate::total_ord::TotalOrd;

unsafe fn assume_init_mut<T>(slice: &mut [MaybeUninit<T>]) -> &mut [T] {
    unsafe { &mut *(slice as *mut [MaybeUninit<T>] as *mut [T]) }
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
    debug_assert_eq!(align_of::<T>(), align_of::<(T, Idx)>());
    let size = size_of::<(T, Idx)>();
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

#[derive(PartialEq, Eq, Clone, Hash)]
#[repr(transparent)]
pub struct ReorderWithNulls<T, const DESCENDING: bool, const NULLS_LAST: bool>(pub Option<T>);

impl<T: PartialOrd, const DESCENDING: bool, const NULLS_LAST: bool> PartialOrd
    for ReorderWithNulls<T, DESCENDING, NULLS_LAST>
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (&self.0, &other.0) {
            (None, None) => Some(Ordering::Equal),
            (None, Some(_)) => {
                if NULLS_LAST {
                    Some(Ordering::Greater)
                } else {
                    Some(Ordering::Less)
                }
            },
            (Some(_), None) => {
                if NULLS_LAST {
                    Some(Ordering::Less)
                } else {
                    Some(Ordering::Greater)
                }
            },
            (Some(l), Some(r)) => {
                if DESCENDING {
                    r.partial_cmp(l)
                } else {
                    l.partial_cmp(r)
                }
            },
        }
    }
}

impl<T: Ord, const DESCENDING: bool, const NULLS_LAST: bool> Ord
    for ReorderWithNulls<T, DESCENDING, NULLS_LAST>
{
    fn cmp(&self, other: &Self) -> Ordering {
        match (&self.0, &other.0) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => {
                if NULLS_LAST {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            },
            (Some(_), None) => {
                if NULLS_LAST {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            },
            (Some(l), Some(r)) => {
                if DESCENDING {
                    r.cmp(l)
                } else {
                    l.cmp(r)
                }
            },
        }
    }
}
