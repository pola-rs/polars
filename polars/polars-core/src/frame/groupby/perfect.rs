use arrow::array::Array;
use num_traits::ToPrimitive;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::sync::SyncPtr;
#[cfg(all(feature = "dtype-categorical", feature = "performant"))]
use polars_utils::unwrap::UnwrapUncheckedRelease;
use polars_utils::IdxSize;
use rayon::prelude::*;

use crate::datatypes::*;
use crate::hashing::{this_partition, AsU64};
use crate::prelude::{ChunkedArray, GroupsIdx, GroupsProxy};
use crate::utils::_set_partition_size;
use crate::POOL;

impl<T> ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: AsU64,
{
    // Use the indexes as perfect groups
    pub fn group_tuples_perfect(
        &self,
        max: usize,
        multithreaded: bool,
        group_capacity: usize,
    ) -> GroupsProxy {
        let len = max + 1;
        let mut groups = Vec::with_capacity(len);
        let mut first = vec![0 as IdxSize; len];
        groups.resize_with(len, || Vec::with_capacity(group_capacity));

        let groups_ptr = unsafe { SyncPtr::new(groups.as_mut_ptr()) };
        let first_ptr = unsafe { SyncPtr::new(first.as_mut_ptr()) };

        if multithreaded {
            let n_parts = _set_partition_size();
            POOL.install(|| {
                (0..n_parts).into_par_iter().for_each(|thread_no| {
                    let mut row_nr = 0 as IdxSize;

                    // safety: we don't alias
                    let groups =
                        unsafe { std::slice::from_raw_parts_mut(groups_ptr.clone().get(), len) };
                    let first = unsafe { std::slice::from_raw_parts_mut(first_ptr.get(), len) };

                    for arr in self.downcast_iter() {
                        assert_eq!(arr.null_count(), 0);
                        let values = arr.values().as_slice();

                        for group_id in values.iter() {
                            if this_partition(group_id.as_u64(), thread_no as u64, n_parts as u64) {
                                let group_id = group_id.as_u64() as usize;

                                let buf = unsafe { groups.get_unchecked_release_mut(group_id) };
                                buf.push(row_nr);

                                // always write first/ branchless
                                unsafe {
                                    // safety: we just  pushed
                                    let first_value = buf.get_unchecked(0);
                                    *first.get_unchecked_release_mut(group_id) = *first_value
                                }
                            }

                            row_nr += 1;
                        }
                    }
                });
            })
        } else {
            let mut row_nr = 0 as IdxSize;
            for arr in self.downcast_iter() {
                assert_eq!(arr.null_count(), 0);
                let values = arr.values().as_slice();

                for group_id in values.iter() {
                    let group_id = group_id.to_usize().unwrap();
                    let buf = unsafe { groups.get_unchecked_release_mut(group_id) };
                    buf.push(row_nr);

                    // always write first/ branchless
                    unsafe {
                        // safety: we just  pushed
                        let first_value = buf.get_unchecked(0);
                        *first.get_unchecked_release_mut(group_id) = *first_value
                    }

                    row_nr += 1;
                }
            }
        }

        GroupsProxy::Idx(GroupsIdx::new(first, groups, false))
    }
}

#[cfg(all(feature = "dtype-categorical", feature = "performant"))]
// Special implementation so that cats can be processed in a single pass
impl CategoricalChunked {
    // Use the indexes as perfect groups
    pub fn group_tuples_perfect(&self, multithreaded: bool, sorted: bool) -> GroupsProxy {
        let DataType::Categorical(Some(rev_map)) = self.dtype() else { unreachable!()};
        if self.is_empty() {
            return GroupsProxy::Idx(GroupsIdx::new(vec![], vec![], true));
        }
        let cats = self.logical();

        let mut out = match &**rev_map {
            RevMapping::Local(cached) => {
                let len = if cats.null_count() > 0 {
                    // we add one to store the null sentinel group
                    cached.len() + 1
                } else {
                    cached.len()
                };
                get_groups_categorical(cats, len, multithreaded, |cat| *cat, self.can_fast_unique())
            }
            RevMapping::Global(mapping, _cached, _) => {
                let len = if cats.null_count() > 0 {
                    // we add one to store the null sentinel group
                    mapping.len() + 1
                } else {
                    mapping.len()
                };
                unsafe {
                    get_groups_categorical(
                        cats,
                        len,
                        multithreaded,
                        |cat| *mapping.get(cat).unwrap_unchecked_release(),
                        self.can_fast_unique(),
                    )
                }
            }
        };
        if sorted {
            out.sort()
        }
        out
    }
}

#[cfg(all(feature = "dtype-categorical", feature = "performant"))]
fn get_groups_categorical<M>(
    cats: &UInt32Chunked,
    len: usize,
    multithreaded: bool,
    get_cat: M,
    can_fast_unique: bool,
) -> GroupsProxy
where
    M: Fn(&u32) -> u32 + Send + Sync,
{
    // the latest index will be used for the null sentinel
    let null_idx = len.saturating_sub(1);
    let mut groups = Vec::with_capacity(len);
    let mut first = vec![IdxSize::MAX; len];
    groups.resize_with(len, Vec::new);

    let groups_ptr = unsafe { SyncPtr::new(groups.as_mut_ptr()) };
    let first_ptr = unsafe { SyncPtr::new(first.as_mut_ptr()) };

    if multithreaded {
        let n_parts = _set_partition_size();
        POOL.install(|| {
            (0..n_parts).into_par_iter().for_each(|thread_no| {
                let mut row_nr = 0 as IdxSize;

                // safety: we don't alias
                let groups =
                    unsafe { std::slice::from_raw_parts_mut(groups_ptr.clone().get(), len) };
                let first = unsafe { std::slice::from_raw_parts_mut(first_ptr.get(), len) };

                for arr in cats.downcast_iter() {
                    if arr.null_count() == 0 {
                        for cat in arr.values().as_slice() {
                            // cannot factor out due to bchk
                            if this_partition(*cat as u64, thread_no as u64, n_parts as u64) {
                                let group_id = get_cat(cat) as usize;

                                let buf = unsafe { groups.get_unchecked_release_mut(group_id) };
                                buf.push(row_nr);

                                // always write first/ branchless
                                unsafe {
                                    // safety: we just  pushed
                                    let first_value = buf.get_unchecked(0);
                                    *first.get_unchecked_release_mut(group_id) = *first_value
                                }
                            }
                            row_nr += 1;
                        }
                    } else {
                        for opt_cat in arr.iter() {
                            if let Some(cat) = opt_cat {
                                // cannot factor out due to bchk
                                if this_partition(*cat as u64, thread_no as u64, n_parts as u64) {
                                    let group_id = get_cat(cat) as usize;

                                    let buf = unsafe { groups.get_unchecked_release_mut(group_id) };
                                    buf.push(row_nr);

                                    // always write first/ branchless
                                    unsafe {
                                        // safety: we just  pushed
                                        let first_value = buf.get_unchecked(0);
                                        *first.get_unchecked_release_mut(group_id) = *first_value
                                    }
                                }
                            }
                            // first thread handles null values
                            else if thread_no == 0 {
                                let buf = unsafe { groups.get_unchecked_release_mut(null_idx) };
                                buf.push(row_nr);
                                unsafe {
                                    let first_value = buf.get_unchecked(0);
                                    *first.get_unchecked_release_mut(null_idx) = *first_value
                                }
                            }

                            row_nr += 1;
                        }
                    }
                }
            });
        })
    } else {
        let mut row_nr = 0 as IdxSize;
        for arr in cats.downcast_iter() {
            for opt_cat in arr.iter() {
                if let Some(cat) = opt_cat {
                    let group_id = get_cat(cat) as usize;
                    let buf = unsafe { groups.get_unchecked_release_mut(group_id) };
                    buf.push(row_nr);

                    // always write first/ branchless
                    unsafe {
                        // safety: we just  pushed
                        let first_value = buf.get_unchecked(0);
                        *first.get_unchecked_release_mut(group_id) = *first_value
                    }
                } else {
                    let buf = unsafe { groups.get_unchecked_release_mut(null_idx) };
                    buf.push(row_nr);
                    unsafe {
                        let first_value = buf.get_unchecked(0);
                        *first.get_unchecked_release_mut(null_idx) = *first_value
                    }
                }

                row_nr += 1;
            }
        }
    }
    if can_fast_unique || first.iter().all(|v| *v != IdxSize::MAX) {
        GroupsProxy::Idx(GroupsIdx::new(first, groups, false))
    } else {
        // remove empty slots
        let first = first.into_iter().filter(|v| *v != IdxSize::MAX).collect();
        let groups = groups.into_iter().filter(|v| !v.is_empty()).collect();
        GroupsProxy::Idx(GroupsIdx::new(first, groups, false))
    }
}
