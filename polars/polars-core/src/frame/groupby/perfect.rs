use arrow::array::Array;
use num_traits::ToPrimitive;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::sync::SyncPtr;
#[cfg(all(feature = "dtype-categorical", feature = "performant"))]
use polars_utils::unwrap::UnwrapUncheckedRelease;
use polars_utils::IdxSize;
use rayon::prelude::*;

#[cfg(all(feature = "dtype-categorical", feature = "performant"))]
use crate::datatypes::{CategoricalChunked, DataType, RevMapping};
use crate::datatypes::{PolarsIntegerType, UInt32Chunked};
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
        use crate::chunked_array::logical::LogicalType;
        let DataType::Categorical(Some(rev_map)) = self.dtype() else { unreachable!()};
        let cats = self.logical();

        let mut out = match &**rev_map {
            RevMapping::Local(cached) => {
                if cats.null_count() == 0 {
                    return cats.group_tuples_perfect(
                        cached.len() - 1,
                        multithreaded,
                        cats.len() / cached.len(),
                    );
                }
                let group_capacity = cats.len() / cached.len();
                let len = cached.len();
                get_groups_categorical(cats, len, group_capacity, multithreaded, |cat| *cat)
            }
            RevMapping::Global(mapping, _cached, _) => {
                let group_capacity = cats.len() / mapping.len();
                let len = if cats.null_count() > 0 {
                    mapping.len() + 1
                } else {
                    mapping.len()
                };
                unsafe {
                    get_groups_categorical(cats, len, group_capacity, multithreaded, |cat| {
                        *mapping.get(cat).unwrap_unchecked_release()
                    })
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
    group_capacity: usize,
    multithreaded: bool,
    get_cat: M,
) -> GroupsProxy
where
    M: Fn(&u32) -> u32 + Send + Sync,
{
    let null_idx = len - 1;
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

                for arr in cats.downcast_iter() {
                    for opt_cat in arr.iter() {
                        if let Some(cat) = opt_cat {
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
    GroupsProxy::Idx(GroupsIdx::new(first, groups, false))
}
