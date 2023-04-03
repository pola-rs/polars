use arrow::array::Array;
use num_traits::ToPrimitive;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::sync::SyncPtr;
use polars_utils::IdxSize;
use rayon::prelude::*;

use crate::datatypes::PolarsIntegerType;
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
