use arrow::array::Array;
use num_traits::ToPrimitive;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::IdxSize;

use crate::datatypes::PolarsIntegerType;
use crate::prelude::{ChunkedArray, GroupsIdx, GroupsProxy};

impl<T> ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: ToPrimitive,
{
    // Use the indexes as perfect groups
    pub fn group_tuples_perfect(
        &self,
        max: usize,
        multithreaded: bool,
        group_capacity: usize,
    ) -> GroupsProxy {
        if multithreaded {
            todo!()
        }

        let mut groups = Vec::with_capacity(max);
        let mut first = vec![0 as IdxSize; max];
        groups.resize_with(max, || Vec::with_capacity(group_capacity));

        let mut row_nr = 0 as IdxSize;
        for arr in self.downcast_iter() {
            assert_eq!(arr.null_count(), 0);
            let values = arr.values().as_slice();

            for partition in values.iter() {
                let partition = partition.to_usize().unwrap();
                let buf = unsafe { groups.get_unchecked_release_mut(partition) };
                buf.push(row_nr);

                // always write first/ branchless
                unsafe {
                    // safety: we just  pushed
                    let first_value = buf.get_unchecked(0);
                    *first.get_unchecked_release_mut(partition) = *first_value
                }

                row_nr += 1;
            }
        }

        GroupsProxy::Idx(GroupsIdx::new(first, groups, false))
    }
}
