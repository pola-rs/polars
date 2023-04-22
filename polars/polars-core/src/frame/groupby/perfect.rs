use arrow::array::Array;
use num_traits::FromPrimitive;
use polars_arrow::bit_util::round_upto_multiple_of_64;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::sync::SyncPtr;
use polars_utils::IdxSize;
use rayon::prelude::*;

#[cfg(feature = "dtype-categorical")]
use crate::config::verbose;
use crate::datatypes::*;
use crate::hashing::AsU64;
use crate::prelude::*;
use crate::POOL;

impl<T> ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: AsU64 + FromPrimitive,
{
    // Use the indexes as perfect groups
    pub fn group_tuples_perfect(
        &self,
        max: usize,
        multithreaded: bool,
        group_capacity: usize,
    ) -> GroupsProxy {
        let len = if self.null_count() > 0 {
            // we add one to store the null sentinel group
            max + 2
        } else {
            max + 1
        };

        // the latest index will be used for the null sentinel
        let null_idx = len.saturating_sub(1);
        let n_threads = POOL.current_num_threads();

        let chunk_size = len / n_threads;

        let (groups, first) = if multithreaded && chunk_size > 1 {
            let mut groups: Vec<Vec<IdxSize>> = unsafe { aligned_vec(len) };
            groups.resize_with(len, || Vec::with_capacity(group_capacity));
            let mut first: Vec<IdxSize> = unsafe { aligned_vec(len) };

            // ensure we keep aligned to cache lines
            let chunk_size = round_upto_multiple_of_64(chunk_size);

            let mut cache_line_offsets = Vec::with_capacity(n_threads + 1);
            cache_line_offsets.push(0);
            let mut current_offset = chunk_size;

            while current_offset < len {
                cache_line_offsets.push(current_offset);
                current_offset += chunk_size;
            }

            let groups_ptr = unsafe { SyncPtr::new(groups.as_mut_ptr()) };
            let first_ptr = unsafe { SyncPtr::new(first.as_mut_ptr()) };

            POOL.install(|| {
                (0..cache_line_offsets.len() - 1)
                    .into_par_iter()
                    .for_each(|thread_no| {
                        let mut row_nr = 0 as IdxSize;
                        let start = cache_line_offsets[thread_no];
                        let start = T::Native::from_usize(start).unwrap();
                        let end = cache_line_offsets[thread_no + 1];
                        let end = T::Native::from_usize(end).unwrap();

                        // safety: we don't alias
                        let groups = unsafe {
                            std::slice::from_raw_parts_mut(groups_ptr.clone().get(), len)
                        };
                        let first = unsafe { std::slice::from_raw_parts_mut(first_ptr.get(), len) };

                        for arr in self.downcast_iter() {
                            if arr.null_count() == 0 {
                                for &cat in arr.values().as_slice() {
                                    if cat >= start && cat < end {
                                        let cat = cat.as_u64() as usize;
                                        let buf = unsafe { groups.get_unchecked_release_mut(cat) };
                                        buf.push(row_nr);

                                        unsafe {
                                            if buf.len() == 1 {
                                                // safety: we just  pushed
                                                let first_value = buf.get_unchecked(0);
                                                *first.get_unchecked_release_mut(cat) = *first_value
                                            }
                                        }
                                    }
                                }
                                row_nr += 1;
                            } else {
                                for opt_cat in arr.iter() {
                                    if let Some(&cat) = opt_cat {
                                        // cannot factor out due to bchk
                                        if cat >= start && cat < end {
                                            let cat = cat.as_u64() as usize;
                                            let buf =
                                                unsafe { groups.get_unchecked_release_mut(cat) };
                                            buf.push(row_nr);

                                            unsafe {
                                                if buf.len() == 1 {
                                                    // safety: we just  pushed
                                                    let first_value = buf.get_unchecked(0);
                                                    *first.get_unchecked_release_mut(cat) =
                                                        *first_value
                                                }
                                            }
                                        }
                                    }
                                    // last thread handles null values
                                    else if thread_no == n_threads - 1 {
                                        let buf =
                                            unsafe { groups.get_unchecked_release_mut(null_idx) };
                                        buf.push(row_nr);
                                        unsafe {
                                            if buf.len() == 1 {
                                                let first_value = buf.get_unchecked(0);
                                                *first.get_unchecked_release_mut(null_idx) =
                                                    *first_value
                                            }
                                        }
                                    }

                                    row_nr += 1;
                                }
                            }
                        }
                    });
            });
            unsafe {
                groups.set_len(len);
                first.set_len(len);
            }
            (groups, first)
        } else {
            let mut groups = Vec::with_capacity(len);
            let mut first = vec![IdxSize::MAX; len];
            groups.resize_with(len, || Vec::with_capacity(group_capacity));

            let mut row_nr = 0 as IdxSize;
            for arr in self.downcast_iter() {
                for opt_cat in arr.iter() {
                    if let Some(cat) = opt_cat {
                        let group_id = cat.as_u64() as usize;
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
            (groups, first)
        };

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
                if self.can_fast_unique() {
                    if verbose() {
                        eprintln!("grouping categoricals, run perfect hash function");
                    }
                    get_groups_categorical(
                        cats,
                        cached.len(),
                        multithreaded,
                        self.can_fast_unique(),
                    )
                } else {
                    self.logical().group_tuples(multithreaded, sorted).unwrap()
                }
            }
            RevMapping::Global(_mapping, _cached, _) => {
                // TODO! see if we can optimize this
                // the problem is that the global categories are not guaranteed packed together
                // so we might need to deref them first to local ones, but that might be more
                // expensive than just hashing (benchmark first)
                self.logical().group_tuples(multithreaded, sorted).unwrap()
            }
        };
        if sorted {
            out.sort()
        }
        out
    }
}

#[repr(C, align(64))]
struct AlignTo64([u8; 64]);

/// There are no guarantees that the Vec<T> will remain aligned if you reallocate the data.
/// This means that you cannot reallocate so you will need to know how big to allocate up front.
unsafe fn aligned_vec<T>(n: usize) -> Vec<T> {
    // Lazy math to ensure we always have enough.
    let n_units = (n * std::mem::size_of::<T>() / std::mem::size_of::<AlignTo64>()) + 1;

    let mut aligned: Vec<AlignTo64> = Vec::with_capacity(n_units);

    let ptr = aligned.as_mut_ptr();
    let len_units = aligned.len();
    let cap_units = aligned.capacity();

    std::mem::forget(aligned);

    Vec::from_raw_parts(
        ptr as *mut T,
        len_units * std::mem::size_of::<AlignTo64>(),
        cap_units * std::mem::size_of::<AlignTo64>(),
    )
}

#[cfg(all(feature = "dtype-categorical", feature = "performant"))]
fn get_groups_categorical(
    cats: &UInt32Chunked,
    len: usize,
    multithreaded: bool,
    can_fast_unique: bool,
) -> GroupsProxy {
    let GroupsProxy::Idx(mut groups) = cats.group_tuples_perfect(len - 1, multithreaded, 0) else {unreachable!()};
    let first = std::mem::take(groups.first_mut());
    let groups = unsafe { std::mem::take(groups.all_mut()) };
    if can_fast_unique || first.iter().all(|v| *v != IdxSize::MAX) {
        GroupsProxy::Idx(GroupsIdx::new(first, groups, false))
    } else {
        // remove empty slots
        let first = first.into_iter().filter(|v| *v != IdxSize::MAX).collect();
        let groups = groups.into_iter().filter(|v| !v.is_empty()).collect();
        GroupsProxy::Idx(GroupsIdx::new(first, groups, false))
    }
}
