use std::mem::ManuallyDrop;
use std::ops::Deref;

use arrow::offset::OffsetsBuffer;
use polars_utils::idx_vec::IdxVec;
use polars_utils::sync::SyncPtr;
use rayon::iter::plumbing::UnindexedConsumer;
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::{flatten, slice_slice, NoNull};
use crate::POOL;

/// Indexes of the groups, the first index is stored separately.
/// this make sorting fast.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GroupsIdx {
    pub(crate) sorted: bool,
    first: Vec<IdxSize>,
    all: Vec<IdxVec>,
}

pub type IdxItem = (IdxSize, IdxVec);
pub type BorrowIdxItem<'a> = (IdxSize, &'a IdxVec);

impl Drop for GroupsIdx {
    fn drop(&mut self) {
        let v = std::mem::take(&mut self.all);
        // ~65k took approximately 1ms on local machine, so from that point we drop on other thread
        // to stop query from being blocked
        #[cfg(not(target_family = "wasm"))]
        if v.len() > 1 << 16 {
            std::thread::spawn(move || drop(v));
        } else {
            drop(v);
        }

        #[cfg(target_family = "wasm")]
        drop(v);
    }
}

impl From<Vec<IdxItem>> for GroupsIdx {
    fn from(v: Vec<IdxItem>) -> Self {
        v.into_iter().collect()
    }
}

impl From<Vec<(Vec<IdxSize>, Vec<IdxVec>)>> for GroupsIdx {
    fn from(v: Vec<(Vec<IdxSize>, Vec<IdxVec>)>) -> Self {
        // we have got the hash tables so we can determine the final
        let cap = v.iter().map(|v| v.0.len()).sum::<usize>();
        let offsets = v
            .iter()
            .scan(0_usize, |acc, v| {
                let out = *acc;
                *acc += v.0.len();
                Some(out)
            })
            .collect::<Vec<_>>();
        let mut global_first = Vec::with_capacity(cap);
        let global_first_ptr = unsafe { SyncPtr::new(global_first.as_mut_ptr()) };
        let mut global_all = Vec::with_capacity(cap);
        let global_all_ptr = unsafe { SyncPtr::new(global_all.as_mut_ptr()) };

        POOL.install(|| {
            v.into_par_iter().zip(offsets).for_each(
                |((local_first_vals, mut local_all_vals), offset)| unsafe {
                    let global_first: *mut IdxSize = global_first_ptr.get();
                    let global_all: *mut IdxVec = global_all_ptr.get();
                    let global_first = global_first.add(offset);
                    let global_all = global_all.add(offset);

                    std::ptr::copy_nonoverlapping(
                        local_first_vals.as_ptr(),
                        global_first,
                        local_first_vals.len(),
                    );
                    std::ptr::copy_nonoverlapping(
                        local_all_vals.as_ptr(),
                        global_all,
                        local_all_vals.len(),
                    );
                    // local_all_vals: Vec<Vec<IdxSize>>
                    // we just copied the contents: Vec<IdxSize> to a new buffer
                    // now, we want to free the outer vec, without freeing
                    // the inner vecs as they are moved, so we set the len to 0
                    local_all_vals.set_len(0);
                },
            );
        });
        unsafe {
            global_all.set_len(cap);
            global_first.set_len(cap);
        }
        GroupsIdx {
            sorted: false,
            first: global_first,
            all: global_all,
        }
    }
}

impl From<Vec<Vec<IdxItem>>> for GroupsIdx {
    fn from(v: Vec<Vec<IdxItem>>) -> Self {
        // single threaded flatten: 10% faster than `iter().flatten().collect()
        // this is the multi-threaded impl of that
        let (cap, offsets) = flatten::cap_and_offsets(&v);
        let mut first = Vec::with_capacity(cap);
        let first_ptr = first.as_ptr() as usize;
        let mut all = Vec::with_capacity(cap);
        let all_ptr = all.as_ptr() as usize;

        POOL.install(|| {
            v.into_par_iter()
                .zip(offsets)
                .for_each(|(mut inner, offset)| {
                    unsafe {
                        let first = (first_ptr as *const IdxSize as *mut IdxSize).add(offset);
                        let all = (all_ptr as *const IdxVec as *mut IdxVec).add(offset);

                        let inner_ptr = inner.as_mut_ptr();
                        for i in 0..inner.len() {
                            let (first_val, vals) = std::ptr::read(inner_ptr.add(i));
                            std::ptr::write(first.add(i), first_val);
                            std::ptr::write(all.add(i), vals);
                        }
                        // set len to 0 so that the contents will not get dropped
                        // they are moved to `first` and `all`
                        inner.set_len(0);
                    }
                });
        });
        unsafe {
            all.set_len(cap);
            first.set_len(cap);
        }
        GroupsIdx {
            sorted: false,
            first,
            all,
        }
    }
}

impl GroupsIdx {
    pub fn new(first: Vec<IdxSize>, all: Vec<IdxVec>, sorted: bool) -> Self {
        Self { sorted, first, all }
    }

    pub fn sort(&mut self) {
        let mut idx = 0;
        let first = std::mem::take(&mut self.first);
        // store index and values so that we can sort those
        let mut idx_vals = first
            .into_iter()
            .map(|v| {
                let out = [idx, v];
                idx += 1;
                out
            })
            .collect_trusted::<Vec<_>>();
        idx_vals.sort_unstable_by_key(|v| v[1]);

        let take_first = || idx_vals.iter().map(|v| v[1]).collect_trusted::<Vec<_>>();
        let take_all = || {
            idx_vals
                .iter()
                .map(|v| unsafe {
                    let idx = v[0] as usize;
                    std::mem::take(self.all.get_unchecked_mut(idx))
                })
                .collect_trusted::<Vec<_>>()
        };
        let (first, all) = POOL.install(|| rayon::join(take_first, take_all));
        self.first = first;
        self.all = all;
        self.sorted = true
    }
    pub fn is_sorted_flag(&self) -> bool {
        self.sorted
    }

    pub fn iter(
        &self,
    ) -> std::iter::Zip<std::iter::Copied<std::slice::Iter<IdxSize>>, std::slice::Iter<IdxVec>>
    {
        self.into_iter()
    }

    pub fn all(&self) -> &[IdxVec] {
        &self.all
    }

    pub fn first(&self) -> &[IdxSize] {
        &self.first
    }

    pub fn first_mut(&mut self) -> &mut Vec<IdxSize> {
        &mut self.first
    }

    pub(crate) fn len(&self) -> usize {
        self.first.len()
    }

    pub(crate) unsafe fn get_unchecked(&self, index: usize) -> BorrowIdxItem {
        let first = *self.first.get_unchecked(index);
        let all = self.all.get_unchecked(index);
        (first, all)
    }
}

impl FromIterator<IdxItem> for GroupsIdx {
    fn from_iter<T: IntoIterator<Item = IdxItem>>(iter: T) -> Self {
        let (first, all) = iter.into_iter().unzip();
        GroupsIdx {
            sorted: false,
            first,
            all,
        }
    }
}

impl<'a> IntoIterator for &'a GroupsIdx {
    type Item = BorrowIdxItem<'a>;
    type IntoIter = std::iter::Zip<
        std::iter::Copied<std::slice::Iter<'a, IdxSize>>,
        std::slice::Iter<'a, IdxVec>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.first.iter().copied().zip(self.all.iter())
    }
}

impl IntoIterator for GroupsIdx {
    type Item = IdxItem;
    type IntoIter = std::iter::Zip<std::vec::IntoIter<IdxSize>, std::vec::IntoIter<IdxVec>>;

    fn into_iter(mut self) -> Self::IntoIter {
        let first = std::mem::take(&mut self.first);
        let all = std::mem::take(&mut self.all);
        first.into_iter().zip(all)
    }
}

impl FromParallelIterator<IdxItem> for GroupsIdx {
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = IdxItem>,
    {
        let (first, all) = par_iter.into_par_iter().unzip();
        GroupsIdx {
            sorted: false,
            first,
            all,
        }
    }
}

impl<'a> IntoParallelIterator for &'a GroupsIdx {
    type Iter = rayon::iter::Zip<
        rayon::iter::Copied<rayon::slice::Iter<'a, IdxSize>>,
        rayon::slice::Iter<'a, IdxVec>,
    >;
    type Item = BorrowIdxItem<'a>;

    fn into_par_iter(self) -> Self::Iter {
        self.first.par_iter().copied().zip(self.all.par_iter())
    }
}

impl IntoParallelIterator for GroupsIdx {
    type Iter = rayon::iter::Zip<rayon::vec::IntoIter<IdxSize>, rayon::vec::IntoIter<IdxVec>>;
    type Item = IdxItem;

    fn into_par_iter(mut self) -> Self::Iter {
        let first = std::mem::take(&mut self.first);
        let all = std::mem::take(&mut self.all);
        first.into_par_iter().zip(all.into_par_iter())
    }
}

/// Every group is indicated by an array where the
///  - first value is an index to the start of the group
///  - second value is the length of the group
///
/// Only used when group values are stored together
///
/// This type should have the invariant that it is always sorted in ascending order.
pub type GroupsSlice = Vec<[IdxSize; 2]>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupsProxy {
    Idx(GroupsIdx),
    /// Slice is always sorted in ascending order.
    Slice {
        // the groups slices
        groups: GroupsSlice,
        // indicates if we do a rolling group_by
        rolling: bool,
    },
}

impl Default for GroupsProxy {
    fn default() -> Self {
        GroupsProxy::Idx(GroupsIdx::default())
    }
}

impl GroupsProxy {
    pub fn into_idx(self) -> GroupsIdx {
        match self {
            GroupsProxy::Idx(groups) => groups,
            GroupsProxy::Slice { groups, .. } => {
                polars_warn!("Had to reallocate groups, missed an optimization opportunity. Please open an issue.");
                groups
                    .iter()
                    .map(|&[first, len]| (first, (first..first + len).collect::<IdxVec>()))
                    .collect()
            },
        }
    }

    pub(crate) fn prepare_list_agg(
        &self,
        total_len: usize,
    ) -> (Option<IdxCa>, OffsetsBuffer<i64>, bool) {
        let mut can_fast_explode = true;
        match self {
            GroupsProxy::Idx(groups) => {
                let mut list_offset = Vec::with_capacity(self.len() + 1);
                let mut gather_offsets = Vec::with_capacity(total_len);

                let mut len_so_far = 0i64;
                list_offset.push(len_so_far);

                for idx in groups {
                    let idx = idx.1;
                    gather_offsets.extend_from_slice(idx);
                    len_so_far += idx.len() as i64;
                    list_offset.push(len_so_far);
                    can_fast_explode &= !idx.is_empty();
                }
                unsafe {
                    (
                        Some(IdxCa::from_vec("", gather_offsets)),
                        OffsetsBuffer::new_unchecked(list_offset.into()),
                        can_fast_explode,
                    )
                }
            },
            GroupsProxy::Slice { groups, .. } => {
                let mut list_offset = Vec::with_capacity(self.len() + 1);
                let mut gather_offsets = Vec::with_capacity(total_len);
                let mut len_so_far = 0i64;
                list_offset.push(len_so_far);

                for g in groups {
                    let len = g[1];
                    let offset = g[0];
                    gather_offsets.extend(offset..offset + len);

                    len_so_far += len as i64;
                    list_offset.push(len_so_far);
                    can_fast_explode &= len > 0;
                }

                unsafe {
                    (
                        Some(IdxCa::from_vec("", gather_offsets)),
                        OffsetsBuffer::new_unchecked(list_offset.into()),
                        can_fast_explode,
                    )
                }
            },
        }
    }

    pub fn iter(&self) -> GroupsProxyIter {
        GroupsProxyIter::new(self)
    }

    pub fn sort(&mut self) {
        match self {
            GroupsProxy::Idx(groups) => {
                if !groups.is_sorted_flag() {
                    groups.sort()
                }
            },
            GroupsProxy::Slice { .. } => {
                // invariant of the type
            },
        }
    }

    pub(crate) fn is_sorted_flag(&self) -> bool {
        match self {
            GroupsProxy::Idx(groups) => groups.is_sorted_flag(),
            GroupsProxy::Slice { .. } => true,
        }
    }

    pub fn take_group_firsts(self) -> Vec<IdxSize> {
        match self {
            GroupsProxy::Idx(mut groups) => std::mem::take(&mut groups.first),
            GroupsProxy::Slice { groups, .. } => {
                groups.into_iter().map(|[first, _len]| first).collect()
            },
        }
    }

    /// # Safety
    /// This will not do any bounds checks. The caller must ensure
    /// all groups have members.
    pub unsafe fn take_group_lasts(self) -> Vec<IdxSize> {
        match self {
            GroupsProxy::Idx(groups) => groups
                .all
                .iter()
                .map(|idx| *idx.get_unchecked(idx.len() - 1))
                .collect(),
            GroupsProxy::Slice { groups, .. } => groups
                .into_iter()
                .map(|[first, len]| first + len - 1)
                .collect(),
        }
    }

    pub fn par_iter(&self) -> GroupsProxyParIter {
        GroupsProxyParIter::new(self)
    }

    /// Get a reference to the `GroupsIdx`.
    ///
    /// # Panic
    ///
    /// panics if the groups are a slice.
    pub fn unwrap_idx(&self) -> &GroupsIdx {
        match self {
            GroupsProxy::Idx(groups) => groups,
            GroupsProxy::Slice { .. } => panic!("groups are slices not index"),
        }
    }

    /// Get a reference to the `GroupsSlice`.
    ///
    /// # Panic
    ///
    /// panics if the groups are an idx.
    pub fn unwrap_slice(&self) -> &GroupsSlice {
        match self {
            GroupsProxy::Slice { groups, .. } => groups,
            GroupsProxy::Idx(_) => panic!("groups are index not slices"),
        }
    }

    pub fn get(&self, index: usize) -> GroupsIndicator {
        match self {
            GroupsProxy::Idx(groups) => {
                let first = groups.first[index];
                let all = &groups.all[index];
                GroupsIndicator::Idx((first, all))
            },
            GroupsProxy::Slice { groups, .. } => GroupsIndicator::Slice(groups[index]),
        }
    }

    /// Get a mutable reference to the `GroupsIdx`.
    ///
    /// # Panic
    ///
    /// panics if the groups are a slice.
    pub fn idx_mut(&mut self) -> &mut GroupsIdx {
        match self {
            GroupsProxy::Idx(groups) => groups,
            GroupsProxy::Slice { .. } => panic!("groups are slices not index"),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            GroupsProxy::Idx(groups) => groups.len(),
            GroupsProxy::Slice { groups, .. } => groups.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn group_count(&self) -> IdxCa {
        match self {
            GroupsProxy::Idx(groups) => {
                let ca: NoNull<IdxCa> = groups
                    .iter()
                    .map(|(_first, idx)| idx.len() as IdxSize)
                    .collect_trusted();
                ca.into_inner()
            },
            GroupsProxy::Slice { groups, .. } => {
                let ca: NoNull<IdxCa> = groups.iter().map(|[_first, len]| *len).collect_trusted();
                ca.into_inner()
            },
        }
    }
    pub fn as_list_chunked(&self) -> ListChunked {
        match self {
            GroupsProxy::Idx(groups) => groups
                .iter()
                .map(|(_first, idx)| {
                    let ca: NoNull<IdxCa> = idx.iter().map(|&v| v as IdxSize).collect();
                    ca.into_inner().into_series()
                })
                .collect_trusted(),
            GroupsProxy::Slice { groups, .. } => groups
                .iter()
                .map(|&[first, len]| {
                    let ca: NoNull<IdxCa> = (first..first + len).collect_trusted();
                    ca.into_inner().into_series()
                })
                .collect_trusted(),
        }
    }

    pub fn unroll(self) -> GroupsProxy {
        match self {
            GroupsProxy::Idx(_) => self,
            GroupsProxy::Slice { rolling: false, .. } => self,
            GroupsProxy::Slice { mut groups, .. } => {
                let mut offset = 0 as IdxSize;
                for g in groups.iter_mut() {
                    g[0] = offset;
                    offset += g[1];
                }
                GroupsProxy::Slice {
                    groups,
                    rolling: false,
                }
            },
        }
    }

    pub fn slice(&self, offset: i64, len: usize) -> SlicedGroups {
        // SAFETY:
        // we create new `Vec`s from the sliced groups. But we wrap them in ManuallyDrop
        // so that we never call drop on them.
        // These groups lifetimes are bounded to the `self`. This must remain valid
        // for the scope of the aggregation.
        let sliced = match self {
            GroupsProxy::Idx(groups) => {
                let first = unsafe {
                    let first = slice_slice(groups.first(), offset, len);
                    let ptr = first.as_ptr() as *mut _;
                    Vec::from_raw_parts(ptr, first.len(), first.len())
                };

                let all = unsafe {
                    let all = slice_slice(groups.all(), offset, len);
                    let ptr = all.as_ptr() as *mut _;
                    Vec::from_raw_parts(ptr, all.len(), all.len())
                };
                ManuallyDrop::new(GroupsProxy::Idx(GroupsIdx::new(
                    first,
                    all,
                    groups.is_sorted_flag(),
                )))
            },
            GroupsProxy::Slice { groups, rolling } => {
                let groups = unsafe {
                    let groups = slice_slice(groups, offset, len);
                    let ptr = groups.as_ptr() as *mut _;
                    Vec::from_raw_parts(ptr, groups.len(), groups.len())
                };

                ManuallyDrop::new(GroupsProxy::Slice {
                    groups,
                    rolling: *rolling,
                })
            },
        };

        SlicedGroups {
            sliced,
            borrowed: self,
        }
    }
}

impl From<GroupsIdx> for GroupsProxy {
    fn from(groups: GroupsIdx) -> Self {
        GroupsProxy::Idx(groups)
    }
}

pub enum GroupsIndicator<'a> {
    Idx(BorrowIdxItem<'a>),
    Slice([IdxSize; 2]),
}

impl<'a> GroupsIndicator<'a> {
    pub fn len(&self) -> usize {
        match self {
            GroupsIndicator::Idx(g) => g.1.len(),
            GroupsIndicator::Slice([_, len]) => *len as usize,
        }
    }
    pub fn first(&self) -> IdxSize {
        match self {
            GroupsIndicator::Idx(g) => g.0,
            GroupsIndicator::Slice([first, _]) => *first,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct GroupsProxyIter<'a> {
    vals: &'a GroupsProxy,
    len: usize,
    idx: usize,
}

impl<'a> GroupsProxyIter<'a> {
    fn new(vals: &'a GroupsProxy) -> Self {
        let len = vals.len();
        let idx = 0;
        GroupsProxyIter { vals, len, idx }
    }
}

impl<'a> Iterator for GroupsProxyIter<'a> {
    type Item = GroupsIndicator<'a>;

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.idx = self.idx.saturating_add(n);
        self.next()
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.len {
            return None;
        }

        let out = unsafe {
            match self.vals {
                GroupsProxy::Idx(groups) => {
                    let item = groups.get_unchecked(self.idx);
                    Some(GroupsIndicator::Idx(item))
                },
                GroupsProxy::Slice { groups, .. } => {
                    Some(GroupsIndicator::Slice(*groups.get_unchecked(self.idx)))
                },
            }
        };
        self.idx += 1;
        out
    }
}

pub struct GroupsProxyParIter<'a> {
    vals: &'a GroupsProxy,
    len: usize,
}

impl<'a> GroupsProxyParIter<'a> {
    fn new(vals: &'a GroupsProxy) -> Self {
        let len = vals.len();
        GroupsProxyParIter { vals, len }
    }
}

impl<'a> ParallelIterator for GroupsProxyParIter<'a> {
    type Item = GroupsIndicator<'a>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        (0..self.len)
            .into_par_iter()
            .map(|i| unsafe {
                match self.vals {
                    GroupsProxy::Idx(groups) => GroupsIndicator::Idx(groups.get_unchecked(i)),
                    GroupsProxy::Slice { groups, .. } => {
                        GroupsIndicator::Slice(*groups.get_unchecked(i))
                    },
                }
            })
            .drive_unindexed(consumer)
    }
}

pub struct SlicedGroups<'a> {
    sliced: ManuallyDrop<GroupsProxy>,
    #[allow(dead_code)]
    // we need the lifetime to ensure the slice remains valid
    borrowed: &'a GroupsProxy,
}

impl Deref for SlicedGroups<'_> {
    type Target = GroupsProxy;

    fn deref(&self) -> &Self::Target {
        self.sliced.deref()
    }
}
