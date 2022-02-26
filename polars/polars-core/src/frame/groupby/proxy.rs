use crate::prelude::*;
use crate::utils::NoNull;
use crate::POOL;
use polars_arrow::utils::CustomIterTools;
use rayon::iter::plumbing::UnindexedConsumer;
use rayon::prelude::*;

/// Indexes of the groups, the first index is stored separately.
/// this make sorting fast.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GroupsIdx {
    pub(crate) sorted: bool,
    first: Vec<IdxSize>,
    all: Vec<Vec<IdxSize>>,
}

pub type IdxItem = (IdxSize, Vec<IdxSize>);
pub type BorrowIdxItem<'a> = (IdxSize, &'a Vec<IdxSize>);

impl Drop for GroupsIdx {
    fn drop(&mut self) {
        let v = std::mem::take(&mut self.all);
        // ~65k took approximately 1ms on local machine, so from that point we drop on other thread
        // to stop query from being blocked
        if v.len() > 1 << 16 {
            std::thread::spawn(move || drop(v));
        } else {
            drop(v);
        }
    }
}

impl From<Vec<IdxItem>> for GroupsIdx {
    fn from(v: Vec<IdxItem>) -> Self {
        v.into_iter().collect()
    }
}

impl From<Vec<Vec<IdxItem>>> for GroupsIdx {
    fn from(v: Vec<Vec<IdxItem>>) -> Self {
        v.into_iter().flatten().collect()
    }
}

impl GroupsIdx {
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
    pub fn is_sorted(&self) -> bool {
        self.sorted
    }

    pub fn iter(
        &self,
    ) -> std::iter::Zip<std::iter::Copied<std::slice::Iter<IdxSize>>, std::slice::Iter<Vec<IdxSize>>>
    {
        self.into_iter()
    }

    pub fn all(&self) -> &[Vec<IdxSize>] {
        &self.all
    }

    pub fn first(&self) -> &[IdxSize] {
        &self.first
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
        std::slice::Iter<'a, Vec<IdxSize>>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.first.iter().copied().zip(self.all.iter())
    }
}

impl<'a> IntoIterator for GroupsIdx {
    type Item = IdxItem;
    type IntoIter = std::iter::Zip<std::vec::IntoIter<IdxSize>, std::vec::IntoIter<Vec<IdxSize>>>;

    fn into_iter(mut self) -> Self::IntoIter {
        let first = std::mem::take(&mut self.first);
        let all = std::mem::take(&mut self.all);
        first.into_iter().zip(all.into_iter())
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
        rayon::slice::Iter<'a, Vec<IdxSize>>,
    >;
    type Item = BorrowIdxItem<'a>;

    fn into_par_iter(self) -> Self::Iter {
        self.first.par_iter().copied().zip(self.all.par_iter())
    }
}

impl IntoParallelIterator for GroupsIdx {
    type Iter = rayon::iter::Zip<rayon::vec::IntoIter<IdxSize>, rayon::vec::IntoIter<Vec<IdxSize>>>;
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
/// Only used when group values are stored together
pub type GroupsSlice = Vec<[IdxSize; 2]>;

#[derive(Debug, Clone, PartialEq)]
pub enum GroupsProxy {
    Idx(GroupsIdx),
    Slice(GroupsSlice),
}

impl Default for GroupsProxy {
    fn default() -> Self {
        GroupsProxy::Idx(GroupsIdx::default())
    }
}

impl GroupsProxy {
    #[cfg(feature = "private")]
    pub fn into_idx(self) -> GroupsIdx {
        match self {
            GroupsProxy::Idx(groups) => groups,
            GroupsProxy::Slice(groups) => groups
                .iter()
                .map(|&[first, len]| (first, (first..first + len).collect_trusted::<Vec<_>>()))
                .collect(),
        }
    }

    pub fn iter(&self) -> GroupsProxyIter {
        GroupsProxyIter::new(self)
    }

    #[cfg(feature = "private")]
    pub fn sort(&mut self) {
        match self {
            GroupsProxy::Idx(groups) => groups.sort(),
            GroupsProxy::Slice(groups) => {
                groups.sort_unstable_by_key(|[first, _]| *first);
            }
        }
    }

    pub fn group_lengths(&self, name: &str) -> IdxCa {
        let ca: NoNull<IdxCa> = match self {
            GroupsProxy::Idx(groups) => groups
                .iter()
                .map(|(_, groups)| groups.len() as IdxSize)
                .collect_trusted(),
            GroupsProxy::Slice(groups) => groups.iter().map(|g| g[1]).collect_trusted(),
        };
        let mut ca = ca.into_inner();
        ca.rename(name);
        ca
    }

    #[cfg(feature = "private")]
    pub fn par_iter(&self) -> GroupsProxyParIter {
        GroupsProxyParIter::new(self)
    }

    /// Get a reference to the `GroupsIdx`.
    ///
    /// # Panic
    ///
    /// panics if the groups are a slice.
    pub fn idx_ref(&self) -> &GroupsIdx {
        match self {
            GroupsProxy::Idx(groups) => groups,
            GroupsProxy::Slice(_) => panic!("groups are slices not index"),
        }
    }

    pub fn get(&self, index: usize) -> GroupsIndicator {
        match self {
            GroupsProxy::Idx(groups) => {
                let first = groups.first[index];
                let all = &groups.all[index];
                GroupsIndicator::Idx((first, all))
            }
            GroupsProxy::Slice(groups) => GroupsIndicator::Slice(groups[index]),
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
            GroupsProxy::Slice(_) => panic!("groups are slices not index"),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            GroupsProxy::Idx(groups) => groups.len(),
            GroupsProxy::Slice(groups) => groups.len(),
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
            }
            GroupsProxy::Slice(groups) => {
                let ca: NoNull<IdxCa> = groups.iter().map(|[_first, len]| *len).collect_trusted();
                ca.into_inner()
            }
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
            GroupsProxy::Slice(groups) => groups
                .iter()
                .map(|&[first, len]| {
                    let ca: NoNull<IdxCa> = (first..first + len).collect_trusted();
                    ca.into_inner().into_series()
                })
                .collect_trusted(),
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

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == self.len {
            return None;
        }

        let out = unsafe {
            match self.vals {
                GroupsProxy::Idx(groups) => {
                    let item = groups.get_unchecked(self.idx);
                    Some(GroupsIndicator::Idx(item))
                }
                GroupsProxy::Slice(groups) => {
                    Some(GroupsIndicator::Slice(*groups.get_unchecked(self.idx)))
                }
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
                    GroupsProxy::Slice(groups) => GroupsIndicator::Slice(*groups.get_unchecked(i)),
                }
            })
            .drive_unindexed(consumer)
    }
}
