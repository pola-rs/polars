use crate::prelude::*;
use crate::utils::NoNull;
use polars_arrow::utils::CustomIterTools;
use rayon::iter::plumbing::UnindexedConsumer;
use rayon::prelude::*;

/// Indexes of the groups, the first index is stored separately.
/// this make sorting fast.
pub type GroupsIdx = Vec<(u32, Vec<u32>)>;
/// Every group is indicated by an array where the
///  - first value is an index to the start of the group
///  - second value is the length of the group
/// Only used when group values are stored together
pub type GroupsSlice = Vec<[u32; 2]>;

#[derive(Debug, Clone, PartialEq)]
pub enum GroupsProxy {
    Idx(GroupsIdx),
    Slice(GroupsSlice),
}

impl Default for GroupsProxy {
    fn default() -> Self {
        GroupsProxy::Idx(vec![])
    }
}

impl GroupsProxy {
    pub(crate) fn into_idx(self) -> GroupsIdx {
        match self {
            GroupsProxy::Idx(groups) => groups,
            GroupsProxy::Slice(groups) => groups
                .iter()
                .map(|&[first, len]| (first, (first..first + len).collect()))
                .collect(),
        }
    }

    pub fn iter(&self) -> GroupsProxyIter {
        GroupsProxyIter::new(self)
    }

    #[cfg(feature = "private")]
    pub fn sort(&mut self) {
        match self {
            GroupsProxy::Idx(groups) => {
                groups.sort_unstable_by_key(|t| t.0);
            }
            GroupsProxy::Slice(groups) => {
                groups.sort_unstable_by_key(|[first, _]| *first);
            }
        }
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
            GroupsProxy::Idx(groups) => GroupsIndicator::Idx(&groups[index]),
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

    pub fn group_count(&self) -> UInt32Chunked {
        match self {
            GroupsProxy::Idx(groups) => {
                let ca: NoNull<UInt32Chunked> = groups
                    .iter()
                    .map(|(_first, idx)| idx.len() as u32)
                    .collect_trusted();
                ca.into_inner()
            }
            GroupsProxy::Slice(groups) => {
                let ca: NoNull<UInt32Chunked> =
                    groups.iter().map(|[_first, len]| *len).collect_trusted();
                ca.into_inner()
            }
        }
    }
    pub fn as_list_chunked(&self) -> ListChunked {
        match self {
            GroupsProxy::Idx(groups) => groups
                .iter()
                .map(|(_first, idx)| {
                    let ca: NoNull<UInt32Chunked> = idx.iter().map(|&v| v as u32).collect();
                    ca.into_inner().into_series()
                })
                .collect_trusted(),
            GroupsProxy::Slice(groups) => groups
                .iter()
                .map(|&[first, len]| {
                    let ca: NoNull<UInt32Chunked> = (first..first + len).collect_trusted();
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
    Idx(&'a (u32, Vec<u32>)),
    Slice([u32; 2]),
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
                    Some(GroupsIndicator::Idx(groups.get_unchecked(self.idx)))
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
