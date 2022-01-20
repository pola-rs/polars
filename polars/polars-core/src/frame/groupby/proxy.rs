/// Indexes of the groups, the first index is stored separately.
/// this make sorting fast.
pub type GroupsIdx = Vec<(u32, Vec<u32>)>;
/// Every group is indicated by an array where the
///  - first value is an index to the start of the group
///  - second value is the length of the group
/// Only used when group values are stored together
pub type GroupsSlice = Vec<[u32; 2]>;

pub enum GroupsProxy {
    Idx(GroupsIdx),
    Slice(GroupsSlice),
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

    /// Get a reference to the `GroupsIdx`.
    ///
    /// # Panic
    ///
    /// panics if the groups are a slice.
    pub(crate) fn idx_ref(&self) -> &GroupsIdx {
        match self {
            GroupsProxy::Idx(groups) => groups,
            GroupsProxy::Slice(_) => panic!("groups are slices not index"),
        }
    }

    pub(crate) fn idx_mut(&mut self) -> &mut GroupsIdx {
        match self {
            GroupsProxy::Idx(groups) => groups,
            GroupsProxy::Slice(_) => panic!("groups are slices not index"),
        }
    }
}

impl From<GroupsIdx> for GroupsProxy {
    fn from(groups: GroupsIdx) -> Self {
        GroupsProxy::Idx(groups)
    }
}
