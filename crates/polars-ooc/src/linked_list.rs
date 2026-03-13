use crate::memory_manager::DfKey;

/// Lock-free linked list for tracking token locations during spill.
///
/// TODO: not yet implemented — push is a no-op and iter returns empty.
pub(crate) struct LockFreeLinkedList {
    _private: (),
}

unsafe impl Send for LockFreeLinkedList {}
unsafe impl Sync for LockFreeLinkedList {}

impl LockFreeLinkedList {
    pub(crate) fn new() -> Self {
        Self { _private: () }
    }

    pub(crate) fn push(&self, _value: (u64, DfKey)) {}

    pub(crate) fn iter(&self) -> Iter<'_> {
        Iter {
            _marker: std::marker::PhantomData,
        }
    }
}

pub(crate) struct Iter<'a> {
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a (u64, DfKey);

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}
