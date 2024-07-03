use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use super::{Metadata, MetadataTrait};
use crate::chunked_array::PolarsDataType;

// I have attempted multiple times to move this interior mutability to a per metadata field basis.
// While this might allow the use of Atomics instead of RwLocks, it suffers several problems:
//
// 1. The amount of boilerplate explodes. For example, you want read, read_blocking, write,
//    write_blocking, get_mut, set for each field.
// 2. It is also very difficult to combine with the dynamic dispatch.
// 3. It is difficult to combine with types that do not allow for atomics (e.g. Box<[u8]>).
// 4. You actually have 2 fields per field: the Option and the Value. You run into critical section
//    problems if you try to separate these.

/// An interiorally mutable [`Metadata`]
///
/// This is essentially a more convenient API around `RwLock<Metadata>`. This also allows it to be
/// `Clone`.
pub struct IMMetadata<T: PolarsDataType>(RwLock<Metadata<T>>);

impl<'a, T: PolarsDataType + 'a> IMMetadata<T>
where
    Metadata<T>: MetadataTrait + 'a,
{
    /// Cast the [`IMMetadata`] to a trait object of [`MetadataTrait`]
    pub fn upcast(&'a self) -> &'a RwLock<dyn MetadataTrait + 'a> {
        &self.0 as &RwLock<dyn MetadataTrait + 'a>
    }
}

impl<T: PolarsDataType> IMMetadata<T> {
    pub const fn new(md: Metadata<T>) -> Self {
        Self(RwLock::new(md))
    }

    /// Try to grab a read guard to the [`Metadata`], this fails if this blocks.
    pub fn try_read(&self) -> Option<RwLockReadGuard<Metadata<T>>> {
        self.0.try_read().ok()
    }
    /// Block to grab a read guard the [`Metadata`]
    pub fn read(&self) -> RwLockReadGuard<Metadata<T>> {
        self.0.read().unwrap()
    }

    /// Try to grab a write guard to the [`Metadata`], this fails if this blocks.
    pub fn try_write(&self) -> Option<RwLockWriteGuard<Metadata<T>>> {
        self.0.try_write().ok()
    }
    /// Block to grab a write guard the [`Metadata`]
    pub fn write(&self) -> RwLockWriteGuard<Metadata<T>> {
        self.0.write().unwrap()
    }

    /// Take the internal [`Metadata`]
    pub fn take(self) -> Metadata<T> {
        self.0.into_inner().unwrap()
    }
    /// Get the mutable to the internal [`Metadata`]
    pub fn get_mut(&mut self) -> &mut Metadata<T> {
        self.0.get_mut().unwrap()
    }
}

impl<T: PolarsDataType> Clone for IMMetadata<T> {
    fn clone(&self) -> Self {
        Self::new(self.read().clone())
    }
}

impl<T: PolarsDataType> Default for IMMetadata<T> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}
