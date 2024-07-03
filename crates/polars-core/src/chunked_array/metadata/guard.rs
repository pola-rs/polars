use std::ops::Deref;
use std::sync::RwLockReadGuard;

use super::Metadata;
use crate::chunked_array::PolarsDataType;

/// A read guard for [`Metadata`]
pub enum MetadataReadGuard<'a, T: PolarsDataType + 'a> {
    Unlocked(RwLockReadGuard<'a, Metadata<T>>),
    Locked(&'a Metadata<T>),
}

impl<'a, T: PolarsDataType + 'a> Deref for MetadataReadGuard<'a, T> {
    type Target = Metadata<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Unlocked(v) => v.deref(),
            Self::Locked(v) => v,
        }
    }
}
