use std::fmt::{Debug, Formatter};

use super::Scalar;
use crate::array::ViewType;
use crate::datatypes::ArrowDataType;

/// The implementation of [`Scalar`] for utf8, semantically equivalent to [`Option<String>`].
#[derive(PartialEq, Eq)]
pub struct BinaryViewScalar<T: ViewType + ?Sized> {
    value: Option<T::Owned>,
    phantom: std::marker::PhantomData<T>,
}

impl<T: ViewType + ?Sized> Debug for BinaryViewScalar<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Scalar({:?})", self.value)
    }
}

impl<T: ViewType + ?Sized> Clone for BinaryViewScalar<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            phantom: Default::default(),
        }
    }
}

impl<T: ViewType + ?Sized> BinaryViewScalar<T> {
    /// Returns a new [`BinaryViewScalar`]
    #[inline]
    pub fn new(value: Option<&T>) -> Self {
        Self {
            value: value.map(|x| x.into_owned()),
            phantom: std::marker::PhantomData,
        }
    }

    /// Returns the value irrespectively of the validity.
    #[inline]
    pub fn value(&self) -> Option<&T> {
        self.value.as_ref().map(|x| x.as_ref())
    }
}

impl<T: ViewType + ?Sized> From<Option<&T>> for BinaryViewScalar<T> {
    #[inline]
    fn from(v: Option<&T>) -> Self {
        Self::new(v)
    }
}

impl<T: ViewType + ?Sized> Scalar for BinaryViewScalar<T> {
    #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[inline]
    fn is_valid(&self) -> bool {
        self.value.is_some()
    }

    #[inline]
    fn data_type(&self) -> &ArrowDataType {
        if T::IS_UTF8 {
            &ArrowDataType::Utf8View
        } else {
            &ArrowDataType::BinaryView
        }
    }
}
