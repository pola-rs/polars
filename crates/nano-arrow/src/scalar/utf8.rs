use crate::{datatypes::DataType, offset::Offset};

use super::Scalar;

/// The implementation of [`Scalar`] for utf8, semantically equivalent to [`Option<String>`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Utf8Scalar<O: Offset> {
    value: Option<String>,
    phantom: std::marker::PhantomData<O>,
}

impl<O: Offset> Utf8Scalar<O> {
    /// Returns a new [`Utf8Scalar`]
    #[inline]
    pub fn new<P: Into<String>>(value: Option<P>) -> Self {
        Self {
            value: value.map(|x| x.into()),
            phantom: std::marker::PhantomData,
        }
    }

    /// Returns the value irrespectively of the validity.
    #[inline]
    pub fn value(&self) -> Option<&str> {
        self.value.as_ref().map(|x| x.as_ref())
    }
}

impl<O: Offset, P: Into<String>> From<Option<P>> for Utf8Scalar<O> {
    #[inline]
    fn from(v: Option<P>) -> Self {
        Self::new(v)
    }
}

impl<O: Offset> Scalar for Utf8Scalar<O> {
    #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[inline]
    fn is_valid(&self) -> bool {
        self.value.is_some()
    }

    #[inline]
    fn data_type(&self) -> &DataType {
        if O::IS_LARGE {
            &DataType::LargeUtf8
        } else {
            &DataType::Utf8
        }
    }
}
