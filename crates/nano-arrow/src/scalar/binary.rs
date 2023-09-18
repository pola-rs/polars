use crate::{datatypes::DataType, offset::Offset};

use super::Scalar;

/// The [`Scalar`] implementation of binary ([`Option<Vec<u8>>`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryScalar<O: Offset> {
    value: Option<Vec<u8>>,
    phantom: std::marker::PhantomData<O>,
}

impl<O: Offset> BinaryScalar<O> {
    /// Returns a new [`BinaryScalar`].
    #[inline]
    pub fn new<P: Into<Vec<u8>>>(value: Option<P>) -> Self {
        Self {
            value: value.map(|x| x.into()),
            phantom: std::marker::PhantomData,
        }
    }

    /// Its value
    #[inline]
    pub fn value(&self) -> Option<&[u8]> {
        self.value.as_ref().map(|x| x.as_ref())
    }
}

impl<O: Offset, P: Into<Vec<u8>>> From<Option<P>> for BinaryScalar<O> {
    #[inline]
    fn from(v: Option<P>) -> Self {
        Self::new(v)
    }
}

impl<O: Offset> Scalar for BinaryScalar<O> {
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
            &DataType::LargeBinary
        } else {
            &DataType::Binary
        }
    }
}
