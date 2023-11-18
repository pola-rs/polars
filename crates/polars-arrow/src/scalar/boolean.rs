use super::Scalar;
use crate::datatypes::ArrowDataType;

/// The [`Scalar`] implementation of a boolean.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanScalar {
    value: Option<bool>,
}

impl BooleanScalar {
    /// Returns a new [`BooleanScalar`]
    #[inline]
    pub fn new(value: Option<bool>) -> Self {
        Self { value }
    }

    /// The value
    #[inline]
    pub fn value(&self) -> Option<bool> {
        self.value
    }
}

impl Scalar for BooleanScalar {
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
        &ArrowDataType::Boolean
    }
}

impl From<Option<bool>> for BooleanScalar {
    #[inline]
    fn from(v: Option<bool>) -> Self {
        Self::new(v)
    }
}
