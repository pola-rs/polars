use super::Scalar;
use crate::datatypes::ArrowDataType;

/// The representation of a single entry of a [`crate::array::NullArray`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NullScalar {}

impl NullScalar {
    /// A new [`NullScalar`]
    #[inline]
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for NullScalar {
    fn default() -> Self {
        Self::new()
    }
}

impl Scalar for NullScalar {
    #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[inline]
    fn is_valid(&self) -> bool {
        false
    }

    #[inline]
    fn data_type(&self) -> &ArrowDataType {
        &ArrowDataType::Null
    }
}
