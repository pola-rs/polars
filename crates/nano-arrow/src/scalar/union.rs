use crate::datatypes::DataType;

use super::Scalar;

/// A single entry of a [`crate::array::UnionArray`].
#[derive(Debug, Clone, PartialEq)]
pub struct UnionScalar {
    value: Box<dyn Scalar>,
    type_: i8,
    data_type: DataType,
}

impl UnionScalar {
    /// Returns a new [`UnionScalar`]
    #[inline]
    pub fn new(data_type: DataType, type_: i8, value: Box<dyn Scalar>) -> Self {
        Self {
            value,
            type_,
            data_type,
        }
    }

    /// Returns the inner value
    #[inline]
    pub fn value(&self) -> &Box<dyn Scalar> {
        &self.value
    }

    /// Returns the type of the union scalar
    #[inline]
    pub fn type_(&self) -> i8 {
        self.type_
    }
}

impl Scalar for UnionScalar {
    #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[inline]
    fn is_valid(&self) -> bool {
        true
    }

    #[inline]
    fn data_type(&self) -> &DataType {
        &self.data_type
    }
}
