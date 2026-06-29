use super::Scalar;
use crate::datatypes::ArrowDataType;

#[derive(Debug, Clone, PartialEq, Eq)]
/// The [`Scalar`] implementation of fixed size binary ([`Option<Box<[u8]>>`]).
pub struct FixedSizeBinaryScalar {
    value: Option<Box<[u8]>>,
    dtype: ArrowDataType,
}

impl FixedSizeBinaryScalar {
    /// Returns a new [`FixedSizeBinaryScalar`].
    /// # Panics
    /// iff
    /// * the `dtype` is not `FixedSizeBinary`
    /// * the size of child binary is not equal
    #[inline]
    pub fn new<P: Into<Vec<u8>>>(dtype: ArrowDataType, value: Option<P>) -> Self {
        assert_eq!(
            dtype.to_physical_type(),
            crate::datatypes::PhysicalType::FixedSizeBinary
        );
        Self {
            value: value.map(|x| {
                let x: Vec<u8> = x.into();
                assert_eq!(dtype.to_storage(), &ArrowDataType::FixedSizeBinary(x.len()));
                x.into_boxed_slice()
            }),
            dtype,
        }
    }

    /// Its value
    #[inline]
    pub fn value(&self) -> Option<&[u8]> {
        self.value.as_ref().map(|x| x.as_ref())
    }
}

impl Scalar for FixedSizeBinaryScalar {
    #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[inline]
    fn is_valid(&self) -> bool {
        self.value.is_some()
    }

    #[inline]
    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }
}
