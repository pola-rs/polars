use super::Scalar;
use crate::datatypes::DataType;
use crate::types::NativeType;

/// The implementation of [`Scalar`] for primitive, semantically equivalent to [`Option<T>`]
/// with [`DataType`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimitiveScalar<T: NativeType> {
    value: Option<T>,
    data_type: DataType,
}

impl<T: NativeType> PrimitiveScalar<T> {
    /// Returns a new [`PrimitiveScalar`].
    #[inline]
    pub fn new(data_type: DataType, value: Option<T>) -> Self {
        if !data_type.to_physical_type().eq_primitive(T::PRIMITIVE) {
            panic!(
                "Type {} does not support logical type {:?}",
                std::any::type_name::<T>(),
                data_type
            )
        }
        Self { value, data_type }
    }

    /// Returns the optional value.
    #[inline]
    pub fn value(&self) -> &Option<T> {
        &self.value
    }

    /// Returns a new `PrimitiveScalar` with the same value but different [`DataType`]
    /// # Panic
    /// This function panics if the `data_type` is not valid for self's physical type `T`.
    pub fn to(self, data_type: DataType) -> Self {
        Self::new(data_type, self.value)
    }
}

impl<T: NativeType> From<Option<T>> for PrimitiveScalar<T> {
    #[inline]
    fn from(v: Option<T>) -> Self {
        Self::new(T::PRIMITIVE.into(), v)
    }
}

impl<T: NativeType> Scalar for PrimitiveScalar<T> {
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
        &self.data_type
    }
}
