#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{ArrowDataType, Metadata};

/// Represents Arrow's metadata of a "column".
///
/// A [`Field`] is the closest representation of the traditional "column": a logical type
/// ([`ArrowDataType`]) with a name and nullability.
/// A Field has optional [`Metadata`] that can be used to annotate the field with custom metadata.
///
/// Almost all IO in this crate uses [`Field`] to represent logical information about the data
/// to be serialized.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Field {
    /// Its name
    pub name: String,
    /// Its logical [`ArrowDataType`]
    pub data_type: ArrowDataType,
    /// Its nullability
    pub is_nullable: bool,
    /// Additional custom (opaque) metadata.
    pub metadata: Metadata,
}

impl Field {
    /// Creates a new [`Field`].
    pub fn new<T: Into<String>>(name: T, data_type: ArrowDataType, is_nullable: bool) -> Self {
        Field {
            name: name.into(),
            data_type,
            is_nullable,
            metadata: Default::default(),
        }
    }

    /// Creates a new [`Field`] with metadata.
    #[inline]
    pub fn with_metadata(self, metadata: Metadata) -> Self {
        Self {
            name: self.name,
            data_type: self.data_type,
            is_nullable: self.is_nullable,
            metadata,
        }
    }

    /// Returns the [`Field`]'s [`ArrowDataType`].
    #[inline]
    pub fn data_type(&self) -> &ArrowDataType {
        &self.data_type
    }
}

#[cfg(feature = "arrow_rs")]
impl From<Field> for arrow_schema::Field {
    fn from(value: Field) -> Self {
        Self::new(value.name, value.data_type.into(), value.is_nullable)
            .with_metadata(value.metadata.into_iter().collect())
    }
}

#[cfg(feature = "arrow_rs")]
impl From<arrow_schema::Field> for Field {
    fn from(value: arrow_schema::Field) -> Self {
        (&value).into()
    }
}

#[cfg(feature = "arrow_rs")]
impl From<&arrow_schema::Field> for Field {
    fn from(value: &arrow_schema::Field) -> Self {
        let data_type = value.data_type().clone().into();
        let metadata = value
            .metadata()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        Self::new(value.name(), data_type, value.is_nullable()).with_metadata(metadata)
    }
}

#[cfg(feature = "arrow_rs")]
impl From<arrow_schema::FieldRef> for Field {
    fn from(value: arrow_schema::FieldRef) -> Self {
        value.as_ref().into()
    }
}

#[cfg(feature = "arrow_rs")]
impl From<&arrow_schema::FieldRef> for Field {
    fn from(value: &arrow_schema::FieldRef) -> Self {
        value.as_ref().into()
    }
}
