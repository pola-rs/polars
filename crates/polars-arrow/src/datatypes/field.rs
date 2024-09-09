use polars_utils::pl_str::PlSmallStr;
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
    pub name: PlSmallStr,
    /// Its logical [`ArrowDataType`]
    pub dtype: ArrowDataType,
    /// Its nullability
    pub is_nullable: bool,
    /// Additional custom (opaque) metadata.
    pub metadata: Metadata,
}

/// Support for `ArrowSchema::from_iter([field, ..])`
impl From<Field> for (PlSmallStr, Field) {
    fn from(value: Field) -> Self {
        (value.name.clone(), value)
    }
}

impl Field {
    /// Creates a new [`Field`].
    pub fn new(name: PlSmallStr, dtype: ArrowDataType, is_nullable: bool) -> Self {
        Field {
            name,
            dtype,
            is_nullable,
            metadata: Default::default(),
        }
    }

    /// Creates a new [`Field`] with metadata.
    #[inline]
    pub fn with_metadata(self, metadata: Metadata) -> Self {
        Self {
            name: self.name,
            dtype: self.dtype,
            is_nullable: self.is_nullable,
            metadata,
        }
    }

    /// Returns the [`Field`]'s [`ArrowDataType`].
    #[inline]
    pub fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }
}

#[cfg(feature = "arrow_rs")]
impl From<Field> for arrow_schema::Field {
    fn from(value: Field) -> Self {
        Self::new(
            value.name.to_string(),
            value.dtype.into(),
            value.is_nullable,
        )
        .with_metadata(
            value
                .metadata
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        )
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
        let dtype = value.data_type().clone().into();
        let metadata = value
            .metadata()
            .iter()
            .map(|(k, v)| (PlSmallStr::from_str(k), PlSmallStr::from_str(v)))
            .collect();
        Self::new(
            PlSmallStr::from_str(value.name().as_str()),
            dtype,
            value.is_nullable(),
        )
        .with_metadata(metadata)
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
