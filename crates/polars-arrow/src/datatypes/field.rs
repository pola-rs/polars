use std::sync::Arc;

use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{ArrowDataType, Metadata};

pub static DTYPE_ENUM_VALUES: &str = "_PL_ENUM_VALUES";
pub static DTYPE_CATEGORICAL: &str = "_PL_CATEGORICAL";

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
    pub metadata: Option<Arc<Metadata>>,
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
        if metadata.is_empty() {
            return self;
        }
        Self {
            name: self.name,
            dtype: self.dtype,
            is_nullable: self.is_nullable,
            metadata: Some(Arc::new(metadata)),
        }
    }

    /// Returns the [`Field`]'s [`ArrowDataType`].
    #[inline]
    pub fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    pub fn is_enum(&self) -> bool {
        if let Some(md) = &self.metadata {
            md.get(DTYPE_ENUM_VALUES).is_some()
        } else {
            false
        }
    }

    pub fn is_categorical(&self) -> bool {
        if let Some(md) = &self.metadata {
            md.get(DTYPE_CATEGORICAL).is_some()
        } else {
            false
        }
    }

    pub fn map_dtype(mut self, f: impl FnOnce(ArrowDataType) -> ArrowDataType) -> Self {
        self.dtype = f(self.dtype);
        self
    }

    pub fn map_dtype_mut(&mut self, f: impl FnOnce(&mut ArrowDataType)) {
        f(&mut self.dtype);
    }

    pub fn with_dtype(&self, dtype: ArrowDataType) -> Self {
        let mut field = self.clone();
        field.dtype = dtype;
        field
    }
}
