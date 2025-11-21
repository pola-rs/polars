use std::ops::Deref;
use std::sync::Arc;

use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::parquet::schema::types::{ParquetType, PrimitiveType};

/// A descriptor of a parquet column. It contains the necessary information to deserialize
/// a parquet column.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Descriptor {
    /// The [`PrimitiveType`] of this column
    pub primitive_type: PrimitiveType,

    /// The maximum definition level
    pub max_def_level: i16,

    /// The maximum repetition level
    pub max_rep_level: i16,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum BaseType {
    Owned(ParquetType),
    Arc(Arc<ParquetType>),
}

impl BaseType {
    pub fn into_arc(self) -> Self {
        match self {
            BaseType::Owned(t) => Self::Arc(Arc::new(t)),
            BaseType::Arc(t) => Self::Arc(t),
        }
    }
}

impl PartialEq for BaseType {
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl Deref for BaseType {
    type Target = ParquetType;

    fn deref(&self) -> &Self::Target {
        match self {
            BaseType::Owned(i) => i,
            BaseType::Arc(i) => i.as_ref(),
        }
    }
}

/// A descriptor for leaf-level primitive columns.
/// This encapsulates information such as definition and repetition levels and is used to
/// re-assemble nested data.
#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct ColumnDescriptor {
    /// The descriptor this columns' leaf.
    pub descriptor: Descriptor,

    /// The path of this column. For instance, "a.b.c.d".
    pub path_in_schema: Vec<PlSmallStr>,

    /// The [`ParquetType`] this descriptor is a leaf of
    pub base_type: BaseType,
}

impl ColumnDescriptor {
    /// Creates new descriptor for leaf-level column.
    pub fn new(
        descriptor: Descriptor,
        path_in_schema: Vec<PlSmallStr>,
        base_type: BaseType,
    ) -> Self {
        Self {
            descriptor,
            path_in_schema,
            base_type,
        }
    }
}
