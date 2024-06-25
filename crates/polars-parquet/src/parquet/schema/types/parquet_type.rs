// see https://github.com/apache/parquet-format/blob/master/LogicalTypes.md
use polars_utils::aliases::*;
#[cfg(feature = "serde_types")]
use serde::{Deserialize, Serialize};

use super::super::Repetition;
use super::{
    spec, FieldInfo, GroupConvertedType, GroupLogicalType, PhysicalType, PrimitiveConvertedType,
    PrimitiveLogicalType,
};
use crate::parquet::error::ParquetResult;

/// The complete description of a parquet column
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde_types", derive(Deserialize, Serialize))]
pub struct PrimitiveType {
    /// The fields' generic information
    pub field_info: FieldInfo,
    /// The optional logical type
    pub logical_type: Option<PrimitiveLogicalType>,
    /// The optional converted type
    pub converted_type: Option<PrimitiveConvertedType>,
    /// The physical type
    pub physical_type: PhysicalType,
}

impl PrimitiveType {
    /// Helper method to create an optional field with no logical or converted types.
    pub fn from_physical(name: String, physical_type: PhysicalType) -> Self {
        let field_info = FieldInfo {
            name,
            repetition: Repetition::Optional,
            id: None,
        };
        Self {
            field_info,
            converted_type: None,
            logical_type: None,
            physical_type,
        }
    }
}

/// Representation of a Parquet type describing primitive and nested fields,
/// including the top-level schema of the parquet file.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde_types", derive(Deserialize, Serialize))]
pub enum ParquetType {
    PrimitiveType(PrimitiveType),
    GroupType {
        field_info: FieldInfo,
        logical_type: Option<GroupLogicalType>,
        converted_type: Option<GroupConvertedType>,
        fields: Vec<ParquetType>,
    },
}

/// Accessors
impl ParquetType {
    /// Returns [`FieldInfo`] information about the type.
    pub fn get_field_info(&self) -> &FieldInfo {
        match self {
            Self::PrimitiveType(primitive) => &primitive.field_info,
            Self::GroupType { field_info, .. } => field_info,
        }
    }

    /// Returns this type's field name.
    pub fn name(&self) -> &str {
        &self.get_field_info().name
    }

    /// Checks if `sub_type` schema is part of current schema.
    /// This method can be used to check if projected columns are part of the root schema.
    pub fn check_contains(&self, sub_type: &ParquetType) -> bool {
        let basic_match = self.get_field_info() == sub_type.get_field_info();

        match (self, sub_type) {
            (
                Self::PrimitiveType(PrimitiveType { physical_type, .. }),
                Self::PrimitiveType(PrimitiveType {
                    physical_type: other_physical_type,
                    ..
                }),
            ) => basic_match && physical_type == other_physical_type,
            (
                Self::GroupType { fields, .. },
                Self::GroupType {
                    fields: other_fields,
                    ..
                },
            ) => {
                // build hashmap of name -> Type
                let mut field_map = PlHashMap::new();
                for field in fields {
                    field_map.insert(field.name(), field);
                }

                for field in other_fields {
                    if !field_map
                        .get(field.name())
                        .map(|tpe| tpe.check_contains(field))
                        .unwrap_or(false)
                    {
                        return false;
                    }
                }
                true
            },
            _ => false,
        }
    }
}

/// Constructors
impl ParquetType {
    pub(crate) fn new_root(name: String, fields: Vec<ParquetType>) -> Self {
        let field_info = FieldInfo {
            name,
            repetition: Repetition::Optional,
            id: None,
        };
        ParquetType::GroupType {
            field_info,
            fields,
            logical_type: None,
            converted_type: None,
        }
    }

    pub fn from_converted(
        name: String,
        fields: Vec<ParquetType>,
        repetition: Repetition,
        converted_type: Option<GroupConvertedType>,
        id: Option<i32>,
    ) -> Self {
        let field_info = FieldInfo {
            name,
            repetition,
            id,
        };

        ParquetType::GroupType {
            field_info,
            fields,
            converted_type,
            logical_type: None,
        }
    }

    /// # Error
    /// Errors iff the combination of physical, logical and converted type is not valid.
    pub fn try_from_primitive(
        name: String,
        physical_type: PhysicalType,
        repetition: Repetition,
        converted_type: Option<PrimitiveConvertedType>,
        logical_type: Option<PrimitiveLogicalType>,
        id: Option<i32>,
    ) -> ParquetResult<Self> {
        spec::check_converted_invariants(&physical_type, &converted_type)?;
        spec::check_logical_invariants(&physical_type, &logical_type)?;

        let field_info = FieldInfo {
            name,
            repetition,
            id,
        };

        Ok(ParquetType::PrimitiveType(PrimitiveType {
            field_info,
            converted_type,
            logical_type,
            physical_type,
        }))
    }

    /// Helper method to create a [`ParquetType::PrimitiveType`] optional field
    /// with no logical or converted types.
    pub fn from_physical(name: String, physical_type: PhysicalType) -> Self {
        ParquetType::PrimitiveType(PrimitiveType::from_physical(name, physical_type))
    }

    pub fn from_group(
        name: String,
        repetition: Repetition,
        converted_type: Option<GroupConvertedType>,
        logical_type: Option<GroupLogicalType>,
        fields: Vec<ParquetType>,
        id: Option<i32>,
    ) -> Self {
        let field_info = FieldInfo {
            name,
            repetition,
            id,
        };

        ParquetType::GroupType {
            field_info,
            logical_type,
            converted_type,
            fields,
        }
    }
}
