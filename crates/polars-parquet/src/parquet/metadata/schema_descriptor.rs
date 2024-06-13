use parquet_format_safe::SchemaElement;
#[cfg(feature = "serde_types")]
use serde::{Deserialize, Serialize};

use super::column_descriptor::{ColumnDescriptor, Descriptor};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::schema::io_message::from_message;
use crate::parquet::schema::types::{FieldInfo, ParquetType};
use crate::parquet::schema::Repetition;

/// A schema descriptor. This encapsulates the top-level schemas for all the columns,
/// as well as all descriptors for all the primitive columns.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_types", derive(Deserialize, Serialize))]
pub struct SchemaDescriptor {
    name: String,
    // The top-level schema (the "message" type).
    fields: Vec<ParquetType>,

    // All the descriptors for primitive columns in this schema, constructed from
    // `schema` in DFS order.
    leaves: Vec<ColumnDescriptor>,
}

impl SchemaDescriptor {
    /// Creates new schema descriptor from Parquet schema.
    pub fn new(name: String, fields: Vec<ParquetType>) -> Self {
        let mut leaves = vec![];
        for f in &fields {
            let mut path = vec![];
            build_tree(f, f, 0, 0, &mut leaves, &mut path);
        }

        Self {
            name,
            fields,
            leaves,
        }
    }

    /// The [`ColumnDescriptor`] (leaves) of this schema.
    ///
    /// Note that, for nested fields, this may contain more entries than the number of fields
    /// in the file - e.g. a struct field may have two columns.
    pub fn columns(&self) -> &[ColumnDescriptor] {
        &self.leaves
    }

    /// The schemas' name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The schemas' fields.
    pub fn fields(&self) -> &[ParquetType] {
        &self.fields
    }

    pub(crate) fn into_thrift(self) -> Vec<SchemaElement> {
        ParquetType::GroupType {
            field_info: FieldInfo {
                name: self.name,
                repetition: Repetition::Optional,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            fields: self.fields,
        }
        .to_thrift()
    }

    fn try_from_type(type_: ParquetType) -> ParquetResult<Self> {
        match type_ {
            ParquetType::GroupType {
                field_info, fields, ..
            } => Ok(Self::new(field_info.name, fields)),
            _ => Err(ParquetError::oos("The parquet schema MUST be a group type")),
        }
    }

    pub(crate) fn try_from_thrift(elements: &[SchemaElement]) -> ParquetResult<Self> {
        let schema = ParquetType::try_from_thrift(elements)?;
        Self::try_from_type(schema)
    }

    /// Creates a schema from
    pub fn try_from_message(message: &str) -> ParquetResult<Self> {
        let schema = from_message(message)?;
        Self::try_from_type(schema)
    }
}

fn build_tree<'a>(
    tp: &'a ParquetType,
    base_tp: &ParquetType,
    mut max_rep_level: i16,
    mut max_def_level: i16,
    leaves: &mut Vec<ColumnDescriptor>,
    path_so_far: &mut Vec<&'a str>,
) {
    path_so_far.push(tp.name());
    match tp.get_field_info().repetition {
        Repetition::Optional => {
            max_def_level += 1;
        },
        Repetition::Repeated => {
            max_def_level += 1;
            max_rep_level += 1;
        },
        _ => {},
    }

    match tp {
        ParquetType::PrimitiveType(p) => {
            let path_in_schema = path_so_far.iter().copied().map(String::from).collect();
            leaves.push(ColumnDescriptor::new(
                Descriptor {
                    primitive_type: p.clone(),
                    max_def_level,
                    max_rep_level,
                },
                path_in_schema,
                base_tp.clone(),
            ));
        },
        ParquetType::GroupType { ref fields, .. } => {
            for f in fields {
                build_tree(
                    f,
                    base_tp,
                    max_rep_level,
                    max_def_level,
                    leaves,
                    path_so_far,
                );
                path_so_far.pop();
            }
        },
    }
}
