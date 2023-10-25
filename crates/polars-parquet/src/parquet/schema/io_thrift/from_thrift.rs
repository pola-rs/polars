use parquet_format_safe::SchemaElement;

use crate::{
    error::{Error, Result},
    schema::types::FieldInfo,
};

use super::super::types::ParquetType;

impl ParquetType {
    /// Method to convert from Thrift.
    pub fn try_from_thrift(elements: &[SchemaElement]) -> Result<ParquetType> {
        let mut index = 0;
        let mut schema_nodes = Vec::new();
        while index < elements.len() {
            let t = from_thrift_helper(elements, index)?;
            index = t.0;
            schema_nodes.push(t.1);
        }
        if schema_nodes.len() != 1 {
            return Err(Error::oos(format!(
                "Expected exactly one root node, but found {}",
                schema_nodes.len()
            )));
        }

        Ok(schema_nodes.remove(0))
    }
}

/// Constructs a new Type from the `elements`, starting at index `index`.
/// The first result is the starting index for the next Type after this one. If it is
/// equal to `elements.len()`, then this Type is the last one.
/// The second result is the result Type.
fn from_thrift_helper(elements: &[SchemaElement], index: usize) -> Result<(usize, ParquetType)> {
    // Whether or not the current node is root (message type).
    // There is only one message type node in the schema tree.
    let is_root_node = index == 0;

    let element = elements
        .get(index)
        .ok_or_else(|| Error::oos(format!("index {} on SchemaElement is not valid", index)))?;
    let name = element.name.clone();
    let converted_type = element.converted_type;

    let id = element.field_id;
    match element.num_children {
        // From parquet-format:
        //   The children count is used to construct the nested relationship.
        //   This field is not set when the element is a primitive type
        // Sometimes parquet-cpp sets num_children field to 0 for primitive types, so we
        // have to handle this case too.
        None | Some(0) => {
            // primitive type
            let repetition = element
                .repetition_type
                .ok_or_else(|| Error::oos("Repetition level must be defined for a primitive type"))?
                .try_into()?;
            let physical_type = element
                .type_
                .ok_or_else(|| Error::oos("Physical type must be defined for a primitive type"))?;

            let converted_type = converted_type
                .map(|converted_type| {
                    let maybe_decimal = match (element.precision, element.scale) {
                        (Some(precision), Some(scale)) => Some((precision, scale)),
                        (None, None) => None,
                        _ => {
                            return Err(Error::oos(
                                "When precision or scale are defined, both must be defined",
                            ))
                        }
                    };
                    (converted_type, maybe_decimal).try_into()
                })
                .transpose()?;

            let logical_type = element
                .logical_type
                .clone()
                .map(|x| x.try_into())
                .transpose()?;

            let tp = ParquetType::try_from_primitive(
                name,
                (physical_type, element.type_length).try_into()?,
                repetition,
                converted_type,
                logical_type,
                id,
            )?;

            Ok((index + 1, tp))
        }
        Some(n) => {
            let mut fields = vec![];
            let mut next_index = index + 1;
            for _ in 0..n {
                let child_result = from_thrift_helper(elements, next_index)?;
                next_index = child_result.0;
                fields.push(child_result.1);
            }

            let tp = if is_root_node {
                ParquetType::new_root(name, fields)
            } else {
                let repetition = if let Some(repetition) = element.repetition_type {
                    repetition.try_into()?
                } else {
                    return Err(Error::oos(
                        "The repetition level of a non-root must be non-null",
                    ));
                };

                let converted_type = converted_type.map(|x| x.try_into()).transpose()?;

                let logical_type = element
                    .logical_type
                    .clone()
                    .map(|x| x.try_into())
                    .transpose()?;

                ParquetType::GroupType {
                    field_info: FieldInfo {
                        name,
                        repetition,
                        id,
                    },
                    fields,
                    converted_type,
                    logical_type,
                }
            };
            Ok((next_index, tp))
        }
    }
}
