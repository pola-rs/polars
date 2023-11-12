use parquet_format_safe::{ConvertedType, SchemaElement};

use super::super::types::ParquetType;
use crate::parquet::schema::types::PrimitiveType;

impl ParquetType {
    /// Method to convert to Thrift.
    pub(crate) fn to_thrift(&self) -> Vec<SchemaElement> {
        let mut elements: Vec<SchemaElement> = Vec::new();
        to_thrift_helper(self, &mut elements, true);
        elements
    }
}

/// Constructs list of `SchemaElement` from the schema using depth-first traversal.
/// Here we assume that schema is always valid and starts with group type.
fn to_thrift_helper(schema: &ParquetType, elements: &mut Vec<SchemaElement>, is_root: bool) {
    match schema {
        ParquetType::PrimitiveType(PrimitiveType {
            field_info,
            logical_type,
            converted_type,
            physical_type,
        }) => {
            let (type_, type_length) = (*physical_type).into();
            let (converted_type, maybe_decimal) = converted_type
                .map(|x| x.into())
                .map(|x: (ConvertedType, Option<(i32, i32)>)| (Some(x.0), x.1))
                .unwrap_or((None, None));

            let element = SchemaElement {
                type_: Some(type_),
                type_length,
                repetition_type: Some(field_info.repetition.into()),
                name: field_info.name.clone(),
                num_children: None,
                converted_type,
                precision: maybe_decimal.map(|x| x.0),
                scale: maybe_decimal.map(|x| x.1),
                field_id: field_info.id,
                logical_type: logical_type.map(|x| x.into()),
            };

            elements.push(element);
        },
        ParquetType::GroupType {
            field_info,
            fields,
            logical_type,
            converted_type,
        } => {
            let converted_type = converted_type.map(|x| x.into());

            let repetition_type = if is_root {
                // https://github.com/apache/parquet-format/blob/7f06e838cbd1b7dbd722ff2580b9c2525e37fc46/src/main/thrift/parquet.thrift#L363
                None
            } else {
                Some(field_info.repetition)
            };

            let element = SchemaElement {
                type_: None,
                type_length: None,
                repetition_type: repetition_type.map(|x| x.into()),
                name: field_info.name.clone(),
                num_children: Some(fields.len() as i32),
                converted_type,
                scale: None,
                precision: None,
                field_id: field_info.id,
                logical_type: logical_type.map(|x| x.into()),
            };

            elements.push(element);

            // Add child elements for a group
            for field in fields {
                to_thrift_helper(field, elements, false);
            }
        },
    }
}
