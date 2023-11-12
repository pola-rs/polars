use super::super::api;

use crate::datatypes::{DataType, Field};
use crate::error::{Error, Result};

/// Infers the [`api::ColumnDescription`] from the fields
pub fn infer_descriptions(fields: &[Field]) -> Result<Vec<api::ColumnDescription>> {
    fields
        .iter()
        .map(|field| {
            let nullability = if field.is_nullable {
                api::Nullability::Nullable
            } else {
                api::Nullability::NoNulls
            };
            let data_type = data_type_to(field.data_type())?;
            Ok(api::ColumnDescription {
                name: api::U16String::from_str(&field.name).into_vec(),
                nullability,
                data_type,
            })
        })
        .collect()
}

fn data_type_to(data_type: &DataType) -> Result<api::DataType> {
    Ok(match data_type {
        DataType::Boolean => api::DataType::Bit,
        DataType::Int16 => api::DataType::SmallInt,
        DataType::Int32 => api::DataType::Integer,
        DataType::Float32 => api::DataType::Float { precision: 24 },
        DataType::Float64 => api::DataType::Float { precision: 53 },
        DataType::FixedSizeBinary(length) => api::DataType::Binary { length: *length },
        DataType::Binary | DataType::LargeBinary => api::DataType::Varbinary { length: 0 },
        DataType::Utf8 | DataType::LargeUtf8 => api::DataType::Varchar { length: 0 },
        other => return Err(Error::nyi(format!("{other:?} to ODBC"))),
    })
}
