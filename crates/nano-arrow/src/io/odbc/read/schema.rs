use crate::datatypes::{DataType, Field, TimeUnit};
use crate::error::Result;

use super::super::api;
use super::super::api::ResultSetMetadata;

/// Infers the Arrow [`Field`]s from a [`ResultSetMetadata`]
pub fn infer_schema(resut_set_metadata: &impl ResultSetMetadata) -> Result<Vec<Field>> {
    let num_cols: u16 = resut_set_metadata.num_result_cols().unwrap() as u16;

    let fields = (0..num_cols)
        .map(|index| {
            let mut column_description = api::ColumnDescription::default();
            resut_set_metadata
                .describe_col(index + 1, &mut column_description)
                .unwrap();

            column_to_field(&column_description)
        })
        .collect();
    Ok(fields)
}

fn column_to_field(column_description: &api::ColumnDescription) -> Field {
    Field::new(
        column_description
            .name_to_string()
            .expect("Column name must be representable in utf8"),
        column_to_data_type(&column_description.data_type),
        column_description.could_be_nullable(),
    )
}

fn column_to_data_type(data_type: &api::DataType) -> DataType {
    use api::DataType as OdbcDataType;
    match data_type {
        OdbcDataType::Numeric {
            precision: p @ 0..=38,
            scale,
        }
        | OdbcDataType::Decimal {
            precision: p @ 0..=38,
            scale,
        } => DataType::Decimal(*p, (*scale) as usize),
        OdbcDataType::Integer => DataType::Int32,
        OdbcDataType::SmallInt => DataType::Int16,
        OdbcDataType::Real | OdbcDataType::Float { precision: 0..=24 } => DataType::Float32,
        OdbcDataType::Float { precision: _ } | OdbcDataType::Double => DataType::Float64,
        OdbcDataType::Date => DataType::Date32,
        OdbcDataType::Timestamp { precision: 0 } => DataType::Timestamp(TimeUnit::Second, None),
        OdbcDataType::Timestamp { precision: 1..=3 } => {
            DataType::Timestamp(TimeUnit::Millisecond, None)
        }
        OdbcDataType::Timestamp { precision: 4..=6 } => {
            DataType::Timestamp(TimeUnit::Microsecond, None)
        }
        OdbcDataType::Timestamp { precision: _ } => DataType::Timestamp(TimeUnit::Nanosecond, None),
        OdbcDataType::BigInt => DataType::Int64,
        OdbcDataType::TinyInt => DataType::Int8,
        OdbcDataType::Bit => DataType::Boolean,
        OdbcDataType::Binary { length } => DataType::FixedSizeBinary(*length),
        OdbcDataType::LongVarbinary { length: _ } | OdbcDataType::Varbinary { length: _ } => {
            DataType::Binary
        }
        OdbcDataType::Unknown
        | OdbcDataType::Time { precision: _ }
        | OdbcDataType::Numeric { .. }
        | OdbcDataType::Decimal { .. }
        | OdbcDataType::Other {
            data_type: _,
            column_size: _,
            decimal_digits: _,
        }
        | OdbcDataType::WChar { length: _ }
        | OdbcDataType::Char { length: _ }
        | OdbcDataType::WVarchar { length: _ }
        | OdbcDataType::LongVarchar { length: _ }
        | OdbcDataType::Varchar { length: _ } => DataType::Utf8,
    }
}
