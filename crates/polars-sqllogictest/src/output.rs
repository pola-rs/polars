use polars_core::prelude::*;
use sqllogictest::{DBOutput, DefaultColumnType};

fn column_type(dtype: &DataType) -> DefaultColumnType {
    match dtype {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Int128
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => DefaultColumnType::Integer,
        DataType::Float32 | DataType::Float64 | DataType::Decimal(_, _) => {
            DefaultColumnType::FloatingPoint
        },
        _ => DefaultColumnType::Text,
    }
}

fn format_value(av: &AnyValue, floating: bool) -> String {
    if matches!(av, AnyValue::Null) {
        return "NULL".to_string();
    }
    if floating {
        return match av {
            AnyValue::Float32(v) => format!("{:.3}", v),
            AnyValue::Float64(v) => format!("{:.3}", v),
            _ => av
                .try_extract::<f64>()
                .map(|v| format!("{:.3}", v))
                .unwrap_or_else(|_| av.to_string()),
        };
    }
    match av {
        AnyValue::String(s) => {
            if s.is_empty() {
                "(empty)".to_string()
            } else {
                s.to_string()
            }
        },
        AnyValue::StringOwned(s) => {
            if s.is_empty() {
                "(empty)".to_string()
            } else {
                s.to_string()
            }
        },
        AnyValue::Boolean(b) => b.to_string(),
        other => other.to_string(),
    }
}

pub fn dataframe_to_output(df: &DataFrame) -> DBOutput<DefaultColumnType> {
    let columns = df.columns();
    let types: Vec<DefaultColumnType> = columns.iter().map(|c| column_type(c.dtype())).collect();
    let floating: Vec<bool> = types
        .iter()
        .map(|t| matches!(t, DefaultColumnType::FloatingPoint))
        .collect();

    let mut rows = Vec::with_capacity(df.height());
    for row_idx in 0..df.height() {
        let mut row = Vec::with_capacity(columns.len());
        for (col_idx, col) in columns.iter().enumerate() {
            let av = col.get(row_idx).unwrap_or(AnyValue::Null);
            row.push(format_value(&av, floating[col_idx]));
        }
        rows.push(row);
    }

    DBOutput::Rows { types, rows }
}
