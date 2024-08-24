#![forbid(unsafe_code)]
mod arrow;
pub(crate) mod read;
mod roundtrip;
mod write;

use std::io::Cursor;
use std::path::PathBuf;

use polars::prelude::*;

// The dynamic representation of values in native Rust. This is not exhaustive.
// todo: maybe refactor this into serde/json?
#[derive(Debug, PartialEq)]
pub enum Array {
    Int32(Vec<Option<i32>>),
    Int64(Vec<Option<i64>>),
    Int96(Vec<Option<[u32; 3]>>),
    Float(Vec<Option<f32>>),
    Double(Vec<Option<f64>>),
    Boolean(Vec<Option<bool>>),
    Binary(Vec<Option<Vec<u8>>>),
    FixedLenBinary(Vec<Option<Vec<u8>>>),
    List(Vec<Option<Array>>),
    Struct(Vec<Array>, Vec<bool>),
}

use polars_parquet::parquet::schema::types::{PhysicalType, PrimitiveType};
use polars_parquet::parquet::statistics::*;

pub fn alltypes_plain(column: &str) -> Array {
    match column {
        "id" => {
            let expected = vec![4, 5, 6, 7, 2, 3, 0, 1];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Int32(expected)
        },
        "id-short-array" => {
            let expected = vec![4];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Int32(expected)
        },
        "bool_col" => {
            let expected = vec![true, false, true, false, true, false, true, false];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Boolean(expected)
        },
        "tinyint_col" => {
            let expected = vec![0, 1, 0, 1, 0, 1, 0, 1];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Int32(expected)
        },
        "smallint_col" => {
            let expected = vec![0, 1, 0, 1, 0, 1, 0, 1];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Int32(expected)
        },
        "int_col" => {
            let expected = vec![0, 1, 0, 1, 0, 1, 0, 1];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Int32(expected)
        },
        "bigint_col" => {
            let expected = vec![0, 10, 0, 10, 0, 10, 0, 10];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Int64(expected)
        },
        "float_col" => {
            let expected = vec![0.0, 1.1, 0.0, 1.1, 0.0, 1.1, 0.0, 1.1];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Float(expected)
        },
        "double_col" => {
            let expected = vec![0.0, 10.1, 0.0, 10.1, 0.0, 10.1, 0.0, 10.1];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Double(expected)
        },
        "date_string_col" => {
            let expected = vec![
                vec![48, 51, 47, 48, 49, 47, 48, 57],
                vec![48, 51, 47, 48, 49, 47, 48, 57],
                vec![48, 52, 47, 48, 49, 47, 48, 57],
                vec![48, 52, 47, 48, 49, 47, 48, 57],
                vec![48, 50, 47, 48, 49, 47, 48, 57],
                vec![48, 50, 47, 48, 49, 47, 48, 57],
                vec![48, 49, 47, 48, 49, 47, 48, 57],
                vec![48, 49, 47, 48, 49, 47, 48, 57],
            ];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Binary(expected)
        },
        "string_col" => {
            let expected = vec![
                vec![48],
                vec![49],
                vec![48],
                vec![49],
                vec![48],
                vec![49],
                vec![48],
                vec![49],
            ];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Binary(expected)
        },
        "timestamp_col" => {
            todo!()
        },
        _ => unreachable!(),
    }
}

pub fn alltypes_statistics(column: &str) -> Statistics {
    match column {
        "id" => PrimitiveStatistics::<i32> {
            primitive_type: PrimitiveType::from_physical("col".to_string(), PhysicalType::Int32),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(0),
            max_value: Some(7),
        }
        .into(),
        "id-short-array" => PrimitiveStatistics::<i32> {
            primitive_type: PrimitiveType::from_physical("col".to_string(), PhysicalType::Int32),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(4),
            max_value: Some(4),
        }
        .into(),
        "bool_col" => BooleanStatistics {
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(false),
            max_value: Some(true),
        }
        .into(),
        "tinyint_col" | "smallint_col" | "int_col" => PrimitiveStatistics::<i32> {
            primitive_type: PrimitiveType::from_physical("col".to_string(), PhysicalType::Int32),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(0),
            max_value: Some(1),
        }
        .into(),
        "bigint_col" => PrimitiveStatistics::<i64> {
            primitive_type: PrimitiveType::from_physical("col".to_string(), PhysicalType::Int64),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(0),
            max_value: Some(10),
        }
        .into(),
        "float_col" => PrimitiveStatistics::<f32> {
            primitive_type: PrimitiveType::from_physical("col".to_string(), PhysicalType::Float),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(0.0),
            max_value: Some(1.1),
        }
        .into(),
        "double_col" => PrimitiveStatistics::<f64> {
            primitive_type: PrimitiveType::from_physical("col".to_string(), PhysicalType::Double),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(0.0),
            max_value: Some(10.1),
        }
        .into(),
        "date_string_col" => BinaryStatistics {
            primitive_type: PrimitiveType::from_physical(
                "col".to_string(),
                PhysicalType::ByteArray,
            ),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(vec![48, 49, 47, 48, 49, 47, 48, 57]),
            max_value: Some(vec![48, 52, 47, 48, 49, 47, 48, 57]),
        }
        .into(),
        "string_col" => BinaryStatistics {
            primitive_type: PrimitiveType::from_physical(
                "col".to_string(),
                PhysicalType::ByteArray,
            ),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(vec![48]),
            max_value: Some(vec![49]),
        }
        .into(),
        "timestamp_col" => {
            todo!()
        },
        _ => unreachable!(),
    }
}

#[test]
fn test_vstack_empty_3220() -> PolarsResult<()> {
    let df1 = df! {
        "a" => ["1", "2"],
        "b" => [1, 2]
    }?;
    let empty_df = df1.head(Some(0));
    let mut stacked = df1.clone();
    stacked.vstack_mut(&empty_df)?;
    stacked.vstack_mut(&df1)?;
    let mut buf = Cursor::new(Vec::new());
    ParquetWriter::new(&mut buf).finish(&mut stacked)?;
    let read_df = ParquetReader::new(buf).finish()?;
    assert!(stacked.equals(&read_df));
    Ok(())
}
