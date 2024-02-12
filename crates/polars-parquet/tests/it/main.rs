#![forbid(unsafe_code)]
mod roundtrip;

mod read;
mod write;

// The dynamic representation of values in native Rust. This is not exhaustive.
// todo: maybe refactor this into serde/json?
#[derive(Debug, PartialEq)]
pub enum Array {
    UInt32(Vec<Option<u32>>),
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

impl Array {
    pub fn len(&self) -> usize {
        match self {
            Array::UInt32(a) => a.len(),
            Array::Int32(a) => a.len(),
            Array::Int64(a) => a.len(),
            Array::Int96(a) => a.len(),
            Array::Float(a) => a.len(),
            Array::Double(a) => a.len(),
            Array::Boolean(a) => a.len(),
            Array::Binary(a) => a.len(),
            Array::FixedLenBinary(a) => a.len(),
            Array::List(a) => a.len(),
            Array::Struct(a, _) => a[0].len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// The dynamic representation of values in native Rust. This is not exhaustive.
// todo: maybe refactor this into serde/json?
#[derive(Debug, PartialEq)]
pub enum Value {
    UInt32(Option<u32>),
    Int32(Option<i32>),
    Int64(Option<i64>),
    Int96(Option<[u32; 3]>),
    Float32(Option<f32>),
    Float64(Option<f64>),
    Boolean(Option<bool>),
    Binary(Option<Vec<u8>>),
    FixedLenBinary(Option<Vec<u8>>),
    List(Option<Array>),
}

use std::path::PathBuf;
use std::sync::Arc;

use polars_parquet::parquet::schema::types::{PhysicalType, PrimitiveType};
use polars_parquet::parquet::statistics::*;

pub fn get_path() -> PathBuf {
    let dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(dir).join("testing/parquet-testing/data")
}

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

pub fn alltypes_statistics(column: &str) -> Arc<dyn Statistics> {
    match column {
        "id" => Arc::new(PrimitiveStatistics::<i32> {
            primitive_type: PrimitiveType::from_physical("col".to_string(), PhysicalType::Int32),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(0),
            max_value: Some(7),
        }),
        "id-short-array" => Arc::new(PrimitiveStatistics::<i32> {
            primitive_type: PrimitiveType::from_physical("col".to_string(), PhysicalType::Int32),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(4),
            max_value: Some(4),
        }),
        "bool_col" => Arc::new(BooleanStatistics {
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(false),
            max_value: Some(true),
        }),
        "tinyint_col" | "smallint_col" | "int_col" => Arc::new(PrimitiveStatistics::<i32> {
            primitive_type: PrimitiveType::from_physical("col".to_string(), PhysicalType::Int32),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(0),
            max_value: Some(1),
        }),
        "bigint_col" => Arc::new(PrimitiveStatistics::<i64> {
            primitive_type: PrimitiveType::from_physical("col".to_string(), PhysicalType::Int64),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(0),
            max_value: Some(10),
        }),
        "float_col" => Arc::new(PrimitiveStatistics::<f32> {
            primitive_type: PrimitiveType::from_physical("col".to_string(), PhysicalType::Float),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(0.0),
            max_value: Some(1.1),
        }),
        "double_col" => Arc::new(PrimitiveStatistics::<f64> {
            primitive_type: PrimitiveType::from_physical("col".to_string(), PhysicalType::Double),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(0.0),
            max_value: Some(10.1),
        }),
        "date_string_col" => Arc::new(BinaryStatistics {
            primitive_type: PrimitiveType::from_physical(
                "col".to_string(),
                PhysicalType::ByteArray,
            ),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(vec![48, 49, 47, 48, 49, 47, 48, 57]),
            max_value: Some(vec![48, 52, 47, 48, 49, 47, 48, 57]),
        }),
        "string_col" => Arc::new(BinaryStatistics {
            primitive_type: PrimitiveType::from_physical(
                "col".to_string(),
                PhysicalType::ByteArray,
            ),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(vec![48]),
            max_value: Some(vec![49]),
        }),
        "timestamp_col" => {
            todo!()
        },
        _ => unreachable!(),
    }
}

// these values match the values in `integration`
pub fn pyarrow_optional(column: &str) -> Array {
    let i64_values = &[
        Some(0),
        Some(1),
        None,
        Some(3),
        None,
        Some(5),
        Some(6),
        Some(7),
        None,
        Some(9),
    ];
    let f64_values = &[
        Some(0.0),
        Some(1.0),
        None,
        Some(3.0),
        None,
        Some(5.0),
        Some(6.0),
        Some(7.0),
        None,
        Some(9.0),
    ];
    let string_values = &[
        Some(b"Hello".to_vec()),
        None,
        Some(b"aa".to_vec()),
        Some(b"".to_vec()),
        None,
        Some(b"abc".to_vec()),
        None,
        None,
        Some(b"def".to_vec()),
        Some(b"aaa".to_vec()),
    ];
    let bool_values = &[
        Some(true),
        None,
        Some(false),
        Some(false),
        None,
        Some(true),
        None,
        None,
        Some(true),
        Some(true),
    ];
    let binary_values = &[
        Some(b"aa".to_vec()),
        None,
        Some(b"cc".to_vec()),
        Some(b"dd".to_vec()),
        None,
        Some(b"ff".to_vec()),
        None,
        None,
        Some(b"ii".to_vec()),
        Some(b"jj".to_vec()),
    ];

    match column {
        "int64" => Array::Int64(i64_values.to_vec()),
        "float64" => Array::Double(f64_values.to_vec()),
        "string" => Array::Binary(string_values.to_vec()),
        "bool" => Array::Boolean(bool_values.to_vec()),
        "date" => Array::Int64(i64_values.to_vec()),
        "uint32" => Array::Int32(i64_values.iter().map(|i| i.map(|x| x as i32)).collect()),
        "fixed_binary" => Array::FixedLenBinary(binary_values.to_vec()),
        _ => unreachable!(),
    }
}

pub fn pyarrow_optional_stats(column: &str) -> (Option<i64>, Value, Value) {
    match column {
        "int64" => (Some(3), Value::Int64(Some(0)), Value::Int64(Some(9))),
        "float64" => (
            Some(3),
            Value::Float64(Some(0.0)),
            Value::Float64(Some(9.0)),
        ),
        "string" => (
            Some(4),
            Value::Binary(Some(b"".to_vec())),
            Value::Binary(Some(b"def".to_vec())),
        ),
        "bool" => (
            Some(4),
            Value::Boolean(Some(false)),
            Value::Boolean(Some(true)),
        ),
        "date" => (Some(3), Value::Int64(Some(0)), Value::Int64(Some(9))),
        "fixed_binary" => (
            Some(3),
            Value::FixedLenBinary(Some(b"aa".to_vec())),
            Value::FixedLenBinary(Some(b"jj".to_vec())),
        ),
        _ => unreachable!(),
    }
}

// these values match the values in `integration`
pub fn pyarrow_required(column: &str) -> Array {
    let i64_values = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let f64_values = &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let string_values = &[
        "Hello", "bbb", "aa", "", "bbb", "abc", "bbb", "bbb", "def", "aaa",
    ];
    let bool_values = &[
        true, true, false, false, false, true, true, true, true, true,
    ];
    let binary_values = &["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj"];

    match column {
        "int64" => Array::Int64(i64_values.iter().map(|i| Some(*i as i64)).collect()),
        "float64" => Array::Double(f64_values.iter().map(|f| Some(*f)).collect()),
        "string" => Array::Binary(
            string_values
                .iter()
                .map(|s| Some(s.as_bytes().to_vec()))
                .collect(),
        ),
        "bool" => Array::Boolean(bool_values.iter().map(|b| Some(*b)).collect()),
        "date" => Array::Int64(i64_values.iter().map(|i| Some(*i as i64)).collect()),
        "uint32" => Array::Int32(i64_values.iter().map(|i| Some(*i)).collect()),
        "fixed_binary" => Array::FixedLenBinary(
            binary_values
                .iter()
                .map(|s| Some(s.as_bytes().to_vec()))
                .collect(),
        ),
        _ => unreachable!(),
    }
}

pub fn pyarrow_required_stats(column: &str) -> (Option<i64>, Value, Value) {
    match column {
        "int64" => (Some(0), Value::Int64(Some(0)), Value::Int64(Some(9))),
        "float64" => (
            Some(3),
            Value::Float64(Some(0.0)),
            Value::Float64(Some(9.0)),
        ),
        "string" => (
            Some(4),
            Value::Binary(Some(b"".to_vec())),
            Value::Binary(Some(b"def".to_vec())),
        ),
        "bool" => (
            Some(4),
            Value::Boolean(Some(false)),
            Value::Boolean(Some(true)),
        ),
        "date" => (Some(3), Value::Int64(Some(0)), Value::Int64(Some(9))),
        "uint32" => (Some(0), Value::Int32(Some(0)), Value::Int32(Some(9))),
        "fixed_binary" => (
            Some(4),
            Value::FixedLenBinary(Some(b"aa".to_vec())),
            Value::FixedLenBinary(Some(b"jj".to_vec())),
        ),
        _ => unreachable!(),
    }
}

// these values match the values in `integration`
pub fn pyarrow_nested_optional(column: &str) -> Array {
    //    [[0, 1], None, [2, None, 3], [4, 5, 6], [], [7, 8, 9], None, [10]]
    // def: 3, 3,  0,     3, 2,    3,   3, 3, 3,  1    3  3  3   0      3
    // rep: 0, 1,  0,     0, 1,    1,   0, 1, 1,  0,   0, 1, 1,  0,     0
    let data = vec![
        Some(Array::Int64(vec![Some(0), Some(1)])),
        None,
        Some(Array::Int64(vec![Some(2), None, Some(3)])),
        Some(Array::Int64(vec![Some(4), Some(5), Some(6)])),
        Some(Array::Int64(vec![])),
        Some(Array::Int64(vec![Some(7), Some(8), Some(9)])),
        None,
        Some(Array::Int64(vec![Some(10)])),
    ];

    match column {
        "list_int64" => Array::List(data),
        _ => unreachable!(),
    }
}

// these values match the values in `integration`
pub fn pyarrow_struct_optional(column: &str) -> Array {
    let validity = vec![false, true, true, true, true, true, true, true, true, true];

    let string = vec![
        Some("Hello".to_string()),
        None,
        Some("aa".to_string()),
        Some("".to_string()),
        None,
        Some("abc".to_string()),
        None,
        None,
        Some("def".to_string()),
        Some("aaa".to_string()),
    ]
    .into_iter()
    .map(|s| s.map(|s| s.as_bytes().to_vec()))
    .collect::<Vec<_>>();
    let boolean = vec![
        Some(true),
        None,
        Some(false),
        Some(false),
        None,
        Some(true),
        None,
        None,
        Some(true),
        Some(true),
    ];

    match column {
        "struct_nullable" => {
            let string = string
                .iter()
                .zip(validity.iter())
                .map(|(item, valid)| if *valid { item.clone() } else { None })
                .collect();
            let boolean = boolean
                .iter()
                .zip(validity.iter())
                .map(|(item, valid)| if *valid { *item } else { None })
                .collect();
            Array::Struct(
                vec![Array::Binary(string), Array::Boolean(boolean)],
                validity,
            )
        },
        "struct_required" => Array::Struct(
            vec![Array::Binary(string), Array::Boolean(boolean)],
            vec![true; validity.len()],
        ),
        _ => unreachable!(),
    }
}
