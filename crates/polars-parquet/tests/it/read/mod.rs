/// Serialization to Rust's Native types.
/// In comparison to Arrow, this in-memory format does not leverage logical types nor SIMD operations,
/// but OTOH it has no external dependencies and is very familiar to Rust developers.
mod binary;
mod boolean;
mod deserialize;
mod dictionary;
mod fixed_binary;
mod indexes;
mod primitive;
mod primitive_nested;
mod struct_;
mod utils;

#[cfg(any(feature = "lz4", feature = "lz4_flex"))]
mod lz4_legacy;

use std::fs::File;

use dictionary::{deserialize as deserialize_dict, DecodedDictPage};
#[cfg(feature = "async")]
use futures::StreamExt;
use polars_parquet::parquet::error::{Error, Result};
use polars_parquet::parquet::metadata::ColumnChunkMetaData;
use polars_parquet::parquet::page::{CompressedPage, DataPage, Page};
use polars_parquet::parquet::schema::types::{GroupConvertedType, ParquetType, PhysicalType};
use polars_parquet::parquet::schema::Repetition;
use polars_parquet::parquet::statistics::{
    BinaryStatistics, BooleanStatistics, FixedLenStatistics, PrimitiveStatistics, Statistics,
};
use polars_parquet::parquet::types::int96_to_i64_ns;
use polars_parquet::parquet::FallibleStreamingIterator;
#[cfg(feature = "async")]
use polars_parquet::read::get_page_stream;
#[cfg(feature = "async")]
use polars_parquet::read::read_metadata_async;
use polars_parquet::read::{
    get_column_iterator, get_field_columns, read_metadata, BasicDecompressor, MutStreamingIterator,
    State,
};

use super::*;

/// Reads a page into an [`Array`].
/// This is CPU-intensive: decompress, decode and de-serialize.
pub fn page_to_array(page: &DataPage, dict: Option<&DecodedDictPage>) -> Result<Array> {
    let physical_type = page.descriptor.primitive_type.physical_type;
    match page.descriptor.max_rep_level {
        0 => match physical_type {
            PhysicalType::Boolean => Ok(Array::Boolean(boolean::page_to_vec(page)?)),
            PhysicalType::Int32 => {
                let dict = dict.map(|dict| {
                    if let DecodedDictPage::Int32(dict) = dict {
                        dict
                    } else {
                        panic!()
                    }
                });
                primitive::page_to_vec(page, dict).map(Array::Int32)
            },
            PhysicalType::Int64 => {
                let dict = dict.map(|dict| {
                    if let DecodedDictPage::Int64(dict) = dict {
                        dict
                    } else {
                        panic!()
                    }
                });
                primitive::page_to_vec(page, dict).map(Array::Int64)
            },
            PhysicalType::Int96 => {
                let dict = dict.map(|dict| {
                    if let DecodedDictPage::Int96(dict) = dict {
                        dict
                    } else {
                        panic!()
                    }
                });
                primitive::page_to_vec(page, dict).map(Array::Int96)
            },
            PhysicalType::Float => {
                let dict = dict.map(|dict| {
                    if let DecodedDictPage::Float(dict) = dict {
                        dict
                    } else {
                        panic!()
                    }
                });
                primitive::page_to_vec(page, dict).map(Array::Float)
            },
            PhysicalType::Double => {
                let dict = dict.map(|dict| {
                    if let DecodedDictPage::Double(dict) = dict {
                        dict
                    } else {
                        panic!()
                    }
                });
                primitive::page_to_vec(page, dict).map(Array::Double)
            },
            PhysicalType::ByteArray => {
                let dict = dict.map(|dict| {
                    if let DecodedDictPage::ByteArray(dict) = dict {
                        dict
                    } else {
                        panic!()
                    }
                });

                binary::page_to_vec(page, dict).map(Array::Binary)
            },
            PhysicalType::FixedLenByteArray(_) => {
                let dict = dict.map(|dict| {
                    if let DecodedDictPage::FixedLenByteArray(dict) = dict {
                        dict
                    } else {
                        panic!()
                    }
                });

                fixed_binary::page_to_vec(page, dict).map(Array::FixedLenBinary)
            },
        },
        _ => match dict {
            None => match physical_type {
                PhysicalType::Int64 => Ok(primitive_nested::page_to_array::<i64>(page, None)?),
                _ => todo!(),
            },
            Some(_) => match physical_type {
                PhysicalType::Int64 => {
                    let dict = dict.map(|dict| {
                        if let DecodedDictPage::Int64(dict) = dict {
                            dict
                        } else {
                            panic!()
                        }
                    });
                    Ok(primitive_nested::page_dict_to_array::<i64>(page, dict)?)
                },
                _ => todo!(),
            },
        },
    }
}

pub fn collect<I: FallibleStreamingIterator<Item = Page, Error = Error>>(
    mut iterator: I,
    type_: PhysicalType,
) -> Result<Vec<Array>> {
    let mut arrays = vec![];
    let mut dict = None;
    while let Some(page) = iterator.next()? {
        match page {
            Page::Data(page) => arrays.push(page_to_array(page, dict.as_ref())?),
            Page::Dict(page) => {
                dict = Some(deserialize_dict(page, type_)?);
            },
        }
    }
    Ok(arrays)
}

/// Reads columns into an [`Array`].
/// This is CPU-intensive: decompress, decode and de-serialize.
pub fn columns_to_array<II, I>(mut columns: I, field: &ParquetType) -> Result<Array>
where
    II: Iterator<Item = Result<CompressedPage>>,
    I: MutStreamingIterator<Item = (II, ColumnChunkMetaData), Error = Error>,
{
    let mut validity = vec![];
    let mut has_filled = false;
    let mut arrays = vec![];
    while let State::Some(mut new_iter) = columns.advance()? {
        if let Some((pages, column)) = new_iter.get() {
            let mut iterator = BasicDecompressor::new(pages, vec![]);

            let mut dict = None;
            while let Some(page) = iterator.next()? {
                match page {
                    polars_parquet::parquet::page::Page::Data(page) => {
                        if !has_filled {
                            struct_::extend_validity(&mut validity, page)?;
                        }
                        arrays.push(page_to_array(page, dict.as_ref())?)
                    },
                    polars_parquet::parquet::page::Page::Dict(page) => {
                        dict = Some(deserialize_dict(page, column.physical_type())?);
                    },
                }
            }
        }
        has_filled = true;
        columns = new_iter;
    }

    match field {
        ParquetType::PrimitiveType { .. } => {
            arrays.pop().ok_or_else(|| Error::OutOfSpec("".to_string()))
        },
        ParquetType::GroupType { converted_type, .. } => {
            if let Some(converted_type) = converted_type {
                match converted_type {
                    GroupConvertedType::List => Ok(arrays.pop().unwrap()),
                    _ => todo!(),
                }
            } else {
                Ok(Array::Struct(arrays, validity))
            }
        },
    }
}

pub fn read_column<R: std::io::Read + std::io::Seek>(
    reader: &mut R,
    row_group: usize,
    field_name: &str,
) -> Result<(Array, Option<std::sync::Arc<dyn Statistics>>)> {
    let metadata = read_metadata(reader)?;

    let field = metadata
        .schema()
        .fields()
        .iter()
        .find_map(|field| (field.name() == field_name).then_some(field))
        .ok_or_else(|| Error::OutOfSpec("column does not exist".to_string()))?;

    let columns = get_column_iterator(
        reader,
        &metadata.row_groups[row_group],
        field.name(),
        None,
        vec![],
        usize::MAX,
    );

    let mut statistics = get_field_columns(metadata.row_groups[row_group].columns(), field.name())
        .iter()
        .map(|column_meta| column_meta.statistics().transpose())
        .collect::<Result<Vec<_>>>()?;

    let array = columns_to_array(columns, field)?;

    Ok((array, statistics.pop().unwrap()))
}

#[cfg(feature = "async")]
pub async fn read_column_async<
    R: futures::AsyncRead + futures::AsyncSeek + Send + std::marker::Unpin,
>(
    reader: &mut R,
    row_group: usize,
    field_name: &str,
) -> Result<(Array, Option<std::sync::Arc<dyn Statistics>>)> {
    let metadata = read_metadata_async(reader).await?;

    let field = metadata
        .schema()
        .fields()
        .iter()
        .find_map(|field| (field.name() == field_name).then(|| field))
        .ok_or_else(|| Error::OutOfSpec("column does not exist".to_string()))?;

    let columns = get_field_columns(metadata.row_groups[row_group].columns(), field.name());
    let column = columns.first().unwrap();

    let pages = get_page_stream(column, reader, vec![], Arc::new(|_, _| true), usize::MAX).await?;

    let mut statistics = get_field_columns(metadata.row_groups[row_group].columns(), field.name())
        .iter()
        .map(|column_meta| column_meta.statistics().transpose())
        .collect::<Result<Vec<_>>>()?;

    let pages = pages.collect::<Vec<_>>().await;

    let iterator = BasicDecompressor::new(pages.into_iter(), vec![]);

    let mut arrays = collect(iterator, column.physical_type())?;

    Ok((arrays.pop().unwrap(), statistics.pop().unwrap()))
}

fn get_column(path: &str, column: &str) -> Result<(Array, Option<std::sync::Arc<dyn Statistics>>)> {
    let mut file = File::open(path).unwrap();
    read_column(&mut file, 0, column)
}

fn test_column(column: &str) -> Result<()> {
    let mut path = get_path();
    path.push("alltypes_plain.parquet");
    let path = path.to_str().unwrap();
    let (result, statistics) = get_column(path, column)?;
    // the file does not have statistics
    assert_eq!(statistics.as_ref().map(|x| x.as_ref()), None);
    assert_eq!(result, alltypes_plain(column));
    Ok(())
}

#[test]
fn int32() -> Result<()> {
    test_column("id")
}

#[test]
fn bool() -> Result<()> {
    test_column("bool_col")
}

#[test]
fn tinyint_col() -> Result<()> {
    test_column("tinyint_col")
}

#[test]
fn smallint_col() -> Result<()> {
    test_column("smallint_col")
}

#[test]
fn int_col() -> Result<()> {
    test_column("int_col")
}

#[test]
fn bigint_col() -> Result<()> {
    test_column("bigint_col")
}

#[test]
fn float_col() -> Result<()> {
    test_column("float_col")
}

#[test]
fn double_col() -> Result<()> {
    test_column("double_col")
}

#[test]
fn date_string_col() -> Result<()> {
    test_column("date_string_col")
}

#[test]
fn string_col() -> Result<()> {
    test_column("string_col")
}

#[test]
fn timestamp_col() -> Result<()> {
    let mut path = get_path();
    path.push("alltypes_plain.parquet");
    let path = path.to_str().unwrap();

    let expected = vec![
        1235865600000000000i64,
        1235865660000000000,
        1238544000000000000,
        1238544060000000000,
        1233446400000000000,
        1233446460000000000,
        1230768000000000000,
        1230768060000000000,
    ];

    let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
    let (array, _) = get_column(path, "timestamp_col")?;
    if let Array::Int96(array) = array {
        let a = array
            .into_iter()
            .map(|x| x.map(int96_to_i64_ns))
            .collect::<Vec<_>>();
        assert_eq!(expected, a);
    } else {
        panic!("Timestamp expected");
    };
    Ok(())
}

fn assert_eq_stats(expected: (Option<i64>, Value, Value), stats: &dyn Statistics) {
    match (expected.1, expected.2) {
        (Value::Int32(min), Value::Int32(max)) => {
            let s = stats
                .as_any()
                .downcast_ref::<PrimitiveStatistics<i32>>()
                .unwrap();
            assert_eq!(expected.0, s.null_count);
            assert_eq!(s.min_value, min);
            assert_eq!(s.max_value, max);
        },
        (Value::Int64(min), Value::Int64(max)) => {
            let s = stats
                .as_any()
                .downcast_ref::<PrimitiveStatistics<i64>>()
                .unwrap();
            assert_eq!(expected.0, s.null_count);
            assert_eq!(s.min_value, min);
            assert_eq!(s.max_value, max);
        },
        (Value::Float64(min), Value::Float64(max)) => {
            let s = stats
                .as_any()
                .downcast_ref::<PrimitiveStatistics<f64>>()
                .unwrap();
            assert_eq!(expected.0, s.null_count);
            assert_eq!(s.min_value, min);
            assert_eq!(s.max_value, max);
        },
        (Value::Binary(min), Value::Binary(max)) => {
            let s = stats.as_any().downcast_ref::<BinaryStatistics>().unwrap();

            assert_eq!(s.min_value, min);
            assert_eq!(s.max_value, max);
        },
        (Value::FixedLenBinary(min), Value::FixedLenBinary(max)) => {
            let s = stats.as_any().downcast_ref::<FixedLenStatistics>().unwrap();

            assert_eq!(s.min_value, min);
            assert_eq!(s.max_value, max);
        },
        (Value::Boolean(min), Value::Boolean(max)) => {
            let s = stats.as_any().downcast_ref::<BooleanStatistics>().unwrap();

            assert_eq!(s.min_value, min);
            assert_eq!(s.max_value, max);
        },

        _ => todo!(),
    }
}

fn test_pyarrow_integration(
    file: &str,
    column: &str,
    version: usize,
    required: bool,
    use_dictionary: bool,
    compression: &str,
) -> Result<()> {
    if std::env::var("PARQUET2_IGNORE_PYARROW_TESTS").is_ok() {
        return Ok(());
    }
    let required_s = if required { "required" } else { "nullable" };
    let use_dictionary_s = if use_dictionary { "dict" } else { "non_dict" };

    let path = format!(
        "fixtures/pyarrow3/v{}/{}{}/{}_{}_10.parquet",
        version, use_dictionary_s, compression, file, required_s
    );

    let (array, statistics) = get_column(&path, column)?;

    let expected = match (file, required) {
        ("basic", true) => pyarrow_required(column),
        ("basic", false) => pyarrow_optional(column),
        ("nested", false) => pyarrow_nested_optional(column),
        ("struct", false) => pyarrow_struct_optional(column),
        _ => todo!(),
    };

    assert_eq!(expected, array);

    let expected_stats = match (file, required) {
        ("basic", true) => pyarrow_required_stats(column),
        ("basic", false) => pyarrow_optional_stats(column),
        ("nested", false) => (Some(4), Value::Int64(Some(0)), Value::Int64(Some(10))),
        // incorrect: it is only picking the first stats
        ("struct", false) => (
            Some(4),
            Value::Boolean(Some(false)),
            Value::Boolean(Some(true)),
        ),
        _ => todo!(),
    };

    assert_eq_stats(expected_stats, statistics.unwrap().as_ref());

    Ok(())
}

#[test]
fn pyarrow_v1_dict_int64_required() -> Result<()> {
    test_pyarrow_integration("basic", "int64", 1, true, true, "")
}

#[test]
fn pyarrow_v1_dict_int64_optional() -> Result<()> {
    test_pyarrow_integration("basic", "int64", 1, false, true, "")
}

#[test]
fn pyarrow_v1_non_dict_int64_required() -> Result<()> {
    test_pyarrow_integration("basic", "int64", 1, true, false, "")
}

#[test]
fn pyarrow_v1_non_dict_int64_optional() -> Result<()> {
    test_pyarrow_integration("basic", "int64", 1, false, false, "")
}

#[test]
fn pyarrow_v1_non_dict_int64_optional_brotli() -> Result<()> {
    test_pyarrow_integration("basic", "int64", 1, false, false, "/brotli")
}

#[test]
fn pyarrow_v1_non_dict_int64_optional_gzip() -> Result<()> {
    test_pyarrow_integration("basic", "int64", 1, false, false, "/gzip")
}

#[test]
fn pyarrow_v1_non_dict_int64_optional_snappy() -> Result<()> {
    test_pyarrow_integration("basic", "int64", 1, false, false, "/snappy")
}

#[test]
fn pyarrow_v1_non_dict_int64_optional_lz4() -> Result<()> {
    test_pyarrow_integration("basic", "int64", 1, false, false, "/lz4")
}

#[test]
fn pyarrow_v1_non_dict_int64_optional_zstd() -> Result<()> {
    test_pyarrow_integration("basic", "int64", 1, false, false, "/zstd")
}

#[test]
fn pyarrow_v2_non_dict_int64_optional() -> Result<()> {
    test_pyarrow_integration("basic", "int64", 2, false, false, "")
}

#[test]
fn pyarrow_v2_non_dict_int64_required() -> Result<()> {
    test_pyarrow_integration("basic", "int64", 2, true, false, "")
}

#[test]
fn pyarrow_v2_dict_int64_optional() -> Result<()> {
    test_pyarrow_integration("basic", "int64", 2, false, true, "")
}

#[test]
fn pyarrow_v2_non_dict_int64_optional_compressed() -> Result<()> {
    test_pyarrow_integration("basic", "int64", 2, false, false, "/snappy")
}

#[test]
fn pyarrow_v1_boolean_optional() -> Result<()> {
    test_pyarrow_integration("basic", "bool", 1, false, false, "")
}

#[test]
fn pyarrow_v1_boolean_required() -> Result<()> {
    test_pyarrow_integration("basic", "bool", 1, true, false, "")
}

#[test]
fn pyarrow_v1_dict_string_required() -> Result<()> {
    test_pyarrow_integration("basic", "string", 1, true, true, "")
}

#[test]
fn pyarrow_v1_dict_string_optional() -> Result<()> {
    test_pyarrow_integration("basic", "string", 1, false, true, "")
}

#[test]
fn pyarrow_v1_non_dict_string_required() -> Result<()> {
    test_pyarrow_integration("basic", "string", 1, true, false, "")
}

#[test]
fn pyarrow_v1_non_dict_string_optional() -> Result<()> {
    test_pyarrow_integration("basic", "string", 1, false, false, "")
}

#[test]
fn pyarrow_v1_dict_fixed_binary_required() -> Result<()> {
    test_pyarrow_integration("basic", "fixed_binary", 1, true, true, "")
}

#[test]
fn pyarrow_v1_dict_fixed_binary_optional() -> Result<()> {
    test_pyarrow_integration("basic", "fixed_binary", 1, false, true, "")
}

#[test]
fn pyarrow_v1_non_dict_fixed_binary_required() -> Result<()> {
    test_pyarrow_integration("basic", "fixed_binary", 1, true, false, "")
}

#[test]
fn pyarrow_v1_non_dict_fixed_binary_optional() -> Result<()> {
    test_pyarrow_integration("basic", "fixed_binary", 1, false, false, "")
}

#[test]
fn pyarrow_v1_dict_list_optional() -> Result<()> {
    test_pyarrow_integration("nested", "list_int64", 1, false, true, "")
}

#[test]
fn pyarrow_v1_non_dict_list_optional() -> Result<()> {
    test_pyarrow_integration("nested", "list_int64", 1, false, false, "")
}

#[test]
fn pyarrow_v1_struct_optional() -> Result<()> {
    test_pyarrow_integration("struct", "struct_nullable", 1, false, false, "")
}

#[test]
fn pyarrow_v2_struct_optional() -> Result<()> {
    test_pyarrow_integration("struct", "struct_nullable", 2, false, false, "")
}

#[test]
fn pyarrow_v1_struct_required() -> Result<()> {
    test_pyarrow_integration("struct", "struct_required", 1, false, false, "")
}

#[test]
fn pyarrow_v2_struct_required() -> Result<()> {
    test_pyarrow_integration("struct", "struct_required", 2, false, false, "")
}

#[test]
fn test_metadata() -> Result<()> {
    let mut testdata = get_path();
    testdata.push("alltypes_plain.parquet");
    let mut file = File::open(testdata).unwrap();

    let metadata = read_metadata(&mut file)?;

    let columns = metadata.schema_descr.columns();

    /*
    from pyarrow:
    required group field_id=0 schema {
        optional int32 field_id=1 id;
        optional boolean field_id=2 bool_col;
        optional int32 field_id=3 tinyint_col;
        optional int32 field_id=4 smallint_col;
        optional int32 field_id=5 int_col;
        optional int64 field_id=6 bigint_col;
        optional float field_id=7 float_col;
        optional double field_id=8 double_col;
        optional binary field_id=9 date_string_col;
        optional binary field_id=10 string_col;
        optional int96 field_id=11 timestamp_col;
    }
    */
    let expected = vec![
        PhysicalType::Int32,
        PhysicalType::Boolean,
        PhysicalType::Int32,
        PhysicalType::Int32,
        PhysicalType::Int32,
        PhysicalType::Int64,
        PhysicalType::Float,
        PhysicalType::Double,
        PhysicalType::ByteArray,
        PhysicalType::ByteArray,
        PhysicalType::Int96,
    ];

    let result = columns
        .iter()
        .map(|column| {
            assert_eq!(
                column.descriptor.primitive_type.field_info.repetition,
                Repetition::Optional
            );
            column.descriptor.primitive_type.physical_type
        })
        .collect::<Vec<_>>();

    assert_eq!(expected, result);
    Ok(())
}
