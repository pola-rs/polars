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

use std::fs::File;

use dictionary::{deserialize as deserialize_dict, DecodedDictPage};
#[cfg(feature = "async")]
use futures::StreamExt;
use polars_parquet::parquet::error::{Error, Result};
use polars_parquet::parquet::metadata::ColumnChunkMetaData;
use polars_parquet::parquet::page::{CompressedPage, DataPage, Page};
#[cfg(feature = "async")]
use polars_parquet::parquet::read::get_page_stream;
#[cfg(feature = "async")]
use polars_parquet::parquet::read::read_metadata_async;
use polars_parquet::parquet::read::{
    get_column_iterator, get_field_columns, read_metadata, BasicDecompressor, MutStreamingIterator,
    State,
};
use polars_parquet::parquet::schema::types::{GroupConvertedType, ParquetType, PhysicalType};
use polars_parquet::parquet::schema::Repetition;
use polars_parquet::parquet::statistics::Statistics;
use polars_parquet::parquet::types::int96_to_i64_ns;
use polars_parquet::parquet::FallibleStreamingIterator;

use super::*;

pub fn get_path() -> PathBuf {
    let dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(dir).join("../../docs/data")
}

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
                    Ok(primitive_nested::page_dict_to_array(page, dict)?)
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
        .find(|field| field.name() == field_name)
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
        .find(|field| field.name() == field_name)
        .ok_or_else(|| Error::OutOfSpec("column does not exist".to_string()))?;

    let column = get_field_columns(metadata.row_groups[row_group].columns(), field.name())
        .next()
        .unwrap();

    let pages = get_page_stream(column, reader, vec![], Arc::new(|_, _| true), usize::MAX).await?;

    let mut statistics = get_field_columns(metadata.row_groups[row_group].columns(), field.name())
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
            optional int32// pub enum Value {
    //     UInt32(Option<u32>),
    //     Int32(Option<i32>),
    //     Int64(Option<i64>),
    //     Int96(Option<[u32; 3]>),
    //     Float32(Option<f32>),
    //     Float64(Option<f64>),
    //     Boolean(Option<bool>),
    //     Binary(Option<Vec<u8>>),
    //     FixedLenBinary(Option<Vec<u8>>),
    //     List(Option<Array>),
    // }
     field_id=5 int_col;
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
