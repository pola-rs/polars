mod binary;
/// Serialization to Rust's Native types.
/// In comparison to Arrow, this in-memory format does not leverage logical types nor SIMD operations,
/// but OTOH it has no external dependencies and is very familiar to Rust developers.
mod boolean;
mod dictionary;
pub(crate) mod file;
mod fixed_binary;
mod primitive;
mod primitive_nested;
pub(crate) mod row_group;
mod struct_;
mod utils;

use std::fs::File;

use dictionary::DecodedDictPage;
use polars_parquet::parquet::encoding::hybrid_rle::HybridRleDecoder;
use polars_parquet::parquet::error::{ParquetError, ParquetResult};
use polars_parquet::parquet::metadata::ColumnChunkMetaData;
use polars_parquet::parquet::page::DataPage;
use polars_parquet::parquet::read::{
    get_column_iterator, get_field_columns, read_metadata, BasicDecompressor,
};
use polars_parquet::parquet::schema::types::{GroupConvertedType, ParquetType};
use polars_parquet::parquet::schema::Repetition;
use polars_parquet::parquet::types::int96_to_i64_ns;
use polars_parquet::read::PageReader;
use polars_utils::mmap::MemReader;

use super::*;

pub fn hybrid_rle_iter(d: HybridRleDecoder) -> ParquetResult<std::vec::IntoIter<u32>> {
    Ok(d.collect()?.into_iter())
}

pub fn get_path() -> PathBuf {
    let dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(dir).join("../../docs/data")
}

/// Reads a page into an [`Array`].
/// This is CPU-intensive: decompress, decode and de-serialize.
pub fn page_to_array(page: &DataPage, dict: Option<&DecodedDictPage>) -> ParquetResult<Array> {
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

/// Reads columns into an [`Array`].
/// This is CPU-intensive: decompress, decode and de-serialize.
pub fn columns_to_array<I>(mut columns: I, field: &ParquetType) -> ParquetResult<Array>
where
    I: Iterator<Item = ParquetResult<(PageReader, ColumnChunkMetaData)>>,
{
    let mut validity = vec![];
    let mut has_filled = false;
    let mut arrays = vec![];
    while let Some((pages, column)) = columns.next().transpose()? {
        let mut iterator = BasicDecompressor::new(pages, vec![]);

        let dict = iterator
            .read_dict_page()?
            .map(|dict| dictionary::deserialize(&dict, column.physical_type()))
            .transpose()?;
        while let Some(page) = iterator.next().transpose()? {
            let page = page.decompress(&mut iterator)?;
            if !has_filled {
                struct_::extend_validity(&mut validity, &page)?;
            }
            arrays.push(page_to_array(&page, dict.as_ref())?)
        }
        has_filled = true;
    }

    match field {
        ParquetType::PrimitiveType { .. } => arrays
            .pop()
            .ok_or_else(|| ParquetError::OutOfSpec("".to_string())),
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

pub fn read_column(
    mut reader: MemReader,
    row_group: usize,
    field_name: &str,
) -> ParquetResult<(Array, Option<Statistics>)> {
    let metadata = read_metadata(&mut reader)?;

    let field = metadata
        .schema()
        .fields()
        .iter()
        .find(|field| field.name() == field_name)
        .ok_or_else(|| ParquetError::OutOfSpec("column does not exist".to_string()))?;

    let columns = get_column_iterator(
        reader,
        &metadata.row_groups[row_group],
        field.name(),
        usize::MAX,
    );

    let mut statistics = get_field_columns(metadata.row_groups[row_group].columns(), field.name())
        .map(|column_meta| column_meta.statistics().transpose())
        .collect::<ParquetResult<Vec<_>>>()?;

    let array = columns_to_array(columns, field)?;

    Ok((array, statistics.pop().unwrap()))
}

fn get_column(path: &str, column: &str) -> ParquetResult<(Array, Option<Statistics>)> {
    let file = File::open(path).unwrap();
    let memreader = MemReader::from_reader(file).unwrap();
    read_column(memreader, 0, column)
}

fn test_column(column: &str) -> ParquetResult<()> {
    let mut path = get_path();
    path.push("alltypes_plain.parquet");
    let path = path.to_str().unwrap();
    let (result, statistics) = get_column(path, column)?;
    // the file does not have statistics
    assert_eq!(statistics.as_ref(), None);
    assert_eq!(result, alltypes_plain(column));
    Ok(())
}

#[test]
fn int32() -> ParquetResult<()> {
    test_column("id")
}

#[test]
fn bool() -> ParquetResult<()> {
    test_column("bool_col")
}

#[test]
fn tinyint_col() -> ParquetResult<()> {
    test_column("tinyint_col")
}

#[test]
fn smallint_col() -> ParquetResult<()> {
    test_column("smallint_col")
}

#[test]
fn int_col() -> ParquetResult<()> {
    test_column("int_col")
}

#[test]
fn bigint_col() -> ParquetResult<()> {
    test_column("bigint_col")
}

#[test]
fn float_col() -> ParquetResult<()> {
    test_column("float_col")
}

#[test]
fn double_col() -> ParquetResult<()> {
    test_column("double_col")
}

#[test]
fn timestamp_col() -> ParquetResult<()> {
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
fn test_metadata() -> ParquetResult<()> {
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
