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

use dictionary::{deserialize as deserialize_dict, DecodedDictPage};
#[cfg(feature = "async")]
use futures::StreamExt;
use polars_parquet::parquet::error::{Error, Result};
use polars_parquet::parquet::metadata::ColumnChunkMetaData;
use polars_parquet::parquet::page::{CompressedPage, DataPage, Page};
use polars_parquet::parquet::schema::types::{GroupConvertedType, ParquetType, PhysicalType};
use polars_parquet::parquet::statistics::Statistics;
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
        .iter()
        .map(|column_meta| column_meta.statistics().transpose())
        .collect::<Result<Vec<_>>>()?;

    let array = columns_to_array(columns, field)?;

    Ok((array, statistics.pop().unwrap()))
}

#[cfg(feature = "async")]
#[allow(dead_code)]
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
