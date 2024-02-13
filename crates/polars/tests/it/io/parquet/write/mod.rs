mod binary;
mod indexes;
mod primitive;
mod sidecar;

use std::io::{Cursor, Read, Seek};
use std::sync::Arc;

use polars_parquet::parquet::compression::{BrotliLevel, CompressionOptions};
use polars_parquet::parquet::error::Result;
use polars_parquet::parquet::metadata::{Descriptor, SchemaDescriptor};
use polars_parquet::parquet::page::Page;
use polars_parquet::parquet::schema::types::{ParquetType, PhysicalType};
use polars_parquet::parquet::statistics::Statistics;
#[cfg(feature = "async")]
use polars_parquet::parquet::write::FileStreamer;
use polars_parquet::parquet::write::{
    Compressor, DynIter, DynStreamingIterator, FileWriter, Version, WriteOptions,
};
use polars_parquet::read::read_metadata;
use primitive::array_to_page_v1;

use super::{alltypes_plain, alltypes_statistics, Array};

pub fn array_to_page(
    array: &Array,
    options: &WriteOptions,
    descriptor: &Descriptor,
) -> Result<Page> {
    // using plain encoding format
    match array {
        Array::Int32(array) => primitive::array_to_page_v1(array, options, descriptor),
        Array::Int64(array) => primitive::array_to_page_v1(array, options, descriptor),
        Array::Int96(array) => primitive::array_to_page_v1(array, options, descriptor),
        Array::Float(array) => primitive::array_to_page_v1(array, options, descriptor),
        Array::Double(array) => primitive::array_to_page_v1(array, options, descriptor),
        Array::Binary(array) => binary::array_to_page_v1(array, options, descriptor),
        _ => todo!(),
    }
}

fn read_column<R: Read + Seek>(reader: &mut R) -> Result<(Array, Option<Arc<dyn Statistics>>)> {
    let (a, statistics) = super::read::read_column(reader, 0, "col")?;
    Ok((a, statistics))
}

#[cfg(feature = "async")]
#[allow(dead_code)]
async fn read_column_async<
    R: futures::AsyncRead + futures::AsyncSeek + Send + std::marker::Unpin,
>(
    reader: &mut R,
) -> Result<(Array, Option<Arc<dyn Statistics>>)> {
    let (a, statistics) = super::read::read_column_async(reader, 0, "col").await?;
    Ok((a, statistics))
}

fn test_column(column: &str, compression: CompressionOptions) -> Result<()> {
    let array = alltypes_plain(column);

    let options = WriteOptions {
        write_statistics: true,
        version: Version::V1,
    };

    // prepare schema
    let type_ = match array {
        Array::Int32(_) => PhysicalType::Int32,
        Array::Int64(_) => PhysicalType::Int64,
        Array::Int96(_) => PhysicalType::Int96,
        Array::Float(_) => PhysicalType::Float,
        Array::Double(_) => PhysicalType::Double,
        Array::Binary(_) => PhysicalType::ByteArray,
        _ => todo!(),
    };

    let schema = SchemaDescriptor::new(
        "schema".to_string(),
        vec![ParquetType::from_physical("col".to_string(), type_)],
    );

    let a = schema.columns();

    let pages = DynStreamingIterator::new(Compressor::new_from_vec(
        DynIter::new(std::iter::once(array_to_page(
            &array,
            &options,
            &a[0].descriptor,
        ))),
        compression,
        vec![],
    ));
    let columns = std::iter::once(Ok(pages));

    let writer = Cursor::new(vec![]);
    let mut writer = FileWriter::new(writer, schema, options, None);

    writer.write(DynIter::new(columns))?;
    writer.end(None)?;

    let data = writer.into_inner().into_inner();

    let (result, statistics) = read_column(&mut Cursor::new(data))?;
    assert_eq!(array, result);
    let stats = alltypes_statistics(column);
    assert_eq!(
        statistics.as_ref().map(|x| x.as_ref()),
        Some(stats).as_ref().map(|x| x.as_ref())
    );
    Ok(())
}

#[test]
fn int32() -> Result<()> {
    test_column("id", CompressionOptions::Uncompressed)
}

#[test]
fn int32_snappy() -> Result<()> {
    test_column("id", CompressionOptions::Snappy)
}

#[test]
fn int32_lz4() -> Result<()> {
    test_column("id", CompressionOptions::Lz4Raw)
}

#[test]
fn int32_lz4_short_i32_array() -> Result<()> {
    test_column("id-short-array", CompressionOptions::Lz4Raw)
}

#[test]
fn int32_brotli() -> Result<()> {
    test_column(
        "id",
        CompressionOptions::Brotli(Some(BrotliLevel::default())),
    )
}

#[test]
#[ignore = "Native boolean writer not yet implemented"]
fn bool() -> Result<()> {
    test_column("bool_col", CompressionOptions::Uncompressed)
}

#[test]
fn tinyint() -> Result<()> {
    test_column("tinyint_col", CompressionOptions::Uncompressed)
}

#[test]
fn smallint_col() -> Result<()> {
    test_column("smallint_col", CompressionOptions::Uncompressed)
}

#[test]
fn int_col() -> Result<()> {
    test_column("int_col", CompressionOptions::Uncompressed)
}

#[test]
fn bigint_col() -> Result<()> {
    test_column("bigint_col", CompressionOptions::Uncompressed)
}

#[test]
fn float_col() -> Result<()> {
    test_column("float_col", CompressionOptions::Uncompressed)
}

#[test]
fn double_col() -> Result<()> {
    test_column("double_col", CompressionOptions::Uncompressed)
}

#[test]
fn basic() -> Result<()> {
    let array = vec![
        Some(0),
        Some(1),
        Some(2),
        Some(3),
        Some(4),
        Some(5),
        Some(6),
    ];

    let options = WriteOptions {
        write_statistics: false,
        version: Version::V1,
    };

    let schema = SchemaDescriptor::new(
        "schema".to_string(),
        vec![ParquetType::from_physical(
            "col".to_string(),
            PhysicalType::Int32,
        )],
    );

    let pages = DynStreamingIterator::new(Compressor::new_from_vec(
        DynIter::new(std::iter::once(array_to_page_v1(
            &array,
            &options,
            &schema.columns()[0].descriptor,
        ))),
        CompressionOptions::Uncompressed,
        vec![],
    ));
    let columns = std::iter::once(Ok(pages));

    let writer = Cursor::new(vec![]);
    let mut writer = FileWriter::new(writer, schema, options, None);

    writer.write(DynIter::new(columns))?;
    writer.end(None)?;

    let data = writer.into_inner().into_inner();
    let mut reader = Cursor::new(data);

    let metadata = read_metadata(&mut reader)?;

    // validated against an equivalent array produced by pyarrow.
    let expected = 51;
    assert_eq!(
        metadata.row_groups[0].columns()[0].uncompressed_size(),
        expected
    );

    Ok(())
}

#[cfg(feature = "async")]
#[allow(dead_code)]
async fn test_column_async(column: &str, compression: CompressionOptions) -> Result<()> {
    let array = alltypes_plain(column);

    let options = WriteOptions {
        write_statistics: true,
        version: Version::V1,
    };

    // prepare schema
    let type_ = match array {
        Array::Int32(_) => PhysicalType::Int32,
        Array::Int64(_) => PhysicalType::Int64,
        Array::Int96(_) => PhysicalType::Int96,
        Array::Float(_) => PhysicalType::Float,
        Array::Double(_) => PhysicalType::Double,
        Array::Binary(_) => PhysicalType::ByteArray,
        _ => todo!(),
    };

    let schema = SchemaDescriptor::new(
        "schema".to_string(),
        vec![ParquetType::from_physical("col".to_string(), type_)],
    );

    let a = schema.columns();

    let pages = DynStreamingIterator::new(Compressor::new_from_vec(
        DynIter::new(std::iter::once(array_to_page(
            &array,
            &options,
            &a[0].descriptor,
        ))),
        compression,
        vec![],
    ));
    let columns = std::iter::once(Ok(pages));

    let writer = futures::io::Cursor::new(vec![]);
    let mut writer = FileStreamer::new(writer, schema, options, None);

    writer.write(DynIter::new(columns)).await?;
    writer.end(None).await?;

    let data = writer.into_inner().into_inner();

    let (result, statistics) = read_column_async(&mut futures::io::Cursor::new(data)).await?;
    assert_eq!(array, result);
    let stats = alltypes_statistics(column);
    assert_eq!(
        statistics.as_ref().map(|x| x.as_ref()),
        Some(stats).as_ref().map(|x| x.as_ref())
    );
    Ok(())
}
