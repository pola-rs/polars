mod binary;
mod primitive;
mod sidecar;

use std::io::{Cursor, Read, Seek};

use polars::io::parquet::read::ParquetReader;
use polars::io::parquet::write::ParquetWriter;
use polars::io::SerReader;
use polars_core::df;
use polars_core::prelude::*;
use polars_parquet::parquet::compression::{BrotliLevel, CompressionOptions};
use polars_parquet::parquet::error::ParquetResult;
use polars_parquet::parquet::metadata::{Descriptor, SchemaDescriptor};
use polars_parquet::parquet::page::Page;
use polars_parquet::parquet::schema::types::{ParquetType, PhysicalType};
use polars_parquet::parquet::statistics::Statistics;
use polars_parquet::parquet::write::{
    Compressor, DynIter, DynStreamingIterator, FileWriter, Version, WriteOptions,
};
use polars_parquet::read::read_metadata;
use polars_utils::mmap::MemReader;
use primitive::array_to_page_v1;

use super::{alltypes_plain, alltypes_statistics, Array};

pub fn array_to_page(
    array: &Array,
    options: &WriteOptions,
    descriptor: &Descriptor,
) -> ParquetResult<Page> {
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

fn read_column<R: Read + Seek>(reader: &mut R) -> ParquetResult<(Array, Option<Statistics>)> {
    let memreader = MemReader::from_reader(reader)?;
    let (a, statistics) = super::read::read_column(memreader, 0, "col")?;
    Ok((a, statistics))
}

fn test_column(column: &str, compression: CompressionOptions) -> ParquetResult<()> {
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
    assert_eq!(statistics.as_ref(), Some(stats).as_ref(),);
    Ok(())
}

#[test]
fn int32() -> ParquetResult<()> {
    test_column("id", CompressionOptions::Uncompressed)
}

#[test]
fn int32_snappy() -> ParquetResult<()> {
    test_column("id", CompressionOptions::Snappy)
}

#[test]
fn int32_lz4() -> ParquetResult<()> {
    test_column("id", CompressionOptions::Lz4Raw)
}

#[test]
fn int32_lz4_short_i32_array() -> ParquetResult<()> {
    test_column("id-short-array", CompressionOptions::Lz4Raw)
}

#[test]
fn int32_brotli() -> ParquetResult<()> {
    test_column(
        "id",
        CompressionOptions::Brotli(Some(BrotliLevel::default())),
    )
}

#[test]
#[ignore = "Native boolean writer not yet implemented"]
fn bool() -> ParquetResult<()> {
    test_column("bool_col", CompressionOptions::Uncompressed)
}

#[test]
fn tinyint() -> ParquetResult<()> {
    test_column("tinyint_col", CompressionOptions::Uncompressed)
}

#[test]
fn smallint_col() -> ParquetResult<()> {
    test_column("smallint_col", CompressionOptions::Uncompressed)
}

#[test]
fn int_col() -> ParquetResult<()> {
    test_column("int_col", CompressionOptions::Uncompressed)
}

#[test]
fn bigint_col() -> ParquetResult<()> {
    test_column("bigint_col", CompressionOptions::Uncompressed)
}

#[test]
fn float_col() -> ParquetResult<()> {
    test_column("float_col", CompressionOptions::Uncompressed)
}

#[test]
fn double_col() -> ParquetResult<()> {
    test_column("double_col", CompressionOptions::Uncompressed)
}

#[test]
fn basic() -> ParquetResult<()> {
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

#[test]
fn test_parquet() {
    // In CI: This test will be skipped because the file does not exist.
    if let Ok(r) = polars_utils::open_file("data/simple.parquet".as_ref()) {
        let reader = ParquetReader::new(r);
        let df = reader.finish().unwrap();
        assert_eq!(df.get_column_names(), ["a", "b"]);
        assert_eq!(df.shape(), (3, 2));
    }
}

#[test]
#[cfg(feature = "dtype-datetime")]
fn test_parquet_datetime_round_trip() -> PolarsResult<()> {
    use std::io::{Cursor, Seek, SeekFrom};

    let mut f = Cursor::new(vec![]);

    let mut df = df![
        "datetime" => [Some(191845729i64), Some(89107598), None, Some(3158971092)]
    ]?;

    df.try_apply("datetime", |s| {
        s.cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
    })?;

    ParquetWriter::new(&mut f).finish(&mut df)?;

    f.seek(SeekFrom::Start(0))?;

    let read = ParquetReader::new(f).finish()?;
    assert!(read.equals_missing(&df));
    Ok(())
}

#[test]
fn test_read_parquet_with_projection() {
    let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
    let mut df = df!("a" => [1, 2, 3], "b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();

    ParquetWriter::new(&mut buf)
        .finish(&mut df)
        .expect("parquet writer");
    buf.set_position(0);

    let expected = df!("b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();
    let df_read = ParquetReader::new(buf)
        .with_projection(Some(vec![1, 2]))
        .finish()
        .unwrap();
    assert_eq!(df_read.shape(), (3, 2));
    df_read.equals(&expected);
}

#[test]
fn test_read_parquet_with_columns() {
    let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
    let mut df = df!("a" => [1, 2, 3], "b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();

    ParquetWriter::new(&mut buf)
        .finish(&mut df)
        .expect("parquet writer");
    buf.set_position(0);

    let expected = df!("b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();
    let df_read = ParquetReader::new(buf)
        .with_columns(Some(vec!["c".to_string(), "b".to_string()]))
        .finish()
        .unwrap();
    assert_eq!(df_read.shape(), (3, 2));
    df_read.equals(&expected);
}
