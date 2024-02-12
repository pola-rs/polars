use std::fs::File;

use polars_arrow::chunk::Chunk;
use polars_arrow::error::Result;
use polars_arrow::io::ipc::read::*;

use crate::io::ipc::common::read_gzip_json;

fn test_file(version: &str, file_name: &str) -> Result<()> {
    let testdata = crate::test_util::arrow_test_data();
    let mut file = File::open(format!(
        "{testdata}/arrow-ipc-stream/integration/{version}/{file_name}.stream"
    ))?;

    let metadata = read_stream_metadata(&mut file)?;
    let reader = StreamReader::new(file, metadata, None);

    // read expected JSON output
    let (schema, ipc_fields, batches) = read_gzip_json(version, file_name)?;

    assert_eq!(&schema, &reader.metadata().schema);
    assert_eq!(&ipc_fields, &reader.metadata().ipc_schema.fields);

    batches
        .iter()
        .zip(reader.map(|x| x.unwrap().unwrap()))
        .for_each(|(lhs, rhs)| {
            assert_eq!(lhs, &rhs);
        });
    Ok(())
}

#[test]
fn read_generated_100_primitive() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_primitive")
}

#[test]
fn read_generated_100_datetime() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_datetime")
}

#[test]
fn read_generated_100_null_trivial() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_null_trivial")
}

#[test]
fn read_generated_100_null() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_null")
}

#[test]
fn read_generated_100_primitive_zerolength() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_primitive_zerolength")
}

#[test]
fn read_generated_100_primitive_primitive_no_batches() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_primitive_no_batches")
}

#[test]
fn read_generated_100_dictionary() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_dictionary")
}

#[test]
fn read_generated_100_nested() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_nested")
}

#[test]
fn read_generated_100_interval() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_interval")
}

#[test]
fn read_generated_100_decimal() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_decimal")
}

#[test]
fn read_generated_100_union() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_union")?;
    test_file("1.0.0-bigendian", "generated_union")
}

#[test]
fn read_generated_017_union() -> Result<()> {
    test_file("0.17.1", "generated_union")
}

#[test]
fn read_generated_200_compression_lz4() -> Result<()> {
    test_file("2.0.0-compression", "generated_lz4")
}

#[test]
fn read_generated_200_compression_zstd() -> Result<()> {
    test_file("2.0.0-compression", "generated_zstd")
}

fn test_projection(version: &str, file_name: &str, columns: Vec<usize>) -> Result<()> {
    let testdata = crate::test_util::arrow_test_data();
    let mut file = File::open(format!(
        "{testdata}/arrow-ipc-stream/integration/{version}/{file_name}.stream"
    ))?;

    let metadata = read_stream_metadata(&mut file)?;

    let (_, _, chunks) = read_gzip_json(version, file_name)?;

    let expected_fields = columns
        .iter()
        .copied()
        .map(|x| metadata.schema.fields[x].clone())
        .collect::<Vec<_>>();

    let expected_chunks = chunks
        .into_iter()
        .map(|chunk| {
            let columns = columns
                .iter()
                .copied()
                .map(|x| chunk.arrays()[x].clone())
                .collect::<Vec<_>>();
            Chunk::new(columns)
        })
        .collect::<Vec<_>>();

    let reader = StreamReader::new(&mut file, metadata, Some(columns.clone()));

    assert_eq!(reader.schema().fields, expected_fields);

    expected_chunks
        .iter()
        .zip(reader.map(|x| x.unwrap().unwrap()))
        .for_each(|(lhs, rhs)| {
            assert_eq!(lhs, &rhs);
        });
    Ok(())
}

#[test]
fn read_projected() -> Result<()> {
    test_projection("1.0.0-littleendian", "generated_primitive", vec![1])?;
    test_projection("1.0.0-littleendian", "generated_dictionary", vec![2])?;
    test_projection("1.0.0-littleendian", "generated_nested", vec![0])?;
    test_projection("1.0.0-littleendian", "generated_primitive", vec![2, 1])?;
    test_projection("1.0.0-littleendian", "generated_primitive", vec![0, 2, 1])
}
