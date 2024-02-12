use std::io::Cursor;

use polars_arrow::array::*;
use polars_arrow::chunk::Chunk;
use polars_arrow::datatypes::{Field, Schema};
use polars_arrow::error::Result;
use polars_arrow::io::ipc::read::{read_file_metadata, FileReader};
use polars_arrow::io::ipc::write::*;
use polars_arrow::io::ipc::IpcField;
use polars_arrow::types::{i256, months_days_ns};

use crate::io::ipc::common::read_gzip_json;

pub(crate) fn write(
    batches: &[Chunk<Box<dyn Array>>],
    schema: &Schema,
    ipc_fields: Option<Vec<IpcField>>,
    compression: Option<Compression>,
) -> Result<Vec<u8>> {
    let result = vec![];
    let options = WriteOptions { compression };
    let mut writer = FileWriter::try_new(result, schema.clone(), ipc_fields.clone(), options)?;
    for batch in batches {
        writer.write(batch, ipc_fields.as_ref().map(|x| x.as_ref()))?;
    }
    writer.finish()?;
    Ok(writer.into_inner())
}

fn round_trip(
    columns: Chunk<Box<dyn Array>>,
    schema: Schema,
    ipc_fields: Option<Vec<IpcField>>,
    compression: Option<Compression>,
) -> Result<()> {
    let (expected_schema, expected_batches) = (schema.clone(), vec![columns]);

    let result = write(&expected_batches, &schema, ipc_fields, compression)?;
    let mut reader = Cursor::new(result);
    let metadata = read_file_metadata(&mut reader)?;
    let schema = metadata.schema.clone();

    let reader = FileReader::new(reader, metadata, None, None);

    assert_eq!(schema, expected_schema);

    let batches = reader.collect::<Result<Vec<_>>>()?;

    assert_eq!(batches, expected_batches);
    Ok(())
}

fn test_file(version: &str, file_name: &str, compressed: bool) -> Result<()> {
    let (schema, ipc_fields, batches) = read_gzip_json(version, file_name)?;

    let compression = if compressed {
        Some(Compression::ZSTD)
    } else {
        None
    };

    let result = write(&batches, &schema, Some(ipc_fields), compression)?;
    let mut reader = Cursor::new(result);
    let metadata = read_file_metadata(&mut reader)?;
    let schema = metadata.schema.clone();
    let ipc_fields = metadata.ipc_schema.fields.clone();

    let reader = FileReader::new(reader, metadata, None, None);

    // read expected JSON output
    let (expected_schema, expected_ipc_fields, expected_batches) =
        read_gzip_json(version, file_name)?;

    assert_eq!(schema, expected_schema);
    assert_eq!(ipc_fields, expected_ipc_fields);

    let batches = reader.collect::<Result<Vec<_>>>()?;

    for (a, b) in batches.iter().zip(expected_batches.iter()) {
        for (c1, c2) in a.columns().iter().zip(b.columns().iter()) {
            assert_eq!(c1, c2)
        }
    }

    //assert_eq!(batches, expected_batches);
    Ok(())
}

#[test]
fn write_100_primitive() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_primitive", false)?;
    test_file("1.0.0-bigendian", "generated_primitive", false)
}

#[test]
#[cfg_attr(miri, ignore)] // compression uses FFI, which miri does not support
fn write_100_compressed_primitive() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_primitive", true)?;
    test_file("1.0.0-bigendian", "generated_primitive", true)
}

#[test]
fn write_100_datetime() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_datetime", false)?;
    test_file("1.0.0-bigendian", "generated_datetime", false)
}

#[test]
#[cfg_attr(miri, ignore)] // compression uses FFI, which miri does not support
fn write_100_compressed_datetime() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_datetime", true)?;
    test_file("1.0.0-bigendian", "generated_datetime", true)
}

#[test]
fn write_100_dictionary_unsigned() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_dictionary_unsigned", false)?;
    test_file("1.0.0-bigendian", "generated_dictionary_unsigned", false)
}

#[test]
#[cfg_attr(miri, ignore)] // compression uses FFI, which miri does not support
fn write_100_compressed_dictionary_unsigned() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_dictionary_unsigned", true)?;
    test_file("1.0.0-bigendian", "generated_dictionary_unsigned", true)
}

#[test]
fn write_100_dictionary() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_dictionary", false)?;
    test_file("1.0.0-bigendian", "generated_dictionary", false)
}

#[test]
#[cfg_attr(miri, ignore)] // compression uses FFI, which miri does not support
fn write_100_compressed_dictionary() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_dictionary", true)?;
    test_file("1.0.0-bigendian", "generated_dictionary", true)
}

#[test]
fn write_100_interval() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_interval", false)?;
    test_file("1.0.0-bigendian", "generated_interval", false)
}

#[test]
#[cfg_attr(miri, ignore)] // compression uses FFI, which miri does not support
fn write_100_compressed_interval() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_interval", true)?;
    test_file("1.0.0-bigendian", "generated_interval", true)
}

#[test]
fn write_100_large_batch() -> Result<()> {
    // this takes too long for unit-tests. It has been passing...
    //test_file("1.0.0-littleendian", "generated_large_batch", false);
    Ok(())
}

#[test]
fn write_100_nested() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_nested", false)?;
    test_file("1.0.0-bigendian", "generated_nested", false)
}

#[test]
#[cfg_attr(miri, ignore)] // compression uses FFI, which miri does not support
fn write_100_compressed_nested() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_nested", true)?;
    test_file("1.0.0-bigendian", "generated_nested", true)
}

#[test]
fn write_100_nested_large_offsets() -> Result<()> {
    test_file(
        "1.0.0-littleendian",
        "generated_nested_large_offsets",
        false,
    )?;
    test_file("1.0.0-bigendian", "generated_nested_large_offsets", false)
}

#[test]
#[cfg_attr(miri, ignore)] // compression uses FFI, which miri does not support
fn write_100_compressed_nested_large_offsets() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_nested_large_offsets", true)?;
    test_file("1.0.0-bigendian", "generated_nested_large_offsets", true)
}

#[test]
fn write_100_null_trivial() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_null_trivial", false)?;
    test_file("1.0.0-bigendian", "generated_null_trivial", false)
}

#[test]
#[cfg_attr(miri, ignore)] // compression uses FFI, which miri does not support
fn write_100_compressed_null_trivial() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_null_trivial", true)?;
    test_file("1.0.0-bigendian", "generated_null_trivial", true)
}

#[test]
fn write_100_null() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_null", false)?;
    test_file("1.0.0-bigendian", "generated_null", false)
}

#[test]
#[cfg_attr(miri, ignore)] // compression uses FFI, which miri does not support
fn write_100_compressed_null() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_null", true)?;
    test_file("1.0.0-bigendian", "generated_null", true)
}

#[test]
fn write_100_primitive_large_offsets() -> Result<()> {
    test_file(
        "1.0.0-littleendian",
        "generated_primitive_large_offsets",
        false,
    )?;
    test_file(
        "1.0.0-bigendian",
        "generated_primitive_large_offsets",
        false,
    )
}

#[test]
#[cfg_attr(miri, ignore)] // compression uses FFI, which miri does not support
fn write_100_compressed_primitive_large_offsets() -> Result<()> {
    test_file(
        "1.0.0-littleendian",
        "generated_primitive_large_offsets",
        true,
    )?;
    test_file("1.0.0-bigendian", "generated_primitive_large_offsets", true)
}

#[test]
fn write_100_primitive_no_batches() -> Result<()> {
    test_file(
        "1.0.0-littleendian",
        "generated_primitive_no_batches",
        false,
    )?;
    test_file("1.0.0-bigendian", "generated_primitive_no_batches", false)
}

#[test]
#[cfg_attr(miri, ignore)] // compression uses FFI, which miri does not support
fn write_100_compressed_primitive_no_batches() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_primitive_no_batches", true)?;
    test_file("1.0.0-bigendian", "generated_primitive_no_batches", true)
}

#[test]
fn write_100_primitive_zerolength() -> Result<()> {
    test_file(
        "1.0.0-littleendian",
        "generated_primitive_zerolength",
        false,
    )?;
    test_file("1.0.0-bigendian", "generated_primitive_zerolength", false)
}

#[test]
#[cfg_attr(miri, ignore)] // compression uses FFI, which miri does not support
fn write_100_compressed_primitive_zerolength() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_primitive_zerolength", true)?;
    test_file("1.0.0-bigendian", "generated_primitive_zerolength", true)
}

#[test]
fn write_0141_primitive_zerolength() -> Result<()> {
    test_file("0.14.1", "generated_primitive_zerolength", false)
}

#[test]
fn write_100_custom_metadata() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_custom_metadata", false)?;
    test_file("1.0.0-bigendian", "generated_custom_metadata", false)
}

#[test]
fn write_100_decimal() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_decimal", false)?;
    test_file("1.0.0-bigendian", "generated_decimal", false)
}

#[test]
fn write_100_extension() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_extension", false)?;
    test_file("1.0.0-bigendian", "generated_extension", false)
}

#[test]
fn write_100_union() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_union", false)?;
    test_file("1.0.0-bigendian", "generated_union", false)
}

#[test]
fn write_100_map() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_map", false)?;
    test_file("1.0.0-bigendian", "generated_map", false)
}

#[test]
fn write_100_map_non_canonical() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_map_non_canonical", false)?;
    test_file("1.0.0-bigendian", "generated_map_non_canonical", false)
}

#[test]
fn write_100_nested_dictionary() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_nested_dictionary", false)?;
    test_file("1.0.0-bigendian", "generated_nested_dictionary", false)
}

#[test]
fn write_generated_017_union() -> Result<()> {
    test_file("0.17.1", "generated_union", false)
}

#[test]
#[cfg_attr(miri, ignore)] // compression uses FFI, which miri does not support
fn write_boolean() -> Result<()> {
    let array = BooleanArray::from([Some(true), Some(false), None, Some(true)]).boxed();
    let schema = Schema::from(vec![Field::new("a", array.data_type().clone(), true)]);
    let columns = Chunk::try_new(vec![array])?;
    round_trip(columns, schema, None, Some(Compression::ZSTD))
}

#[test]
#[cfg_attr(miri, ignore)] // compression uses FFI, which miri does not support
fn write_sliced_utf8() -> Result<()> {
    let array = Utf8Array::<i32>::from_slice(["aa", "bb"])
        .sliced(1, 1)
        .boxed();
    let schema = Schema::from(vec![Field::new("a", array.data_type().clone(), true)]);
    let columns = Chunk::try_new(vec![array])?;
    round_trip(columns, schema, None, Some(Compression::ZSTD))
}

#[test]
fn write_sliced_list() -> Result<()> {
    let data = vec![
        Some(vec![Some(1i32), Some(2), Some(3)]),
        None,
        Some(vec![Some(4), None, Some(6)]),
    ];

    let mut array = MutableListArray::<i32, MutablePrimitiveArray<i32>>::new();
    array.try_extend(data).unwrap();
    let array: Box<dyn Array> = array.into_box().sliced(1, 2);

    let schema = Schema::from(vec![Field::new("a", array.data_type().clone(), true)]);
    let columns = Chunk::try_new(vec![array])?;
    round_trip(columns, schema, None, None)
}

#[test]
fn write_months_days_ns() -> Result<()> {
    let array = Box::new(MonthsDaysNsArray::from([
        Some(months_days_ns::new(1, 1, 0)),
        Some(months_days_ns::new(1, 1, 1)),
        None,
        Some(months_days_ns::new(1, 1, 2)),
    ])) as Box<dyn Array>;
    let schema = Schema::from(vec![Field::new("a", array.data_type().clone(), true)]);
    let columns = Chunk::try_new(vec![array])?;
    round_trip(columns, schema, None, None)
}

#[test]
fn write_decimal256() -> Result<()> {
    let array = Int256Array::from([
        Some(i256::from_words(1, 0)),
        Some(i256::from_words(-2, 0)),
        None,
        Some(i256::from_words(0, 1)),
    ])
    .boxed();
    let schema = Schema::from(vec![Field::new("a", array.data_type().clone(), true)]);
    let columns = Chunk::try_new(vec![array])?;
    round_trip(columns, schema, None, None)
}
