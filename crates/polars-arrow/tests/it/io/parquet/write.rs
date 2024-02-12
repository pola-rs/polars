use std::io::Cursor;

use polars_arrow::error::Result;
use polars_arrow::io::parquet::write::*;

use super::*;

fn round_trip(
    column: &str,
    file: &str,
    version: Version,
    compression: CompressionOptions,
    encodings: Vec<Encoding>,
) -> Result<()> {
    round_trip_opt_stats(column, file, version, compression, encodings, true)
}

fn round_trip_opt_stats(
    column: &str,
    file: &str,
    version: Version,
    compression: CompressionOptions,
    encodings: Vec<Encoding>,
    check_stats: bool,
) -> Result<()> {
    let (array, statistics) = match file {
        "nested" => (
            pyarrow_nested_nullable(column),
            pyarrow_nested_nullable_statistics(column),
        ),
        "nullable" => (
            pyarrow_nullable(column),
            pyarrow_nullable_statistics(column),
        ),
        "required" => (
            pyarrow_required(column),
            pyarrow_required_statistics(column),
        ),
        "struct" => (pyarrow_struct(column), pyarrow_struct_statistics(column)),
        "map" => (pyarrow_map(column), pyarrow_map_statistics(column)),
        "nested_edge" => (
            pyarrow_nested_edge(column),
            pyarrow_nested_edge_statistics(column),
        ),
        _ => unreachable!(),
    };

    let field = Field::new("a1", array.data_type().clone(), true);
    let schema = ArrowSchema::from(vec![field]);

    let options = WriteOptions {
        write_statistics: true,
        compression,
        version,
        data_pagesize_limit: None,
    };

    let iter = vec![Chunk::try_new(vec![array.clone()])];

    let row_groups =
        RowGroupIterator::try_new(iter.into_iter(), &schema, options, vec![encodings])?;

    let writer = Cursor::new(vec![]);
    let mut writer = FileWriter::try_new(writer, schema, options)?;

    for group in row_groups {
        writer.write(group?)?;
    }
    writer.end(None)?;

    let data = writer.into_inner().into_inner();

    let (result, stats) = read_column(&mut Cursor::new(data), "a1")?;

    assert_eq!(array.as_ref(), result.as_ref());
    if check_stats {
        assert_eq!(statistics, stats);
    }
    Ok(())
}

#[test]
fn int64_optional_v1() -> Result<()> {
    round_trip(
        "int64",
        "nullable",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn int64_required_v1() -> Result<()> {
    round_trip(
        "int64",
        "required",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn int64_optional_v2() -> Result<()> {
    round_trip(
        "int64",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn int64_optional_delta() -> Result<()> {
    round_trip(
        "int64",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::DeltaBinaryPacked],
    )
}

#[test]
fn int64_required_delta() -> Result<()> {
    round_trip(
        "int64",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::DeltaBinaryPacked],
    )
}

#[cfg(feature = "io_parquet_compression")]
#[test]
fn int64_optional_v2_compressed() -> Result<()> {
    round_trip(
        "int64",
        "nullable",
        Version::V2,
        CompressionOptions::Snappy,
        vec![Encoding::Plain],
    )
}

#[test]
fn utf8_optional_v1() -> Result<()> {
    round_trip(
        "string",
        "nullable",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn utf8_required_v1() -> Result<()> {
    round_trip(
        "string",
        "required",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn utf8_optional_v2() -> Result<()> {
    round_trip(
        "string",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn utf8_required_v2() -> Result<()> {
    round_trip(
        "string",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[cfg(feature = "io_parquet_compression")]
#[test]
fn utf8_optional_v2_compressed() -> Result<()> {
    round_trip(
        "string",
        "nullable",
        Version::V2,
        CompressionOptions::Snappy,
        vec![Encoding::Plain],
    )
}

#[cfg(feature = "io_parquet_compression")]
#[test]
fn utf8_required_v2_compressed() -> Result<()> {
    round_trip(
        "string",
        "required",
        Version::V2,
        CompressionOptions::Snappy,
        vec![Encoding::Plain],
    )
}

#[test]
fn bool_optional_v1() -> Result<()> {
    round_trip(
        "bool",
        "nullable",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn bool_required_v1() -> Result<()> {
    round_trip(
        "bool",
        "required",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn bool_optional_v2_uncompressed() -> Result<()> {
    round_trip(
        "bool",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn bool_required_v2_uncompressed() -> Result<()> {
    round_trip(
        "bool",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[cfg(feature = "io_parquet_compression")]
#[test]
fn bool_required_v2_compressed() -> Result<()> {
    round_trip(
        "bool",
        "required",
        Version::V2,
        CompressionOptions::Snappy,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_int64_optional_v2() -> Result<()> {
    round_trip(
        "list_int64",
        "nested",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_int64_optional_v1() -> Result<()> {
    round_trip(
        "list_int64",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_int64_required_required_v1() -> Result<()> {
    round_trip(
        "list_int64_required_required",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_int64_required_required_v2() -> Result<()> {
    round_trip(
        "list_int64_required_required",
        "nested",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_bool_optional_v2() -> Result<()> {
    round_trip(
        "list_bool",
        "nested",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_bool_optional_v1() -> Result<()> {
    round_trip(
        "list_bool",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_utf8_optional_v2() -> Result<()> {
    round_trip(
        "list_utf8",
        "nested",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_utf8_optional_v1() -> Result<()> {
    round_trip(
        "list_utf8",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_large_binary_optional_v2() -> Result<()> {
    round_trip(
        "list_large_binary",
        "nested",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_large_binary_optional_v1() -> Result<()> {
    round_trip(
        "list_large_binary",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_nested_inner_required_required_i64() -> Result<()> {
    round_trip_opt_stats(
        "list_nested_inner_required_required_i64",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
        false,
    )
}

#[test]
fn list_struct_nullable() -> Result<()> {
    round_trip_opt_stats(
        "list_struct_nullable",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
        true,
    )
}

#[test]
fn list_decimal_nullable() -> Result<()> {
    round_trip_opt_stats(
        "list_decimal",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
        true,
    )
}

#[test]
fn list_decimal256_nullable() -> Result<()> {
    round_trip_opt_stats(
        "list_decimal256",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
        true,
    )
}

#[test]
fn v1_nested_struct_list_nullable() -> Result<()> {
    round_trip_opt_stats(
        "struct_list_nullable",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
        true,
    )
}

#[test]
fn v1_nested_list_struct_list_nullable() -> Result<()> {
    round_trip_opt_stats(
        "list_struct_list_nullable",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
        true,
    )
}

#[test]
fn utf8_optional_v2_delta() -> Result<()> {
    round_trip(
        "string",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::DeltaLengthByteArray],
    )
}

#[test]
fn utf8_required_v2_delta() -> Result<()> {
    round_trip(
        "string",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::DeltaLengthByteArray],
    )
}

#[test]
fn i32_optional_v2_dict() -> Result<()> {
    round_trip(
        "int32_dict",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::RleDictionary],
    )
}

#[cfg(feature = "io_parquet_compression")]
#[test]
fn i32_optional_v2_dict_compressed() -> Result<()> {
    round_trip(
        "int32_dict",
        "nullable",
        Version::V2,
        CompressionOptions::Snappy,
        vec![Encoding::RleDictionary],
    )
}

// Decimal Testing
#[test]
fn decimal_9_optional_v1() -> Result<()> {
    round_trip(
        "decimal_9",
        "nullable",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal_9_required_v1() -> Result<()> {
    round_trip(
        "decimal_9",
        "required",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal_18_optional_v1() -> Result<()> {
    round_trip(
        "decimal_18",
        "nullable",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal_18_required_v1() -> Result<()> {
    round_trip(
        "decimal_18",
        "required",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal_26_optional_v1() -> Result<()> {
    round_trip(
        "decimal_26",
        "nullable",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal_26_required_v1() -> Result<()> {
    round_trip(
        "decimal_26",
        "required",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_9_optional_v1() -> Result<()> {
    round_trip(
        "decimal256_9",
        "nullable",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_9_required_v1() -> Result<()> {
    round_trip(
        "decimal256_9",
        "required",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_18_optional_v1() -> Result<()> {
    round_trip(
        "decimal256_18",
        "nullable",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_18_required_v1() -> Result<()> {
    round_trip(
        "decimal256_18",
        "required",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_26_optional_v1() -> Result<()> {
    round_trip(
        "decimal256_26",
        "nullable",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_26_required_v1() -> Result<()> {
    round_trip(
        "decimal256_26",
        "required",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_39_optional_v1() -> Result<()> {
    round_trip(
        "decimal256_39",
        "nullable",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_39_required_v1() -> Result<()> {
    round_trip(
        "decimal256_39",
        "required",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_76_optional_v1() -> Result<()> {
    round_trip(
        "decimal256_76",
        "nullable",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_76_required_v1() -> Result<()> {
    round_trip(
        "decimal256_76",
        "required",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal_9_optional_v2() -> Result<()> {
    round_trip(
        "decimal_9",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal_9_required_v2() -> Result<()> {
    round_trip(
        "decimal_9",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal_18_optional_v2() -> Result<()> {
    round_trip(
        "decimal_18",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal_18_required_v2() -> Result<()> {
    round_trip(
        "decimal_18",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal_26_optional_v2() -> Result<()> {
    round_trip(
        "decimal_26",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal_26_required_v2() -> Result<()> {
    round_trip(
        "decimal_26",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_9_optional_v2() -> Result<()> {
    round_trip(
        "decimal256_9",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_9_required_v2() -> Result<()> {
    round_trip(
        "decimal256_9",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_18_optional_v2() -> Result<()> {
    round_trip(
        "decimal256_18",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_18_required_v2() -> Result<()> {
    round_trip(
        "decimal256_18",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_26_optional_v2() -> Result<()> {
    round_trip(
        "decimal256_26",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_26_required_v2() -> Result<()> {
    round_trip(
        "decimal256_26",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_39_optional_v2() -> Result<()> {
    round_trip(
        "decimal256_39",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_39_required_v2() -> Result<()> {
    round_trip(
        "decimal256_39",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_76_optional_v2() -> Result<()> {
    round_trip(
        "decimal256_76",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn decimal256_76_required_v2() -> Result<()> {
    round_trip(
        "decimal256_76",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn struct_v1() -> Result<()> {
    round_trip(
        "struct",
        "struct",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain, Encoding::Plain],
    )
}

#[test]
fn struct_v2() -> Result<()> {
    round_trip(
        "struct",
        "struct",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain, Encoding::Plain],
    )
}

#[test]
fn map_v1() -> Result<()> {
    round_trip(
        "map",
        "map",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain, Encoding::Plain],
    )
}

#[test]
fn map_v2() -> Result<()> {
    round_trip(
        "map",
        "map",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain, Encoding::Plain],
    )
}

#[test]
fn nested_edge_simple() -> Result<()> {
    round_trip(
        "simple",
        "nested_edge",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn nested_edge_null() -> Result<()> {
    round_trip(
        "null",
        "nested_edge",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn v1_nested_edge_struct_list_nullable() -> Result<()> {
    round_trip(
        "struct_list_nullable",
        "nested_edge",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn nested_edge_list_struct_list_nullable() -> Result<()> {
    round_trip(
        "list_struct_list_nullable",
        "nested_edge",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}
