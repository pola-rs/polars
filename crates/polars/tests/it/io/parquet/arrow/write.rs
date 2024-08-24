use polars_parquet::arrow::write::*;

use super::*;

fn round_trip(
    column: &str,
    file: &str,
    version: Version,
    compression: CompressionOptions,
    encodings: Vec<Encoding>,
) -> PolarsResult<()> {
    round_trip_opt_stats(column, file, version, compression, encodings)
}

fn round_trip_opt_stats(
    column: &str,
    file: &str,
    version: Version,
    compression: CompressionOptions,
    encodings: Vec<Encoding>,
) -> PolarsResult<()> {
    let (array, _statistics) = match file {
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
        "nested_edge" => (
            pyarrow_nested_edge(column),
            pyarrow_nested_edge_statistics(column),
        ),
        _ => unreachable!(),
    };

    let field = Field::new("a1", array.data_type().clone(), true);
    let schema = ArrowSchema::from(vec![field]);

    let options = WriteOptions {
        statistics: StatisticsOptions::full(),
        compression,
        version,
        data_page_size: None,
    };

    let iter = vec![RecordBatchT::try_new(vec![array.clone()])];

    let row_groups =
        RowGroupIterator::try_new(iter.into_iter(), &schema, options, vec![encodings])?;

    let writer = Cursor::new(vec![]);
    let mut writer = FileWriter::try_new(writer, schema, options)?;

    for group in row_groups {
        writer.write(group?)?;
    }
    writer.end(None)?;

    let data = writer.into_inner().into_inner();

    std::fs::write("list_struct_list_nullable.parquet", &data).unwrap();

    let result = read_column(&mut Cursor::new(data), "a1")?;

    assert_eq!(array.as_ref(), result.as_ref());
    Ok(())
}

#[test]
fn int64_optional_v1() -> PolarsResult<()> {
    round_trip(
        "int64",
        "nullable",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn int64_required_v1() -> PolarsResult<()> {
    round_trip(
        "int64",
        "required",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn int64_optional_v2() -> PolarsResult<()> {
    round_trip(
        "int64",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn int64_optional_delta() -> PolarsResult<()> {
    round_trip(
        "int64",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::DeltaBinaryPacked],
    )
}

#[test]
fn int64_required_delta() -> PolarsResult<()> {
    round_trip(
        "int64",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::DeltaBinaryPacked],
    )
}

#[cfg(feature = "parquet")]
#[test]
fn int64_optional_v2_compressed() -> PolarsResult<()> {
    round_trip(
        "int64",
        "nullable",
        Version::V2,
        CompressionOptions::Snappy,
        vec![Encoding::Plain],
    )
}

#[test]
fn utf8_optional_v1() -> PolarsResult<()> {
    round_trip(
        "string",
        "nullable",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn utf8_required_v1() -> PolarsResult<()> {
    round_trip(
        "string",
        "required",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn utf8_optional_v2() -> PolarsResult<()> {
    round_trip(
        "string",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn utf8_required_v2() -> PolarsResult<()> {
    round_trip(
        "string",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[cfg(feature = "parquet")]
#[test]
fn utf8_optional_v2_compressed() -> PolarsResult<()> {
    round_trip(
        "string",
        "nullable",
        Version::V2,
        CompressionOptions::Snappy,
        vec![Encoding::Plain],
    )
}

#[cfg(feature = "parquet")]
#[test]
fn utf8_required_v2_compressed() -> PolarsResult<()> {
    round_trip(
        "string",
        "required",
        Version::V2,
        CompressionOptions::Snappy,
        vec![Encoding::Plain],
    )
}

#[test]
fn bool_optional_v1() -> PolarsResult<()> {
    round_trip(
        "bool",
        "nullable",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn bool_required_v1() -> PolarsResult<()> {
    round_trip(
        "bool",
        "required",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn bool_optional_v2_uncompressed() -> PolarsResult<()> {
    round_trip(
        "bool",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn bool_required_v2_uncompressed() -> PolarsResult<()> {
    round_trip(
        "bool",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[cfg(feature = "parquet")]
#[test]
fn bool_required_v2_compressed() -> PolarsResult<()> {
    round_trip(
        "bool",
        "required",
        Version::V2,
        CompressionOptions::Snappy,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_int64_optional_v2() -> PolarsResult<()> {
    round_trip(
        "list_int64",
        "nested",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_int64_optional_v1() -> PolarsResult<()> {
    round_trip(
        "list_int64",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_int64_required_required_v1() -> PolarsResult<()> {
    round_trip(
        "list_int64_required_required",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_int64_required_required_v2() -> PolarsResult<()> {
    round_trip(
        "list_int64_required_required",
        "nested",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_bool_optional_v2() -> PolarsResult<()> {
    round_trip(
        "list_bool",
        "nested",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_bool_optional_v1() -> PolarsResult<()> {
    round_trip(
        "list_bool",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_utf8_optional_v2() -> PolarsResult<()> {
    round_trip(
        "list_utf8",
        "nested",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_utf8_optional_v1() -> PolarsResult<()> {
    round_trip(
        "list_utf8",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn list_nested_inner_required_required_i64() -> PolarsResult<()> {
    round_trip_opt_stats(
        "list_nested_inner_required_required_i64",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn v1_nested_struct_list_nullable() -> PolarsResult<()> {
    round_trip_opt_stats(
        "struct_list_nullable",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn v1_nested_list_struct_list_nullable() -> PolarsResult<()> {
    round_trip_opt_stats(
        "list_struct_list_nullable",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn utf8_optional_v2_delta() -> PolarsResult<()> {
    round_trip(
        "string",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::DeltaLengthByteArray],
    )
}

#[test]
fn utf8_required_v2_delta() -> PolarsResult<()> {
    round_trip(
        "string",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::DeltaLengthByteArray],
    )
}

#[cfg(feature = "parquet")]
#[test]
fn i64_optional_v2_dict_compressed() -> PolarsResult<()> {
    round_trip(
        "int32_dict",
        "nullable",
        Version::V2,
        CompressionOptions::Snappy,
        vec![Encoding::RleDictionary],
    )
}

#[test]
fn struct_v1() -> PolarsResult<()> {
    round_trip(
        "struct",
        "struct",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain, Encoding::Plain],
    )
}

#[test]
fn struct_v2() -> PolarsResult<()> {
    round_trip(
        "struct",
        "struct",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain, Encoding::Plain],
    )
}

#[test]
fn nested_edge_simple() -> PolarsResult<()> {
    round_trip(
        "simple",
        "nested_edge",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn nested_edge_null() -> PolarsResult<()> {
    round_trip(
        "null",
        "nested_edge",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn v1_nested_edge_struct_list_nullable() -> PolarsResult<()> {
    round_trip(
        "struct_list_nullable",
        "nested_edge",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

#[test]
fn nested_edge_list_struct_list_nullable() -> PolarsResult<()> {
    round_trip(
        "list_struct_list_nullable",
        "nested_edge",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}
