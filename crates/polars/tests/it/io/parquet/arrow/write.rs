use polars_parquet::arrow::write::*;

use super::*;

fn round_trip(
    column: &str,
    file: &str,
    version: Version,
    compression: CompressionOptions,
    column_options: Vec<ColumnWriteOptions>,
) -> PolarsResult<()> {
    round_trip_opt_stats(column, file, version, compression, column_options)
}

fn round_trip_opt_stats(
    column: &str,
    file: &str,
    version: Version,
    compression: CompressionOptions,
    column_options: Vec<ColumnWriteOptions>,
) -> PolarsResult<()> {
    let array = match file {
        "nested" => pyarrow_nested_nullable(column),
        "nullable" => pyarrow_nullable(column),
        "required" => pyarrow_required(column),
        "struct" => pyarrow_struct(column),
        "nested_edge" => pyarrow_nested_edge(column),
        _ => unreachable!(),
    };

    let field = Field::new("a1".into(), array.dtype().clone(), true);
    let schema = ArrowSchema::from_iter([field]);

    let options = WriteOptions {
        statistics: StatisticsOptions::full(),
        compression,
        version,
        data_page_size: None,
    };

    let iter = vec![RecordBatchT::try_new(
        array.len(),
        Arc::new(schema.clone()),
        vec![array.clone()],
    )];

    let row_groups =
        RowGroupIterator::try_new(iter.into_iter(), &schema, options, column_options.clone())?;

    let writer = Cursor::new(vec![]);
    let mut writer = FileWriter::try_new(writer, schema, options, &column_options)?;

    for group in row_groups {
        writer.write(group?)?;
    }
    writer.end(None, &column_options)?;

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
        vec![
            FieldWriteOptions::default_with_encoding(Encoding::Plain)
                .into_default_column_write_options(),
        ],
    )
}

#[test]
fn int64_required_v1() -> PolarsResult<()> {
    round_trip(
        "int64",
        "required",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![
            FieldWriteOptions::default_with_encoding(Encoding::Plain)
                .into_default_column_write_options(),
        ],
    )
}

#[test]
fn int64_optional_v2() -> PolarsResult<()> {
    round_trip(
        "int64",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![
            FieldWriteOptions::default_with_encoding(Encoding::Plain)
                .into_default_column_write_options(),
        ],
    )
}

#[test]
fn int64_optional_delta() -> PolarsResult<()> {
    round_trip(
        "int64",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![
            FieldWriteOptions::default_with_encoding(Encoding::DeltaBinaryPacked)
                .into_default_column_write_options(),
        ],
    )
}

#[test]
fn int64_required_delta() -> PolarsResult<()> {
    round_trip(
        "int64",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![
            FieldWriteOptions::default_with_encoding(Encoding::DeltaBinaryPacked)
                .into_default_column_write_options(),
        ],
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
        vec![
            FieldWriteOptions::default_with_encoding(Encoding::Plain)
                .into_default_column_write_options(),
        ],
    )
}

#[test]
fn utf8_optional_v1() -> PolarsResult<()> {
    round_trip(
        "string",
        "nullable",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![
            FieldWriteOptions::default_with_encoding(Encoding::Plain)
                .into_default_column_write_options(),
        ],
    )
}

#[test]
fn utf8_required_v1() -> PolarsResult<()> {
    round_trip(
        "string",
        "required",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![
            FieldWriteOptions::default_with_encoding(Encoding::Plain)
                .into_default_column_write_options(),
        ],
    )
}

#[test]
fn utf8_optional_v2() -> PolarsResult<()> {
    round_trip(
        "string",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![
            FieldWriteOptions::default_with_encoding(Encoding::Plain)
                .into_default_column_write_options(),
        ],
    )
}

#[test]
fn utf8_required_v2() -> PolarsResult<()> {
    round_trip(
        "string",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![
            FieldWriteOptions::default_with_encoding(Encoding::Plain)
                .into_default_column_write_options(),
        ],
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
        vec![
            FieldWriteOptions::default_with_encoding(Encoding::Plain)
                .into_default_column_write_options(),
        ],
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
        vec![
            FieldWriteOptions::default_with_encoding(Encoding::Plain)
                .into_default_column_write_options(),
        ],
    )
}

#[test]
fn bool_optional_v1() -> PolarsResult<()> {
    round_trip(
        "bool",
        "nullable",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![
            FieldWriteOptions::default_with_encoding(Encoding::Plain)
                .into_default_column_write_options(),
        ],
    )
}

#[test]
fn bool_required_v1() -> PolarsResult<()> {
    round_trip(
        "bool",
        "required",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![
            FieldWriteOptions::default_with_encoding(Encoding::Plain)
                .into_default_column_write_options(),
        ],
    )
}

#[test]
fn bool_optional_v2_uncompressed() -> PolarsResult<()> {
    round_trip(
        "bool",
        "nullable",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![
            FieldWriteOptions::default_with_encoding(Encoding::Plain)
                .into_default_column_write_options(),
        ],
    )
}

#[test]
fn bool_required_v2_uncompressed() -> PolarsResult<()> {
    round_trip(
        "bool",
        "required",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![
            FieldWriteOptions::default_with_encoding(Encoding::Plain)
                .into_default_column_write_options(),
        ],
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
        vec![
            FieldWriteOptions::default_with_encoding(Encoding::Plain)
                .into_default_column_write_options(),
        ],
    )
}

#[test]
fn list_int64_optional_v2() -> PolarsResult<()> {
    round_trip(
        "list_int64",
        "nested",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![ColumnWriteOptions::default_with(
            ChildWriteOptions::ListLike(Box::new(ListLikeFieldWriteOptions {
                child: FieldWriteOptions::default_with_encoding(Encoding::Plain)
                    .into_default_column_write_options(),
            })),
        )],
    )
}

#[test]
fn list_int64_optional_v1() -> PolarsResult<()> {
    round_trip(
        "list_int64",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![ColumnWriteOptions::default_with(
            ChildWriteOptions::ListLike(Box::new(ListLikeFieldWriteOptions {
                child: FieldWriteOptions::default_with_encoding(Encoding::Plain)
                    .into_default_column_write_options(),
            })),
        )],
    )
}

#[test]
fn list_int64_required_required_v1() -> PolarsResult<()> {
    round_trip(
        "list_int64_required_required",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![ColumnWriteOptions::default_with(
            ChildWriteOptions::ListLike(Box::new(ListLikeFieldWriteOptions {
                child: FieldWriteOptions::default_with_encoding(Encoding::Plain)
                    .into_default_column_write_options(),
            })),
        )],
    )
}

#[test]
fn list_int64_required_required_v2() -> PolarsResult<()> {
    round_trip(
        "list_int64_required_required",
        "nested",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![ColumnWriteOptions::default_with(
            ChildWriteOptions::ListLike(Box::new(ListLikeFieldWriteOptions {
                child: FieldWriteOptions::default_with_encoding(Encoding::Plain)
                    .into_default_column_write_options(),
            })),
        )],
    )
}

#[test]
fn list_bool_optional_v2() -> PolarsResult<()> {
    round_trip(
        "list_bool",
        "nested",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![ColumnWriteOptions::default_with(
            ChildWriteOptions::ListLike(Box::new(ListLikeFieldWriteOptions {
                child: FieldWriteOptions::default_with_encoding(Encoding::Plain)
                    .into_default_column_write_options(),
            })),
        )],
    )
}

#[test]
fn list_bool_optional_v1() -> PolarsResult<()> {
    round_trip(
        "list_bool",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![ColumnWriteOptions::default_with(
            ChildWriteOptions::ListLike(Box::new(ListLikeFieldWriteOptions {
                child: FieldWriteOptions::default_with_encoding(Encoding::Plain)
                    .into_default_column_write_options(),
            })),
        )],
    )
}

#[test]
fn list_utf8_optional_v2() -> PolarsResult<()> {
    round_trip(
        "list_utf8",
        "nested",
        Version::V2,
        CompressionOptions::Uncompressed,
        vec![ColumnWriteOptions::default_with(
            ChildWriteOptions::ListLike(Box::new(ListLikeFieldWriteOptions {
                child: FieldWriteOptions::default_with_encoding(Encoding::Plain)
                    .into_default_column_write_options(),
            })),
        )],
    )
}

#[test]
fn list_utf8_optional_v1() -> PolarsResult<()> {
    round_trip(
        "list_utf8",
        "nested",
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![ColumnWriteOptions::default_with(
            ChildWriteOptions::ListLike(Box::new(ListLikeFieldWriteOptions {
                child: FieldWriteOptions::default_with_encoding(Encoding::Plain)
                    .into_default_column_write_options(),
            })),
        )],
    )
}
