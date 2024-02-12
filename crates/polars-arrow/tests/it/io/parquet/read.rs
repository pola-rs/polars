use std::fs::File;

use polars_arrow::array::*;
use polars_arrow::error::*;
use polars_arrow::io::parquet::read::*;

use super::*;

fn test_pyarrow_integration(
    column: &str,
    version: usize,
    type_: &str,
    use_dict: bool,
    required: bool,
    compression: Option<&str>,
) -> Result<()> {
    if std::env::var("ARROW2_IGNORE_PARQUET").is_ok() {
        return Ok(());
    }
    let use_dict = if use_dict { "dict/" } else { "" };
    let compression = if let Some(compression) = compression {
        format!("{compression}/")
    } else {
        "".to_string()
    };
    let required_str = if required { "required" } else { "nullable" };
    let path = format!(
        "fixtures/pyarrow3/v{version}/{use_dict}{compression}{type_}_{required_str}_10.parquet"
    );

    let mut file = File::open(path).unwrap();
    let (array, statistics) = read_column(&mut file, column)?;

    let expected = match (type_, required) {
        ("basic", true) => pyarrow_required(column),
        ("basic", false) => pyarrow_nullable(column),
        ("nested", false) => pyarrow_nested_nullable(column),
        ("nested_edge", false) => pyarrow_nested_edge(column),
        ("struct", false) => pyarrow_struct(column),
        ("map", true) => pyarrow_map(column),
        _ => unreachable!(),
    };

    let expected_statistics = match (type_, required) {
        ("basic", true) => pyarrow_required_statistics(column),
        ("basic", false) => pyarrow_nullable_statistics(column),
        ("nested", false) => pyarrow_nested_nullable_statistics(column),
        ("nested_edge", false) => pyarrow_nested_edge_statistics(column),
        ("struct", false) => pyarrow_struct_statistics(column),
        ("map", true) => pyarrow_map_statistics(column),
        _ => unreachable!(),
    };

    assert_eq!(expected.as_ref(), array.as_ref());
    if ![
        // pyarrow outputs an incorrect number of null count for nested types - ARROW-16299
        "list_int16",
        "list_large_binary",
        "list_int64",
        "list_int64_required",
        "list_int64_required_required",
        "list_nested_i64",
        "list_utf8",
        "list_bool",
        "list_decimal",
        "list_decimal256",
        "list_nested_inner_required_required_i64",
        "list_nested_inner_required_i64",
        // pyarrow counts null struct items as nulls
        "struct_nullable",
        "list_struct_nullable",
        "list_struct_list_nullable",
        "struct_list_nullable",
        "null",
        // pyarrow reports an incorrect min/max for MapArray
        "map",
        "map_nullable",
    ]
    .contains(&column)
    {
        assert_eq!(expected_statistics, statistics);
    }

    Ok(())
}

#[test]
fn v1_int64_nullable() -> Result<()> {
    test_pyarrow_integration("int64", 1, "basic", false, false, None)
}

#[test]
#[ignore] // see https://issues.apache.org/jira/browse/ARROW-15073
fn v1_int64_lz4_nullable() -> Result<()> {
    test_pyarrow_integration("int64", 1, "basic", false, false, Some("lz4"))
}

#[test]
#[ignore] // see https://issues.apache.org/jira/browse/ARROW-15073
fn v1_int64_lz4_required() -> Result<()> {
    test_pyarrow_integration("int64", 1, "basic", false, true, Some("lz4"))
}

#[test]
fn v1_int64_required() -> Result<()> {
    test_pyarrow_integration("int64", 1, "basic", false, true, None)
}

#[test]
fn v1_float64_nullable() -> Result<()> {
    test_pyarrow_integration("float64", 1, "basic", false, false, None)
}

#[test]
fn v1_utf8_nullable() -> Result<()> {
    test_pyarrow_integration("string", 1, "basic", false, false, None)
}

#[test]
fn v1_utf8_required() -> Result<()> {
    test_pyarrow_integration("string", 1, "basic", false, true, None)
}

#[test]
fn v1_boolean_nullable() -> Result<()> {
    test_pyarrow_integration("bool", 1, "basic", false, false, None)
}

#[test]
fn v1_boolean_required() -> Result<()> {
    test_pyarrow_integration("bool", 1, "basic", false, true, None)
}

#[test]
fn v1_timestamp_ms_nullable() -> Result<()> {
    test_pyarrow_integration("timestamp_ms", 1, "basic", false, false, None)
}

#[test]
fn v1_u32_nullable() -> Result<()> {
    test_pyarrow_integration("uint32", 1, "basic", false, false, None)
}

#[test]
fn v2_u32_nullable() -> Result<()> {
    test_pyarrow_integration("uint32", 2, "basic", false, false, None)
}

#[test]
fn v2_int64_nullable() -> Result<()> {
    test_pyarrow_integration("int64", 2, "basic", false, false, None)
}

#[test]
fn v2_int64_nullable_dict() -> Result<()> {
    test_pyarrow_integration("int64", 2, "basic", true, false, None)
}

#[test]
#[ignore] // see https://issues.apache.org/jira/browse/ARROW-15073
fn v2_int64_nullable_dict_lz4() -> Result<()> {
    test_pyarrow_integration("int64", 2, "basic", true, false, Some("lz4"))
}

#[test]
fn v1_int64_nullable_dict() -> Result<()> {
    test_pyarrow_integration("int64", 1, "basic", true, false, None)
}

#[test]
fn v2_int64_required_dict() -> Result<()> {
    test_pyarrow_integration("int64", 2, "basic", true, true, None)
}

#[test]
fn v1_int64_required_dict() -> Result<()> {
    test_pyarrow_integration("int64", 1, "basic", true, true, None)
}

#[test]
fn v2_utf8_nullable() -> Result<()> {
    test_pyarrow_integration("string", 2, "basic", false, false, None)
}

#[test]
fn v2_utf8_required() -> Result<()> {
    test_pyarrow_integration("string", 2, "basic", false, true, None)
}

#[test]
fn v2_utf8_nullable_dict() -> Result<()> {
    test_pyarrow_integration("string", 2, "basic", true, false, None)
}

#[test]
fn v1_utf8_nullable_dict() -> Result<()> {
    test_pyarrow_integration("string", 1, "basic", true, false, None)
}

#[test]
fn v2_utf8_required_dict() -> Result<()> {
    test_pyarrow_integration("string", 2, "basic", true, true, None)
}

#[test]
fn v1_utf8_required_dict() -> Result<()> {
    test_pyarrow_integration("string", 1, "basic", true, true, None)
}

#[test]
fn v2_boolean_nullable() -> Result<()> {
    test_pyarrow_integration("bool", 2, "basic", false, false, None)
}

#[test]
fn v2_boolean_required() -> Result<()> {
    test_pyarrow_integration("bool", 2, "basic", false, true, None)
}

#[test]
fn v2_nested_int64_nullable() -> Result<()> {
    test_pyarrow_integration("list_int64", 2, "nested", false, false, None)
}

#[test]
fn v1_nested_int64_nullable() -> Result<()> {
    test_pyarrow_integration("list_int64", 1, "nested", false, false, None)
}

#[test]
fn v2_nested_int64_nullable_required() -> Result<()> {
    test_pyarrow_integration("list_int64", 2, "nested", false, false, None)
}

#[test]
fn v2_nested_int64_required_required() -> Result<()> {
    test_pyarrow_integration("list_int64_required", 2, "nested", false, false, None)
}

#[test]
fn v1_nested_int64_required_required() -> Result<()> {
    test_pyarrow_integration("list_int64_required", 1, "nested", false, false, None)
}

#[test]
fn v2_list_int64_required_required() -> Result<()> {
    test_pyarrow_integration(
        "list_int64_required_required",
        2,
        "nested",
        false,
        false,
        None,
    )
}

#[test]
#[ignore] // see https://issues.apache.org/jira/browse/ARROW-15073
fn v2_list_int64_optional_required() -> Result<()> {
    test_pyarrow_integration(
        "list_int64_optional_required",
        2,
        "nested",
        false,
        false,
        None,
    )
}

#[test]
fn v1_nested_i16() -> Result<()> {
    test_pyarrow_integration("list_int16", 1, "nested", false, false, None)
}

#[test]
fn v1_nested_i16_dict() -> Result<()> {
    test_pyarrow_integration("list_int16", 1, "nested", true, false, None)
}

#[test]
fn v1_nested_i16_required_dict() -> Result<()> {
    test_pyarrow_integration(
        "list_int64_required_required",
        1,
        "nested",
        true,
        false,
        None,
    )
}

#[test]
fn v2_nested_bool() -> Result<()> {
    test_pyarrow_integration("list_bool", 2, "nested", false, false, None)
}

#[test]
fn v1_nested_bool() -> Result<()> {
    test_pyarrow_integration("list_bool", 1, "nested", false, false, None)
}

#[test]
fn v2_nested_utf8() -> Result<()> {
    test_pyarrow_integration("list_utf8", 2, "nested", false, false, None)
}

#[test]
fn v1_nested_utf8() -> Result<()> {
    test_pyarrow_integration("list_utf8", 1, "nested", false, false, None)
}

#[test]
fn v1_nested_utf8_dict() -> Result<()> {
    test_pyarrow_integration("list_utf8", 1, "nested", true, false, None)
}

#[test]
fn v2_nested_large_binary() -> Result<()> {
    test_pyarrow_integration("list_large_binary", 2, "nested", false, false, None)
}

#[test]
fn v1_nested_large_binary() -> Result<()> {
    test_pyarrow_integration("list_large_binary", 1, "nested", false, false, None)
}

#[test]
fn v2_nested_decimal_nullable() -> Result<()> {
    test_pyarrow_integration("list_decimal", 2, "nested", false, false, None)
}

#[test]
fn v2_nested_decimal256_nullable() -> Result<()> {
    test_pyarrow_integration("list_decimal256", 2, "nested", false, false, None)
}

#[test]
fn v2_nested_nested() -> Result<()> {
    test_pyarrow_integration("list_nested_i64", 2, "nested", false, false, None)
}

#[test]
fn v2_nested_nested_required() -> Result<()> {
    test_pyarrow_integration(
        "list_nested_inner_required_i64",
        2,
        "nested",
        false,
        false,
        None,
    )
}

#[test]
fn v2_nested_nested_required_required() -> Result<()> {
    test_pyarrow_integration(
        "list_nested_inner_required_required_i64",
        2,
        "nested",
        false,
        false,
        None,
    )
}

#[test]
fn v1_nested_list_struct_nullable() -> Result<()> {
    test_pyarrow_integration("list_struct_nullable", 1, "nested", false, false, None)
}

#[test]
fn v1_nested_struct_list_nullable() -> Result<()> {
    test_pyarrow_integration("struct_list_nullable", 1, "nested", false, false, None)
}

#[test]
fn v1_nested_list_struct_list_nullable() -> Result<()> {
    test_pyarrow_integration("list_struct_list_nullable", 1, "nested", false, false, None)
}

#[test]
fn v1_decimal_9_nullable() -> Result<()> {
    test_pyarrow_integration("decimal_9", 1, "basic", false, false, None)
}

#[test]
fn v1_decimal_9_required() -> Result<()> {
    test_pyarrow_integration("decimal_9", 1, "basic", false, true, None)
}

#[test]
fn v1_decimal_9_nullable_dict() -> Result<()> {
    test_pyarrow_integration("decimal_9", 1, "basic", true, false, None)
}

#[test]
fn v1_decimal_18_nullable() -> Result<()> {
    test_pyarrow_integration("decimal_18", 1, "basic", false, false, None)
}

#[test]
fn v1_decimal_18_required() -> Result<()> {
    test_pyarrow_integration("decimal_18", 1, "basic", false, true, None)
}

#[test]
fn v1_decimal_26_nullable() -> Result<()> {
    test_pyarrow_integration("decimal_26", 1, "basic", false, false, None)
}

#[test]
fn v1_decimal_26_required() -> Result<()> {
    test_pyarrow_integration("decimal_26", 1, "basic", false, true, None)
}

#[test]
fn v1_decimal256_9_nullable() -> Result<()> {
    test_pyarrow_integration("decimal256_9", 1, "basic", false, false, None)
}

#[test]
fn v1_decimal256_9_required() -> Result<()> {
    test_pyarrow_integration("decimal256_9", 1, "basic", false, true, None)
}

#[test]
fn v1_decimal256_18_nullable() -> Result<()> {
    test_pyarrow_integration("decimal256_18", 1, "basic", false, false, None)
}

#[test]
fn v1_decimal256_18_required() -> Result<()> {
    test_pyarrow_integration("decimal256_18", 1, "basic", false, true, None)
}

#[test]
fn v1_decimal256_26_nullable() -> Result<()> {
    test_pyarrow_integration("decimal256_26", 1, "basic", false, false, None)
}

#[test]
fn v1_decimal256_26_required() -> Result<()> {
    test_pyarrow_integration("decimal256_26", 1, "basic", false, true, None)
}

#[test]
fn v1_decimal256_39_nullable() -> Result<()> {
    test_pyarrow_integration("decimal256_39", 1, "basic", false, false, None)
}

#[test]
fn v1_decimal256_39_required() -> Result<()> {
    test_pyarrow_integration("decimal256_39", 1, "basic", false, true, None)
}

#[test]
fn v1_decimal256_76_nullable() -> Result<()> {
    test_pyarrow_integration("decimal256_76", 1, "basic", false, false, None)
}

#[test]
fn v1_decimal256_76_required() -> Result<()> {
    test_pyarrow_integration("decimal256_76", 1, "basic", false, true, None)
}

#[test]
fn v2_decimal_9_nullable() -> Result<()> {
    test_pyarrow_integration("decimal_9", 2, "basic", false, false, None)
}

#[test]
fn v2_decimal_9_required() -> Result<()> {
    test_pyarrow_integration("decimal_9", 2, "basic", false, true, None)
}

#[test]
fn v2_decimal_9_required_dict() -> Result<()> {
    test_pyarrow_integration("decimal_9", 2, "basic", true, true, None)
}

#[test]
fn v2_decimal_18_nullable() -> Result<()> {
    test_pyarrow_integration("decimal_18", 2, "basic", false, false, None)
}

#[test]
fn v2_decimal_18_required() -> Result<()> {
    test_pyarrow_integration("decimal_18", 2, "basic", false, true, None)
}

#[test]
fn v2_decimal_18_required_dict() -> Result<()> {
    test_pyarrow_integration("decimal_18", 2, "basic", true, true, None)
}

#[test]
fn v2_decimal_26_nullable() -> Result<()> {
    test_pyarrow_integration("decimal_26", 2, "basic", false, false, None)
}

#[test]
fn v2_decimal256_9_nullable() -> Result<()> {
    test_pyarrow_integration("decimal256_9", 2, "basic", false, false, None)
}

#[test]
fn v2_decimal256_18_nullable() -> Result<()> {
    test_pyarrow_integration("decimal256_18", 2, "basic", false, false, None)
}

#[test]
fn v2_decimal256_26_nullable() -> Result<()> {
    test_pyarrow_integration("decimal256_26", 2, "basic", false, false, None)
}

#[test]
fn v2_decimal256_39_nullable() -> Result<()> {
    test_pyarrow_integration("decimal256_39", 2, "basic", false, false, None)
}

#[test]
fn v2_decimal256_76_nullable() -> Result<()> {
    test_pyarrow_integration("decimal256_76", 2, "basic", false, false, None)
}

#[test]
fn v1_timestamp_us_nullable() -> Result<()> {
    test_pyarrow_integration("timestamp_us", 1, "basic", false, false, None)
}

#[test]
fn v1_timestamp_s_nullable() -> Result<()> {
    test_pyarrow_integration("timestamp_s", 1, "basic", false, false, None)
}

#[test]
fn v1_timestamp_s_nullable_dict() -> Result<()> {
    test_pyarrow_integration("timestamp_s", 1, "basic", true, false, None)
}

#[test]
fn v1_timestamp_s_utc_nullable() -> Result<()> {
    test_pyarrow_integration("timestamp_s_utc", 1, "basic", false, false, None)
}

#[test]
fn v2_decimal_26_required() -> Result<()> {
    test_pyarrow_integration("decimal_26", 2, "basic", false, true, None)
}

#[test]
fn v2_decimal_26_required_dict() -> Result<()> {
    test_pyarrow_integration("decimal_26", 2, "basic", true, true, None)
}

#[test]
fn v2_decimal256_9_required() -> Result<()> {
    test_pyarrow_integration("decimal256_9", 2, "basic", false, true, None)
}

#[test]
fn v2_decimal256_9_required_dict() -> Result<()> {
    test_pyarrow_integration("decimal256_9", 2, "basic", true, true, None)
}

#[test]
fn v2_decimal256_18_required() -> Result<()> {
    test_pyarrow_integration("decimal256_18", 2, "basic", false, true, None)
}

#[test]
fn v2_decimal256_18_required_dict() -> Result<()> {
    test_pyarrow_integration("decimal256_18", 2, "basic", true, true, None)
}

#[test]
fn v2_decimal256_26_required() -> Result<()> {
    test_pyarrow_integration("decimal256_26", 2, "basic", false, true, None)
}

#[test]
fn v2_decimal256_26_required_dict() -> Result<()> {
    test_pyarrow_integration("decimal256_26", 2, "basic", true, true, None)
}

#[test]
fn v2_decimal256_39_required() -> Result<()> {
    test_pyarrow_integration("decimal256_39", 2, "basic", false, true, None)
}

#[test]
fn v2_decimal256_39_required_dict() -> Result<()> {
    test_pyarrow_integration("decimal256_39", 2, "basic", true, true, None)
}

#[test]
fn v2_decimal256_76_required() -> Result<()> {
    test_pyarrow_integration("decimal256_76", 2, "basic", false, true, None)
}

#[test]
fn v2_decimal256_76_required_dict() -> Result<()> {
    test_pyarrow_integration("decimal256_76", 2, "basic", true, true, None)
}

#[test]
fn v1_struct_required_optional() -> Result<()> {
    test_pyarrow_integration("struct", 1, "struct", false, false, None)
}

#[test]
fn v1_struct_struct() -> Result<()> {
    test_pyarrow_integration("struct_struct", 1, "struct", false, false, None)
}

#[test]
fn v1_struct_optional_optional() -> Result<()> {
    test_pyarrow_integration("struct_nullable", 1, "struct", false, false, None)
}

#[test]
fn v1_struct_struct_optional() -> Result<()> {
    test_pyarrow_integration("struct_struct_nullable", 1, "struct", false, false, None)
}

#[test]
fn v1_nested_edge_simple() -> Result<()> {
    test_pyarrow_integration("simple", 1, "nested_edge", false, false, None)
}

#[test]
fn v1_nested_edge_null() -> Result<()> {
    test_pyarrow_integration("null", 1, "nested_edge", false, false, None)
}

#[test]
fn v1_nested_edge_struct_list_nullable() -> Result<()> {
    test_pyarrow_integration("struct_list_nullable", 1, "nested_edge", false, false, None)
}

#[test]
fn v1_nested_edge_list_struct_list_nullable() -> Result<()> {
    test_pyarrow_integration(
        "list_struct_list_nullable",
        1,
        "nested_edge",
        false,
        false,
        None,
    )
}

#[test]
fn v1_map() -> Result<()> {
    test_pyarrow_integration("map", 1, "map", false, true, None)
}

#[test]
fn v1_map_nullable() -> Result<()> {
    test_pyarrow_integration("map_nullable", 1, "map", false, true, None)
}

#[cfg(feature = "io_parquet_compression")]
#[test]
fn all_types() -> Result<()> {
    let path = "testing/parquet-testing/data/alltypes_plain.parquet";
    let mut reader = std::fs::File::open(path)?;

    let metadata = read_metadata(&mut reader)?;
    let schema = infer_schema(&metadata)?;
    let reader = FileReader::new(reader, metadata.row_groups, schema, None, None, None);

    let batches = reader.collect::<Result<Vec<_>>>()?;
    assert_eq!(batches.len(), 1);

    let result = batches[0].columns()[0]
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(result, &Int32Array::from_slice([4, 5, 6, 7, 2, 3, 0, 1]));

    let result = batches[0].columns()[6]
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_eq!(
        result,
        &Float32Array::from_slice([0.0, 1.1, 0.0, 1.1, 0.0, 1.1, 0.0, 1.1])
    );

    let result = batches[0].columns()[9]
        .as_any()
        .downcast_ref::<BinaryArray<i32>>()
        .unwrap();
    assert_eq!(
        result,
        &BinaryArray::<i32>::from_slice([[48], [49], [48], [49], [48], [49], [48], [49]])
    );

    Ok(())
}

#[cfg(feature = "io_parquet_compression")]
#[test]
fn all_types_chunked() -> Result<()> {
    // this has one batch with 8 elements
    let path = "testing/parquet-testing/data/alltypes_plain.parquet";
    let mut reader = std::fs::File::open(path)?;

    let metadata = read_metadata(&mut reader)?;
    let schema = infer_schema(&metadata)?;
    // chunk it in 5 (so, (5,3))
    let reader = FileReader::new(reader, metadata.row_groups, schema, Some(5), None, None);

    let batches = reader.collect::<Result<Vec<_>>>()?;
    assert_eq!(batches.len(), 2);

    assert_eq!(batches[0].len(), 5);
    assert_eq!(batches[1].len(), 3);

    let result = batches[0].columns()[0]
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(result, &Int32Array::from_slice([4, 5, 6, 7, 2]));

    let result = batches[1].columns()[0]
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(result, &Int32Array::from_slice([3, 0, 1]));

    let result = batches[0].columns()[6]
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_eq!(result, &Float32Array::from_slice([0.0, 1.1, 0.0, 1.1, 0.0]));

    let result = batches[0].columns()[9]
        .as_any()
        .downcast_ref::<BinaryArray<i32>>()
        .unwrap();
    assert_eq!(
        result,
        &BinaryArray::<i32>::from_slice([[48], [49], [48], [49], [48]])
    );

    let result = batches[1].columns()[9]
        .as_any()
        .downcast_ref::<BinaryArray<i32>>()
        .unwrap();
    assert_eq!(result, &BinaryArray::<i32>::from_slice([[49], [48], [49]]));

    Ok(())
}

#[cfg(feature = "io_parquet_compression")]
#[test]
fn invalid_utf8() -> Result<()> {
    let invalid_data = &[
        0x50, 0x41, 0x52, 0x31, 0x15, 0x00, 0x15, 0x24, 0x15, 0x28, 0x2c, 0x15, 0x02, 0x15, 0x00,
        0x15, 0x06, 0x15, 0x08, 0x00, 0x00, 0x12, 0x44, 0x02, 0x00, 0x00, 0x00, 0x03, 0xff, 0x08,
        0x00, 0x00, 0x00, 0x67, 0x6f, 0x75, 0x67, 0xe8, 0x72, 0x65, 0x73, 0x15, 0x02, 0x19, 0x2c,
        0x48, 0x0d, 0x64, 0x75, 0x63, 0x6b, 0x64, 0x62, 0x5f, 0x73, 0x63, 0x68, 0x65, 0x6d, 0x61,
        0x15, 0x02, 0x00, 0x15, 0x0c, 0x25, 0x02, 0x18, 0x02, 0x63, 0x31, 0x15, 0x00, 0x15, 0x00,
        0x00, 0x16, 0x02, 0x19, 0x1c, 0x19, 0x1c, 0x26, 0x00, 0x1c, 0x15, 0x0c, 0x19, 0x05, 0x19,
        0x18, 0x02, 0x63, 0x31, 0x15, 0x02, 0x16, 0x02, 0x16, 0x00, 0x16, 0x4a, 0x26, 0x08, 0x00,
        0x00, 0x16, 0x00, 0x16, 0x02, 0x26, 0x08, 0x00, 0x28, 0x06, 0x44, 0x75, 0x63, 0x6b, 0x44,
        0x42, 0x00, 0x51, 0x00, 0x00, 0x00, 0x50, 0x41, 0x52, 0x31,
    ];

    let mut reader = Cursor::new(invalid_data);

    let metadata = read_metadata(&mut reader)?;
    let schema = infer_schema(&metadata)?;
    let reader = FileReader::new(reader, metadata.row_groups, schema, Some(5), None, None);

    let error = reader.collect::<Result<Vec<_>>>().unwrap_err();
    assert!(
        error.to_string().contains("invalid utf-8"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn read_int96_timestamps() -> Result<()> {
    use std::collections::BTreeMap;

    let timestamp_data = &[
        0x50, 0x41, 0x52, 0x31, 0x15, 0x04, 0x15, 0x48, 0x15, 0x3c, 0x4c, 0x15, 0x06, 0x15, 0x00,
        0x12, 0x00, 0x00, 0x24, 0x00, 0x00, 0x0d, 0x01, 0x08, 0x9f, 0xd5, 0x1f, 0x0d, 0x0a, 0x44,
        0x00, 0x00, 0x59, 0x68, 0x25, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14,
        0xfb, 0x2a, 0x00, 0x15, 0x00, 0x15, 0x14, 0x15, 0x18, 0x2c, 0x15, 0x06, 0x15, 0x10, 0x15,
        0x06, 0x15, 0x06, 0x1c, 0x00, 0x00, 0x00, 0x0a, 0x24, 0x02, 0x00, 0x00, 0x00, 0x06, 0x01,
        0x02, 0x03, 0x24, 0x00, 0x26, 0x9e, 0x01, 0x1c, 0x15, 0x06, 0x19, 0x35, 0x10, 0x00, 0x06,
        0x19, 0x18, 0x0a, 0x74, 0x69, 0x6d, 0x65, 0x73, 0x74, 0x61, 0x6d, 0x70, 0x73, 0x15, 0x02,
        0x16, 0x06, 0x16, 0x9e, 0x01, 0x16, 0x96, 0x01, 0x26, 0x60, 0x26, 0x08, 0x29, 0x2c, 0x15,
        0x04, 0x15, 0x00, 0x15, 0x02, 0x00, 0x15, 0x00, 0x15, 0x10, 0x15, 0x02, 0x00, 0x00, 0x00,
        0x15, 0x04, 0x19, 0x2c, 0x35, 0x00, 0x18, 0x06, 0x73, 0x63, 0x68, 0x65, 0x6d, 0x61, 0x15,
        0x02, 0x00, 0x15, 0x06, 0x25, 0x02, 0x18, 0x0a, 0x74, 0x69, 0x6d, 0x65, 0x73, 0x74, 0x61,
        0x6d, 0x70, 0x73, 0x00, 0x16, 0x06, 0x19, 0x1c, 0x19, 0x1c, 0x26, 0x9e, 0x01, 0x1c, 0x15,
        0x06, 0x19, 0x35, 0x10, 0x00, 0x06, 0x19, 0x18, 0x0a, 0x74, 0x69, 0x6d, 0x65, 0x73, 0x74,
        0x61, 0x6d, 0x70, 0x73, 0x15, 0x02, 0x16, 0x06, 0x16, 0x9e, 0x01, 0x16, 0x96, 0x01, 0x26,
        0x60, 0x26, 0x08, 0x29, 0x2c, 0x15, 0x04, 0x15, 0x00, 0x15, 0x02, 0x00, 0x15, 0x00, 0x15,
        0x10, 0x15, 0x02, 0x00, 0x00, 0x00, 0x16, 0x9e, 0x01, 0x16, 0x06, 0x26, 0x08, 0x16, 0x96,
        0x01, 0x14, 0x00, 0x00, 0x28, 0x20, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2d, 0x63,
        0x70, 0x70, 0x2d, 0x61, 0x72, 0x72, 0x6f, 0x77, 0x20, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6f,
        0x6e, 0x20, 0x31, 0x32, 0x2e, 0x30, 0x2e, 0x30, 0x19, 0x1c, 0x1c, 0x00, 0x00, 0x00, 0x95,
        0x00, 0x00, 0x00, 0x50, 0x41, 0x52, 0x31,
    ];

    let parse = |time_unit: TimeUnit| {
        let mut reader = Cursor::new(timestamp_data);
        let metadata = read_metadata(&mut reader)?;
        let schema = polars_arrow::datatypes::ArrowSchema {
            fields: vec![polars_arrow::datatypes::Field::new(
                "timestamps",
                polars_arrow::datatypes::ArrowDataType::Timestamp(time_unit, None),
                false,
            )],
            metadata: BTreeMap::new(),
        };
        let reader = FileReader::new(reader, metadata.row_groups, schema, Some(5), None, None);
        reader.collect::<Result<Vec<_>>>()
    };

    // This data contains int96 timestamps in the year 1000 and 3000, which are out of range for
    // Timestamp(TimeUnit::Nanoseconds) and will cause a panic in dev builds/overflow in release builds
    // However, the code should work for the Microsecond/Millisecond time units
    for time_unit in [
        polars_arrow::datatypes::TimeUnit::Microsecond,
        polars_arrow::datatypes::TimeUnit::Millisecond,
        polars_arrow::datatypes::TimeUnit::Second,
    ] {
        parse(time_unit).expect("Should not error");
    }
    std::panic::catch_unwind(|| parse(polars_arrow::datatypes::TimeUnit::Nanosecond))
        .expect_err("Should be a panic error");

    Ok(())
}
