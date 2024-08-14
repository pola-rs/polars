use std::path::PathBuf;

use polars_parquet::arrow::read::*;

use super::*;
use crate::io::parquet::read::file::FileReader;
#[cfg(feature = "parquet")]
#[test]
fn all_types() -> PolarsResult<()> {
    use crate::io::parquet::read::file::FileReader;

    let dir = env!("CARGO_MANIFEST_DIR");
    let path = PathBuf::from(dir).join("../../docs/data/alltypes_plain.parquet");

    let mut reader = std::fs::File::open(path)?;

    let metadata = read_metadata(&mut reader)?;
    let schema = infer_schema(&metadata)?;
    let reader = FileReader::new(reader, metadata.row_groups, schema, None);

    let batches = reader.collect::<PolarsResult<Vec<_>>>()?;
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
        .downcast_ref::<BinaryViewArray>()
        .unwrap();
    assert_eq!(
        result,
        &BinaryViewArray::from_slice_values([[48], [49], [48], [49], [48], [49], [48], [49]])
    );

    Ok(())
}

#[cfg(feature = "parquet")]
#[test]
fn all_types_chunked() -> PolarsResult<()> {
    // this has one batch with 8 elements

    use crate::io::parquet::read::file::FileReader;
    let dir = env!("CARGO_MANIFEST_DIR");
    let path = PathBuf::from(dir).join("../../docs/data/alltypes_plain.parquet");
    let mut reader = std::fs::File::open(path)?;

    let metadata = read_metadata(&mut reader)?;
    let schema = infer_schema(&metadata)?;
    // chunk it in 5 (so, (5,3))
    let reader = FileReader::new(reader, metadata.row_groups, schema, None);

    let batches = reader.collect::<PolarsResult<Vec<_>>>()?;
    assert_eq!(batches.len(), 1);

    assert_eq!(batches[0].len(), 8);

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
        .downcast_ref::<BinaryViewArray>()
        .unwrap();
    assert_eq!(
        result,
        &BinaryViewArray::from_slice_values([[48], [49], [48], [49], [48], [49], [48], [49]])
    );

    Ok(())
}

#[test]
fn read_int96_timestamps() -> PolarsResult<()> {
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
        let schema = arrow::datatypes::ArrowSchema {
            fields: vec![arrow::datatypes::Field::new(
                "timestamps",
                arrow::datatypes::ArrowDataType::Timestamp(time_unit, None),
                false,
            )],
            metadata: BTreeMap::new(),
        };
        let reader = FileReader::new(reader, metadata.row_groups, schema, None);
        reader.collect::<PolarsResult<Vec<_>>>()
    };

    // This data contains int96 timestamps in the year 1000 and 3000, which are out of range for
    // Timestamp(TimeUnit::Nanoseconds) and will cause a panic in dev builds/overflow in release builds
    // However, the code should work for the Microsecond/Millisecond time units
    for time_unit in [
        arrow::datatypes::TimeUnit::Microsecond,
        arrow::datatypes::TimeUnit::Millisecond,
        arrow::datatypes::TimeUnit::Second,
    ] {
        parse(time_unit).expect("Should not error");
    }
    std::panic::catch_unwind(|| parse(arrow::datatypes::TimeUnit::Nanosecond))
        .expect_err("Should be a panic error");

    Ok(())
}
