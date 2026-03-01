use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, BinaryViewArray, Utf8Array, Utf8ViewArray};
use arrow::datatypes::{ArrowSchema, Field};
use arrow::record_batch::RecordBatchT;
use polars_buffer::Buffer;
use polars_error::PolarsResult;
use polars_parquet::arrow::write::{FileWriter, WriteOptions};
use polars_parquet::read::read_metadata;
use polars_parquet::write::{
    CompressionOptions, Encoding, RowGroupIterator, StatisticsOptions, Version,
};

/// Write an array to a parquet file in memory and return the raw bytes.
fn write_parquet_bytes(
    array: &ArrayRef,
    options: WriteOptions,
    encodings: Vec<Encoding>,
) -> PolarsResult<Vec<u8>> {
    let field = Field::new("a1".into(), array.dtype().clone(), true);
    let schema = ArrowSchema::from_iter([field]);

    let iter = vec![RecordBatchT::try_new(
        array.len(),
        Arc::new(schema.clone()),
        vec![array.clone()],
    )];

    let row_groups = RowGroupIterator::try_new(
        iter.into_iter(),
        &schema,
        options,
        Buffer::from_iter([encodings]),
    )?;

    let writer = Cursor::new(vec![]);
    let mut writer = FileWriter::try_new(writer, schema, options)?;

    for group in row_groups {
        writer.write(u64::MAX, group?)?;
    }
    writer.end(None)?;

    Ok(writer.into_inner().into_inner())
}

/// Write `array` with the given `truncate_length`, read back column chunk
/// statistics, and assert that min/max match `expected_min`/`expected_max`.
fn assert_truncated_statistics(
    array: &ArrayRef,
    truncate_length: Option<usize>,
    expected_min: Option<&[u8]>,
    expected_max: Option<&[u8]>,
) {
    let mut stats_options = StatisticsOptions::full();
    stats_options.statistics_truncate_length = truncate_length;

    let options = WriteOptions {
        statistics: stats_options,
        compression: CompressionOptions::Uncompressed,
        version: Version::V1,
        data_page_size: None,
    };

    let data = write_parquet_bytes(array, options, vec![Encoding::Plain]).unwrap();

    let mut reader = Cursor::new(data);
    let md = read_metadata(&mut reader).unwrap();

    let col = &md.row_groups[0].parquet_columns()[0];
    let stats = col.statistics().unwrap().unwrap();
    let binary_stats = stats.expect_as_binary();

    assert_eq!(binary_stats.min_value.as_deref(), expected_min);
    assert_eq!(binary_stats.max_value.as_deref(), expected_max);
}

/// Test statistics truncation and disabled truncation for a given array type.
/// The value "Blart" (20 bytes) is truncated to 2 bytes.
fn test_truncation_for_array(array: &ArrayRef) {
    // Truncated to 2 bytes:
    //   min = "Bl"
    //   max = "Bm" ('l' 0x6C incremented to 'm' 0x6D)
    assert_truncated_statistics(array, Some(2), Some(b"Bl"), Some(b"Bm"));

    // Truncation disabled: full values preserved.
    assert_truncated_statistics(
        array,
        None,
        Some(b"Blart"),
        Some(b"Blart"),
    );
}

// --- Per-type truncation tests -----------------------------------------------

#[test]
fn statistics_truncation_utf8view() {
    let array = Utf8ViewArray::from_slice([Some("Blart")]);
    test_truncation_for_array(&array.boxed());
}

#[test]
fn statistics_truncation_binaryview() {
    let array = BinaryViewArray::from_slice([Some(b"Blart".as_slice())]);
    test_truncation_for_array(&array.boxed());
}

#[test]
fn statistics_truncation_large_binary() {
    let array: BinaryArray<i64> =
        [Some(b"Blart".as_slice())].into_iter().collect();
    test_truncation_for_array(&array.boxed());
}

#[test]
fn statistics_truncation_large_utf8() {
    let array = Utf8Array::<i64>::from_slice(["Blart"]);
    test_truncation_for_array(&array.boxed());
}

// --- Large-value file size test ----------------------------------------------

#[test]
fn large_binary_file_size_with_truncation() -> PolarsResult<()> {
    // Single row with 16 MiB of 0x42 in a binary column.
    let big_value = vec![0x42u8; 16 * 1024 * 1024];
    let array = BinaryViewArray::from_slice([Some(big_value.as_slice())]);

    // Default options include statistics_truncate_length: Some(64).
    let options = WriteOptions {
        statistics: StatisticsOptions::default(),
        compression: CompressionOptions::Uncompressed,
        version: Version::V1,
        data_page_size: None,
    };

    let data = write_parquet_bytes(&array.boxed(), options, vec![Encoding::Plain])?;

    // Without truncation this would be very large (untruncated stats in page header AND
    // column chunk metadata). With truncation the file should be only slightly
    // larger than 16 MiB (the data itself, as we're not compression).
    assert!(
        data.len() < 17 * 1024 * 1024,
        "File size {} bytes is too large; stats truncation may not be working",
        data.len()
    );

    // Verify statistics values.
    let mut reader = Cursor::new(&data);
    let md = read_metadata(&mut reader)?;

    let col = &md.row_groups[0].parquet_columns()[0];
    let stats = col.statistics().unwrap()?;
    let binary_stats = stats.expect_as_binary();

    // Min: truncated to 64 bytes of 0x42.
    let expected_min = vec![0x42u8; 64];
    assert_eq!(
        binary_stats.min_value.as_deref(),
        Some(expected_min.as_slice())
    );

    // Max: truncated prefix with the last byte incremented: 0x42 -> 0x43.
    let mut expected_max = vec![0x42u8; 64];
    *expected_max.last_mut().unwrap() = 0x43;
    assert_eq!(
        binary_stats.max_value.as_deref(),
        Some(expected_max.as_slice())
    );

    Ok(())
}
