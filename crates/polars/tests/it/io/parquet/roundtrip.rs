use std::io::Cursor;
use std::sync::Arc;

use polars_arrow::array::{
    ArrayRef, Int32Array, ListArray, MutableBinaryViewArray, MutableListArray, TryExtend,
    Utf8Array, Utf8ViewArray,
};
use polars_arrow::datatypes::{ArrowSchema, Field};
use polars_arrow::record_batch::RecordBatchT;
use polars_buffer::Buffer;
use polars_error::PolarsResult;
use polars_parquet::arrow::write::{FileWriter, WriteOptions};
use polars_parquet::read::read_metadata;
use polars_parquet::write::{
    CompressionOptions, Encoding, RowGroupIterator, StatisticsOptions, Version,
};

use crate::io::parquet::read::file::FileReader;

fn round_trip(
    array: &ArrayRef,
    version: Version,
    compression: CompressionOptions,
    encodings: Vec<Encoding>,
) -> PolarsResult<()> {
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

    let row_groups = RowGroupIterator::try_new(
        iter.into_iter(),
        &schema,
        options,
        Buffer::from_iter([encodings]),
    )?;

    let writer = Cursor::new(vec![]);
    let mut writer = FileWriter::try_new(writer, schema.clone(), options, None)?;

    for group in row_groups {
        writer.write(u64::MAX, group?)?;
    }
    writer.end(None)?;

    let data = writer.into_inner().into_inner();

    let mut reader = Cursor::new(data);
    let md = read_metadata(&mut reader).unwrap();
    // say we found that we only need to read the first two row groups, "0" and "1"
    let row_groups = md
        .row_groups
        .into_iter()
        .enumerate()
        .filter(|(index, _)| *index == 0 || *index == 1)
        .map(|(_, row_group)| row_group)
        .collect();

    // we can then read the row groups into chunks
    let chunks = FileReader::new(reader, row_groups, schema, None);

    let mut arrays = vec![];
    for chunk in chunks {
        let chunk = chunk?;
        arrays.push(chunk.first().unwrap().clone())
    }
    assert_eq!(arrays.len(), 1);

    assert_eq!(array.as_ref(), arrays[0].as_ref());
    Ok(())
}

#[test]
fn roundtrip_binview() -> PolarsResult<()> {
    let array = Utf8ViewArray::from_slice([Some("foo"), Some("bar"), None, Some("hamlet")]);

    round_trip(
        &array.boxed(),
        Version::V1,
        CompressionOptions::Uncompressed,
        vec![Encoding::Plain],
    )
}

/// Writes `array` as one column with `encoding`, and returns its size statistics.
fn unencoded_bytes(array: ArrayRef, encoding: Encoding, version: Version) -> Option<i64> {
    let schema = ArrowSchema::from_iter([Field::new("a".into(), array.dtype().clone(), true)]);
    let options = WriteOptions {
        statistics: StatisticsOptions::full(),
        compression: CompressionOptions::Uncompressed,
        version,
        // Small pages, so that the size adds up over several.
        data_page_size: Some(16),
    };
    let batch = RecordBatchT::try_new(array.len(), Arc::new(schema.clone()), vec![array]);
    let row_groups = RowGroupIterator::try_new(
        [batch].into_iter(),
        &schema,
        options,
        Buffer::from_iter([vec![encoding]]),
    )
    .unwrap();
    let mut writer = FileWriter::try_new(Cursor::new(vec![]), schema, options, None).unwrap();
    for group in row_groups {
        writer.write(u64::MAX, group.unwrap()).unwrap();
    }
    writer.end(None).unwrap();

    let md = read_metadata(&mut Cursor::new(writer.into_inner().into_inner())).unwrap();
    md.row_groups[0].parquet_columns()[0]
        .unencoded_byte_array_data_bytes()
        .unwrap()
}

#[test]
fn size_statistics_strings() {
    let values = [Some("foo"), Some("foo"), None, Some("hamlet"), Some("foo")];
    let view = Utf8ViewArray::from_slice(values).boxed();
    let large = Utf8Array::<i64>::from(values).boxed();
    for version in [Version::V1, Version::V2] {
        for encoding in [
            Encoding::Plain,
            Encoding::DeltaLengthByteArray,
            Encoding::RleDictionary,
        ] {
            assert_eq!(unencoded_bytes(view.clone(), encoding, version), Some(15));
            assert_eq!(unencoded_bytes(large.clone(), encoding, version), Some(15));
        }
    }
}

#[test]
fn size_statistics_nested_strings() {
    let mut list = MutableListArray::<i64, MutableBinaryViewArray<str>>::new();
    list.try_extend([
        Some(vec![Some("foo"), Some("bar")]),
        None,
        Some(vec![Some("hamlet"), None]),
    ])
    .unwrap();
    let list: ListArray<i64> = list.into();
    assert_eq!(
        unencoded_bytes(list.boxed(), Encoding::Plain, Version::V1),
        Some(12)
    );
}

#[test]
fn size_statistics_not_byte_array() {
    let ints = Int32Array::from([Some(1), None, Some(3)]).boxed();
    assert_eq!(unencoded_bytes(ints, Encoding::Plain, Version::V1), None);
}
