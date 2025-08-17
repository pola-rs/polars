use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{ArrayRef, Utf8ViewArray};
use arrow::datatypes::{ArrowSchema, Field};
use arrow::record_batch::RecordBatchT;
use polars_error::PolarsResult;
use polars_parquet::arrow::write::{FileWriter, WriteOptions};
use polars_parquet::read::read_metadata;
use polars_parquet::write::{
    ColumnWriteOptions, CompressionOptions, Encoding, FieldWriteOptions, RowGroupIterator,
    StatisticsOptions, Version,
};

use crate::io::parquet::read::file::FileReader;

fn round_trip(
    array: &ArrayRef,
    version: Version,
    compression: CompressionOptions,
    column_options: Vec<ColumnWriteOptions>,
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

    let row_groups =
        RowGroupIterator::try_new(iter.into_iter(), &schema, options, column_options.clone())?;

    let writer = Cursor::new(vec![]);
    let mut writer = FileWriter::try_new(writer, schema.clone(), options, &column_options)?;

    for group in row_groups {
        writer.write(group?)?;
    }
    writer.end(None, &column_options)?;

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
        vec![
            FieldWriteOptions::default_with_encoding(Encoding::Plain)
                .into_default_column_write_options(),
        ],
    )
}
