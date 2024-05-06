use arrow::array::*;
use arrow::datatypes::*;
use arrow::io::avro::write;
use arrow::record_batch::RecordBatchT;
use avro_schema::file::Compression;
use avro_schema::write_async::{write_block, write_metadata};
use polars_error::PolarsResult;

use super::read::read_avro;
use super::write::{data, schema, serialize_to_block};

async fn write_avro<R: AsRef<dyn Array>>(
    columns: &RecordBatchT<R>,
    schema: &ArrowSchema,
    compression: Option<Compression>,
) -> PolarsResult<Vec<u8>> {
    // usually done on a different thread pool
    let compressed_block = serialize_to_block(columns, schema, compression)?;

    let record = write::to_record(schema, "".to_string())?;
    let mut file = vec![];

    write_metadata(&mut file, record, compression).await?;

    write_block(&mut file, &compressed_block).await?;

    Ok(file)
}

async fn roundtrip(compression: Option<Compression>) -> PolarsResult<()> {
    let expected = data();
    let expected_schema = schema();

    let data = write_avro(&expected, &expected_schema, compression).await?;

    let (result, read_schema) = read_avro(&data, None)?;

    assert_eq!(expected_schema, read_schema);
    for (c1, c2) in result.columns().iter().zip(expected.columns().iter()) {
        assert_eq!(c1.as_ref(), c2.as_ref());
    }
    Ok(())
}

#[tokio::test]
async fn no_compression() -> PolarsResult<()> {
    roundtrip(None).await
}
