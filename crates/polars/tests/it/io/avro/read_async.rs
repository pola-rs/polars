use apache_avro::Codec;
use arrow::io::avro::avro_schema::read_async::{block_stream, read_metadata};
use arrow::io::avro::read;
use futures::{pin_mut, StreamExt};
use polars_error::PolarsResult;

use super::read::{schema, write_avro};

async fn test(codec: Codec) -> PolarsResult<()> {
    let avro_data = write_avro(codec).unwrap();
    let (_, expected_schema) = schema();

    let mut reader = &mut &avro_data[..];

    let metadata = read_metadata(&mut reader).await?;
    let schema = read::infer_schema(&metadata.record)?;

    assert_eq!(schema, expected_schema);

    let blocks = block_stream(&mut reader, metadata.marker).await;

    pin_mut!(blocks);
    while let Some(block) = blocks.next().await.transpose()? {
        assert!(block.number_of_rows > 0 || block.data.is_empty())
    }
    Ok(())
}

#[tokio::test]
async fn read_without_codec() -> PolarsResult<()> {
    test(Codec::Null).await
}

#[tokio::test]
async fn read_deflate() -> PolarsResult<()> {
    test(Codec::Deflate).await
}

#[tokio::test]
async fn read_snappy() -> PolarsResult<()> {
    test(Codec::Snappy).await
}
