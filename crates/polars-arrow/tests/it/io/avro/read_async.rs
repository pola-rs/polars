use avro_rs::Codec;
use futures::{pin_mut, StreamExt};
use polars_arrow::error::Result;
use polars_arrow::io::avro::avro_schema::read_async::{block_stream, read_metadata};
use polars_arrow::io::avro::read;

use super::read::{schema, write_avro};

async fn test(codec: Codec) -> Result<()> {
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
async fn read_without_codec() -> Result<()> {
    test(Codec::Null).await
}

#[tokio::test]
async fn read_deflate() -> Result<()> {
    test(Codec::Deflate).await
}

#[tokio::test]
async fn read_snappy() -> Result<()> {
    test(Codec::Snappy).await
}
