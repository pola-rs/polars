use ahash::AHashMap;
use futures::future::BoxFuture;
use futures::io::Cursor;
use futures::SinkExt;
use polars_arrow::array::{Float32Array, Int32Array};
use polars_arrow::chunk::Chunk;
use polars_arrow::datatypes::{ArrowDataType, ArrowSchema, Field};
use polars_arrow::error::Result;
use polars_arrow::io::parquet::read::{
    infer_schema, read_columns_many_async, read_metadata_async, RowGroupDeserializer,
};
use polars_arrow::io::parquet::write::{CompressionOptions, Encoding, Version, WriteOptions};

use super::FileSink;

#[tokio::test]
async fn test_parquet_async_roundtrip() {
    let mut data = vec![];
    for i in 0..5 {
        let a1 = Int32Array::from(&[Some(i), None, Some(i + 1)]);
        let a2 = Float32Array::from(&[None, Some(i as f32), None]);
        let chunk = Chunk::new(vec![a1.boxed(), a2.boxed()]);
        data.push(chunk);
    }
    let schema = ArrowSchema::from(vec![
        Field::new("a1", ArrowDataType::Int32, true),
        Field::new("a2", ArrowDataType::Float32, true),
    ]);
    let encoding = vec![vec![Encoding::Plain], vec![Encoding::Plain]];
    let options = WriteOptions {
        write_statistics: true,
        compression: CompressionOptions::Uncompressed,
        version: Version::V2,
        data_pagesize_limit: None,
    };

    let mut buffer = Cursor::new(Vec::new());
    let mut sink = FileSink::try_new(&mut buffer, schema.clone(), encoding, options).unwrap();
    sink.metadata
        .insert(String::from("key"), Some("value".to_string()));
    for chunk in &data {
        sink.feed(chunk.clone()).await.unwrap();
    }
    sink.close().await.unwrap();
    drop(sink);

    buffer.set_position(0);
    let metadata = read_metadata_async(&mut buffer).await.unwrap();
    let kv = AHashMap::<String, Option<String>>::from_iter(
        metadata
            .key_value_metadata()
            .to_owned()
            .unwrap()
            .into_iter()
            .map(|kv| (kv.key, kv.value)),
    );
    assert_eq!(kv.get("key").unwrap(), &Some("value".to_string()));
    let read_schema = infer_schema(&metadata).unwrap();
    assert_eq!(read_schema, schema);
    let factory = || Box::pin(futures::future::ready(Ok(buffer.clone()))) as BoxFuture<_>;

    let mut out = vec![];
    for group in &metadata.row_groups {
        let column_chunks =
            read_columns_many_async(factory, group, schema.fields.clone(), None, None, None)
                .await
                .unwrap();
        let chunks = RowGroupDeserializer::new(column_chunks, group.num_rows(), None);
        let mut chunks = chunks.collect::<Result<Vec<_>>>().unwrap();
        out.append(&mut chunks);
    }

    for i in 0..5 {
        assert_eq!(data[i], out[i]);
    }
}
