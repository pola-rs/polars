// --8<-- [start:read_parquet]
use aws_config::BehaviorVersion;
use polars::prelude::*;

#[tokio::main]
async fn main() {
    let bucket = "<YOUR_BUCKET>";
    let path = "<YOUR_PATH>";

    let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
    let client = aws_sdk_s3::Client::new(&config);

    let object = client
        .get_object()
        .bucket(bucket)
        .key(path)
        .send()
        .await
        .unwrap();

    let bytes = object.body.collect().await.unwrap().into_bytes();

    let cursor = std::io::Cursor::new(bytes);
    let df = CsvReader::new(cursor).finish().unwrap();

    println!("{:?}", df);
}
// --8<-- [end:read_parquet]

// --8<-- [start:scan_parquet]
// --8<-- [end:scan_parquet]

// --8<-- [start:scan_parquet_query]
// --8<-- [end:scan_parquet_query]

// --8<-- [start:scan_pyarrow_dataset]
// --8<-- [end:scan_pyarrow_dataset]

// --8<-- [start:write_parquet]
// --8<-- [end:write_parquet]
