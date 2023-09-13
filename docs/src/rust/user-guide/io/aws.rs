"""
# --8<-- [start:bucket]
use aws_sdk_s3::Region;

use aws_config::meta::region::RegionProviderChain;
use aws_sdk_s3::Client;
use std::borrow::Cow;

use polars::prelude::*;

#[tokio::main]
async fn main() {
    let bucket = "<YOUR_BUCKET>";
    let path = "<YOUR_PATH>";

    let config = aws_config::from_env().load().await;
    let client = Client::new(&config);

    let req = client.get_object().bucket(bucket).key(path);

    let res = req.clone().send().await.unwrap();
    let bytes = res.body.collect().await.unwrap();
    let bytes = bytes.into_bytes();

    let cursor = std::io::Cursor::new(bytes);

    let df = CsvReader::new(cursor).finish().unwrap();

    println!("{:?}", df);
}
# --8<-- [end:bucket]
"""
