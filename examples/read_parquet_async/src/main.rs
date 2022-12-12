use awscreds::Credentials;
use polars::prelude::*;
use tokio;
use std::env;

#[tokio::main]
async fn main() -> PolarsResult<()> {
    let cred = Credentials::default().unwrap();
    env::set_var("AWS_SECRET_ACCESS_KEY", cred.secret_key.unwrap());
    env::set_var("AWS_REGION", "us-east-2");
    env::set_var("AWS_ACCESS_KEY_ID", cred.access_key.unwrap());

    let df = LazyFrame::scan_parquet_async(
        "s3://lov2test/delta-rs/rust/tests/data/simple_table/part-00190-8ac0ae67-fb1d-461d-a3d3-8dc112766ff5-c000.snappy.parquet",
        ScanArgsParquet::default(),
    ).await?
    .select([
        // select all columns
        all(),
    ])
    .collect()?;

    dbg!(df);
    Ok(())
}
