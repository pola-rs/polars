use std::env;

use awscreds::Credentials;
use polars::prelude::*;

// Login to your aws account and then copy the ../datasets/foods1.parquet file to your own bucket.
// Adjust the link below.
const TEST_S3: &str = "s3://lov2test/polars/datasets/*.parquet";

fn main() -> PolarsResult<()> {
    let cred = Credentials::default().unwrap();
    env::set_var("AWS_SECRET_ACCESS_KEY", cred.secret_key.unwrap());
    env::set_var("AWS_DEFAULT_REGION", "us-west-2");
    env::set_var("AWS_ACCESS_KEY_ID", cred.access_key.unwrap());

    let df = LazyFrame::scan_parquet(TEST_S3, ScanArgsParquet::default())?
        .with_streaming(true)
        .select([
            // select all columns
            all(),
        ])
        .collect()?;

    dbg!(df);
    Ok(())
}
