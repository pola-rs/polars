use cloud::AmazonS3ConfigKey as Key;
use polars::prelude::*;

const TEST_S3_LOCATION: &str = "s3://test-bucket/test-writes/polars_write_example_cloud.ipc";

fn main() -> PolarsResult<()> {
    // You can also use "awscreds::Credentials" to simplify the setup. See the "write_parquet_cloud" example.
    let cloud_options = cloud::CloudOptions::default().with_aws([
        (Key::AccessKeyId, "test".to_string()),
        (Key::SecretAccessKey, "test".to_string()),
        (Key::Endpoint, "http://localhost:4566".to_string()),
        (Key::Region, "us-east-1".to_string()),
    ]);
    let cloud_options = Some(cloud_options);

    let df = df!(
        "foo" => &[1, 2, 3],
        "bar" => &[None, Some("bak"), Some("baz")],
    )
    .unwrap();

    df.lazy()
        .sink_ipc_cloud(
            TEST_S3_LOCATION.to_string(),
            cloud_options,
            Default::default(),
        )
        .unwrap();

    Ok(())
}
