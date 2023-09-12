use awscreds::Credentials;
use cloud::AmazonS3ConfigKey as Key;
use polars::prelude::*;

// Login to your aws account and then copy the ../datasets/foods1.parquet file to your own bucket.
// Adjust the link below.
const TEST_S3_LOCATION: &str = "s3://polarstesting/polars_write_example_cloud.parquet";

fn main() -> PolarsResult<()> {
    sink_file();
    sink_cloud_local();
    sink_aws();

    Ok(())
}

fn sink_file() {
    let df = example_dataframe();

    // Writing to a local file:
    let path = "/tmp/polars_write_example.parquet".into();
    df.lazy().sink_parquet(path, Default::default()).unwrap();
}

fn sink_cloud_local() {
    let df = example_dataframe();

    // Writing to a location that might be in the cloud:
    let uri = "file:///tmp/polars_write_example_cloud.parquet".to_string();
    df.lazy()
        .sink_parquet_cloud(uri, None, Default::default())
        .unwrap();
}

fn sink_aws() {
    let cred = Credentials::default().unwrap();

    // Propagate the credentials and other cloud options.
    let cloud_options = cloud::CloudOptions::default().with_aws([
        (Key::AccessKeyId, &cred.access_key.unwrap()),
        (Key::SecretAccessKey, &cred.secret_key.unwrap()),
        (Key::Region, &"eu-central-1".into()),
    ]);
    let cloud_options = Some(cloud_options);

    let df = example_dataframe();

    df.lazy()
        .sink_parquet_cloud(
            TEST_S3_LOCATION.to_string(),
            cloud_options,
            Default::default(),
        )
        .unwrap();
}

fn example_dataframe() -> DataFrame {
    df!(
        "foo" => &[1, 2, 3],
        "bar" => &[None, Some("bak"), Some("baz")],
    )
    .unwrap()
}
