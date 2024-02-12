use futures::StreamExt;
use polars_arrow::error::Result;
use polars_arrow::io::ipc::read::file_async::*;
use tokio::fs::File;
use tokio_util::compat::*;

use crate::io::ipc::common::read_gzip_json;

async fn test_file(version: &str, file_name: &str) -> Result<()> {
    let testdata = crate::test_util::arrow_test_data();
    let mut file = File::open(format!(
        "{testdata}/arrow-ipc-stream/integration/{version}/{file_name}.arrow_file"
    ))
    .await?
    .compat();

    let metadata = read_file_metadata_async(&mut file).await?;
    let mut reader = FileStream::new(file, metadata, None, None);

    // read expected JSON output
    let (schema, ipc_fields, batches) = read_gzip_json(version, file_name)?;

    assert_eq!(&schema, &reader.metadata().schema);
    assert_eq!(&ipc_fields, &reader.metadata().ipc_schema.fields);

    let mut items = vec![];
    while let Some(item) = reader.next().await {
        items.push(item?)
    }

    batches
        .iter()
        .zip(items.into_iter())
        .for_each(|(lhs, rhs)| {
            assert_eq!(lhs, &rhs);
        });
    Ok(())
}

#[tokio::test]
async fn write_async() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_primitive").await
}
