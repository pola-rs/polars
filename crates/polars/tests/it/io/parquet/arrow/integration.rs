use arrow2::error::Result;

use super::{integration_read, integration_write};
use crate::io::ipc::read_gzip_json;

fn test_file(version: &str, file_name: &str) -> Result<()> {
    let (schema, _, batches) = read_gzip_json(version, file_name)?;

    // empty batches are not written/read from parquet and can be ignored
    let batches = batches
        .into_iter()
        .filter(|x| !x.is_empty())
        .collect::<Vec<_>>();

    let data = integration_write(&schema, &batches)?;

    let (read_schema, read_batches) = integration_read(&data, None)?;

    assert_eq!(schema, read_schema);
    assert_eq!(batches, read_batches);

    Ok(())
}

#[test]
fn roundtrip_100_primitive() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_primitive")?;
    test_file("1.0.0-bigendian", "generated_primitive")
}

#[test]
fn roundtrip_100_dict() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_dictionary")?;
    test_file("1.0.0-bigendian", "generated_dictionary")
}

#[test]
fn roundtrip_100_extension() -> Result<()> {
    test_file("1.0.0-littleendian", "generated_extension")?;
    test_file("1.0.0-bigendian", "generated_extension")
}
