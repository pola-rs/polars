use polars_arrow::array::*;
use polars_arrow::chunk::Chunk;
use polars_arrow::datatypes::*;
use polars_arrow::error::Result;
use polars_arrow::io::ipc::read;
use polars_arrow::io::ipc::write::{FileWriter, WriteOptions};

use super::file::write;

#[test]
fn basic() -> Result<()> {
    // prepare some data
    let array = BooleanArray::from([Some(true), Some(false), None, Some(true)]).boxed();
    let schema = Schema::from(vec![Field::new("a", array.data_type().clone(), true)]);
    let columns = Chunk::try_new(vec![array])?;

    let (expected_schema, expected_batches) = (schema.clone(), vec![columns.clone()]);

    // write to a file
    let result = write(&expected_batches, &schema, None, None)?;

    // read the file to append
    let mut file = std::io::Cursor::new(result);
    let metadata = read::read_file_metadata(&mut file)?;
    let mut writer = FileWriter::try_from_file(file, metadata, WriteOptions { compression: None })?;

    // write a new column
    writer.write(&columns, None)?;
    writer.finish()?;

    let data = writer.into_inner();
    let mut reader = std::io::Cursor::new(data.into_inner());

    // read the file again and confirm that it contains both messages
    let metadata = read::read_file_metadata(&mut reader)?;
    assert_eq!(schema, expected_schema);
    let reader = read::FileReader::new(reader, metadata, None, None);

    let chunks = reader.collect::<Result<Vec<_>>>()?;

    assert_eq!(chunks, vec![columns.clone(), columns]);

    Ok(())
}
