mod read;
mod write;

use polars_arrow::array::*;
use polars_arrow::chunk::Chunk;
use polars_arrow::datatypes::ArrowSchema;
use polars_arrow::error::Result;
use polars_arrow::io::json::write as json_write;

fn write_batch(array: Box<dyn Array>) -> Result<Vec<u8>> {
    let mut serializer = json_write::Serializer::new(vec![Ok(array)].into_iter(), vec![]);

    let mut buf = vec![];
    json_write::write(&mut buf, &mut serializer)?;
    Ok(buf)
}

fn write_record_batch<A: AsRef<dyn Array>>(
    schema: ArrowSchema,
    chunk: Chunk<A>,
) -> Result<Vec<u8>> {
    let mut serializer = json_write::RecordSerializer::new(schema, &chunk, vec![]);

    let mut buf = vec![];
    json_write::write(&mut buf, &mut serializer)?;
    Ok(buf)
}
