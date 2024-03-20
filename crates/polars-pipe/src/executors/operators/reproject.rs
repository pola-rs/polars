use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::schema::Schema;

use crate::operators::DataChunk;

pub(crate) fn reproject_chunk(
    chunk: &mut DataChunk,
    positions: &mut Vec<usize>,
    schema: &Schema,
) -> PolarsResult<()> {
    let out = if positions.is_empty() {
        // use the chunk schema to cache
        // the positions for subsequent calls
        let chunk_schema = chunk.data.schema();

        let out = chunk
            .data
            .select_with_schema_unchecked(schema.iter_names(), &chunk_schema)?;

        *positions = out
            .get_columns()
            .iter()
            .map(|s| chunk_schema.get_full(s.name()).unwrap().0)
            .collect();
        out
    } else {
        let columns = chunk.data.get_columns();
        let cols = positions.iter().map(|i| columns[*i].clone()).collect();
        unsafe { DataFrame::new_no_checks(cols) }
    };
    *chunk = chunk.with_data(out);
    Ok(())
}
