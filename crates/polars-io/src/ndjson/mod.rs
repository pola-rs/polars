use core::{get_file_chunks_json, json_lines};
use std::num::NonZeroUsize;

use arrow::array::StructArray;
use polars_core::POOL;
use polars_core::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub(crate) mod buffer;
pub mod core;

pub fn infer_schema<R: std::io::BufRead>(
    reader: &mut R,
    infer_schema_len: Option<NonZeroUsize>,
) -> PolarsResult<Schema> {
    let dtypes = polars_json::ndjson::iter_unique_dtypes(reader, infer_schema_len)?;
    let dtype =
        crate::json::infer::dtypes_to_supertype(dtypes.map(|dt| DataType::from_arrow_dtype(&dt)))?;
    let schema = StructArray::get_fields(&dtype.to_arrow(CompatLevel::newest()))
        .iter()
        .map(Into::<Field>::into)
        .collect();
    Ok(schema)
}

/// Count the number of rows. The slice passed must represent the entire file. This will
/// potentially parallelize using rayon.
///
/// This does not check if the lines are valid NDJSON - it assumes that is the case.
pub fn count_rows_par(full_bytes: &[u8], n_threads: Option<usize>) -> usize {
    let n_threads = n_threads.unwrap_or(POOL.current_num_threads());
    let file_chunks = get_file_chunks_json(full_bytes, n_threads);

    if file_chunks.len() == 1 {
        count_rows(full_bytes)
    } else {
        let iter = file_chunks
            .into_par_iter()
            .map(|(start_pos, stop_at_nbytes)| count_rows(&full_bytes[start_pos..stop_at_nbytes]));

        POOL.install(|| iter.sum())
    }
}

/// Count the number of rows. The slice passed must represent the entire file.
/// This does not check if the lines are valid NDJSON - it assumes that is the case.
pub fn count_rows(full_bytes: &[u8]) -> usize {
    json_lines(full_bytes).count()
}
