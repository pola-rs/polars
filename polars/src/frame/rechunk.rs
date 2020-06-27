use super::DfColumns;
use crate::prelude::*;

/// Used to create the same chunksizes in a DataFrame
/// Only aggregate smaller chunks to larger ones
struct ReChunker<'a> {
    columns: &'a DfColumns,
    /// Number of chunks in the column with the least chunks. Optimal chunking is: `min_chunks = 1`
    min_chunks: usize,
    /// The indexes of the columns selected for rechunking
    to_rechunk: Vec<usize>,
}

impl<'a> ReChunker<'a> {
    fn new(columns: &'a DfColumns) -> Result<Self> {
        let chunk_lens = columns.iter().map(|s| s.n_chunks()).collect::<Vec<_>>();

        let min_chunks = *chunk_lens.iter().min().ok_or(PolarsError::NoData)?;

        let to_rechunk = chunk_lens
            .into_iter()
            .enumerate()
            .filter_map(|(idx, len)| if len > min_chunks { Some(idx) } else { None })
            .collect::<Vec<_>>();

        Ok(ReChunker {
            columns,
            min_chunks,
            to_rechunk,
        })
    }

    fn rechunk(&self) {
        unimplemented!()
    }
}
