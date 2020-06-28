use super::DfColumns;
use crate::prelude::*;
use itertools::Itertools;

/// Used to create the same chunksizes in a DataFrame
/// Only aggregate smaller chunks to larger ones
pub(crate) struct ReChunker<'a> {
    columns: &'a mut DfColumns,
    /// Number of chunks in the column with the least chunks. Optimal chunking is: `min_chunks = 1`
    min_chunks: usize,
    /// The indexes of the columns selected for rechunking
    to_rechunk: Vec<usize>,
    // idx of a minimal chunked column
    argmin: usize,
}

impl<'a> ReChunker<'a> {
    pub(crate) fn new(columns: &'a mut DfColumns) -> Result<Self> {
        let chunk_lens = columns.iter().map(|s| s.n_chunks()).collect::<Vec<_>>();

        let argmin = chunk_lens
            .iter()
            .position_min()
            .ok_or(PolarsError::NoData)?;
        let min_chunks = chunk_lens[argmin];

        let to_rechunk = chunk_lens
            .into_iter()
            .enumerate()
            .filter_map(|(idx, len)| if len > min_chunks { Some(idx) } else { None })
            .collect::<Vec<_>>();

        Ok(ReChunker {
            columns,
            min_chunks,
            to_rechunk,
            argmin,
        })
    }

    pub(crate) fn rechunk(self) -> Result<()> {
        // clone shouldn't be too expensive as we expect the nr. of chunks to be close to 1.
        let chunk_id = self.columns[self.argmin].chunk_lengths().clone();

        for idx in self.to_rechunk {
            let col = &self.columns[idx];
            let new_col = col.rechunk(Some(&chunk_id))?;
            self.columns[idx] = new_col;
        }
        Ok(())
    }
}
