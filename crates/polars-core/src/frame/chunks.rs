use arrow::record_batch::RecordBatch;
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::_split_offsets;
use crate::POOL;

pub type ArrowChunk = RecordBatch<ArrayRef>;

impl std::convert::TryFrom<(ArrowChunk, &[ArrowField])> for DataFrame {
    type Error = PolarsError;

    fn try_from(arg: (ArrowChunk, &[ArrowField])) -> PolarsResult<DataFrame> {
        let columns: PolarsResult<Vec<Series>> = arg
            .0
            .columns()
            .iter()
            .zip(arg.1)
            .map(|(arr, field)| Series::try_from((field, arr.clone())))
            .collect();

        DataFrame::new(columns?)
    }
}

impl DataFrame {
    pub fn split_chunks(mut self) -> impl Iterator<Item = DataFrame> {
        self.align_chunks();

        (0..self.n_chunks()).map(move |i| unsafe {
            let columns = self
                .get_columns()
                .iter()
                .map(|s| {
                    Series::from_chunks_and_dtype_unchecked(
                        s.name(),
                        vec![s.chunks()[i].clone()],
                        s.dtype(),
                    )
                })
                .collect::<Vec<_>>();

            DataFrame::new_no_checks(columns)
        })
    }

    pub fn split_chunks_by_n(self, n: usize, parallel: bool) -> Vec<DataFrame> {
        let split = _split_offsets(self.height(), n);

        let split_fn = |(offset, len)| self.slice(offset as i64, len);

        if parallel {
            // Parallel so that null_counts run in parallel
            POOL.install(|| split.into_par_iter().map(split_fn).collect())
        } else {
            split.into_iter().map(split_fn).collect()
        }
    }
}
