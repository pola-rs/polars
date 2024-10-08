use arrow::record_batch::RecordBatch;
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::_split_offsets;
use crate::POOL;

impl TryFrom<(RecordBatch, &ArrowSchema)> for DataFrame {
    type Error = PolarsError;

    fn try_from(arg: (RecordBatch, &ArrowSchema)) -> PolarsResult<DataFrame> {
        let columns: PolarsResult<Vec<Column>> = arg
            .0
            .columns()
            .iter()
            .zip(arg.1.iter_values())
            .map(|(arr, field)| Series::try_from((field, arr.clone())).map(Column::from))
            .collect();

        DataFrame::new(columns?)
    }
}

impl DataFrame {
    pub fn split_chunks(&mut self) -> impl Iterator<Item = DataFrame> + '_ {
        self.align_chunks_par();

        (0..self.n_chunks()).map(move |i| unsafe {
            let columns = self
                .get_columns()
                .iter()
                .map(|column| column.as_materialized_series().select_chunk(i))
                .map(Column::from)
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
