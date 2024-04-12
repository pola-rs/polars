use arrow::record_batch::RecordBatch;

use crate::prelude::*;

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
}
