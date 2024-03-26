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
