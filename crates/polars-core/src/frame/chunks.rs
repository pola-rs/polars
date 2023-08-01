use arrow::chunk::Chunk;

use crate::prelude::*;

pub type ArrowChunk = Chunk<ArrayRef>;

impl std::convert::TryFrom<(ArrowChunk, &[ArrowField])> for DataFrame {
    type Error = PolarsError;

    fn try_from(arg: (ArrowChunk, &[ArrowField])) -> PolarsResult<DataFrame> {
        let columns: PolarsResult<Vec<Series>> = arg
            .0
            .columns()
            .iter()
            .zip(arg.1)
            .map(|(arr, field)| Series::try_from((field.name.as_ref(), arr.clone())))
            .collect();

        DataFrame::new(columns?)
    }
}
