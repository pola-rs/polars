use arrow::record_batch::RecordBatch;
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::_split_offsets;
use crate::POOL;

impl TryFrom<(RecordBatch, &[ArrowField])> for DataFrame {
    type Error = PolarsError;

    fn try_from(arg: (RecordBatch, &[ArrowField])) -> PolarsResult<DataFrame> {
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
    pub fn split_chunks(&mut self) -> impl Iterator<Item = DataFrame> + '_ {
        self.align_chunks();

        (0..self.n_chunks()).map(move |i| unsafe {
            let columns = self
                .get_columns()
                .iter()
                .map(|s| match s.dtype() {
                    #[cfg(feature = "dtype-struct")]
                    DataType::Struct(_) => {
                        let mut ca = s.struct_().unwrap().clone();
                        for field in ca.fields_mut().iter_mut() {
                            *field = field.replace_with_chunk(field.chunks()[i].clone())
                        }
                        ca.update_chunks(0);
                        ca.into_series()
                    },
                    _ => s.replace_with_chunk(s.chunks()[i].clone()),
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
