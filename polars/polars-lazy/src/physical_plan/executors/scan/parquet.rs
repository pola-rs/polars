use super::*;
use crate::prelude::file_caching::FileFingerPrint;

pub struct ParquetExec {
    path: PathBuf,
    schema: SchemaRef,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    aggregate: Vec<ScanAggregation>,
    options: ParquetOptions,
}

impl ParquetExec {
    pub(crate) fn new(
        path: PathBuf,
        schema: SchemaRef,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        aggregate: Vec<ScanAggregation>,
        options: ParquetOptions,
    ) -> Self {
        ParquetExec {
            path,
            schema,
            predicate,
            aggregate,
            options,
        }
    }

    fn read(&mut self) -> Result<DataFrame> {
        let (file, projection, n_rows, aggregate, predicate) = prepare_scan_args(
            &self.path,
            &self.predicate,
            &mut self.options.with_columns,
            &mut self.schema,
            self.options.n_rows,
            &self.aggregate,
        );

        ParquetReader::new(file)
            .with_n_rows(n_rows)
            .read_parallel(self.options.parallel)
            .with_row_count(std::mem::take(&mut self.options.row_count))
            .set_rechunk(self.options.rechunk)
            .set_low_memory(self.options.low_memory)
            ._finish_with_scan_ops(
                predicate,
                aggregate,
                projection.as_ref().map(|v| v.as_ref()),
            )
    }
}

impl Executor for ParquetExec {
    fn execute(&mut self, state: &mut ExecutionState) -> Result<DataFrame> {
        let finger_print = FileFingerPrint {
            path: self.path.clone(),
            predicate: self.predicate.as_ref().map(|ae| ae.as_expression().clone()),
            slice: (0, self.options.n_rows),
        };
        state
            .file_cache
            .read(finger_print, self.options.file_counter, &mut || self.read())
    }
}
