use std::path::PathBuf;

use polars_core::cloud::CloudOptions;

use super::*;

#[allow(dead_code)]
pub struct ParquetExec {
    path: PathBuf,
    schema: SchemaRef,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    options: ParquetOptions,
    cloud_options: Option<CloudOptions>,
    file_options: FileScanOptions,
}

impl ParquetExec {
    pub(crate) fn new(
        path: PathBuf,
        schema: SchemaRef,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        options: ParquetOptions,
        cloud_options: Option<CloudOptions>,
        file_options: FileScanOptions,
    ) -> Self {
        ParquetExec {
            path,
            schema,
            predicate,
            options,
            cloud_options,
            file_options,
        }
    }

    fn read(&mut self) -> PolarsResult<DataFrame> {
        let (file, projection, n_rows, predicate) = prepare_scan_args(
            &self.path,
            &self.predicate,
            &mut self.file_options.with_columns,
            &mut self.schema,
            self.file_options.n_rows,
            self.file_options.row_count.is_some(),
        );

        ParquetReader::new(file)
            .with_n_rows(n_rows)
            .read_parallel(self.options.parallel)
            .with_row_count(mem::take(&mut self.file_options.row_count))
            .set_rechunk(self.file_options.rechunk)
            .set_low_memory(self.options.low_memory)
            .use_statistics(self.options.use_statistics)
            ._finish_with_scan_ops(predicate, projection.as_ref().map(|v| v.as_ref()))
    }
}

impl Executor for ParquetExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let finger_print = FileFingerPrint {
            path: self.path.clone(),
            predicate: self
                .predicate
                .as_ref()
                .map(|ae| ae.as_expression().unwrap().clone()),
            slice: (0, self.file_options.n_rows),
        };

        let profile_name = if state.has_node_timer() {
            let mut ids = vec![self.path.to_string_lossy().into()];
            if self.predicate.is_some() {
                ids.push("predicate".into())
            }
            let name = comma_delimited("parquet".to_string(), &ids);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        state.record(
            || {
                state
                    .file_cache
                    .read(finger_print, self.file_options.file_counter, &mut || {
                        self.read()
                    })
            },
            profile_name,
        )
    }
}
