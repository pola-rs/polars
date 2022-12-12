use async_trait::async_trait;

use super::*;

pub struct ParquetExecAsync {
    path: PathBuf,
    schema: SchemaRef,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    options: ParquetOptions,
    async_options: bool,
}

impl ParquetExecAsync {
    pub(crate) fn new(
        path: PathBuf,
        schema: SchemaRef,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        options: ParquetOptions,
        async_options: bool,
    ) -> Self {
        ParquetExecAsync {
            path,
            schema,
            predicate,
            options,
            async_options,
        }
    }

    async fn read(&mut self) -> PolarsResult<DataFrame> {
        let (projection, n_rows, predicate) = prepare_generic_scan_args(
            &self.predicate,
            &mut self.options.with_columns,
            &mut self.schema,
            self.options.n_rows,
        );

        {
            ParquetAsyncReader::from_uri("file")?
                .with_n_rows(n_rows)
                .with_row_count(std::mem::take(&mut self.options.row_count))
                .set_rechunk(self.options.rechunk)
                .set_low_memory(self.options.low_memory)
                ._finish_with_scan_ops(predicate, projection.as_ref().map(|v| v.as_ref()))
                .await
        }
    }
}

#[async_trait]
impl AsyncExecutor for ParquetExecAsync {
    async fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        // --- caching not ported yet for async execution.
        // let finger_print = FileFingerPrint {
        //     path: self.path.clone(),
        //     predicate: self
        //         .predicate
        //         .as_ref()
        //         .map(|ae| ae.as_expression().unwrap().clone()),
        //     slice: (0, self.options.n_rows),
        // };

        // let profile_name = if state.has_node_timer() {
        //     let mut ids = vec![self.path.to_string_lossy().to_string()];
        //     if self.predicate.is_some() {
        //         ids.push("predicate".to_string())
        //     }
        //     let name = column_delimited("parquet".to_string(), &ids);
        //     Cow::Owned(name)
        // } else {
        //     Cow::Borrowed("")
        // };

        // state.record(
        //     || {
        //         state
        //             .file_cache
        //             .read(finger_print, self.options.file_counter, &mut || self.read())
        //     },
        //     profile_name,
        // )
        self.read().await
    }
}
