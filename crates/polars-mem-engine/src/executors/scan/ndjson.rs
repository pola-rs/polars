use std::path::PathBuf;

use polars_core::utils::accumulate_dataframes_vertical;

use super::*;

pub struct JsonExec {
    paths: Arc<[PathBuf]>,
    options: NDJsonReadOptions,
    file_scan_options: FileScanOptions,
    file_info: FileInfo,
    predicate: Option<Arc<dyn PhysicalExpr>>,
}

impl JsonExec {
    pub fn new(
        paths: Arc<[PathBuf]>,
        options: NDJsonReadOptions,
        file_scan_options: FileScanOptions,
        file_info: FileInfo,
        predicate: Option<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        Self {
            paths,
            options,
            file_scan_options,
            file_info,
            predicate,
        }
    }

    fn read(&mut self) -> PolarsResult<DataFrame> {
        let schema = self
            .file_info
            .reader_schema
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap_right();

        let mut n_rows = self.file_scan_options.n_rows;

        let dfs = self
            .paths
            .iter()
            .map_while(|p| {
                if n_rows == Some(0) {
                    return None;
                }

                let reader = match JsonLineReader::from_path(p) {
                    Ok(r) => r,
                    Err(e) => return Some(Err(e)),
                };

                let row_index = self.file_scan_options.row_index.as_mut();

                let df = reader
                    .with_schema(schema.clone())
                    .with_rechunk(self.file_scan_options.rechunk)
                    .with_chunk_size(Some(self.options.chunk_size))
                    .with_row_index(row_index)
                    .with_predicate(self.predicate.clone().map(phys_expr_to_io_expr))
                    .with_projection(self.file_scan_options.with_columns.clone())
                    .low_memory(self.options.low_memory)
                    .with_n_rows(n_rows)
                    .with_ignore_errors(self.options.ignore_errors)
                    .finish();

                let df = match df {
                    Ok(df) => df,
                    Err(e) => return Some(Err(e)),
                };

                if let Some(ref mut n_rows) = n_rows {
                    *n_rows -= df.height();
                }

                Some(Ok(df))
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        accumulate_dataframes_vertical(dfs)
    }
}

impl Executor for JsonExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let profile_name = if state.has_node_timer() {
            let ids = vec![self.paths[0].to_string_lossy().into()];
            let name = comma_delimited("ndjson".to_string(), &ids);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        state.record(|| self.read(), profile_name)
    }
}
