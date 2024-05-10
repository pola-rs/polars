use std::path::PathBuf;
use std::sync::Arc;

use super::*;

pub struct CsvExec {
    pub path: PathBuf,
    pub file_info: FileInfo,
    pub options: CsvReadOptions,
    pub file_options: FileScanOptions,
    pub predicate: Option<Arc<dyn PhysicalExpr>>,
}

impl CsvExec {
    fn read(&self) -> PolarsResult<DataFrame> {
        let with_columns = self
            .file_options
            .with_columns
            .clone()
            // Interpret selecting no columns as selecting all columns.
            .filter(|columns| !columns.is_empty());

        let n_rows = _set_n_rows_for_scan(self.file_options.n_rows);
        let predicate = self.predicate.clone().map(phys_expr_to_io_expr);

        self.options
            .clone()
            .with_skip_rows_after_header(
                // If we don't set it to 0 here, it will skip double the amount of rows.
                // But if we set it to 0, it will still skip the requested amount of rows.
                // The reason I currently cannot fathom.
                // TODO: Find out why. Maybe has something to do with schema inference.
                0,
            )
            .with_schema(Some(
                self.file_info.reader_schema.clone().unwrap().unwrap_right(),
            ))
            .with_n_rows(n_rows)
            .with_columns(with_columns)
            .with_rechunk(self.file_options.rechunk)
            .with_row_index(self.file_options.row_index.clone())
            .with_path(Some(self.path.clone()))
            .try_into_reader_with_file_path(None)?
            ._with_predicate(predicate)
            .finish()
    }
}

impl Executor for CsvExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let profile_name = if state.has_node_timer() {
            let mut ids = vec![self.path.to_string_lossy().into()];
            if self.predicate.is_some() {
                ids.push("predicate".into())
            }
            let name = comma_delimited("csv".to_string(), &ids);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        state.record(|| self.read(), profile_name)
    }
}
