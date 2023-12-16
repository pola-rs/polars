use std::path::PathBuf;

use super::*;

pub struct CsvExec {
    pub path: PathBuf,
    pub schema: SchemaRef,
    pub options: CsvParserOptions,
    pub file_options: FileScanOptions,
    pub predicate: Option<Arc<dyn PhysicalExpr>>,
}

impl CsvExec {
    fn read(&mut self) -> PolarsResult<DataFrame> {
        let mut with_columns = mem::take(&mut self.file_options.with_columns);
        let mut projected_len = 0;
        with_columns.as_ref().map(|columns| {
            projected_len = columns.len();
            columns
        });

        if projected_len == 0 {
            with_columns = None;
        }
        let n_rows = _set_n_rows_for_scan(self.file_options.n_rows);
        let predicate = self.predicate.clone().map(phys_expr_to_io_expr);

        CsvReader::from_path(&self.path)
            .unwrap()
            .has_header(self.options.has_header)
            .with_dtypes(Some(self.schema.clone()))
            .with_separator(self.options.separator)
            .with_ignore_errors(self.options.ignore_errors)
            .with_skip_rows(self.options.skip_rows)
            .with_n_rows(n_rows)
            .with_columns(with_columns.map(|mut cols| std::mem::take(Arc::make_mut(&mut cols))))
            .low_memory(self.options.low_memory)
            .with_null_values(std::mem::take(&mut self.options.null_values))
            .with_predicate(predicate)
            .with_encoding(CsvEncoding::LossyUtf8)
            ._with_comment_prefix(std::mem::take(&mut self.options.comment_prefix))
            .with_quote_char(self.options.quote_char)
            .with_end_of_line_char(self.options.eol_char)
            .with_encoding(self.options.encoding)
            .with_rechunk(self.file_options.rechunk)
            .with_row_count(std::mem::take(&mut self.file_options.row_count))
            .with_try_parse_dates(self.options.try_parse_dates)
            .truncate_ragged_lines(self.options.truncate_ragged_lines)
            .raise_if_empty(self.options.raise_if_empty)
            .finish()
    }
}

impl Executor for CsvExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let finger_print = FileFingerPrint {
            paths: Arc::new([self.path.clone()]),
            predicate: self
                .predicate
                .as_ref()
                .map(|ae| ae.as_expression().unwrap().clone()),
            slice: (self.options.skip_rows, self.file_options.n_rows),
        };

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
