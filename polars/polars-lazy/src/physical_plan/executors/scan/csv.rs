use super::*;

pub struct CsvExec {
    pub path: PathBuf,
    pub schema: SchemaRef,
    pub options: CsvParserOptions,
    pub predicate: Option<Arc<dyn PhysicalExpr>>,
}

impl CsvExec {
    fn read(&mut self) -> PolarsResult<DataFrame> {
        let mut with_columns = mem::take(&mut self.options.with_columns);
        let mut projected_len = 0;
        with_columns.as_ref().map(|columns| {
            projected_len = columns.len();
            columns
        });

        if projected_len == 0 {
            with_columns = None;
        }
        let n_rows = _set_n_rows_for_scan(self.options.n_rows);
        let predicate = self
            .predicate
            .clone()
            .map(|expr| Arc::new(PhysicalIoHelper { expr }) as Arc<dyn PhysicalIoExpr>);

        CsvReader::from_path(&self.path)
            .unwrap()
            .has_header(self.options.has_header)
            .with_schema(&self.schema)
            .with_delimiter(self.options.delimiter)
            .with_ignore_errors(self.options.ignore_errors)
            .with_skip_rows(self.options.skip_rows)
            .with_n_rows(n_rows)
            .with_columns(with_columns.map(|mut cols| std::mem::take(Arc::make_mut(&mut cols))))
            .low_memory(self.options.low_memory)
            .with_null_values(std::mem::take(&mut self.options.null_values))
            .with_predicate(predicate)
            .with_encoding(CsvEncoding::LossyUtf8)
            .with_comment_char(self.options.comment_char)
            .with_quote_char(self.options.quote_char)
            .with_end_of_line_char(self.options.eol_char)
            .with_encoding(self.options.encoding)
            .with_rechunk(self.options.rechunk)
            .with_row_count(std::mem::take(&mut self.options.row_count))
            .with_parse_dates(self.options.parse_dates)
            .finish()
    }
}

impl Executor for CsvExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let finger_print = FileFingerPrint {
            path: self.path.clone(),
            predicate: self
                .predicate
                .as_ref()
                .map(|ae| ae.as_expression().unwrap().clone()),
            slice: (self.options.skip_rows, self.options.n_rows),
        };

        let profile_name = if state.has_node_timer() {
            let mut ids = vec![self.path.to_string_lossy().to_string()];
            if self.predicate.is_some() {
                ids.push("predicate".to_string())
            }
            let name = column_delimited("csv".to_string(), &ids);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        state.record(
            || {
                state
                    .file_cache
                    .read(finger_print, self.options.file_counter, &mut || self.read())
            },
            profile_name,
        )
    }
}
