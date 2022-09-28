use std::fs::File;
use std::path::PathBuf;

use polars_io::csv::read_impl::BatchedCsvReader;
use polars_io::csv::{CsvEncoding, CsvReader};
use polars_lazy::physical_plan::executors::_set_n_rows_for_scan;
use polars_lazy::prelude::{CsvParserOptions, PhysicalExpr};

use super::*;

struct CsvSource {
    schema: SchemaRef,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    reader: *mut CsvReader<'static, File>,
    batched_reader: *mut BatchedCsvReader<'static>,
}

impl CsvSource {
    fn new(
        path: PathBuf,
        schema: SchemaRef,
        options: CsvParserOptions,
        predicate: Option<Arc<dyn PhysicalExpr>>,
    ) -> PolarsResult<Self> {
        let mut with_columns = options.with_columns;
        let mut projected_len = 0;
        with_columns.as_ref().map(|columns| {
            projected_len = columns.len();
            columns
        });

        if projected_len == 0 {
            with_columns = None;
        }
        let n_rows = _set_n_rows_for_scan(options.n_rows);

        // Safety:
        // schema will be owned by CsvSource and have a valid lifetime until CsvSource is dropped
        let schema_ref =
            unsafe { std::mem::transmute::<&Schema, &'static Schema>(schema.as_ref()) };

        let reader = CsvReader::from_path(&path)
            .unwrap()
            .has_header(options.has_header)
            .with_schema(schema_ref)
            .with_delimiter(options.delimiter)
            .with_ignore_parser_errors(options.ignore_errors)
            .with_skip_rows(options.skip_rows)
            .with_n_rows(n_rows)
            .with_columns(with_columns.map(|mut cols| std::mem::take(Arc::make_mut(&mut cols))))
            .low_memory(options.low_memory)
            .with_null_values(options.null_values)
            .with_encoding(CsvEncoding::LossyUtf8)
            .with_comment_char(options.comment_char)
            .with_quote_char(options.quote_char)
            .with_end_of_line_char(options.eol_char)
            .with_encoding(options.encoding)
            .with_rechunk(options.rechunk)
            .with_row_count(options.row_count)
            .with_parse_dates(options.parse_dates);

        let reader = Box::new(reader);
        let reader = Box::leak(reader) as *mut CsvReader<'static, File>;

        let batched_reader = unsafe { Box::new((*reader).batched()?) };

        let batched_reader = Box::leak(batched_reader) as *mut BatchedCsvReader;

        Ok(CsvSource {
            schema,
            predicate,
            reader,
            batched_reader,
        })
    }
}

impl Drop for CsvSource {
    fn drop(&mut self) {
        unsafe {
            let _to_drop = Box::from_raw(self.batched_reader);
            let _to_drop = Box::from_raw(self.reader);
        };
    }
}

impl Source for CsvSource {
    fn get_batches(context: &PExecutionContext) -> PolarsResult<SourceResult> {
        todo!()
    }
}
