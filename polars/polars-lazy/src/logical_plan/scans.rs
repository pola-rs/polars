use polars_io::csv::NullValues;

#[derive(Clone, Debug)]
pub struct CsvParserOptions {
    pub(crate) delimiter: u8,
    pub(crate) comment_char: Option<u8>,
    pub(crate) quote_char: Option<u8>,
    pub(crate) has_header: bool,
    pub(crate) skip_rows: usize,
    pub(crate) n_rows: Option<usize>,
    pub(crate) with_columns: Option<Vec<String>>,
    pub(crate) low_memory: bool,
    pub(crate) ignore_errors: bool,
    pub(crate) cache: bool,
    pub(crate) null_values: Option<NullValues>,
}
#[cfg(feature = "parquet")]
#[derive(Clone, Debug)]
pub struct ParquetOptions {
    pub(crate) n_rows: Option<usize>,
    pub(crate) with_columns: Option<Vec<String>>,
    pub(crate) cache: bool,
    pub(crate) parallel: bool,
}

#[derive(Clone, Debug)]
pub struct ScanOptions {
    pub n_rows: Option<usize>,
    pub with_columns: Option<Vec<String>>,
    pub cache: bool,
}
