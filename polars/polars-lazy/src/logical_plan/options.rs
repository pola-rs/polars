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
    pub(crate) rechunk: bool,
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
pub struct LpScanOptions {
    pub n_rows: Option<usize>,
    pub with_columns: Option<Vec<String>>,
    pub cache: bool,
}

#[derive(Clone, Debug, Copy, Default)]
pub struct UnionOptions {
    pub(crate) slice: bool,
    pub(crate) slice_offset: i64,
    pub(crate) slice_len: u32,
}
