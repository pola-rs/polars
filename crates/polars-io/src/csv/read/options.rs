use polars_core::schema::{IndexOfSchema, Schema, SchemaRef};
use polars_error::PolarsResult;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CsvReaderOptions {
    pub has_header: bool,
    pub separator: u8,
    pub quote_char: Option<u8>,
    pub comment_prefix: Option<CommentPrefix>,
    pub eol_char: u8,
    pub encoding: CsvEncoding,
    pub skip_rows: usize,
    pub skip_rows_after_header: usize,
    pub schema: Option<SchemaRef>,
    pub schema_overwrite: Option<SchemaRef>,
    pub infer_schema_length: Option<usize>,
    pub try_parse_dates: bool,
    pub null_values: Option<NullValues>,
    pub ignore_errors: bool,
    pub raise_if_empty: bool,
    pub truncate_ragged_lines: bool,
    pub decimal_comma: bool,
    pub n_threads: Option<usize>,
    pub low_memory: bool,
}

impl Default for CsvReaderOptions {
    fn default() -> Self {
        Self {
            has_header: true,
            separator: b',',
            quote_char: Some(b'"'),
            comment_prefix: None,
            eol_char: b'\n',
            encoding: CsvEncoding::default(),
            skip_rows: 0,
            skip_rows_after_header: 0,
            schema: None,
            schema_overwrite: None,
            infer_schema_length: Some(100),
            try_parse_dates: false,
            null_values: None,
            ignore_errors: false,
            raise_if_empty: true,
            truncate_ragged_lines: false,
            decimal_comma: false,
            n_threads: None,
            low_memory: false,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CsvEncoding {
    /// Utf8 encoding.
    #[default]
    Utf8,
    /// Utf8 encoding and unknown bytes are replaced with �.
    LossyUtf8,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CommentPrefix {
    /// A single byte character that indicates the start of a comment line.
    Single(u8),
    /// A string that indicates the start of a comment line.
    /// This allows for multiple characters to be used as a comment identifier.
    Multi(String),
}

impl CommentPrefix {
    /// Creates a new `CommentPrefix` for the `Single` variant.
    pub fn new_single(prefix: u8) -> Self {
        CommentPrefix::Single(prefix)
    }

    /// Creates a new `CommentPrefix` for the `Multi` variant.
    pub fn new_multi(prefix: String) -> Self {
        CommentPrefix::Multi(prefix)
    }

    /// Creates a new `CommentPrefix` from a `&str`.
    pub fn new_from_str(prefix: &str) -> Self {
        if prefix.len() == 1 && prefix.chars().next().unwrap().is_ascii() {
            let c = prefix.as_bytes()[0];
            CommentPrefix::Single(c)
        } else {
            CommentPrefix::Multi(prefix.to_string())
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NullValues {
    /// A single value that's used for all columns
    AllColumnsSingle(String),
    /// Multiple values that are used for all columns
    AllColumns(Vec<String>),
    /// Tuples that map column names to null value of that column
    Named(Vec<(String, String)>),
}

impl NullValues {
    pub(super) fn compile(self, schema: &Schema) -> PolarsResult<NullValuesCompiled> {
        Ok(match self {
            NullValues::AllColumnsSingle(v) => NullValuesCompiled::AllColumnsSingle(v),
            NullValues::AllColumns(v) => NullValuesCompiled::AllColumns(v),
            NullValues::Named(v) => {
                let mut null_values = vec!["".to_string(); schema.len()];
                for (name, null_value) in v {
                    let i = schema.try_index_of(&name)?;
                    null_values[i] = null_value;
                }
                NullValuesCompiled::Columns(null_values)
            },
        })
    }
}

pub(super) enum NullValuesCompiled {
    /// A single value that's used for all columns
    AllColumnsSingle(String),
    // Multiple null values that are null for all columns
    AllColumns(Vec<String>),
    /// A different null value per column, computed from `NullValues::Named`
    Columns(Vec<String>),
}

impl NullValuesCompiled {
    pub(super) fn apply_projection(&mut self, projections: &[usize]) {
        if let Self::Columns(nv) = self {
            let nv = projections
                .iter()
                .map(|i| std::mem::take(&mut nv[*i]))
                .collect::<Vec<_>>();

            *self = NullValuesCompiled::Columns(nv);
        }
    }

    /// # Safety
    ///
    /// The caller must ensure that `index` is in bounds
    pub(super) unsafe fn is_null(&self, field: &[u8], index: usize) -> bool {
        use NullValuesCompiled::*;
        match self {
            AllColumnsSingle(v) => v.as_bytes() == field,
            AllColumns(v) => v.iter().any(|v| v.as_bytes() == field),
            Columns(v) => {
                debug_assert!(index < v.len());
                v.get_unchecked(index).as_bytes() == field
            },
        }
    }
}
