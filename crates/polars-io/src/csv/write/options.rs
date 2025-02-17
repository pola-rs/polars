use std::num::NonZeroUsize;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Options for writing CSV files.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CsvWriterOptions {
    pub include_bom: bool,
    pub include_header: bool,
    pub batch_size: NonZeroUsize,
    pub maintain_order: bool,
    pub serialize_options: SerializeOptions,
}

impl Default for CsvWriterOptions {
    fn default() -> Self {
        Self {
            include_bom: false,
            include_header: true,
            batch_size: NonZeroUsize::new(1024).unwrap(),
            maintain_order: false,
            serialize_options: SerializeOptions::default(),
        }
    }
}

/// Options to serialize logical types to CSV.
///
/// The default is to format times and dates as `chrono` crate formats them.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SerializeOptions {
    /// Used for [`DataType::Date`](polars_core::datatypes::DataType::Date).
    pub date_format: Option<String>,
    /// Used for [`DataType::Time`](polars_core::datatypes::DataType::Time).
    pub time_format: Option<String>,
    /// Used for [`DataType::Datetime`](polars_core::datatypes::DataType::Datetime).
    pub datetime_format: Option<String>,
    /// Used for [`DataType::Float64`](polars_core::datatypes::DataType::Float64)
    /// and [`DataType::Float32`](polars_core::datatypes::DataType::Float32).
    pub float_scientific: Option<bool>,
    pub float_precision: Option<usize>,
    /// Used as separator.
    pub separator: u8,
    /// Quoting character.
    pub quote_char: u8,
    /// Null value representation.
    pub null: String,
    /// String appended after every row.
    pub line_terminator: String,
    /// When to insert quotes.
    pub quote_style: QuoteStyle,
}

impl Default for SerializeOptions {
    fn default() -> Self {
        Self {
            date_format: None,
            time_format: None,
            datetime_format: None,
            float_scientific: None,
            float_precision: None,
            separator: b',',
            quote_char: b'"',
            null: String::new(),
            line_terminator: "\n".into(),
            quote_style: Default::default(),
        }
    }
}

/// Quote style indicating when to insert quotes around a field.
#[derive(Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum QuoteStyle {
    /// Quote fields only when necessary.
    ///
    /// Quotes are necessary when fields contain a quote, separator or record terminator.
    /// Quotes are also necessary when writing an empty record (which is indistinguishable
    /// from arecord with one empty field).
    /// This is the default.
    #[default]
    Necessary,
    /// Quote every field. Always.
    Always,
    /// Quote non-numeric fields.
    ///
    /// When writing a field that does not parse as a valid float or integer,
    /// quotes will be used even if they aren't strictly necessary.
    NonNumeric,
    /// Never quote any fields, even if it would produce invalid CSV data.
    Never,
}
