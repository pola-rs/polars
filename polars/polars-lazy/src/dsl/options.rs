use polars_core::datatypes::DataType;
use polars_core::prelude::TimeUnit;

#[derive(Debug, Clone)]
pub struct StrpTimeOptions {
    /// DataType to parse in. One of {Date, Datetime}
    pub date_dtype: DataType,
    /// Formatting string
    pub fmt: Option<String>,
    /// If set then polars will return an error if any date parsing fails
    pub strict: bool,
    /// If polars may parse matches that not contain the whole string
    /// e.g. "foo-2021-01-01-bar" could match "2021-01-01"
    pub exact: bool,
}

impl Default for StrpTimeOptions {
    fn default() -> Self {
        StrpTimeOptions {
            date_dtype: DataType::Datetime(TimeUnit::Milliseconds, None),
            fmt: None,
            strict: false,
            exact: false,
        }
    }
}
