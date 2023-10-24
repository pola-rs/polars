// Formatting environment variables (typically referenced/set from the python-side Config object)
pub(crate) const FMT_MAX_COLS: &str = "POLARS_FMT_MAX_COLS";
pub(crate) const FMT_MAX_ROWS: &str = "POLARS_FMT_MAX_ROWS";
pub(crate) const FMT_STR_LEN: &str = "POLARS_FMT_STR_LEN";
pub(crate) const FMT_TABLE_CELL_ALIGNMENT: &str = "POLARS_FMT_TABLE_CELL_ALIGNMENT";
pub(crate) const FMT_TABLE_CELL_NUMERIC_ALIGNMENT: &str = "POLARS_FMT_TABLE_CELL_NUMERIC_ALIGNMENT";
pub(crate) const FMT_TABLE_DATAFRAME_SHAPE_BELOW: &str = "POLARS_FMT_TABLE_DATAFRAME_SHAPE_BELOW";
pub(crate) const FMT_TABLE_FORMATTING: &str = "POLARS_FMT_TABLE_FORMATTING";
pub(crate) const FMT_TABLE_HIDE_COLUMN_DATA_TYPES: &str = "POLARS_FMT_TABLE_HIDE_COLUMN_DATA_TYPES";
pub(crate) const FMT_TABLE_HIDE_COLUMN_NAMES: &str = "POLARS_FMT_TABLE_HIDE_COLUMN_NAMES";
pub(crate) const FMT_TABLE_HIDE_COLUMN_SEPARATOR: &str = "POLARS_FMT_TABLE_HIDE_COLUMN_SEPARATOR";
pub(crate) const FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION: &str =
    "POLARS_FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION";
pub(crate) const FMT_TABLE_INLINE_COLUMN_DATA_TYPE: &str =
    "POLARS_FMT_TABLE_INLINE_COLUMN_DATA_TYPE";
pub(crate) const FMT_TABLE_ROUNDED_CORNERS: &str = "POLARS_FMT_TABLE_ROUNDED_CORNERS";
pub(crate) const FMT_TABLE_CELL_LIST_LEN: &str = "POLARS_FMT_TABLE_CELL_LIST_LEN";

// Other env vars
#[cfg(feature = "dtype-decimal")]
pub(crate) const DECIMAL_ACTIVE: &str = "POLARS_ACTIVATE_DECIMAL";

#[cfg(feature = "dtype-decimal")]
pub(crate) fn decimal_is_active() -> bool {
    std::env::var(DECIMAL_ACTIVE).as_deref().unwrap_or("") == "1"
}

pub fn verbose() -> bool {
    std::env::var("POLARS_VERBOSE").as_deref().unwrap_or("") == "1"
}

pub fn concurrent_download_limit() -> usize {
    std::env::var("POLARS_CONC_DOWNLOAD_LIMIT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64)
}
