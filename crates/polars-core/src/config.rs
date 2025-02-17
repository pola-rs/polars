use crate::POOL;

// Formatting environment variables (typically referenced/set from the python-side Config object)
#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
pub(crate) const FMT_MAX_COLS: &str = "POLARS_FMT_MAX_COLS";
pub(crate) const FMT_MAX_ROWS: &str = "POLARS_FMT_MAX_ROWS";
pub(crate) const FMT_STR_LEN: &str = "POLARS_FMT_STR_LEN";
#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
pub(crate) const FMT_TABLE_CELL_ALIGNMENT: &str = "POLARS_FMT_TABLE_CELL_ALIGNMENT";
#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
pub(crate) const FMT_TABLE_CELL_NUMERIC_ALIGNMENT: &str = "POLARS_FMT_TABLE_CELL_NUMERIC_ALIGNMENT";
#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
pub(crate) const FMT_TABLE_DATAFRAME_SHAPE_BELOW: &str = "POLARS_FMT_TABLE_DATAFRAME_SHAPE_BELOW";
#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
pub(crate) const FMT_TABLE_FORMATTING: &str = "POLARS_FMT_TABLE_FORMATTING";
#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
pub(crate) const FMT_TABLE_HIDE_COLUMN_DATA_TYPES: &str = "POLARS_FMT_TABLE_HIDE_COLUMN_DATA_TYPES";
#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
pub(crate) const FMT_TABLE_HIDE_COLUMN_NAMES: &str = "POLARS_FMT_TABLE_HIDE_COLUMN_NAMES";
#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
pub(crate) const FMT_TABLE_HIDE_COLUMN_SEPARATOR: &str = "POLARS_FMT_TABLE_HIDE_COLUMN_SEPARATOR";
#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
pub(crate) const FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION: &str =
    "POLARS_FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION";
#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
pub(crate) const FMT_TABLE_INLINE_COLUMN_DATA_TYPE: &str =
    "POLARS_FMT_TABLE_INLINE_COLUMN_DATA_TYPE";
#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
pub(crate) const FMT_TABLE_ROUNDED_CORNERS: &str = "POLARS_FMT_TABLE_ROUNDED_CORNERS";
pub(crate) const FMT_TABLE_CELL_LIST_LEN: &str = "POLARS_FMT_TABLE_CELL_LIST_LEN";

pub fn verbose() -> bool {
    std::env::var("POLARS_VERBOSE").as_deref().unwrap_or("") == "1"
}

/// Prints a log message if sensitive verbose logging has been enabled.
pub fn verbose_print_sensitive<F: Fn() -> String>(create_log_message: F) {
    fn do_log(create_log_message: &dyn Fn() -> String) {
        if std::env::var("POLARS_VERBOSE_SENSITIVE")
            .as_deref()
            .unwrap_or("")
            == "1"
        {
            // Force the message to be a single line.
            let msg = create_log_message().replace('\n', "");
            eprintln!("[SENSITIVE]: {}", msg)
        }
    }

    do_log(&create_log_message)
}

pub fn get_file_prefetch_size() -> usize {
    std::env::var("POLARS_PREFETCH_SIZE")
        .map(|s| s.parse::<usize>().expect("integer"))
        .unwrap_or_else(|_| std::cmp::max(POOL.current_num_threads() * 2, 16))
}

pub fn get_rg_prefetch_size() -> usize {
    std::env::var("POLARS_ROW_GROUP_PREFETCH_SIZE")
        .map(|s| s.parse::<usize>().expect("integer"))
        // Set it to something big, but not unlimited.
        .unwrap_or_else(|_| std::cmp::max(get_file_prefetch_size(), 128))
}

pub fn force_async() -> bool {
    std::env::var("POLARS_FORCE_ASYNC")
        .map(|value| value == "1")
        .unwrap_or_default()
}
