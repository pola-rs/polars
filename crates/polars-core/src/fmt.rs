#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
use std::borrow::Cow;
use std::fmt::{Debug, Display, Formatter, Write};
use std::str::FromStr;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::RwLock;
use std::{fmt, str};

#[cfg(any(
    feature = "dtype-date",
    feature = "dtype-datetime",
    feature = "dtype-time"
))]
use arrow::temporal_conversions::*;
#[cfg(feature = "dtype-datetime")]
use chrono::NaiveDateTime;
#[cfg(feature = "timezones")]
use chrono::TimeZone;
#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
use comfy_table::modifiers::*;
#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
use comfy_table::presets::*;
#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
use comfy_table::*;
#[cfg(feature = "dtype-duration")]
use itoa;
use num_traits::{Num, NumCast};
use polars_error::feature_gated;

use crate::config::*;
use crate::prelude::*;

// Note: see https://github.com/pola-rs/polars/pull/13699 for the rationale
// behind choosing 10 as the default value for default number of rows displayed
const DEFAULT_ROW_LIMIT: usize = 10;
#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
const DEFAULT_COL_LIMIT: usize = 8;
const DEFAULT_STR_LEN_LIMIT: usize = 30;
const DEFAULT_LIST_LEN_LIMIT: usize = 3;

#[derive(Copy, Clone)]
#[repr(u8)]
pub enum FloatFmt {
    Mixed,
    Full,
}
static FLOAT_PRECISION: RwLock<Option<usize>> = RwLock::new(None);
static FLOAT_FMT: AtomicU8 = AtomicU8::new(FloatFmt::Mixed as u8);

static THOUSANDS_SEPARATOR: AtomicU8 = AtomicU8::new(b'\0');
static DECIMAL_SEPARATOR: AtomicU8 = AtomicU8::new(b'.');

// Numeric formatting getters
pub fn get_float_fmt() -> FloatFmt {
    match FLOAT_FMT.load(Ordering::Relaxed) {
        0 => FloatFmt::Mixed,
        1 => FloatFmt::Full,
        _ => panic!(),
    }
}
pub fn get_float_precision() -> Option<usize> {
    *FLOAT_PRECISION.read().unwrap()
}
pub fn get_decimal_separator() -> char {
    DECIMAL_SEPARATOR.load(Ordering::Relaxed) as char
}
pub fn get_thousands_separator() -> String {
    let sep = THOUSANDS_SEPARATOR.load(Ordering::Relaxed) as char;
    if sep == '\0' {
        "".to_string()
    } else {
        sep.to_string()
    }
}
#[cfg(feature = "dtype-decimal")]
pub fn get_trim_decimal_zeros() -> bool {
    arrow::compute::decimal::get_trim_decimal_zeros()
}

// Numeric formatting setters
pub fn set_float_fmt(fmt: FloatFmt) {
    FLOAT_FMT.store(fmt as u8, Ordering::Relaxed)
}
pub fn set_float_precision(precision: Option<usize>) {
    *FLOAT_PRECISION.write().unwrap() = precision;
}
pub fn set_decimal_separator(dec: Option<char>) {
    DECIMAL_SEPARATOR.store(dec.unwrap_or('.') as u8, Ordering::Relaxed)
}
pub fn set_thousands_separator(sep: Option<char>) {
    THOUSANDS_SEPARATOR.store(sep.unwrap_or('\0') as u8, Ordering::Relaxed)
}
#[cfg(feature = "dtype-decimal")]
pub fn set_trim_decimal_zeros(trim: Option<bool>) {
    arrow::compute::decimal::set_trim_decimal_zeros(trim)
}

/// Parses an environment variable value.
fn parse_env_var<T: FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}
/// Parses an environment variable value as a limit or set a default.
///
/// Negative values (e.g. -1) are parsed as 'no limit' or [`usize::MAX`].
fn parse_env_var_limit(name: &str, default: usize) -> usize {
    parse_env_var(name).map_or(
        default,
        |n: i64| {
            if n < 0 {
                usize::MAX
            } else {
                n as usize
            }
        },
    )
}

fn get_row_limit() -> usize {
    parse_env_var_limit(FMT_MAX_ROWS, DEFAULT_ROW_LIMIT)
}
#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
fn get_col_limit() -> usize {
    parse_env_var_limit(FMT_MAX_COLS, DEFAULT_COL_LIMIT)
}
fn get_str_len_limit() -> usize {
    parse_env_var_limit(FMT_STR_LEN, DEFAULT_STR_LEN_LIMIT)
}
fn get_list_len_limit() -> usize {
    parse_env_var_limit(FMT_TABLE_CELL_LIST_LEN, DEFAULT_LIST_LEN_LIMIT)
}
#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
fn get_ellipsis() -> &'static str {
    match std::env::var(FMT_TABLE_FORMATTING).as_deref().unwrap_or("") {
        preset if preset.starts_with("ASCII") => "...",
        _ => "…",
    }
}
#[cfg(not(any(feature = "fmt", feature = "fmt_no_tty")))]
fn get_ellipsis() -> &'static str {
    "…"
}

fn estimate_string_width(s: &str) -> usize {
    // get a slightly more accurate estimate of a string's screen
    // width, accounting (very roughly) for multibyte characters
    let n_chars = s.chars().count();
    let n_bytes = s.len();
    if n_bytes == n_chars {
        n_chars
    } else {
        let adjust = n_bytes as f64 / n_chars as f64;
        std::cmp::min(n_chars * 2, (n_chars as f64 * adjust).ceil() as usize)
    }
}

macro_rules! format_array {
    ($f:ident, $a:expr, $dtype:expr, $name:expr, $array_type:expr) => {{
        write!(
            $f,
            "shape: ({},)\n{}: '{}' [{}]\n[\n",
            fmt_int_string_custom(&$a.len().to_string(), 3, "_"),
            $array_type,
            $name,
            $dtype
        )?;

        let ellipsis = get_ellipsis();
        let truncate = match $a.dtype() {
            DataType::String => true,
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) | DataType::Enum(_, _) => true,
            _ => false,
        };
        let truncate_len = if truncate { get_str_len_limit() } else { 0 };

        let write_fn = |v, f: &mut Formatter| -> fmt::Result {
            if truncate {
                let v = format!("{}", v);
                let v_no_quotes = &v[1..v.len() - 1];
                let v_trunc = &v_no_quotes[..v_no_quotes
                    .char_indices()
                    .take(truncate_len)
                    .last()
                    .map(|(i, c)| i + c.len_utf8())
                    .unwrap_or(0)];
                if v_no_quotes == v_trunc {
                    write!(f, "\t{}\n", v)?;
                } else {
                    write!(f, "\t\"{v_trunc}{ellipsis}\n")?;
                }
            } else {
                write!(f, "\t{v}\n")?;
            };
            Ok(())
        };

        let limit = get_row_limit();

        if $a.len() > limit {
            let half = limit / 2;
            let rest = limit % 2;

            for i in 0..(half + rest) {
                let v = $a.get_any_value(i).unwrap();
                write_fn(v, $f)?;
            }
            write!($f, "\t{ellipsis}\n")?;
            for i in ($a.len() - half)..$a.len() {
                let v = $a.get_any_value(i).unwrap();
                write_fn(v, $f)?;
            }
        } else {
            for i in 0..$a.len() {
                let v = $a.get_any_value(i).unwrap();
                write_fn(v, $f)?;
            }
        }

        write!($f, "]")
    }};
}

#[cfg(feature = "object")]
fn format_object_array(
    f: &mut Formatter<'_>,
    object: &Series,
    name: &str,
    array_type: &str,
) -> fmt::Result {
    match object.dtype() {
        DataType::Object(inner_type, _) => {
            let limit = std::cmp::min(DEFAULT_ROW_LIMIT, object.len());
            write!(
                f,
                "shape: ({},)\n{}: '{}' [o][{}]\n[\n",
                fmt_int_string_custom(&object.len().to_string(), 3, "_"),
                array_type,
                name,
                inner_type
            )?;
            for i in 0..limit {
                let v = object.str_value(i);
                writeln!(f, "\t{}", v.unwrap())?;
            }
            write!(f, "]")
        },
        _ => unreachable!(),
    }
}

impl<T> Debug for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let dt = format!("{}", T::get_dtype());
        format_array!(f, self, dt, self.name(), "ChunkedArray")
    }
}

impl Debug for ChunkedArray<BooleanType> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        format_array!(f, self, "bool", self.name(), "ChunkedArray")
    }
}

impl Debug for StringChunked {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        format_array!(f, self, "str", self.name(), "ChunkedArray")
    }
}

impl Debug for BinaryChunked {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        format_array!(f, self, "binary", self.name(), "ChunkedArray")
    }
}

impl Debug for ListChunked {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        format_array!(f, self, "list", self.name(), "ChunkedArray")
    }
}

#[cfg(feature = "dtype-array")]
impl Debug for ArrayChunked {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        format_array!(f, self, "fixed size list", self.name(), "ChunkedArray")
    }
}

#[cfg(feature = "object")]
impl<T> Debug for ObjectChunked<T>
where
    T: PolarsObject,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let limit = std::cmp::min(DEFAULT_ROW_LIMIT, self.len());
        let ellipsis = get_ellipsis();
        let inner_type = T::type_name();
        write!(
            f,
            "ChunkedArray: '{}' [o][{}]\n[\n",
            self.name(),
            inner_type
        )?;

        if limit < self.len() {
            for i in 0..limit / 2 {
                match self.get(i) {
                    None => writeln!(f, "\tnull")?,
                    Some(val) => writeln!(f, "\t{val}")?,
                };
            }
            writeln!(f, "\t{ellipsis}")?;
            for i in (0..limit / 2).rev() {
                match self.get(self.len() - i - 1) {
                    None => writeln!(f, "\tnull")?,
                    Some(val) => writeln!(f, "\t{val}")?,
                };
            }
        } else {
            for i in 0..limit {
                match self.get(i) {
                    None => writeln!(f, "\tnull")?,
                    Some(val) => writeln!(f, "\t{val}")?,
                };
            }
        }
        Ok(())
    }
}

impl Debug for Series {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.dtype() {
            DataType::Boolean => {
                format_array!(f, self.bool().unwrap(), "bool", self.name(), "Series")
            },
            DataType::String => {
                format_array!(f, self.str().unwrap(), "str", self.name(), "Series")
            },
            DataType::UInt8 => {
                format_array!(f, self.u8().unwrap(), "u8", self.name(), "Series")
            },
            DataType::UInt16 => {
                format_array!(f, self.u16().unwrap(), "u16", self.name(), "Series")
            },
            DataType::UInt32 => {
                format_array!(f, self.u32().unwrap(), "u32", self.name(), "Series")
            },
            DataType::UInt64 => {
                format_array!(f, self.u64().unwrap(), "u64", self.name(), "Series")
            },
            DataType::Int8 => {
                format_array!(f, self.i8().unwrap(), "i8", self.name(), "Series")
            },
            DataType::Int16 => {
                format_array!(f, self.i16().unwrap(), "i16", self.name(), "Series")
            },
            DataType::Int32 => {
                format_array!(f, self.i32().unwrap(), "i32", self.name(), "Series")
            },
            DataType::Int64 => {
                format_array!(f, self.i64().unwrap(), "i64", self.name(), "Series")
            },
            DataType::Int128 => {
                feature_gated!(
                    "dtype-i128",
                    format_array!(f, self.i128().unwrap(), "i128", self.name(), "Series")
                )
            },
            DataType::Float32 => {
                format_array!(f, self.f32().unwrap(), "f32", self.name(), "Series")
            },
            DataType::Float64 => {
                format_array!(f, self.f64().unwrap(), "f64", self.name(), "Series")
            },
            #[cfg(feature = "dtype-date")]
            DataType::Date => format_array!(f, self.date().unwrap(), "date", self.name(), "Series"),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => {
                let dt = format!("{}", self.dtype());
                format_array!(f, self.datetime().unwrap(), &dt, self.name(), "Series")
            },
            #[cfg(feature = "dtype-time")]
            DataType::Time => format_array!(f, self.time().unwrap(), "time", self.name(), "Series"),
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(_) => {
                let dt = format!("{}", self.dtype());
                format_array!(f, self.duration().unwrap(), &dt, self.name(), "Series")
            },
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(_, _) => {
                let dt = format!("{}", self.dtype());
                format_array!(f, self.decimal().unwrap(), &dt, self.name(), "Series")
            },
            #[cfg(feature = "dtype-array")]
            DataType::Array(_, _) => {
                let dt = format!("{}", self.dtype());
                format_array!(f, self.array().unwrap(), &dt, self.name(), "Series")
            },
            DataType::List(_) => {
                let dt = format!("{}", self.dtype());
                format_array!(f, self.list().unwrap(), &dt, self.name(), "Series")
            },
            #[cfg(feature = "object")]
            DataType::Object(_, _) => format_object_array(f, self, self.name(), "Series"),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) => {
                format_array!(f, self.categorical().unwrap(), "cat", self.name(), "Series")
            },

            #[cfg(feature = "dtype-categorical")]
            DataType::Enum(_, _) => format_array!(
                f,
                self.categorical().unwrap(),
                "enum",
                self.name(),
                "Series"
            ),
            #[cfg(feature = "dtype-struct")]
            dt @ DataType::Struct(_) => format_array!(
                f,
                self.struct_().unwrap(),
                format!("{dt}"),
                self.name(),
                "Series"
            ),
            DataType::Null => {
                format_array!(f, self.null().unwrap(), "null", self.name(), "Series")
            },
            DataType::Binary => {
                format_array!(f, self.binary().unwrap(), "binary", self.name(), "Series")
            },
            DataType::BinaryOffset => {
                format_array!(
                    f,
                    self.binary_offset().unwrap(),
                    "binary[offset]",
                    self.name(),
                    "Series"
                )
            },
            dt => panic!("{dt:?} not impl"),
        }
    }
}

impl Display for Series {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(self, f)
    }
}

impl Debug for DataFrame {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}
#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
fn make_str_val(v: &str, truncate: usize, ellipsis: &String) -> String {
    let v_trunc = &v[..v
        .char_indices()
        .take(truncate)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0)];
    if v == v_trunc {
        v.to_string()
    } else {
        format!("{v_trunc}{ellipsis}")
    }
}

#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
fn field_to_str(
    f: &Field,
    str_truncate: usize,
    ellipsis: &String,
    padding: usize,
) -> (String, usize) {
    let name = make_str_val(f.name(), str_truncate, ellipsis);
    let name_length = estimate_string_width(name.as_str());
    let mut column_name = name;
    if env_is_true(FMT_TABLE_HIDE_COLUMN_NAMES) {
        column_name = "".to_string();
    }
    let column_dtype = if env_is_true(FMT_TABLE_HIDE_COLUMN_DATA_TYPES) {
        "".to_string()
    } else if env_is_true(FMT_TABLE_INLINE_COLUMN_DATA_TYPE)
        | env_is_true(FMT_TABLE_HIDE_COLUMN_NAMES)
    {
        format!("{}", f.dtype())
    } else {
        format!("\n{}", f.dtype())
    };
    let mut dtype_length = column_dtype.trim_start().len();
    let mut separator = "\n---";
    if env_is_true(FMT_TABLE_HIDE_COLUMN_SEPARATOR)
        | env_is_true(FMT_TABLE_HIDE_COLUMN_NAMES)
        | env_is_true(FMT_TABLE_HIDE_COLUMN_DATA_TYPES)
    {
        separator = ""
    }
    let s = if env_is_true(FMT_TABLE_INLINE_COLUMN_DATA_TYPE)
        & !env_is_true(FMT_TABLE_HIDE_COLUMN_DATA_TYPES)
    {
        let inline_name_dtype = format!("{column_name} ({column_dtype})");
        dtype_length = inline_name_dtype.len();
        inline_name_dtype
    } else {
        format!("{column_name}{separator}{column_dtype}")
    };
    let mut s_len = std::cmp::max(name_length, dtype_length);
    let separator_length = estimate_string_width(separator.trim());
    if s_len < separator_length {
        s_len = separator_length;
    }
    (s, s_len + padding)
}

#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
fn prepare_row(
    row: Vec<Cow<'_, str>>,
    n_first: usize,
    n_last: usize,
    str_truncate: usize,
    max_elem_lengths: &mut [usize],
    ellipsis: &String,
    padding: usize,
) -> Vec<String> {
    let reduce_columns = n_first + n_last < row.len();
    let n_elems = n_first + n_last + reduce_columns as usize;
    let mut row_strings = Vec::with_capacity(n_elems);

    for (idx, v) in row[0..n_first].iter().enumerate() {
        let elem_str = make_str_val(v, str_truncate, ellipsis);
        let elem_len = estimate_string_width(elem_str.as_str()) + padding;
        if max_elem_lengths[idx] < elem_len {
            max_elem_lengths[idx] = elem_len;
        };
        row_strings.push(elem_str);
    }
    if reduce_columns {
        row_strings.push(ellipsis.to_string());
        max_elem_lengths[n_first] = ellipsis.chars().count() + padding;
    }
    let elem_offset = n_first + reduce_columns as usize;
    for (idx, v) in row[row.len() - n_last..].iter().enumerate() {
        let elem_str = make_str_val(v, str_truncate, ellipsis);
        let elem_len = estimate_string_width(elem_str.as_str()) + padding;
        let elem_idx = elem_offset + idx;
        if max_elem_lengths[elem_idx] < elem_len {
            max_elem_lengths[elem_idx] = elem_len;
        };
        row_strings.push(elem_str);
    }
    row_strings
}

#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
fn env_is_true(varname: &str) -> bool {
    std::env::var(varname).as_deref().unwrap_or("0") == "1"
}

#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
fn fmt_df_shape((shape0, shape1): &(usize, usize)) -> String {
    // e.g. (1_000_000, 4_000)
    format!(
        "({}, {})",
        fmt_int_string_custom(&shape0.to_string(), 3, "_"),
        fmt_int_string_custom(&shape1.to_string(), 3, "_")
    )
}

impl Display for DataFrame {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        #[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
        {
            let height = self.height();
            assert!(
                self.columns.iter().all(|s| s.len() == height),
                "The column lengths in the DataFrame are not equal."
            );

            let table_style = std::env::var(FMT_TABLE_FORMATTING).unwrap_or("DEFAULT".to_string());
            let is_utf8 = !table_style.starts_with("ASCII");
            let preset = match table_style.as_str() {
                "ASCII_FULL" => ASCII_FULL,
                "ASCII_FULL_CONDENSED" => ASCII_FULL_CONDENSED,
                "ASCII_NO_BORDERS" => ASCII_NO_BORDERS,
                "ASCII_BORDERS_ONLY" => ASCII_BORDERS_ONLY,
                "ASCII_BORDERS_ONLY_CONDENSED" => ASCII_BORDERS_ONLY_CONDENSED,
                "ASCII_HORIZONTAL_ONLY" => ASCII_HORIZONTAL_ONLY,
                "ASCII_MARKDOWN" | "MARKDOWN" => ASCII_MARKDOWN,
                "UTF8_FULL" => UTF8_FULL,
                "UTF8_FULL_CONDENSED" => UTF8_FULL_CONDENSED,
                "UTF8_NO_BORDERS" => UTF8_NO_BORDERS,
                "UTF8_BORDERS_ONLY" => UTF8_BORDERS_ONLY,
                "UTF8_HORIZONTAL_ONLY" => UTF8_HORIZONTAL_ONLY,
                "NOTHING" => NOTHING,
                _ => UTF8_FULL_CONDENSED,
            };
            let ellipsis = get_ellipsis().to_string();
            let ellipsis_len = ellipsis.chars().count();
            let max_n_cols = get_col_limit();
            let max_n_rows = get_row_limit();
            let str_truncate = get_str_len_limit();
            let padding = 2; // eg: one char either side of the value

            let (n_first, n_last) = if self.width() > max_n_cols {
                ((max_n_cols + 1) / 2, max_n_cols / 2)
            } else {
                (self.width(), 0)
            };
            let reduce_columns = n_first + n_last < self.width();
            let n_tbl_cols = n_first + n_last + reduce_columns as usize;
            let mut names = Vec::with_capacity(n_tbl_cols);
            let mut name_lengths = Vec::with_capacity(n_tbl_cols);

            let fields = self.fields();
            for field in fields[0..n_first].iter() {
                let (s, l) = field_to_str(field, str_truncate, &ellipsis, padding);
                names.push(s);
                name_lengths.push(l);
            }
            if reduce_columns {
                names.push(ellipsis.clone());
                name_lengths.push(ellipsis_len);
            }
            for field in fields[self.width() - n_last..].iter() {
                let (s, l) = field_to_str(field, str_truncate, &ellipsis, padding);
                names.push(s);
                name_lengths.push(l);
            }

            let mut table = Table::new();
            table
                .load_preset(preset)
                .set_content_arrangement(ContentArrangement::Dynamic);

            if is_utf8 && env_is_true(FMT_TABLE_ROUNDED_CORNERS) {
                table.apply_modifier(UTF8_ROUND_CORNERS);
            }
            let mut constraints = Vec::with_capacity(n_tbl_cols);
            let mut max_elem_lengths: Vec<usize> = vec![0; n_tbl_cols];

            if max_n_rows > 0 {
                if height > max_n_rows {
                    // Truncate the table if we have more rows than the
                    // configured maximum number of rows
                    let mut rows = Vec::with_capacity(std::cmp::max(max_n_rows, 2));
                    let half = max_n_rows / 2;
                    let rest = max_n_rows % 2;

                    for i in 0..(half + rest) {
                        let row = self
                            .get_columns()
                            .iter()
                            .map(|c| c.str_value(i).unwrap())
                            .collect();

                        let row_strings = prepare_row(
                            row,
                            n_first,
                            n_last,
                            str_truncate,
                            &mut max_elem_lengths,
                            &ellipsis,
                            padding,
                        );
                        rows.push(row_strings);
                    }
                    let dots = vec![ellipsis.clone(); rows[0].len()];
                    rows.push(dots);

                    for i in (height - half)..height {
                        let row = self
                            .get_columns()
                            .iter()
                            .map(|c| c.str_value(i).unwrap())
                            .collect();

                        let row_strings = prepare_row(
                            row,
                            n_first,
                            n_last,
                            str_truncate,
                            &mut max_elem_lengths,
                            &ellipsis,
                            padding,
                        );
                        rows.push(row_strings);
                    }
                    table.add_rows(rows);
                } else {
                    for i in 0..height {
                        if self.width() > 0 {
                            let row = self
                                .materialized_column_iter()
                                .map(|s| s.str_value(i).unwrap())
                                .collect();

                            let row_strings = prepare_row(
                                row,
                                n_first,
                                n_last,
                                str_truncate,
                                &mut max_elem_lengths,
                                &ellipsis,
                                padding,
                            );
                            table.add_row(row_strings);
                        } else {
                            break;
                        }
                    }
                }
            } else if height > 0 {
                let dots: Vec<String> = vec![ellipsis.clone(); self.columns.len()];
                table.add_row(dots);
            }
            let tbl_fallback_width = 100;
            let tbl_width = std::env::var("POLARS_TABLE_WIDTH")
                .map(|s| {
                    Some(
                        s.parse::<u16>()
                            .expect("could not parse table width argument"),
                    )
                })
                .unwrap_or(None);

            // column width constraints
            let col_width_exact =
                |w: usize| ColumnConstraint::Absolute(comfy_table::Width::Fixed(w as u16));
            let col_width_bounds = |l: usize, u: usize| ColumnConstraint::Boundaries {
                lower: Width::Fixed(l as u16),
                upper: Width::Fixed(u as u16),
            };
            let min_col_width = std::cmp::max(5, 3 + padding);
            for (idx, elem_len) in max_elem_lengths.iter().enumerate() {
                let mx = std::cmp::min(
                    str_truncate + ellipsis_len + padding,
                    std::cmp::max(name_lengths[idx], *elem_len),
                );
                if mx <= min_col_width {
                    constraints.push(col_width_exact(mx));
                } else {
                    constraints.push(col_width_bounds(min_col_width, mx));
                }
            }

            // insert a header row, unless both column names and dtypes are hidden
            if !(env_is_true(FMT_TABLE_HIDE_COLUMN_NAMES)
                && env_is_true(FMT_TABLE_HIDE_COLUMN_DATA_TYPES))
            {
                table.set_header(names).set_constraints(constraints);
            }

            // if tbl_width is explicitly set, use it
            if let Some(w) = tbl_width {
                table.set_width(w);
            } else {
                // if no tbl_width (it's not tty && width not explicitly set), apply
                // a default value; this is needed to support non-tty applications
                #[cfg(feature = "fmt")]
                if table.width().is_none() && !table.is_tty() {
                    table.set_width(tbl_fallback_width);
                }
                #[cfg(feature = "fmt_no_tty")]
                if table.width().is_none() {
                    table.set_width(tbl_fallback_width);
                }
            }

            // set alignment of cells, if defined
            if std::env::var(FMT_TABLE_CELL_ALIGNMENT).is_ok()
                | std::env::var(FMT_TABLE_CELL_NUMERIC_ALIGNMENT).is_ok()
            {
                let str_preset = std::env::var(FMT_TABLE_CELL_ALIGNMENT)
                    .unwrap_or_else(|_| "DEFAULT".to_string());
                let num_preset = std::env::var(FMT_TABLE_CELL_NUMERIC_ALIGNMENT)
                    .unwrap_or_else(|_| str_preset.to_string());
                for (column_index, column) in table.column_iter_mut().enumerate() {
                    let dtype = fields[column_index].dtype();
                    let mut preset = str_preset.as_str();
                    if dtype.is_primitive_numeric() || dtype.is_decimal() {
                        preset = num_preset.as_str();
                    }
                    match preset {
                        "RIGHT" => column.set_cell_alignment(CellAlignment::Right),
                        "LEFT" => column.set_cell_alignment(CellAlignment::Left),
                        "CENTER" => column.set_cell_alignment(CellAlignment::Center),
                        _ => {},
                    }
                }
            }

            // establish 'shape' information (above/below/hidden)
            if env_is_true(FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION) {
                write!(f, "{table}")?;
            } else {
                let shape_str = fmt_df_shape(&self.shape());
                if env_is_true(FMT_TABLE_DATAFRAME_SHAPE_BELOW) {
                    write!(f, "{table}\nshape: {}", shape_str)?;
                } else {
                    write!(f, "shape: {}\n{}", shape_str, table)?;
                }
            }
        }
        #[cfg(not(any(feature = "fmt", feature = "fmt_no_tty")))]
        {
            write!(
                f,
                "shape: {:?}\nto see more, compile with the 'fmt' or 'fmt_no_tty' feature",
                self.shape()
            )?;
        }
        Ok(())
    }
}

fn fmt_int_string_custom(num: &str, group_size: u8, group_separator: &str) -> String {
    if group_size == 0 || num.len() <= 1 {
        num.to_string()
    } else {
        let mut out = String::new();
        let sign_offset = if num.starts_with('-') || num.starts_with('+') {
            out.push(num.chars().next().unwrap());
            1
        } else {
            0
        };
        let int_body = num[sign_offset..]
            .as_bytes()
            .rchunks(group_size as usize)
            .rev()
            .map(str::from_utf8)
            .collect::<Result<Vec<&str>, _>>()
            .unwrap()
            .join(group_separator);
        out.push_str(&int_body);
        out
    }
}

fn fmt_int_string(num: &str) -> String {
    fmt_int_string_custom(num, 3, &get_thousands_separator())
}

fn fmt_float_string_custom(
    num: &str,
    group_size: u8,
    group_separator: &str,
    decimal: char,
) -> String {
    // Quick exit if no formatting would be applied
    if num.len() <= 1 || (group_size == 0 && decimal == '.') {
        num.to_string()
    } else {
        // Take existing numeric string and apply digit grouping & separator/decimal chars
        // e.g. "1000000" → "1_000_000", "-123456.798" → "-123,456.789", etc
        let (idx, has_fractional) = match num.find('.') {
            Some(i) => (i, true),
            None => (num.len(), false),
        };
        let mut out = String::new();
        let integer_part = &num[..idx];

        out.push_str(&fmt_int_string_custom(
            integer_part,
            group_size,
            group_separator,
        ));
        if has_fractional {
            out.push(decimal);
            out.push_str(&num[idx + 1..]);
        };
        out
    }
}

fn fmt_float_string(num: &str) -> String {
    fmt_float_string_custom(num, 3, &get_thousands_separator(), get_decimal_separator())
}

fn fmt_integer<T: Num + NumCast + Display>(
    f: &mut Formatter<'_>,
    width: usize,
    v: T,
) -> fmt::Result {
    write!(f, "{:>width$}", fmt_int_string(&v.to_string()))
}

const SCIENTIFIC_BOUND: f64 = 999999.0;

fn fmt_float<T: Num + NumCast>(f: &mut Formatter<'_>, width: usize, v: T) -> fmt::Result {
    let v: f64 = NumCast::from(v).unwrap();

    let float_precision = get_float_precision();

    if let Some(precision) = float_precision {
        if format!("{v:.precision$}", precision = precision).len() > 19 {
            return write!(f, "{v:>width$.precision$e}", precision = precision);
        }
        let s = format!("{v:>width$.precision$}", precision = precision);
        return write!(f, "{}", fmt_float_string(s.as_str()));
    }

    if matches!(get_float_fmt(), FloatFmt::Full) {
        let s = format!("{v:>width$}");
        return write!(f, "{}", fmt_float_string(s.as_str()));
    }

    // show integers as 0.0, 1.0 ... 101.0
    if v.fract() == 0.0 && v.abs() < SCIENTIFIC_BOUND {
        let s = format!("{v:>width$.1}");
        write!(f, "{}", fmt_float_string(s.as_str()))
    } else if format!("{v}").len() > 9 {
        // large and small floats in scientific notation.
        // (note: scientific notation does not play well with digit grouping)
        if (!(0.000001..=SCIENTIFIC_BOUND).contains(&v.abs()) | (v.abs() > SCIENTIFIC_BOUND))
            && get_thousands_separator().is_empty()
        {
            let s = format!("{v:>width$.4e}");
            write!(f, "{}", fmt_float_string(s.as_str()))
        } else {
            // this makes sure we don't write 12.00000 in case of a long flt that is 12.0000000001
            // instead we write 12.0
            let s = format!("{v:>width$.6}");

            if s.ends_with('0') {
                let mut s = s.as_str();
                let mut len = s.len() - 1;

                while s.ends_with('0') {
                    s = &s[..len];
                    len -= 1;
                }
                let s = if s.ends_with('.') {
                    format!("{s}0")
                } else {
                    s.to_string()
                };
                write!(f, "{}", fmt_float_string(s.as_str()))
            } else {
                // 12.0934509341243124
                // written as
                // 12.09345
                let s = format!("{v:>width$.6}");
                write!(f, "{}", fmt_float_string(s.as_str()))
            }
        }
    } else {
        let s = if v.fract() == 0.0 {
            format!("{v:>width$e}")
        } else {
            format!("{v:>width$}")
        };
        write!(f, "{}", fmt_float_string(s.as_str()))
    }
}

#[cfg(feature = "dtype-datetime")]
fn fmt_datetime(
    f: &mut Formatter<'_>,
    v: i64,
    tu: TimeUnit,
    tz: Option<&self::datatypes::TimeZone>,
) -> fmt::Result {
    let ndt = match tu {
        TimeUnit::Nanoseconds => timestamp_ns_to_datetime(v),
        TimeUnit::Microseconds => timestamp_us_to_datetime(v),
        TimeUnit::Milliseconds => timestamp_ms_to_datetime(v),
    };
    match tz {
        None => std::fmt::Display::fmt(&ndt, f),
        Some(tz) => PlTzAware::new(ndt, tz).fmt(f),
    }
}

#[cfg(feature = "dtype-duration")]
const DURATION_PARTS: [&str; 4] = ["d", "h", "m", "s"];
#[cfg(feature = "dtype-duration")]
const ISO_DURATION_PARTS: [&str; 4] = ["D", "H", "M", "S"];
#[cfg(feature = "dtype-duration")]
const SIZES_NS: [i64; 4] = [
    86_400_000_000_000, // per day
    3_600_000_000_000,  // per hour
    60_000_000_000,     // per minute
    1_000_000_000,      // per second
];
#[cfg(feature = "dtype-duration")]
const SIZES_US: [i64; 4] = [86_400_000_000, 3_600_000_000, 60_000_000, 1_000_000];
#[cfg(feature = "dtype-duration")]
const SIZES_MS: [i64; 4] = [86_400_000, 3_600_000, 60_000, 1_000];

#[cfg(feature = "dtype-duration")]
pub fn fmt_duration_string<W: Write>(f: &mut W, v: i64, unit: TimeUnit) -> fmt::Result {
    // take the physical/integer duration value and return a
    // friendly/readable duration string, eg: "3d 22m 55s 1ms"
    if v == 0 {
        return match unit {
            TimeUnit::Nanoseconds => f.write_str("0ns"),
            TimeUnit::Microseconds => f.write_str("0µs"),
            TimeUnit::Milliseconds => f.write_str("0ms"),
        };
    };
    // iterate over dtype-specific sizes to appropriately scale
    // and extract 'days', 'hours', 'minutes', and 'seconds' parts.
    let sizes = match unit {
        TimeUnit::Nanoseconds => SIZES_NS.as_slice(),
        TimeUnit::Microseconds => SIZES_US.as_slice(),
        TimeUnit::Milliseconds => SIZES_MS.as_slice(),
    };
    let mut buffer = itoa::Buffer::new();
    for (i, &size) in sizes.iter().enumerate() {
        let whole_num = if i == 0 {
            v / size
        } else {
            (v % sizes[i - 1]) / size
        };
        if whole_num != 0 {
            f.write_str(buffer.format(whole_num))?;
            f.write_str(DURATION_PARTS[i])?;
            if v % size != 0 {
                f.write_char(' ')?;
            }
        }
    }
    // write fractional seconds as integer nano/micro/milliseconds.
    let (v, units) = match unit {
        TimeUnit::Nanoseconds => (v % 1_000_000_000, ["ns", "µs", "ms"]),
        TimeUnit::Microseconds => (v % 1_000_000, ["µs", "ms", ""]),
        TimeUnit::Milliseconds => (v % 1_000, ["ms", "", ""]),
    };
    if v != 0 {
        let (value, suffix) = if v % 1_000 != 0 {
            (v, units[0])
        } else if v % 1_000_000 != 0 {
            (v / 1_000, units[1])
        } else {
            (v / 1_000_000, units[2])
        };
        f.write_str(buffer.format(value))?;
        f.write_str(suffix)?;
    }
    Ok(())
}

#[cfg(feature = "dtype-duration")]
pub fn iso_duration_string(s: &mut String, mut v: i64, unit: TimeUnit) {
    if v == 0 {
        s.push_str("PT0S");
        return;
    }
    let mut buffer = itoa::Buffer::new();
    let mut wrote_part = false;
    if v < 0 {
        // negative sign before "P" indicates entire ISO duration is negative.
        s.push_str("-P");
        v = v.abs();
    } else {
        s.push('P');
    }
    // iterate over dtype-specific sizes to appropriately scale
    // and extract 'days', 'hours', 'minutes', and 'seconds' parts.
    let sizes = match unit {
        TimeUnit::Nanoseconds => SIZES_NS.as_slice(),
        TimeUnit::Microseconds => SIZES_US.as_slice(),
        TimeUnit::Milliseconds => SIZES_MS.as_slice(),
    };
    for (i, &size) in sizes.iter().enumerate() {
        let whole_num = if i == 0 {
            v / size
        } else {
            (v % sizes[i - 1]) / size
        };
        if whole_num != 0 || i == 3 {
            if i != 3 {
                // days, hours, minutes
                s.push_str(buffer.format(whole_num));
                s.push_str(ISO_DURATION_PARTS[i]);
            } else {
                // (index 3 => 'seconds' part): the ISO version writes
                // fractional seconds, not integer nano/micro/milliseconds.
                // if zero, only write out if no other parts written yet.
                let fractional_part = v % size;
                if whole_num == 0 && fractional_part == 0 {
                    if !wrote_part {
                        s.push_str("0S")
                    }
                } else {
                    s.push_str(buffer.format(whole_num));
                    if fractional_part != 0 {
                        let secs = match unit {
                            TimeUnit::Nanoseconds => format!(".{:09}", fractional_part),
                            TimeUnit::Microseconds => format!(".{:06}", fractional_part),
                            TimeUnit::Milliseconds => format!(".{:03}", fractional_part),
                        };
                        s.push_str(secs.trim_end_matches('0'));
                    }
                    s.push_str(ISO_DURATION_PARTS[i]);
                }
            }
            // (index 0 => 'days' part): after writing days above (if non-zero)
            // the ISO duration string requires a `T` before the time part.
            if i == 0 {
                s.push('T');
            }
            wrote_part = true;
        } else if i == 0 {
            // always need to write the `T` separator for ISO
            // durations, even if there is no 'days' part.
            s.push('T');
        }
    }
    // if there was only a 'days' component, no need for time separator.
    if s.ends_with('T') {
        s.pop();
    }
}

fn format_blob(f: &mut Formatter<'_>, bytes: &[u8]) -> fmt::Result {
    let ellipsis = get_ellipsis();
    let width = get_str_len_limit() * 2;
    write!(f, "b\"")?;

    for b in bytes.iter().take(width) {
        if b.is_ascii_alphanumeric() || b.is_ascii_punctuation() {
            write!(f, "{}", *b as char)?;
        } else {
            write!(f, "\\x{:02x}", b)?;
        }
    }
    if bytes.len() > width {
        write!(f, "\"{ellipsis}")?;
    } else {
        f.write_str("\"")?;
    }
    Ok(())
}

impl Display for AnyValue<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let width = 0;
        match self {
            AnyValue::Null => write!(f, "null"),
            AnyValue::UInt8(v) => fmt_integer(f, width, *v),
            AnyValue::UInt16(v) => fmt_integer(f, width, *v),
            AnyValue::UInt32(v) => fmt_integer(f, width, *v),
            AnyValue::UInt64(v) => fmt_integer(f, width, *v),
            AnyValue::Int8(v) => fmt_integer(f, width, *v),
            AnyValue::Int16(v) => fmt_integer(f, width, *v),
            AnyValue::Int32(v) => fmt_integer(f, width, *v),
            AnyValue::Int64(v) => fmt_integer(f, width, *v),
            AnyValue::Int128(v) => feature_gated!("dtype-i128", fmt_integer(f, width, *v)),
            AnyValue::Float32(v) => fmt_float(f, width, *v),
            AnyValue::Float64(v) => fmt_float(f, width, *v),
            AnyValue::Boolean(v) => write!(f, "{}", *v),
            AnyValue::String(v) => write!(f, "{}", format_args!("\"{v}\"")),
            AnyValue::StringOwned(v) => write!(f, "{}", format_args!("\"{v}\"")),
            AnyValue::Binary(d) => format_blob(f, d),
            AnyValue::BinaryOwned(d) => format_blob(f, d),
            #[cfg(feature = "dtype-date")]
            AnyValue::Date(v) => write!(f, "{}", date32_to_date(*v)),
            #[cfg(feature = "dtype-datetime")]
            AnyValue::Datetime(v, tu, tz) => fmt_datetime(f, *v, *tu, *tz),
            #[cfg(feature = "dtype-datetime")]
            AnyValue::DatetimeOwned(v, tu, tz) => {
                fmt_datetime(f, *v, *tu, tz.as_ref().map(|v| v.as_ref()))
            },
            #[cfg(feature = "dtype-duration")]
            AnyValue::Duration(v, tu) => fmt_duration_string(f, *v, *tu),
            #[cfg(feature = "dtype-time")]
            AnyValue::Time(_) => {
                let nt: chrono::NaiveTime = self.into();
                write!(f, "{nt}")
            },
            #[cfg(feature = "dtype-categorical")]
            AnyValue::Categorical(_, _, _)
            | AnyValue::CategoricalOwned(_, _, _)
            | AnyValue::Enum(_, _, _)
            | AnyValue::EnumOwned(_, _, _) => {
                let s = self.get_str().unwrap();
                write!(f, "\"{s}\"")
            },
            #[cfg(feature = "dtype-array")]
            AnyValue::Array(s, _size) => write!(f, "{}", s.fmt_list()),
            AnyValue::List(s) => write!(f, "{}", s.fmt_list()),
            #[cfg(feature = "object")]
            AnyValue::Object(v) => write!(f, "{v}"),
            #[cfg(feature = "object")]
            AnyValue::ObjectOwned(v) => write!(f, "{}", v.0.as_ref()),
            #[cfg(feature = "dtype-struct")]
            av @ AnyValue::Struct(_, _, _) => {
                let mut avs = vec![];
                av._materialize_struct_av(&mut avs);
                fmt_struct(f, &avs)
            },
            #[cfg(feature = "dtype-struct")]
            AnyValue::StructOwned(payload) => fmt_struct(f, &payload.0),
            #[cfg(feature = "dtype-decimal")]
            AnyValue::Decimal(v, scale) => fmt_decimal(f, *v, *scale),
        }
    }
}

/// Utility struct to format a timezone aware datetime.
#[allow(dead_code)]
#[cfg(feature = "dtype-datetime")]
pub struct PlTzAware<'a> {
    ndt: NaiveDateTime,
    tz: &'a str,
}
#[cfg(feature = "dtype-datetime")]
impl<'a> PlTzAware<'a> {
    pub fn new(ndt: NaiveDateTime, tz: &'a str) -> Self {
        Self { ndt, tz }
    }
}

#[cfg(feature = "dtype-datetime")]
impl Display for PlTzAware<'_> {
    #[allow(unused_variables)]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        #[cfg(feature = "timezones")]
        match self.tz.parse::<chrono_tz::Tz>() {
            Ok(tz) => {
                let dt_utc = chrono::Utc.from_local_datetime(&self.ndt).unwrap();
                let dt_tz_aware = dt_utc.with_timezone(&tz);
                write!(f, "{dt_tz_aware}")
            },
            Err(_) => write!(f, "invalid timezone"),
        }
        #[cfg(not(feature = "timezones"))]
        {
            panic!("activate 'timezones' feature")
        }
    }
}

#[cfg(feature = "dtype-struct")]
fn fmt_struct(f: &mut Formatter<'_>, vals: &[AnyValue]) -> fmt::Result {
    write!(f, "{{")?;
    if !vals.is_empty() {
        for v in &vals[..vals.len() - 1] {
            write!(f, "{v},")?;
        }
        // last value has no trailing comma
        write!(f, "{}", vals[vals.len() - 1])?;
    }
    write!(f, "}}")
}

impl Series {
    pub fn fmt_list(&self) -> String {
        if self.is_empty() {
            return "[]".to_owned();
        }
        let mut result = "[".to_owned();
        let max_items = get_list_len_limit();
        let ellipsis = get_ellipsis();

        match max_items {
            0 => write!(result, "{ellipsis}]").unwrap(),
            _ if max_items >= self.len() => {
                // this will always leave a trailing ", " after the last item
                // but for long lists, this is faster than checking against the length each time
                for item in self.iter() {
                    write!(result, "{item}, ").unwrap();
                }
                // remove trailing ", " and replace with closing brace
                result.truncate(result.len() - 2);
                result.push(']');
            },
            _ => {
                let s = self.slice(0, max_items).rechunk();
                for (i, item) in s.iter().enumerate() {
                    if i == max_items.saturating_sub(1) {
                        write!(result, "{ellipsis} {}", self.get(self.len() - 1).unwrap()).unwrap();
                        break;
                    } else {
                        write!(result, "{item}, ").unwrap();
                    }
                }
                result.push(']');
            },
        };
        result
    }
}

#[inline]
#[cfg(feature = "dtype-decimal")]
fn fmt_decimal(f: &mut Formatter<'_>, v: i128, scale: usize) -> fmt::Result {
    use arrow::compute::decimal::format_decimal;

    let trim_zeros = get_trim_decimal_zeros();
    let repr = format_decimal(v, scale, trim_zeros);
    f.write_str(fmt_float_string(repr.as_str()).as_str())
}

#[cfg(all(
    test,
    feature = "temporal",
    feature = "dtype-date",
    feature = "dtype-datetime"
))]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_fmt_list() {
        let mut builder = ListPrimitiveChunkedBuilder::<Int32Type>::new(
            PlSmallStr::from_static("a"),
            10,
            10,
            DataType::Int32,
        );
        builder.append_opt_slice(Some(&[1, 2, 3, 4, 5, 6]));
        builder.append_opt_slice(None);
        let list_long = builder.finish().into_series();

        assert_eq!(
            r#"shape: (2,)
Series: 'a' [list[i32]]
[
	[1, 2, … 6]
	null
]"#,
            format!("{:?}", list_long)
        );

        std::env::set_var("POLARS_FMT_TABLE_CELL_LIST_LEN", "10");

        assert_eq!(
            r#"shape: (2,)
Series: 'a' [list[i32]]
[
	[1, 2, 3, 4, 5, 6]
	null
]"#,
            format!("{:?}", list_long)
        );

        std::env::set_var("POLARS_FMT_TABLE_CELL_LIST_LEN", "-1");

        assert_eq!(
            r#"shape: (2,)
Series: 'a' [list[i32]]
[
	[1, 2, 3, 4, 5, 6]
	null
]"#,
            format!("{:?}", list_long)
        );

        std::env::set_var("POLARS_FMT_TABLE_CELL_LIST_LEN", "0");

        assert_eq!(
            r#"shape: (2,)
Series: 'a' [list[i32]]
[
	[…]
	null
]"#,
            format!("{:?}", list_long)
        );

        std::env::set_var("POLARS_FMT_TABLE_CELL_LIST_LEN", "1");

        assert_eq!(
            r#"shape: (2,)
Series: 'a' [list[i32]]
[
	[… 6]
	null
]"#,
            format!("{:?}", list_long)
        );

        std::env::set_var("POLARS_FMT_TABLE_CELL_LIST_LEN", "4");

        assert_eq!(
            r#"shape: (2,)
Series: 'a' [list[i32]]
[
	[1, 2, 3, … 6]
	null
]"#,
            format!("{:?}", list_long)
        );

        let mut builder = ListPrimitiveChunkedBuilder::<Int32Type>::new(
            PlSmallStr::from_static("a"),
            10,
            10,
            DataType::Int32,
        );
        builder.append_opt_slice(Some(&[1]));
        builder.append_opt_slice(None);
        let list_short = builder.finish().into_series();

        std::env::set_var("POLARS_FMT_TABLE_CELL_LIST_LEN", "");

        assert_eq!(
            r#"shape: (2,)
Series: 'a' [list[i32]]
[
	[1]
	null
]"#,
            format!("{:?}", list_short)
        );

        std::env::set_var("POLARS_FMT_TABLE_CELL_LIST_LEN", "0");

        assert_eq!(
            r#"shape: (2,)
Series: 'a' [list[i32]]
[
	[…]
	null
]"#,
            format!("{:?}", list_short)
        );

        std::env::set_var("POLARS_FMT_TABLE_CELL_LIST_LEN", "-1");

        assert_eq!(
            r#"shape: (2,)
Series: 'a' [list[i32]]
[
	[1]
	null
]"#,
            format!("{:?}", list_short)
        );

        let mut builder = ListPrimitiveChunkedBuilder::<Int32Type>::new(
            PlSmallStr::from_static("a"),
            10,
            10,
            DataType::Int32,
        );
        builder.append_opt_slice(Some(&[]));
        builder.append_opt_slice(None);
        let list_empty = builder.finish().into_series();

        std::env::set_var("POLARS_FMT_TABLE_CELL_LIST_LEN", "");

        assert_eq!(
            r#"shape: (2,)
Series: 'a' [list[i32]]
[
	[]
	null
]"#,
            format!("{:?}", list_empty)
        );
    }

    #[test]
    fn test_fmt_temporal() {
        let s = Int32Chunked::new(PlSmallStr::from_static("Date"), &[Some(1), None, Some(3)])
            .into_date();
        assert_eq!(
            r#"shape: (3,)
Series: 'Date' [date]
[
	1970-01-02
	null
	1970-01-04
]"#,
            format!("{:?}", s.into_series())
        );

        let s = Int64Chunked::new(PlSmallStr::EMPTY, &[Some(1), None, Some(1_000_000_000_000)])
            .into_datetime(TimeUnit::Nanoseconds, None);
        assert_eq!(
            r#"shape: (3,)
Series: '' [datetime[ns]]
[
	1970-01-01 00:00:00.000000001
	null
	1970-01-01 00:16:40
]"#,
            format!("{:?}", s.into_series())
        );
    }

    #[test]
    fn test_fmt_chunkedarray() {
        let ca = Int32Chunked::new(PlSmallStr::from_static("Date"), &[Some(1), None, Some(3)]);
        assert_eq!(
            r#"shape: (3,)
ChunkedArray: 'Date' [i32]
[
	1
	null
	3
]"#,
            format!("{:?}", ca)
        );
        let ca = StringChunked::new(PlSmallStr::from_static("name"), &["a", "b"]);
        assert_eq!(
            r#"shape: (2,)
ChunkedArray: 'name' [str]
[
	"a"
	"b"
]"#,
            format!("{:?}", ca)
        );
    }
}
