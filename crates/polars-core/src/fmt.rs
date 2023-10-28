#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
use std::borrow::Cow;
use std::fmt::{Debug, Display, Formatter, Write};
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
use num_traits::{Num, NumCast};

use crate::config::*;
use crate::prelude::*;
const LIMIT: usize = 25;

#[derive(Copy, Clone)]
#[repr(u8)]
pub enum FloatFmt {
    Mixed,
    Full,
}
static FLOAT_FMT: AtomicU8 = AtomicU8::new(FloatFmt::Mixed as u8);
static FLOAT_PRECISION: RwLock<Option<usize>> = RwLock::new(None);

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

pub fn set_float_fmt(fmt: FloatFmt) {
    FLOAT_FMT.store(fmt as u8, Ordering::Relaxed)
}

pub fn set_float_precision(precision: Option<usize>) {
    *FLOAT_PRECISION.write().unwrap() = precision;
}

macro_rules! format_array {
    ($f:ident, $a:expr, $dtype:expr, $name:expr, $array_type:expr) => {{
        write!(
            $f,
            "shape: ({},)\n{}: '{}' [{}]\n[\n",
            fmt_uint(&$a.len()),
            $array_type,
            $name,
            $dtype
        )?;
        let truncate = matches!($a.dtype(), DataType::Utf8);
        let truncate_len = if truncate {
            std::env::var(FMT_STR_LEN)
                .as_deref()
                .unwrap_or("")
                .parse()
                .unwrap_or(15)
        } else {
            15
        };
        let limit: usize = {
            let limit = std::env::var(FMT_MAX_ROWS)
                .as_deref()
                .unwrap_or("")
                .parse()
                .map_or(LIMIT, |n: i64| if n < 0 { $a.len() } else { n as usize });
            std::cmp::min(limit, $a.len())
        };
        let write_fn = |v, f: &mut Formatter| -> fmt::Result {
            if truncate {
                let v = format!("{}", v);
                let v_trunc = &v[..v
                    .char_indices()
                    .take(truncate_len)
                    .last()
                    .map(|(i, c)| i + c.len_utf8())
                    .unwrap_or(0)];
                if v == v_trunc {
                    write!(f, "\t{}\n", v)?;
                } else {
                    write!(f, "\t{}…\n", v_trunc)?;
                }
            } else {
                write!(f, "\t{}\n", v)?;
            };
            Ok(())
        };
        if (limit == 0 && $a.len() > 0) || ($a.len() > limit + 1) {
            if limit > 0 {
                for i in 0..std::cmp::max((limit / 2), 1) {
                    let v = $a.get_any_value(i).unwrap();
                    write_fn(v, $f)?;
                }
            }
            write!($f, "\t…\n")?;
            if limit > 1 {
                for i in ($a.len() - (limit + 1) / 2)..$a.len() {
                    let v = $a.get_any_value(i).unwrap();
                    write_fn(v, $f)?;
                }
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
        DataType::Object(inner_type) => {
            let limit = std::cmp::min(LIMIT, object.len());

            write!(
                f,
                "shape: ({},)\n{}: '{}' [o][{}]\n[\n",
                fmt_uint(&object.len()),
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

impl Debug for Utf8Chunked {
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
        let limit = std::cmp::min(LIMIT, self.len());
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
            writeln!(f, "\t…")?;
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
            DataType::Utf8 => {
                format_array!(f, self.utf8().unwrap(), "str", self.name(), "Series")
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
            DataType::Object(_) => format_object_array(f, self, self.name(), "Series"),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => {
                format_array!(f, self.categorical().unwrap(), "cat", self.name(), "Series")
            },
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
fn make_str_val(v: &str, truncate: usize) -> String {
    let v_trunc = &v[..v
        .char_indices()
        .take(truncate)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0)];
    if v == v_trunc {
        v.to_string()
    } else {
        format!("{v_trunc}…")
    }
}

#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
fn field_to_str(f: &Field, str_truncate: usize) -> (String, usize) {
    let name = make_str_val(f.name(), str_truncate);
    let name_length = name.len();
    let mut column_name = name;
    if env_is_true(FMT_TABLE_HIDE_COLUMN_NAMES) {
        column_name = "".to_string();
    }
    let column_data_type = if env_is_true(FMT_TABLE_HIDE_COLUMN_DATA_TYPES) {
        "".to_string()
    } else if env_is_true(FMT_TABLE_INLINE_COLUMN_DATA_TYPE)
        | env_is_true(FMT_TABLE_HIDE_COLUMN_NAMES)
    {
        format!("{}", f.data_type())
    } else {
        format!("\n{}", f.data_type())
    };
    let mut dtype_length = column_data_type.trim_start().len();
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
        let inline_name_dtype = format!("{column_name} ({column_data_type})");
        dtype_length = inline_name_dtype.len();
        inline_name_dtype
    } else {
        format!("{column_name}{separator}{column_data_type}")
    };
    let mut s_len = std::cmp::max(name_length, dtype_length);
    let separator_length = separator.trim().len();
    if s_len < separator_length {
        s_len = separator_length;
    }
    (s, s_len + 2)
}

#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
fn prepare_row(
    row: Vec<Cow<'_, str>>,
    n_first: usize,
    n_last: usize,
    str_truncate: usize,
    max_elem_lengths: &mut [usize],
) -> Vec<String> {
    let reduce_columns = n_first + n_last < row.len();
    let n_elems = n_first + n_last + reduce_columns as usize;
    let mut row_strings = Vec::with_capacity(n_elems);

    for (idx, v) in row[0..n_first].iter().enumerate() {
        let elem_str = make_str_val(v, str_truncate);
        let elem_len = elem_str.len() + 2;
        if max_elem_lengths[idx] < elem_len {
            max_elem_lengths[idx] = elem_len;
        };
        row_strings.push(elem_str);
    }
    if reduce_columns {
        row_strings.push("…".to_string());
        max_elem_lengths[n_first] = 3;
    }
    let elem_offset = n_first + reduce_columns as usize;
    for (idx, v) in row[row.len() - n_last..].iter().enumerate() {
        let elem_str = make_str_val(v, str_truncate);
        let elem_len = elem_str.len() + 2;
        let elem_idx = elem_offset + idx;
        if max_elem_lengths[elem_idx] < elem_len {
            max_elem_lengths[elem_idx] = elem_len;
        };
        row_strings.push(elem_str);
    }
    row_strings
}

fn env_is_true(varname: &str) -> bool {
    std::env::var(varname).as_deref().unwrap_or("0") == "1"
}

fn fmt_uint(num: &usize) -> String {
    // Return a string with thousands separated by _
    // e.g. 1_000_000
    num.to_string()
        .as_bytes()
        .rchunks(3)
        .rev()
        .map(str::from_utf8)
        .collect::<Result<Vec<&str>, _>>()
        .unwrap()
        .join("_") // separator
}

fn fmt_df_shape((shape0, shape1): &(usize, usize)) -> String {
    // e.g. (1_000_000, 4_000)
    format!("({}, {})", fmt_uint(shape0), fmt_uint(shape1))
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
            let str_truncate = std::env::var(FMT_STR_LEN)
                .as_deref()
                .unwrap_or("")
                .parse()
                .unwrap_or(32);

            let max_n_cols = std::env::var(FMT_MAX_COLS)
                .as_deref()
                .unwrap_or("")
                .parse()
                .map_or(8, |n: i64| if n < 0 { self.width() } else { n as usize });

            let max_n_rows = std::env::var(FMT_MAX_ROWS)
                .as_deref()
                .unwrap_or("")
                .parse()
                .map_or(8, |n: i64| if n < 0 { height } else { n as usize });

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
                let (s, l) = field_to_str(field, str_truncate);
                names.push(s);
                name_lengths.push(l);
            }
            if reduce_columns {
                names.push("…".into());
                name_lengths.push(3);
            }
            for field in fields[self.width() - n_last..].iter() {
                let (s, l) = field_to_str(field, str_truncate);
                names.push(s);
                name_lengths.push(l);
            }
            let (preset, is_utf8) = match std::env::var(FMT_TABLE_FORMATTING)
                .as_deref()
                .unwrap_or("DEFAULT")
            {
                "ASCII_FULL" => (ASCII_FULL, false),
                "ASCII_FULL_CONDENSED" => (ASCII_FULL_CONDENSED, false),
                "ASCII_NO_BORDERS" => (ASCII_NO_BORDERS, false),
                "ASCII_BORDERS_ONLY" => (ASCII_BORDERS_ONLY, false),
                "ASCII_BORDERS_ONLY_CONDENSED" => (ASCII_BORDERS_ONLY_CONDENSED, false),
                "ASCII_HORIZONTAL_ONLY" => (ASCII_HORIZONTAL_ONLY, false),
                "ASCII_MARKDOWN" => (ASCII_MARKDOWN, false),
                "UTF8_FULL" => (UTF8_FULL, true),
                "UTF8_FULL_CONDENSED" => (UTF8_FULL_CONDENSED, true),
                "UTF8_NO_BORDERS" => (UTF8_NO_BORDERS, true),
                "UTF8_BORDERS_ONLY" => (UTF8_BORDERS_ONLY, true),
                "UTF8_HORIZONTAL_ONLY" => (UTF8_HORIZONTAL_ONLY, true),
                "NOTHING" => (NOTHING, false),
                "DEFAULT" => (UTF8_FULL_CONDENSED, true),
                _ => (UTF8_FULL_CONDENSED, true),
            };

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
                if height > max_n_rows + 1 {
                    // Truncate the table if we have more rows than the configured maximum
                    // number of rows plus the single row which would contain "…".
                    let mut rows = Vec::with_capacity(std::cmp::max(max_n_rows, 2));
                    for i in 0..std::cmp::max(max_n_rows / 2, 1) {
                        let row = self
                            .columns
                            .iter()
                            .map(|s| s.str_value(i).unwrap())
                            .collect();

                        let row_strings =
                            prepare_row(row, n_first, n_last, str_truncate, &mut max_elem_lengths);

                        rows.push(row_strings);
                    }
                    let dots = rows[0].iter().map(|_| "…".to_string()).collect();
                    rows.push(dots);
                    if max_n_rows > 1 {
                        for i in (height - (max_n_rows + 1) / 2)..height {
                            let row = self
                                .columns
                                .iter()
                                .map(|s| s.str_value(i).unwrap())
                                .collect();

                            let row_strings = prepare_row(
                                row,
                                n_first,
                                n_last,
                                str_truncate,
                                &mut max_elem_lengths,
                            );
                            rows.push(row_strings);
                        }
                    }
                    table.add_rows(rows);
                } else {
                    for i in 0..height {
                        if self.width() > 0 {
                            let row = self
                                .columns
                                .iter()
                                .map(|s| s.str_value(i).unwrap())
                                .collect();

                            let row_strings = prepare_row(
                                row,
                                n_first,
                                n_last,
                                str_truncate,
                                &mut max_elem_lengths,
                            );
                            table.add_row(row_strings);
                        } else {
                            break;
                        }
                    }
                }
            } else if height > 0 {
                let dots: Vec<String> = self.columns.iter().map(|_| "…".to_string()).collect();
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
            let min_col_width = 5;
            for (idx, elem_len) in max_elem_lengths.iter().enumerate() {
                let mx = std::cmp::min(
                    str_truncate + 3, // (3 = 2 space chars of padding + ellipsis char)
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
                    let dtype = fields[column_index].data_type();
                    let mut preset = str_preset.as_str();
                    if dtype.is_numeric() {
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

fn fmt_integer<T: Num + NumCast + Display>(
    f: &mut Formatter<'_>,
    width: usize,
    v: T,
) -> fmt::Result {
    write!(f, "{v:>width$}")
}

const SCIENTIFIC_BOUND: f64 = 999999.0;

fn fmt_float<T: Num + NumCast>(f: &mut Formatter<'_>, width: usize, v: T) -> fmt::Result {
    let v: f64 = NumCast::from(v).unwrap();

    let float_precision = get_float_precision();

    if let Some(precision) = float_precision {
        if format!("{v:.precision$}", precision = precision).len() > 19 {
            return write!(f, "{v:>width$.precision$e}", precision = precision);
        }
        return write!(f, "{v:>width$.precision$}", precision = precision);
    }

    if matches!(get_float_fmt(), FloatFmt::Full) {
        return write!(f, "{v:>width$}");
    }

    // show integers as 0.0, 1.0 ... 101.0
    if v.fract() == 0.0 && v.abs() < SCIENTIFIC_BOUND {
        write!(f, "{v:>width$.1}")
    } else if format!("{v}").len() > 9 {
        // large and small floats in scientific notation
        if !(0.000001..=SCIENTIFIC_BOUND).contains(&v.abs()) | (v.abs() > SCIENTIFIC_BOUND) {
            write!(f, "{v:>width$.4e}")
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
                if s.ends_with('.') {
                    write!(f, "{s}0")
                } else {
                    write!(f, "{s}")
                }
            } else {
                // 12.0934509341243124
                // written as
                // 12.09345
                write!(f, "{v:>width$.6}")
            }
        }
    } else if v.fract() == 0.0 {
        write!(f, "{v:>width$e}")
    } else {
        write!(f, "{v:>width$}")
    }
}

const SIZES_NS: [i64; 4] = [
    86_400_000_000_000,
    3_600_000_000_000,
    60_000_000_000,
    1_000_000_000,
];
const NAMES: [&str; 4] = ["d", "h", "m", "s"];
const SIZES_US: [i64; 4] = [86_400_000_000, 3_600_000_000, 60_000_000, 1_000_000];
const SIZES_MS: [i64; 4] = [86_400_000, 3_600_000, 60_000, 1_000];

fn fmt_duration_ns(f: &mut Formatter<'_>, v: i64) -> fmt::Result {
    if v == 0 {
        return write!(f, "0ns");
    }
    format_duration(f, v, SIZES_NS.as_slice(), NAMES.as_slice())?;
    if v % 1000 != 0 {
        write!(f, "{}ns", v % 1_000_000_000)?;
    } else if v % 1_000_000 != 0 {
        write!(f, "{}µs", (v % 1_000_000_000) / 1000)?;
    } else if v % 1_000_000_000 != 0 {
        write!(f, "{}ms", (v % 1_000_000_000) / 1_000_000)?;
    }
    Ok(())
}

fn fmt_duration_us(f: &mut Formatter<'_>, v: i64) -> fmt::Result {
    if v == 0 {
        return write!(f, "0µs");
    }
    format_duration(f, v, SIZES_US.as_slice(), NAMES.as_slice())?;
    if v % 1000 != 0 {
        write!(f, "{}µs", (v % 1_000_000))?;
    } else if v % 1_000_000 != 0 {
        write!(f, "{}ms", (v % 1_000_000) / 1_000)?;
    }
    Ok(())
}

fn fmt_duration_ms(f: &mut Formatter<'_>, v: i64) -> fmt::Result {
    if v == 0 {
        return write!(f, "0ms");
    }
    format_duration(f, v, SIZES_MS.as_slice(), NAMES.as_slice())?;
    if v % 1_000 != 0 {
        write!(f, "{}ms", (v % 1_000))?;
    }
    Ok(())
}

fn format_duration(f: &mut Formatter, v: i64, sizes: &[i64], names: &[&str]) -> fmt::Result {
    for i in 0..4 {
        let whole_num = if i == 0 {
            v / sizes[i]
        } else {
            (v % sizes[i - 1]) / sizes[i]
        };
        if whole_num <= -1 || whole_num >= 1 {
            write!(f, "{}{}", whole_num, names[i])?;
            if v % sizes[i] != 0 {
                write!(f, " ")?;
            }
        }
    }
    Ok(())
}

impl Display for AnyValue<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let width = 0;
        match self {
            AnyValue::Null => write!(f, "null"),
            AnyValue::UInt8(v) => write!(f, "{v}"),
            AnyValue::UInt16(v) => write!(f, "{v}"),
            AnyValue::UInt32(v) => write!(f, "{v}"),
            AnyValue::UInt64(v) => write!(f, "{v}"),
            AnyValue::Int8(v) => fmt_integer(f, width, *v),
            AnyValue::Int16(v) => fmt_integer(f, width, *v),
            AnyValue::Int32(v) => fmt_integer(f, width, *v),
            AnyValue::Int64(v) => fmt_integer(f, width, *v),
            AnyValue::Float32(v) => fmt_float(f, width, *v),
            AnyValue::Float64(v) => fmt_float(f, width, *v),
            AnyValue::Boolean(v) => write!(f, "{}", *v),
            AnyValue::Utf8(v) => write!(f, "{}", format_args!("\"{v}\"")),
            AnyValue::Utf8Owned(v) => write!(f, "{}", format_args!("\"{v}\"")),
            AnyValue::Binary(_) | AnyValue::BinaryOwned(_) => write!(f, "[binary data]"),
            #[cfg(feature = "dtype-date")]
            AnyValue::Date(v) => write!(f, "{}", date32_to_date(*v)),
            #[cfg(feature = "dtype-datetime")]
            AnyValue::Datetime(v, tu, tz) => {
                let ndt = match tu {
                    TimeUnit::Nanoseconds => timestamp_ns_to_datetime(*v),
                    TimeUnit::Microseconds => timestamp_us_to_datetime(*v),
                    TimeUnit::Milliseconds => timestamp_ms_to_datetime(*v),
                };
                match tz {
                    None => write!(f, "{ndt}"),
                    Some(tz) => {
                        write!(f, "{}", PlTzAware::new(ndt, tz))
                    },
                }
            },
            #[cfg(feature = "dtype-duration")]
            AnyValue::Duration(v, tu) => match tu {
                TimeUnit::Nanoseconds => fmt_duration_ns(f, *v),
                TimeUnit::Microseconds => fmt_duration_us(f, *v),
                TimeUnit::Milliseconds => fmt_duration_ms(f, *v),
            },
            #[cfg(feature = "dtype-time")]
            AnyValue::Time(_) => {
                let nt: chrono::NaiveTime = self.into();
                write!(f, "{nt}")
            },
            #[cfg(feature = "dtype-categorical")]
            AnyValue::Categorical(_, _, _) => {
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

        let max_items = std::env::var(FMT_TABLE_CELL_LIST_LEN)
            .as_deref()
            .unwrap_or("")
            .parse()
            .map_or(3, |n: i64| if n < 0 { self.len() } else { n as usize });

        match max_items {
            0 => "[…]".to_owned(),
            _ if max_items >= self.len() => {
                let mut result = "[".to_owned();

                for i in 0..self.len() {
                    let item = self.get(i).unwrap();
                    write!(result, "{item}").unwrap();
                    // this will always leave a trailing ", " after the last item
                    // but for long lists, this is faster than checking against the length each time
                    result.push_str(", ");
                }
                // remove trailing ", " and replace with closing brace
                result.pop();
                result.pop();
                result.push(']');

                result
            },
            _ => {
                let mut result = "[".to_owned();

                for (i, item) in self.iter().enumerate() {
                    if i == max_items.saturating_sub(1) {
                        result.push_str("… ");
                        write!(result, "{}", self.get(self.len() - 1).unwrap()).unwrap();
                        break;
                    } else {
                        write!(result, "{item}").unwrap();
                        result.push_str(", ");
                    }
                }
                result.push(']');

                result
            },
        }
    }
}

#[cfg(feature = "dtype-decimal")]
mod decimal {
    use std::fmt::Formatter;
    use std::{fmt, ptr, str};

    const BUF_LEN: usize = 48;

    #[derive(Clone, Copy)]
    pub struct FormatBuffer {
        data: [u8; BUF_LEN],
        len: usize,
    }

    impl FormatBuffer {
        #[inline]
        pub const fn new() -> Self {
            Self {
                data: [0; BUF_LEN],
                len: 0,
            }
        }

        #[inline]
        pub fn as_str(&self) -> &str {
            unsafe { str::from_utf8_unchecked(&self.data[..self.len]) }
        }
    }

    const POW10: [i128; 38] = [
        1,
        10,
        100,
        1000,
        10000,
        100000,
        1000000,
        10000000,
        100000000,
        1000000000,
        10000000000,
        100000000000,
        1000000000000,
        10000000000000,
        100000000000000,
        1000000000000000,
        10000000000000000,
        100000000000000000,
        1000000000000000000,
        10000000000000000000,
        100000000000000000000,
        1000000000000000000000,
        10000000000000000000000,
        100000000000000000000000,
        1000000000000000000000000,
        10000000000000000000000000,
        100000000000000000000000000,
        1000000000000000000000000000,
        10000000000000000000000000000,
        100000000000000000000000000000,
        1000000000000000000000000000000,
        10000000000000000000000000000000,
        100000000000000000000000000000000,
        1000000000000000000000000000000000,
        10000000000000000000000000000000000,
        100000000000000000000000000000000000,
        1000000000000000000000000000000000000,
        10000000000000000000000000000000000000,
    ];

    pub fn format_decimal(v: i128, scale: usize, trim_zeros: bool) -> FormatBuffer {
        const ZEROS: [u8; BUF_LEN] = [b'0'; BUF_LEN];

        let mut buf = FormatBuffer::new();
        let factor = POW10[scale]; //10_i128.pow(scale as _);
        let (div, rem) = (v / factor, v.abs() % factor);

        unsafe {
            let mut ptr = buf.data.as_mut_ptr();
            if div == 0 && v < 0 {
                *ptr = b'-';
                ptr = ptr.add(1);
                buf.len = 1;
            }
            let n_whole = itoap::write_to_ptr(ptr, div);
            buf.len += n_whole;
            if rem != 0 {
                ptr = ptr.add(n_whole);
                *ptr = b'.';
                ptr = ptr.add(1);
                let mut frac_buf = [0_u8; BUF_LEN];
                let n_frac = itoap::write_to_ptr(frac_buf.as_mut_ptr(), rem);
                ptr::copy_nonoverlapping(ZEROS.as_ptr(), ptr, scale - n_frac);
                ptr = ptr.add(scale - n_frac);
                ptr::copy_nonoverlapping(frac_buf.as_mut_ptr(), ptr, n_frac);
                buf.len += 1 + scale;
                if trim_zeros {
                    ptr = ptr.add(n_frac - 1);
                    while *ptr == b'0' {
                        ptr = ptr.sub(1);
                        buf.len -= 1;
                    }
                }
            }
        }

        buf
    }

    #[inline]
    pub fn fmt_decimal(f: &mut Formatter<'_>, v: i128, scale: usize) -> fmt::Result {
        f.write_str(format_decimal(v, scale, !f.alternate()).as_str())
    }
}

#[cfg(feature = "dtype-decimal")]
pub use decimal::fmt_decimal;

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
        let mut builder =
            ListPrimitiveChunkedBuilder::<Int32Type>::new("a", 10, 10, DataType::Int32);
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

        let mut builder =
            ListPrimitiveChunkedBuilder::<Int32Type>::new("a", 10, 10, DataType::Int32);
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

        let mut builder =
            ListPrimitiveChunkedBuilder::<Int32Type>::new("a", 10, 10, DataType::Int32);
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
        let s = Int32Chunked::new("Date", &[Some(1), None, Some(3)]).into_date();
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

        let s = Int64Chunked::new("", &[Some(1), None, Some(1_000_000_000_000)])
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
        let ca = Int32Chunked::new("Date", &[Some(1), None, Some(3)]);
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
        let ca = Utf8Chunked::new("name", &["a", "b"]);
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
