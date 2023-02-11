#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
use std::borrow::Cow;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::sync::atomic::{AtomicU8, Ordering};

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
use num::{Num, NumCast};

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

fn get_float_fmt() -> FloatFmt {
    match FLOAT_FMT.load(Ordering::Relaxed) {
        0 => FloatFmt::Mixed,
        1 => FloatFmt::Full,
        _ => panic!(),
    }
}

pub fn set_float_fmt(fmt: FloatFmt) {
    FLOAT_FMT.store(fmt as u8, Ordering::Relaxed)
}

macro_rules! format_array {
    ($f:ident, $a:expr, $dtype:expr, $name:expr, $array_type:expr) => {{
        write!(
            $f,
            "shape: ({},)\n{}: '{}' [{}]\n[\n",
            $a.len(),
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
        let write_fn = |v, f: &mut Formatter| {
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
                    write!(f, "\t{}...\n", v_trunc)?;
                }
            } else {
                write!(f, "\t{}\n", v)?;
            };
            Ok(())
        };
        if limit < $a.len() {
            if limit > 0 {
                for i in 0..std::cmp::max((limit / 2), 1) {
                    let v = $a.get_any_value(i).unwrap();
                    write_fn(v, $f)?;
                }
            }
            write!($f, "\t...\n")?;
            if limit > 1 {
                for i in ($a.len() - (limit + 1) / 2)..$a.len() {
                    let v = $a.get_any_value(i).unwrap();
                    write_fn(v, $f)?;
                }
            }
        } else {
            for i in 0..limit {
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
                object.len(),
                array_type,
                name,
                inner_type
            )?;

            for i in 0..limit {
                let v = object.str_value(i);
                writeln!(f, "\t{}", v.unwrap())?;
            }

            write!(f, "]")
        }
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

#[cfg(feature = "dtype-binary")]
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

#[cfg(feature = "object")]
impl<T> Debug for ObjectChunked<T>
where
    T: PolarsObject,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let limit = std::cmp::min(LIMIT, self.len());
        let taker = self.take_rand();
        let inner_type = T::type_name();
        write!(
            f,
            "ChunkedArray: '{}' [o][{}]\n[\n",
            self.name(),
            inner_type
        )?;

        if limit < self.len() {
            for i in 0..limit / 2 {
                match taker.get(i) {
                    None => writeln!(f, "\tnull")?,
                    Some(val) => writeln!(f, "\t{val}")?,
                };
            }
            writeln!(f, "\t...")?;
            for i in (0..limit / 2).rev() {
                match taker.get(self.len() - i - 1) {
                    None => writeln!(f, "\tnull")?,
                    Some(val) => writeln!(f, "\t{val}")?,
                };
            }
        } else {
            for i in 0..limit {
                match taker.get(i) {
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
            }
            DataType::Utf8 => {
                format_array!(f, self.utf8().unwrap(), "str", self.name(), "Series")
            }
            DataType::UInt8 => {
                format_array!(f, self.u8().unwrap(), "u8", self.name(), "Series")
            }
            DataType::UInt16 => {
                format_array!(f, self.u16().unwrap(), "u16", self.name(), "Series")
            }
            DataType::UInt32 => {
                format_array!(f, self.u32().unwrap(), "u32", self.name(), "Series")
            }
            DataType::UInt64 => {
                format_array!(f, self.u64().unwrap(), "u64", self.name(), "Series")
            }
            DataType::Int8 => {
                format_array!(f, self.i8().unwrap(), "i8", self.name(), "Series")
            }
            DataType::Int16 => {
                format_array!(f, self.i16().unwrap(), "i16", self.name(), "Series")
            }
            DataType::Int32 => {
                format_array!(f, self.i32().unwrap(), "i32", self.name(), "Series")
            }
            DataType::Int64 => {
                format_array!(f, self.i64().unwrap(), "i64", self.name(), "Series")
            }
            DataType::Float32 => {
                format_array!(f, self.f32().unwrap(), "f32", self.name(), "Series")
            }
            DataType::Float64 => {
                format_array!(f, self.f64().unwrap(), "f64", self.name(), "Series")
            }
            #[cfg(feature = "dtype-date")]
            DataType::Date => format_array!(f, self.date().unwrap(), "date", self.name(), "Series"),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => {
                let dt = format!("{}", self.dtype());
                format_array!(f, self.datetime().unwrap(), &dt, self.name(), "Series")
            }
            #[cfg(feature = "dtype-time")]
            DataType::Time => format_array!(f, self.time().unwrap(), "time", self.name(), "Series"),
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(_) => {
                let dt = format!("{}", self.dtype());
                format_array!(f, self.duration().unwrap(), &dt, self.name(), "Series")
            }
            DataType::List(_) => {
                let dt = format!("{}", self.dtype());
                format_array!(f, self.list().unwrap(), &dt, self.name(), "Series")
            }
            #[cfg(feature = "object")]
            DataType::Object(_) => format_object_array(f, self, self.name(), "Series"),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => {
                format_array!(f, self.categorical().unwrap(), "cat", self.name(), "Series")
            }
            #[cfg(feature = "dtype-struct")]
            dt @ DataType::Struct(_) => format_array!(
                f,
                self.struct_().unwrap(),
                format!("{dt}"),
                self.name(),
                "Series"
            ),
            DataType::Null => {
                writeln!(f, "nullarray")
            }
            #[cfg(feature = "dtype-binary")]
            DataType::Binary => {
                format_array!(f, self.binary().unwrap(), "binary", self.name(), "Series")
            }
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
        format!("{v_trunc}...")
    }
}

#[cfg(any(feature = "fmt", feature = "fmt_no_tty"))]
fn prepare_row(
    row: Vec<Cow<'_, str>>,
    n_first: usize,
    n_last: usize,
    str_truncate: usize,
) -> Vec<String> {
    let reduce_columns = n_first + n_last < row.len();
    let mut row_str = Vec::with_capacity(n_first + n_last + reduce_columns as usize);
    for v in row[0..n_first].iter() {
        row_str.push(make_str_val(v, str_truncate));
    }
    if reduce_columns {
        row_str.push("...".to_string());
    }
    for v in row[row.len() - n_last..].iter() {
        row_str.push(make_str_val(v, str_truncate));
    }
    row_str
}

fn env_is_true(varname: &str) -> bool {
    std::env::var(varname).as_deref().unwrap_or("0") == "1"
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
            let mut names = Vec::with_capacity(n_first + n_last + reduce_columns as usize);

            let field_to_str = |f: &Field| {
                let name = make_str_val(f.name(), str_truncate);
                let lower_bounds = name.len().clamp(5, 12);
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
                let mut column_separator = "\n---";
                if env_is_true(FMT_TABLE_HIDE_COLUMN_SEPARATOR)
                    | env_is_true(FMT_TABLE_HIDE_COLUMN_NAMES)
                    | env_is_true(FMT_TABLE_HIDE_COLUMN_DATA_TYPES)
                {
                    column_separator = ""
                }
                let s = if env_is_true(FMT_TABLE_INLINE_COLUMN_DATA_TYPE)
                    & !env_is_true(FMT_TABLE_HIDE_COLUMN_DATA_TYPES)
                {
                    format!("{column_name} ({column_data_type})")
                } else {
                    format!("{column_name}{column_separator}{column_data_type}")
                };
                (s, lower_bounds)
            };
            let tbl_lower_bounds =
                |l: usize| ColumnConstraint::LowerBoundary(comfy_table::Width::Fixed(l as u16));

            let mut constraints = Vec::with_capacity(n_first + n_last + reduce_columns as usize);
            let fields = self.fields();
            for field in fields[0..n_first].iter() {
                let (s, l) = field_to_str(field);
                names.push(s);
                constraints.push(tbl_lower_bounds(l));
            }
            if reduce_columns {
                names.push("...".into());
                constraints.push(tbl_lower_bounds(5));
            }
            for field in fields[self.width() - n_last..].iter() {
                let (s, l) = field_to_str(field);
                names.push(s);
                constraints.push(tbl_lower_bounds(l));
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
            if max_n_rows > 0 {
                if height > max_n_rows {
                    let mut rows = Vec::with_capacity(std::cmp::max(max_n_rows, 2));
                    for i in 0..std::cmp::max(max_n_rows / 2, 1) {
                        let row = self
                            .columns
                            .iter()
                            .map(|s| s.str_value(i).unwrap())
                            .collect();
                        rows.push(prepare_row(row, n_first, n_last, str_truncate));
                    }
                    let dots = rows[0].iter().map(|_| "...".to_string()).collect();
                    rows.push(dots);
                    if max_n_rows > 1 {
                        for i in (height - (max_n_rows + 1) / 2)..height {
                            let row = self
                                .columns
                                .iter()
                                .map(|s| s.str_value(i).unwrap())
                                .collect();
                            rows.push(prepare_row(row, n_first, n_last, str_truncate));
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
                            table.add_row(prepare_row(row, n_first, n_last, str_truncate));
                        } else {
                            break;
                        }
                    }
                }
            } else if height > 0 {
                let dots: Vec<String> = self.columns.iter().map(|_| "...".to_string()).collect();
                table.add_row(dots);
            }

            // insert a header row, unless both column names and column data types are already hidden
            if !(env_is_true(FMT_TABLE_HIDE_COLUMN_NAMES)
                && env_is_true(FMT_TABLE_HIDE_COLUMN_DATA_TYPES))
            {
                table.set_header(names).set_constraints(constraints);
            }
            let tbl_width = std::env::var("POLARS_TABLE_WIDTH")
                .map(|s| {
                    Some(
                        s.parse::<u16>()
                            .expect("could not parse table width argument"),
                    )
                })
                .unwrap_or(None);

            // if tbl_width is explicitly set, use it
            if let Some(w) = tbl_width {
                table.set_width(w);
            }

            // if no tbl_width (its not-tty && it is not explicitly set), then set default.
            // this is needed to support non-tty applications
            #[cfg(feature = "fmt")]
            if table.width().is_none() && !table.is_tty() {
                table.set_width(100);
            }
            #[cfg(feature = "fmt_no_tty")]
            if table.width().is_none() {
                table.set_width(100);
            }

            // set alignment of cells, if defined
            if std::env::var(FMT_TABLE_CELL_ALIGNMENT).is_ok() {
                // for (column_index, column) in table.column_iter_mut().enumerate() {
                let str_preset = std::env::var(FMT_TABLE_CELL_ALIGNMENT)
                    .unwrap_or_else(|_| "DEFAULT".to_string());
                for column in table.column_iter_mut() {
                    if str_preset == "RIGHT" {
                        column.set_cell_alignment(CellAlignment::Right);
                    } else if str_preset == "LEFT" {
                        column.set_cell_alignment(CellAlignment::Left);
                    } else if str_preset == "CENTER" {
                        column.set_cell_alignment(CellAlignment::Center);
                    } else {
                        column.set_cell_alignment(CellAlignment::Left);
                    }
                }
            }

            // establish 'shape' information (above/below/hidden)
            if env_is_true(FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION) {
                write!(f, "{table}")?;
            } else if env_is_true(FMT_TABLE_DATAFRAME_SHAPE_BELOW) {
                write!(f, "{table}\nshape: {:?}", self.shape())?;
            } else {
                write!(f, "shape: {:?}\n{}", self.shape(), table)?;
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
            #[cfg(feature = "dtype-binary")]
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
                    }
                }
            }
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
            }
            #[cfg(feature = "dtype-categorical")]
            AnyValue::Categorical(_, _, _) => {
                let s = self.get_str().unwrap();
                write!(f, "\"{s}\"")
            }
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
            }
            #[cfg(feature = "dtype-struct")]
            AnyValue::StructOwned(payload) => fmt_struct(f, &payload.0),
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
            }
            Err(_) => match parse_offset(self.tz) {
                Ok(offset) => {
                    let dt_tz_aware = offset.from_utc_datetime(&self.ndt);
                    write!(f, "{dt_tz_aware}")
                }
                Err(_) => write!(f, "invalid timezone"),
            },
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

macro_rules! impl_fmt_list {
    ($self:ident) => {{
        match $self.len() {
            0 => format!("[]"),
            1 => format!("[{}]", $self.get_any_value(0).unwrap()),
            2 => format!(
                "[{}, {}]",
                $self.get_any_value(0).unwrap(),
                $self.get_any_value(1).unwrap()
            ),
            3 => format!(
                "[{}, {}, {}]",
                $self.get_any_value(0).unwrap(),
                $self.get_any_value(1).unwrap(),
                $self.get_any_value(2).unwrap()
            ),
            _ => format!(
                "[{}, {}, ... {}]",
                $self.get_any_value(0).unwrap(),
                $self.get_any_value(1).unwrap(),
                $self.get_any_value($self.len() - 1).unwrap()
            ),
        }
    }};
}

pub(crate) trait FmtList {
    fn fmt_list(&self) -> String;
}

impl<T> FmtList for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: fmt::Display,
{
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

impl FmtList for BooleanChunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

impl FmtList for Utf8Chunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

#[cfg(feature = "dtype-binary")]
impl FmtList for BinaryChunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

impl FmtList for ListChunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

#[cfg(feature = "dtype-categorical")]
impl FmtList for CategoricalChunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

#[cfg(feature = "dtype-date")]
impl FmtList for DateChunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

#[cfg(feature = "dtype-datetime")]
impl FmtList for DatetimeChunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

#[cfg(feature = "dtype-duration")]
impl FmtList for DurationChunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

#[cfg(feature = "dtype-time")]
impl FmtList for TimeChunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

#[cfg(feature = "dtype-struct")]
impl FmtList for StructChunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> FmtList for ObjectChunked<T> {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
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
        let mut builder =
            ListPrimitiveChunkedBuilder::<Int32Type>::new("a", 10, 10, DataType::Int32);
        builder.append_opt_slice(Some(&[1, 2, 3]));
        builder.append_opt_slice(None);
        let list = builder.finish().into_series();

        assert_eq!(
            r#"shape: (2,)
Series: 'a' [list[i32]]
[
	[1, 2, 3]
	null
]"#,
            format!("{:?}", list)
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
